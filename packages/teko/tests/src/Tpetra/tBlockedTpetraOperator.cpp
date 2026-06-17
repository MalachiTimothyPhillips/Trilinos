// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

// Thyra testing tools
#include "Thyra_TestingTools.hpp"
#include "Thyra_LinearOpTester.hpp"

// Thyra includes
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_DefaultBlockedLinearOp.hpp"
#include "Thyra_ProductVectorBase.hpp"
#include "Thyra_SpmdVectorSpaceBase.hpp"
#include "Thyra_DetachedSpmdVectorView.hpp"

// Teuchos includes
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

// Tpetra includes
#include "Tpetra_Core.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_CrsMatrix.hpp"

// Galeri / Xpetra
#include "Galeri_XpetraMaps.hpp"
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraParameters.hpp"

#include "Xpetra_Map.hpp"
#include "Xpetra_Matrix.hpp"
#include "Xpetra_TpetraMap.hpp"
#include "Xpetra_TpetraCrsMatrix.hpp"

#include "tBlockedTpetraOperator.hpp"

#include "Teko_BlockedTpetraOperator.hpp"
#include "Teko_TpetraHelpers.hpp"
#include "Teko_ConfigDefs.hpp"

#include <random>

namespace Teko::Test {

using Teuchos::null;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Thyra::createMember;
using Thyra::LinearOpBase;
using Thyra::LinearOpTester;
using Thyra::VectorBase;

namespace {

using ST = Teko::ST;
using LO = Teko::LO;
using GO = Teko::GO;
using NT = Teko::NT;

using map_t = Tpetra::Map<LO, GO, NT>;
using crs_t = Tpetra::CrsMatrix<ST, LO, GO, NT>;
using mv_t  = Tpetra::MultiVector<ST, LO, GO, NT>;

RCP<crs_t> buildRecirc2DMatrix(const RCP<const Teuchos::Comm<int>>& comm, GO nx, GO ny) {
  Teuchos::ParameterList galeriList;
  galeriList.set("nx", nx);
  galeriList.set("ny", ny);
  galeriList.set("mx", comm->getSize());
  galeriList.set("my", 1);
  galeriList.set("lx", 1.0);
  galeriList.set("ly", 1.0);
  galeriList.set("diff", 1e-5);
  galeriList.set("conv", 1.0);

  RCP<const Xpetra::Map<LO, GO, NT>> xMap =
      Galeri::Xpetra::CreateMap<LO, GO, NT>(Xpetra::UseTpetra, "Cartesian2D", comm, galeriList);

  RCP<const map_t> tMap = Xpetra::toTpetra(xMap);

  auto problem =
      Galeri::Xpetra::BuildProblem<ST, LO, GO, map_t, crs_t, mv_t>("Recirc2D", tMap, galeriList);

  return problem->BuildMatrix();
}

ST relativeError(const mv_t& a, const mv_t& b) {
  mv_t diff(a, Teuchos::Copy);
  diff.update(ST(-1.0), b, ST(1.0));

  Teuchos::Array<ST> diffNorms(diff.getNumVectors());
  Teuchos::Array<ST> trueNorms(b.getNumVectors());

  diff.norm2(diffNorms());
  b.norm2(trueNorms());

  ST maxRel = ST(0.0);
  for (size_t i = 0; i < size_t(diffNorms.size()); ++i) {
    ST rel = (trueNorms[i] == ST(0.0)) ? diffNorms[i] : diffNorms[i] / trueNorms[i];
    maxRel = std::max(maxRel, rel);
  }
  return maxRel;
}

std::vector<GO> random_gids(const RCP<const crs_t>& contigMat) {
  std::random_device rd;
  std::mt19937 g(rd());

  const auto contigRowMap          = contigMat->getRowMap();
  const auto numGlobalElements     = contigRowMap->getGlobalNumElements();
  const auto numLargestPossibleGid = 10 * numGlobalElements;
  std::vector<GO> randomGids(numLargestPossibleGid);
  std::iota(randomGids.begin(), randomGids.end(), 0);
  std::shuffle(randomGids.begin(), randomGids.end(), g);

  const auto numLocalGids = contigRowMap->getLocalNumElements();
  const auto gidStart     = contigRowMap->getMinGlobalIndex();
  const auto gidEnd       = contigRowMap->getMaxGlobalIndex();

  size_t numGids = gidEnd - gidStart + 1;
  TEUCHOS_ASSERT(numGids == numLocalGids);

  std::vector<GO> localRandomGids(numLocalGids);
  std::copy(randomGids.begin() + gidStart, randomGids.begin() + gidStart + numGids,
            localRandomGids.begin());

  return localRandomGids;
}

RCP<crs_t> assemble_noncontig_matrix(const RCP<const crs_t>& contigMat) {
  const auto contigRowMap = contigMat->getRowMap();
  const auto comm         = contigRowMap->getComm();

  auto randomGlobalGids = random_gids(contigMat);

  const auto indexBase = 0;
  const auto invalid   = Teuchos::OrdinalTraits<GO>::invalid();

  RCP<const map_t> noncontigRowMap = Teuchos::make_rcp<map_t>(
      invalid, Teuchos::ArrayView<const GO>(randomGlobalGids), indexBase, comm);

  auto nrowsLocal = contigRowMap->getLocalNumElements();
  std::vector<size_t> numEntPerRow(nrowsLocal);
  for (size_t row = 0; row < nrowsLocal; ++row) {
    numEntPerRow[row] = contigMat->getNumEntriesInLocalRow(row);
  }

  auto noncontigMat =
      Teuchos::make_rcp<crs_t>(noncontigRowMap, Teuchos::ArrayView<const size_t>(numEntPerRow));

  noncontigMat->resumeFill();

  const auto globalMaxNumRowEntries = contigMat->getGlobalMaxNumRowEntries();
  typename crs_t::nonconst_global_inds_host_view_type contigColumnIndices("contigColumnIndices",
                                                                          globalMaxNumRowEntries);
  typename crs_t::nonconst_global_inds_host_view_type noncontigColumnIndices(
      "noncontigColumnIndices", globalMaxNumRowEntries);
  typename crs_t::nonconst_values_host_view_type columnValues("columnValues",
                                                              globalMaxNumRowEntries);

  for (size_t row = 0; row < nrowsLocal; ++row) {
    const auto contigGlobalRow = contigRowMap->getGlobalElement(row);
    size_t numEntries          = Teuchos::OrdinalTraits<size_t>::invalid();
    contigMat->getGlobalRowCopy(contigGlobalRow, contigColumnIndices, columnValues, numEntries);

    for (size_t index = 0; index < numEntries; ++index) {
      auto localCol                 = contigRowMap->getLocalElement(contigColumnIndices(index));
      noncontigColumnIndices(index) = noncontigRowMap->getGlobalElement(localCol);
    }

    const auto noncontigGlobalRow = noncontigRowMap->getGlobalElement(row);
    noncontigMat->insertGlobalValues(
        noncontigGlobalRow, Teuchos::ArrayView<const GO>(noncontigColumnIndices.data(), numEntries),
        Teuchos::ArrayView<ST>(columnValues.data(), numEntries));
  }

  noncontigMat->fillComplete(noncontigRowMap, noncontigRowMap);

  return noncontigMat;
}

}  // namespace

void tBlockedTpetraOperator::buildBlockGIDs(std::vector<std::vector<GO>>& gids,
                                            const Tpetra::Map<LO, GO, NT>& map,
                                            bool singleBlock) const {
  LO numLocal = map.getLocalNumElements();
  LO numHalf  = numLocal / 2;
  numHalf += ((numHalf % 2 == 0) ? 0 : 1);

  gids.clear();
  gids.resize(singleBlock ? 1 : 3);

  std::vector<GO>& blk0 = gids[0];
  std::vector<GO>& blk1 = singleBlock ? gids[0] : gids[1];
  std::vector<GO>& blk2 = singleBlock ? gids[0] : gids[2];

  GO gid = -1;
  for (LO i = 0; i < numHalf; i += 2) {
    gid = map.getGlobalElement(i);
    blk0.push_back(gid);

    gid = map.getGlobalElement(i + 1);
    blk1.push_back(gid);
  }

  for (LO i = numHalf; i < numLocal; i++) {
    gid = map.getGlobalElement(i);
    blk2.push_back(gid);
  }

  TEUCHOS_ASSERT(LO(singleBlock ? blk0.size() : blk0.size() + blk1.size() + blk2.size()) ==
                 numLocal);
}

void tBlockedTpetraOperator::initializeTest() { tolerance_ = 1e-14; }

int tBlockedTpetraOperator::runTest(int verbosity, std::ostream& stdstrm, std::ostream& failstrm,
                                    int& totalrun) {
  bool allTests = true;
  bool status;
  int failcount = 0;

  failstrm << "tBlockedTpetraOperator";

  status = test_vector_constr(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"vector_constr\" ... PASSED",
                       "   \"vector_constr\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_single_block(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"test_single_block\" ... PASSED",
                       "   \"test_single_block\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_noncontig(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"test_noncontig\" ... PASSED",
                       "   \"test_noncontig\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_reorder(verbosity, failstrm, 0);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"reorder(flat reorder)\" ... PASSED",
                       "   \"reorder(flat reorder)\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_reorder(verbosity, failstrm, 1);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"reorder(composite reorder = " << 1 << ")\" ... PASSED",
                       "   \"reorder(composite reorder)\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_reorder(verbosity, failstrm, 2);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"reorder(composite reorder = " << 2 << ")\" ... PASSED",
                       "   \"reorder(composite reorder)\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = allTests;
  if (verbosity >= 10) {
    Teko_TEST_MSG_tpetra(failstrm, 0, "tBlockedTpetraOperator...PASSED",
                         "tBlockedTpetraOperator...FAILED");
  } else {
    Teko_TEST_MSG_tpetra(failstrm, 0, "...PASSED", "tBlockedTpetraOperator...FAILED");
  }

  return failcount;
}

bool tBlockedTpetraOperator::test_vector_constr(int verbosity, std::ostream& os) {
  bool status    = false;
  bool allPassed = true;

  RCP<const Teuchos::Comm<int>> comm_tpetra = GetComm_tpetra();

  TEST_MSG("\n   tBlockedTpetraOperator::test_vector_constr: "
           << "Running on " << comm_tpetra->getSize() << " processors");

  GO nx = 5;
  GO ny = 5;

  RCP<crs_t> A  = buildRecirc2DMatrix(comm_tpetra, nx, ny);
  ST beforeNorm = A->getFrobeniusNorm();

  int width = 3;
  mv_t x(A->getDomainMap(), width);
  mv_t ys(A->getRangeMap(), width);
  mv_t y(A->getRangeMap(), width);

  std::vector<std::vector<GO>> vars;
  buildBlockGIDs(vars, *A->getRowMap(), false);

  Teko::TpetraHelpers::BlockedTpetraOperator shell(vars, A);

  int numtests = 50;
  ST max       = 0.0;
  ST min       = 1.0;
  for (int i = 0; i < numtests; i++) {
    x.randomize();

    shell.apply(x, y);
    A->apply(x, ys);

    ST err = relativeError(y, ys);
    max    = std::max(max, err);
    min    = std::min(min, err);
  }
  TEST_ASSERT(max >= min, "\n   tBlockedTpetraOperator::test_vector_constr: "
                              << toString(status) << ": "
                              << "sanity checked - " << max << " >= " << min);
  TEST_ASSERT(max <= tolerance_, "\n   tBlockedTpetraOperator::test_vector_constr: "
                                     << toString(status) << ": "
                                     << "testing tolerance over many matrix vector multiplies ( "
                                     << max << " <= " << tolerance_ << " )");

  A->scale(2.0);

  ST afterNorm = A->getFrobeniusNorm();
  TEST_ASSERT(beforeNorm != afterNorm, "\n   tBlockedTpetraOperator::test_vector_constr "
                                           << toString(status) << ": "
                                           << "verify matrix has been modified");

  shell.RebuildOps();

  numtests = 50;
  max      = 0.0;
  min      = 1.0;
  for (int i = 0; i < numtests; i++) {
    x.randomize();

    shell.apply(x, y);
    A->apply(x, ys);

    ST err = relativeError(y, ys);
    max    = std::max(max, err);
    min    = std::min(min, err);
  }
  TEST_ASSERT(max >= min, "\n   tBlockedTpetraOperator::test_vector_constr (rebuild): "
                              << toString(status) << ": "
                              << "sanity checked - " << max << " >= " << min);
  TEST_ASSERT(max <= tolerance_, "\n   tBlockedTpetraOperator::test_vector_constr (rebuild): "
                                     << toString(status) << ": "
                                     << "testing tolerance over many matrix vector multiplies ( "
                                     << max << " <= " << tolerance_ << " )");

  return allPassed;
}

bool tBlockedTpetraOperator::test_single_block(int verbosity, std::ostream& os) {
  bool status    = false;
  bool allPassed = true;

  RCP<const Teuchos::Comm<int>> comm_tpetra = GetComm_tpetra();

  TEST_MSG("\n   tBlockedTpetraOperator::test_single_block: "
           << "Running on " << comm_tpetra->getSize() << " processors");

  GO nx = 5;
  GO ny = 5;

  RCP<crs_t> contigA = buildRecirc2DMatrix(comm_tpetra, nx, ny);
  auto A             = assemble_noncontig_matrix(contigA);

  ST beforeNorm = A->getFrobeniusNorm();

  int width = 3;
  mv_t x(A->getDomainMap(), width);
  mv_t ys(A->getRangeMap(), width);
  mv_t y(A->getRangeMap(), width);

  std::vector<std::vector<GO>> vars;
  buildBlockGIDs(vars, *A->getRowMap(), true);

  Teko::TpetraHelpers::BlockedTpetraOperator shell(vars, A);

  int numtests = 50;
  ST max       = 0.0;
  ST min       = 1.0;
  for (int i = 0; i < numtests; i++) {
    x.randomize();

    shell.apply(x, y);
    A->apply(x, ys);

    ST err = relativeError(y, ys);
    max    = std::max(max, err);
    min    = std::min(min, err);
  }
  TEST_ASSERT(max >= min, "\n   tBlockedTpetraOperator::test_single_block: "
                              << toString(status) << ": "
                              << "sanity checked - " << max << " >= " << min);
  TEST_ASSERT(max <= tolerance_, "\n   tBlockedTpetraOperator::test_single_block: "
                                     << toString(status) << ": "
                                     << "testing tolerance over many matrix vector multiplies ( "
                                     << max << " <= " << tolerance_ << " )");

  A->scale(2.0);

  ST afterNorm = A->getFrobeniusNorm();
  TEST_ASSERT(beforeNorm != afterNorm, "\n   tBlockedTpetraOperator::test_single_block "
                                           << toString(status) << ": "
                                           << "verify matrix has been modified");

  shell.RebuildOps();

  numtests = 50;
  max      = 0.0;
  min      = 1.0;
  for (int i = 0; i < numtests; i++) {
    x.randomize();

    shell.apply(x, y);
    A->apply(x, ys);

    ST err = relativeError(y, ys);
    max    = std::max(max, err);
    min    = std::min(min, err);
  }
  TEST_ASSERT(max >= min, "\n   tBlockedTpetraOperator::test_single_block (rebuild): "
                              << toString(status) << ": "
                              << "sanity checked - " << max << " >= " << min);
  TEST_ASSERT(max <= tolerance_, "\n   tBlockedTpetraOperator::test_single_block (rebuild): "
                                     << toString(status) << ": "
                                     << "testing tolerance over many matrix vector multiplies ( "
                                     << max << " <= " << tolerance_ << " )");

  return allPassed;
}

bool tBlockedTpetraOperator::test_noncontig(int verbosity, std::ostream& os) {
  bool status    = false;
  bool allPassed = true;

  RCP<const Teuchos::Comm<int>> comm_tpetra = GetComm_tpetra();

  TEST_MSG("\n   tBlockedTpetraOperator::test_noncontig: "
           << "Running on " << comm_tpetra->getSize() << " processors");

  GO nx = 15;
  GO ny = 15;

  RCP<crs_t> contigA = buildRecirc2DMatrix(comm_tpetra, nx, ny);
  auto A             = assemble_noncontig_matrix(contigA);

  ST beforeNorm = A->getFrobeniusNorm();

  int width = 3;
  mv_t x(A->getDomainMap(), width);
  mv_t ys(A->getRangeMap(), width);
  mv_t y(A->getRangeMap(), width);

  std::vector<std::vector<GO>> vars;
  buildBlockGIDs(vars, *A->getRowMap(), false);

  Teko::TpetraHelpers::BlockedTpetraOperator shell(vars, A);

  int numtests = 50;
  ST max       = 0.0;
  ST min       = 1.0;
  for (int i = 0; i < numtests; i++) {
    x.randomize();

    shell.apply(x, y);
    A->apply(x, ys);

    ST err = relativeError(y, ys);
    max    = std::max(max, err);
    min    = std::min(min, err);
  }
  TEST_ASSERT(max >= min, "\n   tBlockedTpetraOperator::test_noncontig: "
                              << toString(status) << ": "
                              << "sanity checked - " << max << " >= " << min);
  TEST_ASSERT(max <= tolerance_, "\n   tBlockedTpetraOperator::test_noncontig: "
                                     << toString(status) << ": "
                                     << "testing tolerance over many matrix vector multiplies ( "
                                     << max << " <= " << tolerance_ << " )");

  A->scale(2.0);

  ST afterNorm = A->getFrobeniusNorm();
  TEST_ASSERT(beforeNorm != afterNorm, "\n   tBlockedTpetraOperator::test_noncontig "
                                           << toString(status) << ": "
                                           << "verify matrix has been modified");

  shell.RebuildOps();

  numtests = 50;
  max      = 0.0;
  min      = 1.0;
  for (int i = 0; i < numtests; i++) {
    x.randomize();

    shell.apply(x, y);
    A->apply(x, ys);

    ST err = relativeError(y, ys);
    max    = std::max(max, err);
    min    = std::min(min, err);
  }
  TEST_ASSERT(max >= min, "\n   tBlockedTpetraOperator::test_noncontig (rebuild): "
                              << toString(status) << ": "
                              << "sanity checked - " << max << " >= " << min);
  TEST_ASSERT(max <= tolerance_, "\n   tBlockedTpetraOperator::test_noncontig (rebuild): "
                                     << toString(status) << ": "
                                     << "testing tolerance over many matrix vector multiplies ( "
                                     << max << " <= " << tolerance_ << " )");

  return allPassed;
}

bool tBlockedTpetraOperator::test_reorder(int verbosity, std::ostream& os, int total) {
  bool status    = false;
  bool allPassed = true;

  RCP<const Teuchos::Comm<int>> comm_tpetra = GetComm_tpetra();

  std::string tstr = total ? "(composite reorder)" : "(flat reorder)";

  TEST_MSG("\n   tBlockedTpetraOperator::test_reorder" << tstr << ": "
                                                       << "Running on " << comm_tpetra->getSize()
                                                       << " processors");

  GO nx = 5;
  GO ny = 5;

  RCP<crs_t> A = buildRecirc2DMatrix(comm_tpetra, nx, ny);

  int width = 3;
  mv_t x(A->getDomainMap(), width);
  mv_t yf(A->getRangeMap(), width);
  mv_t yr(A->getRangeMap(), width);

  std::vector<std::vector<GO>> vars;
  buildBlockGIDs(vars, *A->getRowMap(), false);

  Teko::TpetraHelpers::BlockedTpetraOperator flatShell(vars, A, "Af");
  Teko::TpetraHelpers::BlockedTpetraOperator reorderShell(vars, A, "Ar");

  Teko::BlockReorderManager brm;
  switch (total) {
    case 0:
      brm.SetNumBlocks(3);
      brm.SetBlock(0, 1);
      brm.SetBlock(1, 0);
      brm.SetBlock(2, 2);
      break;
    case 1:
      brm.SetNumBlocks(2);
      brm.SetBlock(0, 1);
      brm.GetBlock(1)->SetNumBlocks(2);
      brm.GetBlock(1)->SetBlock(0, 0);
      brm.GetBlock(1)->SetBlock(1, 2);
      break;
    case 2:
      brm.SetNumBlocks(2);
      brm.GetBlock(0)->SetNumBlocks(2);
      brm.GetBlock(0)->SetBlock(0, 0);
      brm.GetBlock(0)->SetBlock(1, 2);
      brm.SetBlock(1, 1);
      break;
  }
  reorderShell.Reorder(brm);
  TEST_MSG("\n   tBlockedTpetraOperator::test_reorder" << tstr << ": patern = " << brm.toString());

  TEST_MSG("\n   tBlockedTpetraOperator::test_reorder" << tstr << ":\n");
  TEST_MSG("\n      " << Teuchos::describe(*reorderShell.getThyraOp(), Teuchos::VERB_HIGH)
                      << std::endl);

  int numtests = 10;
  ST max       = 0.0;
  ST min       = 1.0;
  for (int i = 0; i < numtests; i++) {
    x.randomize();

    flatShell.apply(x, yf);
    reorderShell.apply(x, yr);

    ST err = relativeError(yf, yr);
    max    = std::max(max, err);
    min    = std::min(min, err);
  }
  TEST_ASSERT(max >= min, "   tBlockedTpetraOperator::test_reorder"
                              << tstr << ": " << toString(status) << ": "
                              << "sanity checked - " << max << " >= " << min);
  TEST_ASSERT(max <= tolerance_, "   tBlockedTpetraOperator::test_reorder"
                                     << tstr << ": " << toString(status) << ": "
                                     << "testing tolerance over many matrix vector multiplies ( "
                                     << max << " <= " << tolerance_ << " )");

  return allPassed;
}

}  // namespace Teko::Test
