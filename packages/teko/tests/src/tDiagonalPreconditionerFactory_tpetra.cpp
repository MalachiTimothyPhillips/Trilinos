// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "tDiagonalPreconditionerFactory_tpetra.hpp"
#include "Teko_DiagonalPreconditionerFactory.hpp"
#include "Teko_DiagonalPreconditionerOp.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

// Thyra includes
#include "Thyra_LinearOpBase.hpp"
#include "Thyra_DefaultBlockedLinearOp.hpp"
#include "Thyra_DefaultIdentityLinearOp.hpp"
#include "Thyra_DefaultZeroLinearOp.hpp"
#include "Thyra_DefaultLinearOpSource.hpp"
#include "Thyra_DefaultPreconditioner.hpp"
#include "Thyra_DefaultDiagonalLinearOp.hpp"
#include "Thyra_DefaultMultipliedLinearOp.hpp"
#include "Thyra_DefaultScaledAdjointLinearOp.hpp"
#include "Thyra_DefaultLinearOpSource.hpp"
#include "Thyra_LinearOpTester.hpp"
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"

// Tpetra includes
#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Vector.hpp"

// Galeri / Xpetra
#include "Galeri_XpetraMaps.hpp"
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraParameters.hpp"

#include "Xpetra_Map.hpp"
#include "Xpetra_Matrix.hpp"
#include "Xpetra_TpetraMap.hpp"
#include "Xpetra_TpetraCrsMatrix.hpp"

#include "Teko_Utilities.hpp"
#include "Teko_TpetraHelpers.hpp"
#include "Teko_ConfigDefs.hpp"

#include <vector>
#include <math.h>

namespace Teko::Test {

using namespace Teuchos;
using namespace Thyra;

namespace {

using ST = Teko::ST;
using LO = Teko::LO;
using GO = Teko::GO;
using NT = Teko::NT;

using map_t = Tpetra::Map<LO, GO, NT>;
using crs_t = Tpetra::CrsMatrix<ST, LO, GO, NT>;
using vec_t = Tpetra::Vector<ST, LO, GO, NT>;
using mv_t  = Tpetra::MultiVector<ST, LO, GO, NT>;

RCP<crs_t> buildLaplace2DMatrix(const RCP<const Teuchos::Comm<int> >& comm, GO nx, GO ny) {
  Teuchos::ParameterList galeriList;
  galeriList.set("nx", nx);
  galeriList.set("ny", ny);
  galeriList.set("mx", comm->getSize());
  galeriList.set("my", 1);

  RCP<const Xpetra::Map<LO, GO, NT> > xMap =
      Galeri::Xpetra::CreateMap<LO, GO, NT>(Xpetra::UseTpetra, "Cartesian2D", comm, galeriList);
  RCP<const map_t> tMap = Xpetra::toTpetra(xMap);

  auto problem =
      Galeri::Xpetra::BuildProblem<ST, LO, GO, map_t, crs_t, mv_t>("Laplace2D", tMap, galeriList);

  return problem->BuildMatrix();
}

}  // namespace

tDiagonalPreconditionerFactory_tpetra::~tDiagonalPreconditionerFactory_tpetra() {
  delete fact;
  delete pstate;
  delete[] block_starts;
  delete[] block_gids;
}

void tDiagonalPreconditionerFactory_tpetra::initializeTest() {
  const RCP<const Teuchos::Comm<int> > comm_tpetra = GetComm_tpetra();

  tolerance_ = 1.0e-14;

  GO nx = 39;
  GO ny = 53;

  tpetraF = buildLaplace2DMatrix(comm_tpetra, nx, ny);
  F_      = Thyra::constTpetraLinearOp<ST, LO, GO, NT>(
      Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraF->getDomainMap()),
      Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraF->getRangeMap()), tpetraF);
}

void tDiagonalPreconditionerFactory_tpetra::buildParameterList(int blocksize) {
  const crs_t* F = &*tpetraF;
  TEUCHOS_ASSERT(F);

  if (blocksize > 0) {
    delete[] block_starts;
    delete[] block_gids;

    LO Nr        = F->getLocalNumRows();
    int Nb       = (int)ceil(((double)Nr) / ((double)blocksize));
    block_starts = new GO[Nb + 1];
    block_gids   = new GO[Nr];

    block_starts[0] = 0;
    for (int i = 0; i < Nb; i++) block_starts[i + 1] = block_starts[i] + blocksize;
    block_starts[Nb] = Nr;

    for (LO i = 0; i < Nr; i++) block_gids[i] = F->getRowMap()->getGlobalElement(i);

    Teuchos::ParameterList sublist;
    List_.set("number of local blocks", Nb);
    List_.set("block start index", block_starts);
    List_.set("block entry gids", block_gids);
    sublist.set("apply mode", "invert");
    List_.set("blockdiagmatrix: list", sublist);
  } else {
    List_.set("Diagonal Type", "Diagonal");
  }
}

int tDiagonalPreconditionerFactory_tpetra::runTest(int verbosity, std::ostream& stdstrm,
                                                   std::ostream& failstrm, int& totalrun) {
  bool allTests = true;
  bool status;
  int failcount = 0;

  failstrm << "tDiagonalPreconditionerFactory_tpetra";

  status = test_createPrec(verbosity, failstrm, 0);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"createPrec\" ... PASSED", "   \"createPrec\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_initializePrec(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"initializePrec\" ... PASSED",
                       "   \"initializePrec\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_canApply(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"canApply\" ... PASSED", "   \"canApply\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = allTests;
  if (verbosity >= 10) {
    Teko_TEST_MSG_tpetra(failstrm, 0, "tDiagonalPreconditionedFactory...PASSED",
                         "tDiagonalPreconditionedFactory...FAILED");
  } else {
    Teko_TEST_MSG_tpetra(failstrm, 0, "...PASSED", "tDiagonalPreconditionedFactory...FAILED");
  }

  return failcount;
}

bool tDiagonalPreconditionerFactory_tpetra::test_initializePrec(int verbosity, std::ostream& os) {
  delete pstate;

  pstate = new DiagonalPrecondState();
  pop    = fact->buildPreconditionerOperator(F_, *pstate);

  RCP<const Thyra::DiagonalLinearOpBase<ST> > dop =
      rcp_dynamic_cast<const Thyra::DiagonalLinearOpBase<ST> >(pop);
  if (dop.is_null()) return false;

  return true;
}

bool tDiagonalPreconditionerFactory_tpetra::test_createPrec(int verbosity, std::ostream& os,
                                                            int blocksize) {
  buildParameterList(blocksize);

  delete fact;

  fact = new DiagonalPreconditionerFactory();
  fact->initializeFromParameterList(List_);
  if (!fact) return false;

  return true;
}

bool tDiagonalPreconditionerFactory_tpetra::test_canApply(int verbosity, std::ostream& os) {
  RCP<const map_t> domain_ = tpetraF->getDomainMap();
  RCP<const map_t> range_  = tpetraF->getRangeMap();

  RCP<vec_t> X = Tpetra::createVector<ST, LO, GO, NT>(domain_);
  RCP<vec_t> Y = Tpetra::createVector<ST, LO, GO, NT>(range_);
  RCP<vec_t> Z = Tpetra::createVector<ST, LO, GO, NT>(range_);
  Y->putScalar(0.0);
  Z->putScalar(1.0);

  tpetraF->getLocalDiagCopy(*X);

  MultiVector tX =
      Thyra::createVector<ST, LO, GO, NT>(X, Thyra::createVectorSpace<ST, LO, GO, NT>(domain_));
  MultiVector tY =
      Thyra::createVector<ST, LO, GO, NT>(Y, Thyra::createVectorSpace<ST, LO, GO, NT>(range_));

  Teko::applyOp(pop, tX, tY, 1.0, 0.0);

  double znrm = Z->norm2();
  Z->update(-1.0, *Y, 1.0);
  double dnrm = Z->norm2();

  if (!tpetraF->getComm()->getRank()) std::cout << "||Z-Y||/||Z|| = " << dnrm / znrm << std::endl;
  if (dnrm / znrm > 1e-12)
    return false;
  else
    return true;
}

}  // end namespace Teko::Test