// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "tExplicitOps_tpetra.hpp"

#include <string>

// Tpetra includes
#include "Tpetra_Core.hpp"
#include "Tpetra_Export.hpp"

// Galeri / Xpetra
#include "Galeri_XpetraMaps.hpp"
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraParameters.hpp"

#include "Xpetra_Map.hpp"
#include "Xpetra_Matrix.hpp"
#include "Xpetra_TpetraMap.hpp"
#include "Xpetra_TpetraCrsMatrix.hpp"

// Teko-Package includes
#include "Teko_Utilities.hpp"
#include "Teko_TpetraHelpers.hpp"
#include "Teko_ConfigDefs.hpp"

// Thyra includes
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_DefaultDiagonalLinearOp.hpp"
#include "Thyra_LinearOpTester.hpp"

// Test-rig
#include "Test_Utils.hpp"

namespace Teko::Test {

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::rcpFromRef;

namespace {

using ST = Teko::ST;
using LO = Teko::LO;
using GO = Teko::GO;
using NT = Teko::NT;

using map_t = Tpetra::Map<LO, GO, NT>;
using crs_t = Tpetra::CrsMatrix<ST, LO, GO, NT>;
using vec_t = Tpetra::Vector<ST, LO, GO, NT>;
using mv_t  = Tpetra::MultiVector<ST, LO, GO, NT>;

RCP<crs_t> buildGaleriMatrix(const std::string& matrixType,
                             const RCP<const Teuchos::Comm<int> >& comm, GO nx, GO ny,
                             double diff = 1e-5, double conv = 1.0, double diagA = 1.0) {
  Teuchos::ParameterList galeriList;
  galeriList.set("nx", nx);
  galeriList.set("ny", ny);
  galeriList.set("mx", comm->getSize());
  galeriList.set("my", 1);

  if (matrixType == "Recirc2D") {
    galeriList.set("lx", 1.0);
    galeriList.set("ly", 1.0);
    galeriList.set("diff", diff);
    galeriList.set("conv", conv);
  }

  if (matrixType == "Identity" || matrixType == "Diag") {
    galeriList.set("a", diagA);
  }

  RCP<const Xpetra::Map<LO, GO, NT> > xMap =
      Galeri::Xpetra::CreateMap<LO, GO, NT>(Xpetra::UseTpetra, "Cartesian2D", comm, galeriList);
  RCP<const map_t> tMap = Xpetra::toTpetra(xMap);

  auto problem =
      Galeri::Xpetra::BuildProblem<ST, LO, GO, map_t, crs_t, mv_t>(matrixType, tMap, galeriList);

  return problem->BuildMatrix();
}

}  // namespace

void tExplicitOps_tpetra::initializeTest() {
  const RCP<const Teuchos::Comm<int> > comm_tpetra = GetComm_tpetra();

  tolerance_ = 1.0e-4;

  GO nx = 39;
  GO ny = 53;

  RCP<crs_t> tpetraF = buildGaleriMatrix("Recirc2D", comm_tpetra, nx, ny, 1e-5, 1.0);
  F_                 = Thyra::tpetraLinearOp<ST, LO, GO, NT>(
      Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraF->getDomainMap()),
      Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraF->getRangeMap()), tpetraF);

  RCP<crs_t> tpetraG = buildGaleriMatrix("Laplace2D", comm_tpetra, nx, ny);
  G_                 = Thyra::tpetraLinearOp<ST, LO, GO, NT>(
      Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraG->getDomainMap()),
      Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraG->getRangeMap()), tpetraG);

  RCP<crs_t> tpetraH = buildGaleriMatrix("Identity", comm_tpetra, nx, ny, 0.0, 0.0, 1.0);
  H_                 = Thyra::tpetraLinearOp<ST, LO, GO, NT>(
      Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraH->getDomainMap()),
      Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraH->getRangeMap()), tpetraH);

  RCP<vec_t> v = rcp(new vec_t(tpetraF->getRangeMap()));
  v->randomize();
  RCP<Thyra::VectorBase<ST> > tV = Thyra::createVector<ST, LO, GO, NT>(
      v, Thyra::createVectorSpace<ST, LO, GO, NT>(tpetraF->getRowMap()));
  D_ = Thyra::diagonal(tV);
}

int tExplicitOps_tpetra::runTest(int verbosity, std::ostream& stdstrm, std::ostream& failstrm,
                                 int& totalrun) {
  bool allTests = true;
  bool status;
  int failcount = 0;

  failstrm << "tExplicitOps_tpetra";

  status = test_mult_diagScaleMatProd(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"mult_diagScaleMatProd\" ... PASSED",
                       "   \"mult_diagScaleMatProd\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_mult_diagScaling(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"mult_diagScaling\" ... PASSED",
                       "   \"mult_diagScaling\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_add(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"add\" ... PASSED", "   \"add\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_mult_modScaleMatProd(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"mult_modScaleMatProd\" ... PASSED",
                       "   \"mult_modScaleMatProd\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = test_add_mod(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"add_mod\" ... PASSED", "   \"add_mod\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = allTests;
  if (verbosity >= 10) {
    Teko_TEST_MSG_tpetra(failstrm, 0, "tExplicitOps_tpetra...PASSED",
                         "tExplicitOps_tpetra...FAILED");
  } else {
    Teko_TEST_MSG_tpetra(failstrm, 0, "...PASSED", "tExplicitOps_tpetra...FAILED");
  }

  return failcount;
}

bool tExplicitOps_tpetra::test_mult_diagScaleMatProd(int verbosity, std::ostream& os) {
  bool status    = false;
  bool allPassed = true;

  Thyra::LinearOpTester<ST> tester;
  tester.set_all_error_tol(1e-10);
  tester.show_all_tests(true);

  RCP<const Thyra::LinearOpBase<ST> > thyOp;
  Teko::LinearOp expOp;

  thyOp = Thyra::multiply(Teko::scale(-4.0, F_), D_, Teko::adjoint(G_));
  expOp = Teko::explicitMultiply(Teko::scale(-4.0, F_), D_, Teko::adjoint(G_));

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    const bool result = tester.compare(*thyOp, *expOp, Teuchos::ptrFromRef(fos));
    TEST_ASSERT(result, std::endl
                            << "   tExplicitOps_tpetra::test_diagScaleMatProd "
                            << ": Testing triple matrix product");
    if (not result || verbosity >= 10) os << ss.str();
  }

  thyOp = Teko::multiply(Teko::scale(-4.0, F_), Teko::adjoint(G_));
  expOp = Teko::explicitMultiply(Teko::scale(-4.0, F_), Teko::adjoint(G_));

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    const bool result = tester.compare(*thyOp, *expOp, Teuchos::ptrFromRef(fos));
    TEST_ASSERT(result, std::endl
                            << "   tExplicitOps_tpetra::test_diagScaleMatProd "
                            << ": Testing double matrix product");
    if (not result || verbosity >= 10) os << ss.str();
  }

  return allPassed;
}

bool tExplicitOps_tpetra::test_mult_diagScaling(int verbosity, std::ostream& os) {
  bool status    = false;
  bool allPassed = true;

  Thyra::LinearOpTester<ST> tester;
  tester.set_all_error_tol(1e-10);
  tester.show_all_tests(true);

  RCP<const Thyra::LinearOpBase<ST> > thyOp;
  Teko::LinearOp expOp;

  thyOp = Teko::multiply(Teko::scale(-4.0, F_), D_);
  expOp = Teko::explicitMultiply(Teko::scale(-4.0, F_), D_);

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    const bool result = tester.compare(*thyOp, *expOp, Teuchos::ptrFromRef(fos));
    TEST_ASSERT(result, std::endl
                            << "   tExplicitOps_tpetra::test_diagScaleMatProd "
                            << ": Testing diagonal scaling");
    if (not result || verbosity >= 10) os << ss.str();
  }

  thyOp = Teko::multiply(D_, Teko::scale(-9.0, F_));
  expOp = Teko::explicitMultiply(D_, Teko::scale(-9.0, F_));

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    const bool result = tester.compare(*thyOp, *expOp, Teuchos::ptrFromRef(fos));
    TEST_ASSERT(result, std::endl
                            << "   tExplicitOps_tpetra::test_diagScaleMatProd "
                            << ": Testing diagonal scaling");
    if (not result || verbosity >= 10) os << ss.str();
  }

  return allPassed;
}

bool tExplicitOps_tpetra::test_mult_modScaleMatProd(int verbosity, std::ostream& os) {
  bool status    = false;
  bool allPassed = true;

  Thyra::LinearOpTester<ST> tester;
  tester.set_all_error_tol(1e-10);
  tester.show_all_tests(true);

  Teko::LinearOp thyOp;
  Teko::ModifiableLinearOp expOp;

  thyOp = Teko::multiply(Teko::scale(-4.0, F_), D_, Teko::adjoint(G_));
  expOp = Teko::explicitMultiply(Teko::scale(-4.0, F_), D_, Teko::adjoint(G_), expOp);

  RCP<const Thyra::TpetraLinearOp<ST, LO, GO, NT> > tOp1 =
      Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp, true);
  RCP<const Tpetra::Operator<ST, LO, GO, NT> > eop1 = tOp1->getConstTpetraOperator();

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    const bool result = tester.compare(*thyOp, *expOp, Teuchos::ptrFromRef(fos));
    TEST_ASSERT(result, std::endl
                            << "   tExplicitOps_tpetra::test_modScaleMatProd "
                            << ": Testing triple matrix product");
    if (not result || verbosity >= 10) os << ss.str();
  }

  RCP<Thyra::TpetraLinearOp<ST, LO, GO, NT> > tF =
      Teuchos::rcp_dynamic_cast<Thyra::TpetraLinearOp<ST, LO, GO, NT> >(F_);
  RCP<Tpetra::CrsMatrix<ST, LO, GO, NT> > crsF =
      Teuchos::rcp_dynamic_cast<Tpetra::CrsMatrix<ST, LO, GO, NT> >(tF->getTpetraOperator(), true);
  crsF->scale(5.0);

  RCP<Thyra::TpetraLinearOp<ST, LO, GO, NT> > tG =
      Teuchos::rcp_dynamic_cast<Thyra::TpetraLinearOp<ST, LO, GO, NT> >(G_);
  RCP<Tpetra::CrsMatrix<ST, LO, GO, NT> > crsG =
      Teuchos::rcp_dynamic_cast<Tpetra::CrsMatrix<ST, LO, GO, NT> >(tG->getTpetraOperator(), true);
  crsG->scale(2.0);

  size_t numEntries = crsF->getNumEntriesInLocalRow(3);
  auto indices1 = typename Tpetra::CrsMatrix<ST, LO, GO, NT>::nonconst_local_inds_host_view_type(
      Kokkos::ViewAllocateWithoutInitializing("rowIndices"), numEntries);
  auto values1 = typename Tpetra::CrsMatrix<ST, LO, GO, NT>::nonconst_values_host_view_type(
      Kokkos::ViewAllocateWithoutInitializing("rowValues"), numEntries);
  crsF->getLocalRowCopy(3, indices1, values1, numEntries);
  for (size_t i = 0; i < numEntries; i++) values1(i) *= values1(i) * ST(i + 1) * 0.92;
  crsF->replaceLocalValues(3, indices1, values1);

  numEntries    = crsF->getNumEntriesInLocalRow(7);
  auto indices2 = typename Tpetra::CrsMatrix<ST, LO, GO, NT>::nonconst_local_inds_host_view_type(
      Kokkos::ViewAllocateWithoutInitializing("rowIndices"), numEntries);
  auto values2 = typename Tpetra::CrsMatrix<ST, LO, GO, NT>::nonconst_values_host_view_type(
      Kokkos::ViewAllocateWithoutInitializing("rowValues"), numEntries);
  crsF->getLocalRowCopy(7, indices2, values2, numEntries);
  for (size_t i = 0; i < numEntries; i++) values2(i) *= values2(i) * ST(i + 1) * 0.92;
  crsF->replaceLocalValues(7, indices2, values2);

  thyOp = Teko::multiply(Teko::scale(-4.0, F_), D_, Teko::adjoint(G_));
  expOp = Teko::explicitMultiply(Teko::scale(-4.0, F_), D_, Teko::adjoint(G_), expOp);

  RCP<const Thyra::TpetraLinearOp<ST, LO, GO, NT> > tOp2 =
      Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp, true);
  RCP<const Tpetra::Operator<ST, LO, GO, NT> > eop2 = tOp2->getConstTpetraOperator();

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    const bool result = tester.compare(*thyOp, *expOp, Teuchos::ptrFromRef(fos));
    TEST_ASSERT(result, std::endl
                            << "   tExplicitOps_tpetra::test_modScaleMatProd "
                            << ": Testing triple matrix product");
    if (not result || verbosity >= 10) os << ss.str();
  }

  return allPassed;
}

bool tExplicitOps_tpetra::test_add(int verbosity, std::ostream& os) {
  bool status    = false;
  bool allPassed = true;

  Thyra::LinearOpTester<ST> tester;
  tester.set_all_error_tol(1e-10);
  tester.show_all_tests(true);

  RCP<const Thyra::LinearOpBase<ST> > thyOp;
  Teko::LinearOp expOp;

  thyOp = Teko::add(Teko::scale(-4.0, F_), Teko::adjoint(G_));
  expOp = Teko::explicitAdd(Teko::scale(-4.0, F_), Teko::adjoint(G_));

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    const bool result = tester.compare(*thyOp, *expOp, Teuchos::ptrFromRef(fos));
    TEST_ASSERT(result, std::endl
                            << "   tExplicitOps_tpetra::test_add "
                            << ": Testing explicit add");
    if (not result || verbosity >= 10) os << ss.str();
  }

  return allPassed;
}

bool tExplicitOps_tpetra::test_add_mod(int verbosity, std::ostream& os) {
  bool status    = false;
  bool allPassed = true;

  Thyra::LinearOpTester<ST> tester;
  tester.set_all_error_tol(1e-10);
  tester.show_all_tests(true);

  RCP<const Thyra::LinearOpBase<ST> > thyOp;
  Teko::ModifiableLinearOp expOp;

  thyOp = Teko::add(Teko::scale(-4.0, F_), Teko::adjoint(G_));
  expOp = Teko::explicitAdd(Teko::scale(-4.0, F_), Teko::adjoint(G_), expOp);

  RCP<const Thyra::TpetraLinearOp<ST, LO, GO, NT> > tOp1 =
      Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp, true);
  RCP<const Tpetra::Operator<ST, LO, GO, NT> > eop1 = tOp1->getConstTpetraOperator();

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    const bool result = tester.compare(*thyOp, *expOp, Teuchos::ptrFromRef(fos));
    TEST_ASSERT(result, std::endl
                            << "   tExplicitOps_tpetra::test_add_mod"
                            << ": Testing explicit add");
    if (not result || verbosity >= 10) os << ss.str();
  }

  RCP<Thyra::TpetraLinearOp<ST, LO, GO, NT> > tF =
      Teuchos::rcp_dynamic_cast<Thyra::TpetraLinearOp<ST, LO, GO, NT> >(F_);
  RCP<Tpetra::CrsMatrix<ST, LO, GO, NT> > crsF =
      Teuchos::rcp_dynamic_cast<Tpetra::CrsMatrix<ST, LO, GO, NT> >(tF->getTpetraOperator(), true);
  crsF->scale(5.0);

  RCP<Thyra::TpetraLinearOp<ST, LO, GO, NT> > tG =
      Teuchos::rcp_dynamic_cast<Thyra::TpetraLinearOp<ST, LO, GO, NT> >(G_);
  RCP<Tpetra::CrsMatrix<ST, LO, GO, NT> > crsG =
      Teuchos::rcp_dynamic_cast<Tpetra::CrsMatrix<ST, LO, GO, NT> >(tG->getTpetraOperator(), true);
  crsG->scale(2.0);

  size_t numEntries = crsF->getNumEntriesInLocalRow(3);
  auto indices1 = typename Tpetra::CrsMatrix<ST, LO, GO, NT>::nonconst_local_inds_host_view_type(
      Kokkos::ViewAllocateWithoutInitializing("rowIndices"), numEntries);
  auto values1 = typename Tpetra::CrsMatrix<ST, LO, GO, NT>::nonconst_values_host_view_type(
      Kokkos::ViewAllocateWithoutInitializing("rowValues"), numEntries);
  crsF->getLocalRowCopy(3, indices1, values1, numEntries);
  for (size_t i = 0; i < numEntries; i++) values1(i) *= values1(i) * ST(i + 1) * 0.92;
  crsF->replaceLocalValues(3, indices1, values1);

  numEntries    = crsF->getNumEntriesInLocalRow(7);
  auto indices2 = typename Tpetra::CrsMatrix<ST, LO, GO, NT>::nonconst_local_inds_host_view_type(
      Kokkos::ViewAllocateWithoutInitializing("rowIndices"), numEntries);
  auto values2 = typename Tpetra::CrsMatrix<ST, LO, GO, NT>::nonconst_values_host_view_type(
      Kokkos::ViewAllocateWithoutInitializing("rowValues"), numEntries);
  crsF->getLocalRowCopy(7, indices2, values2, numEntries);
  for (size_t i = 0; i < numEntries; i++) values2(i) *= values2(i) * ST(i + 1) * 0.92;
  crsF->replaceLocalValues(7, indices2, values2);

  thyOp = Teko::add(Teko::scale(-4.0, F_), Teko::adjoint(G_));
  expOp = Teko::explicitAdd(Teko::scale(-4.0, F_), Teko::adjoint(G_), expOp);

  RCP<const Thyra::TpetraLinearOp<ST, LO, GO, NT> > tOp2 =
      Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp, true);
  RCP<const Tpetra::Operator<ST, LO, GO, NT> > eop2 = tOp2->getConstTpetraOperator();

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    TEST_ASSERT(
        eop2.getRawPtr() == eop1.getRawPtr(), std::endl
                                                  << " tExplicitOps_tpetra::test_add_mod"
                                                  << ": Testing matrix addition preserves pointer");
    if (not(eop2.getRawPtr() == eop1.getRawPtr()) || verbosity >= 10) os << ss.str();
  }

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    const bool result = tester.compare(*thyOp, *expOp, Teuchos::ptrFromRef(fos));
    TEST_ASSERT(result, std::endl
                            << "   tExplicitOps_tpetra::test_add_mod"
                            << ": Testing matrix addition");
    if (not result || verbosity >= 10) os << ss.str();
  }

  Teko::ModifiableLinearOp expOp2;

  expOp2 = Teko::explicitAdd(H_, H_, expOp2);
  RCP<const Thyra::TpetraLinearOp<ST, LO, GO, NT> > tOp3 =
      Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp2, true);
  RCP<const Tpetra::Operator<ST, LO, GO, NT> > eop3 = tOp3->getConstTpetraOperator();

  expOp2 = Teko::explicitAdd(H_, H_, expOp2);
  RCP<const Thyra::TpetraLinearOp<ST, LO, GO, NT> > tOp3b =
      Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp2, true);
  RCP<const Tpetra::Operator<ST, LO, GO, NT> > eop3b = tOp3b->getConstTpetraOperator();

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    TEST_ASSERT(
        eop3.getRawPtr() == eop3b.getRawPtr(), std::endl
                                                   << " tExplicitOps_tpetra::test_add_mod"
                                                   << ": Testing matrix addition reuses pointer");
    if (not(eop3.getRawPtr() == eop3b.getRawPtr()) || verbosity >= 10) os << ss.str();
  }

  expOp2 = Teko::explicitAdd(Teko::scale(-4.0, F_), Teko::adjoint(G_), expOp2);
  RCP<const Thyra::TpetraLinearOp<ST, LO, GO, NT> > tOp4 =
      Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp2, true);
  RCP<const Tpetra::Operator<ST, LO, GO, NT> > eop4 = tOp4->getConstTpetraOperator();

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    TEST_ASSERT(eop3.getRawPtr() != eop4.getRawPtr(),
                std::endl
                    << " tExplicitOps_tpetra::test_add_mod"
                    << ": Testing matrix addition returns new pointer");
    if (not(eop3.getRawPtr() != eop4.getRawPtr()) || verbosity >= 10) os << ss.str();
  }

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    const bool result = tester.compare(*thyOp, *expOp2, Teuchos::ptrFromRef(fos));
    TEST_ASSERT(result, std::endl
                            << "   tExplicitOps_tpetra::test_add_mod"
                            << ": Testing matrix addition");
    if (not result || verbosity >= 10) os << ss.str();
  }

  expOp2 = Teko::explicitAdd(H_, H_, expOp2);
  tOp3   = Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp2, true);
  eop3   = tOp3->getConstTpetraOperator();
  auto expOp3 = Teko::explicitAdd(H_, expOp2, expOp2);
  RCP<const Thyra::TpetraLinearOp<ST, LO, GO, NT> > tOp5 =
      Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp3, true);
  RCP<const Tpetra::Operator<ST, LO, GO, NT> > eop5 = tOp5->getConstTpetraOperator();

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    TEST_ASSERT(eop3.getRawPtr() != eop5.getRawPtr(),
                std::endl
                    << " tExplicitOps_tpetra::test_add_mod2"
                    << ": Testing aliased matrix addition returns new pointer");
    if (not(eop3.getRawPtr() != eop5.getRawPtr()) || verbosity >= 10) os << ss.str();
  }

  expOp2 = Teko::explicitAdd(H_, H_, expOp2);
  tOp3   = Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp2, true);
  eop3   = tOp3->getConstTpetraOperator();
  auto expOp4 = Teko::explicitAdd(expOp2, H_, expOp2);
  RCP<const Thyra::TpetraLinearOp<ST, LO, GO, NT> > tOp6 =
      Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(expOp4, true);
  RCP<const Tpetra::Operator<ST, LO, GO, NT> > eop6 = tOp6->getConstTpetraOperator();

  {
    std::stringstream ss;
    Teuchos::FancyOStream fos(rcpFromRef(ss), "      |||");
    TEST_ASSERT(eop3.getRawPtr() != eop6.getRawPtr(),
                std::endl
                    << " tExplicitOps_tpetra::test_add_mod3"
                    << ": Testing aliased matrix addition returns new pointer");
    if (not(eop3.getRawPtr() != eop6.getRawPtr()) || verbosity >= 10) os << ss.str();
  }

  return allPassed;
}

}  // namespace Teko::Test