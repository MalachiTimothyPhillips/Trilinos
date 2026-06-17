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
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_DefaultBlockedLinearOp.hpp"
#include "Thyra_ProductVectorBase.hpp"
#include "Thyra_ProductMultiVectorBase.hpp"
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

#include "Teko_TpetraOperatorWrapper.hpp"
#include "Teko_TpetraHelpers.hpp"
#include "Teko_ConfigDefs.hpp"

#include "tTpetraOperatorWrapper.hpp"

namespace Teko::Test {

using Teuchos::null;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::rcpFromRef;
using Thyra::createMember;
using Thyra::LinearOpBase;
using Thyra::LinearOpTester;
using Thyra::MultiVectorBase;
using Thyra::VectorBase;

namespace {

using ST = Teko::ST;
using LO = Teko::LO;
using GO = Teko::GO;
using NT = Teko::NT;

using map_t = Tpetra::Map<LO, GO, NT>;
using crs_t = Tpetra::CrsMatrix<ST, LO, GO, NT>;
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

void tTpetraOperatorWrapper::initializeTest() {}

int tTpetraOperatorWrapper::runTest(int verbosity, std::ostream& stdstrm, std::ostream& failstrm,
                                    int& totalrun) {
  bool allTests = true;
  bool status;
  int failcount = 0;

  failstrm << "tTpetraOperatorWrapper";

  status = test_functionality(verbosity, failstrm);
  Teko_TEST_MSG_tpetra(stdstrm, 1, "   \"functionality\" ... PASSED",
                       "   \"functionality\" ... FAILED");
  allTests &= status;
  failcount += status ? 0 : 1;
  totalrun++;

  status = allTests;
  if (verbosity >= 10) {
    Teko_TEST_MSG_tpetra(failstrm, 0, "tTpetraOperatorWrapper...PASSED",
                         "tTpetraOperatorWrapper...FAILED");
  } else {
    Teko_TEST_MSG_tpetra(failstrm, 0, "...PASSED", "tTpetraOperatorWrapper...FAILED");
  }

  return failcount;
}

bool tTpetraOperatorWrapper::test_functionality(int verbosity, std::ostream& os) {
  bool status    = false;
  bool allPassed = true;

  RCP<const Teuchos::Comm<int> > comm_tpetra = GetComm_tpetra();

  TEST_MSG("\n   tTpetraOperatorWrapper::test_functionality: "
           << "Running on " << comm_tpetra->getSize() << " processors");

  GO nx = 39;
  GO ny = 53;

  TEST_MSG("   tTpetraOperatorWrapper::test_functionality: "
           << "Using Galeri/Xpetra to create test matrices");

  RCP<const crs_t> tpetraF = buildGaleriMatrix("Recirc2D", comm_tpetra, nx, ny, 1e-5, 1.0);

  RCP<const crs_t> tpetraC = buildGaleriMatrix("Laplace2D", comm_tpetra, nx, ny);

  RCP<const crs_t> tpetraB = buildGaleriMatrix("Identity", comm_tpetra, nx, ny, 0.0, 0.0, 5.0);

  RCP<const crs_t> tpetraBt = buildGaleriMatrix("Identity", comm_tpetra, nx, ny, 0.0, 0.0, 3.0);

  TEST_MSG("   tTpetraOperatorWrapper::test_functionality: "
           << " Building block2x2 Thyra matrix ... wrapping in TpetraOperatorWrapper");

  const RCP<const LinearOpBase<double> > A = Thyra::block2x2<double>(
      Thyra::constTpetraLinearOp<ST, LO, GO, NT>(
          Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraF->getDomainMap()),
          Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraF->getRangeMap()), tpetraF),
      Thyra::constTpetraLinearOp<ST, LO, GO, NT>(
          Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraBt->getDomainMap()),
          Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraBt->getRangeMap()), tpetraBt),
      Thyra::constTpetraLinearOp<ST, LO, GO, NT>(
          Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraB->getDomainMap()),
          Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraB->getRangeMap()), tpetraB),
      Thyra::constTpetraLinearOp<ST, LO, GO, NT>(
          Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraC->getDomainMap()),
          Thyra::tpetraVectorSpace<ST, LO, GO, NT>(tpetraC->getRangeMap()), tpetraC),
      "A");

  const RCP<Teko::TpetraHelpers::TpetraOperatorWrapper> tpetra_A =
      rcp(new Teko::TpetraHelpers::TpetraOperatorWrapper(A));

  const RCP<const Tpetra::Map<LO, GO, NT> >& rangeMap  = tpetra_A->getRangeMap();
  const RCP<const Tpetra::Map<LO, GO, NT> >& domainMap = tpetra_A->getDomainMap();

  TEST_EQUALITY(rangeMap->getGlobalNumElements(), (Tpetra::global_size_t)2 * nx * ny,
                "   tTpetraOperatorWrapper::test_functionality: "
                    << toString(status) << ": "
                    << "checking rangeMap size "
                    << "( map = " << rangeMap->getGlobalNumElements() << ", true = " << 2 * nx * ny
                    << " )");

  TEST_EQUALITY(domainMap->getGlobalNumElements(), (Tpetra::global_size_t)2 * nx * ny,
                "   tTpetraOperatorWrapper::test_functionality: "
                    << toString(status) << ": "
                    << "checking domainMap size "
                    << "( map = " << domainMap->getGlobalNumElements() << ", true = " << 2 * nx * ny
                    << " )");

  TEST_EQUALITY(rangeMap->getGlobalNumElements() - 1,
                (Tpetra::global_size_t)rangeMap->getMaxAllGlobalIndex(),
                "   tTpetraOperatorWrapper::test_functionality: "
                    << toString(status) << ": "
                    << " checking largest range element "
                    << "( largest = " << rangeMap->getMaxAllGlobalIndex()
                    << ", true = " << rangeMap->getGlobalNumElements() - 1 << " )");
  TEST_EQUALITY(domainMap->getGlobalNumElements() - 1,
                (Tpetra::global_size_t)domainMap->getMaxAllGlobalIndex(),
                "   tTpetraOperatorWrapper::test_functionality: "
                    << toString(status) << ": "
                    << " checking largest domain element "
                    << "( largest = " << domainMap->getMaxAllGlobalIndex()
                    << ", true = " << domainMap->getGlobalNumElements() - 1 << " )");

  RCP<const Teko::TpetraHelpers::MappingStrategy> ms = tpetra_A->getMapStrategy();

  {
    const RCP<MultiVectorBase<ST> > tv = Thyra::createMembers(A->domain(), 1);
    Thyra::randomize(-100.0, 100.0, tv.ptr());

    const RCP<const MultiVectorBase<ST> > tv_0 =
        Teuchos::rcp_dynamic_cast<const Thyra::ProductMultiVectorBase<ST> >(tv)
            ->getMultiVectorBlock(0);
    const RCP<const MultiVectorBase<ST> > tv_1 =
        Teuchos::rcp_dynamic_cast<const Thyra::ProductMultiVectorBase<ST> >(tv)
            ->getMultiVectorBlock(1);

    const Thyra::ConstDetachedSpmdVectorView<ST> vv_0(tv_0->col(0));
    const Thyra::ConstDetachedSpmdVectorView<ST> vv_1(tv_1->col(0));

    LO off_0 = vv_0.globalOffset();
    LO off_1 = vv_1.globalOffset();

    const RCP<Tpetra::Vector<ST, LO, GO, NT> > ev =
        rcp(new Tpetra::Vector<ST, LO, GO, NT>(tpetra_A->getDomainMap()));
    ms->copyThyraIntoTpetra(tv, *ev);

    TEST_EQUALITY((Tpetra::global_size_t)tv->range()->dim(), ev->getGlobalLength(),
                  "   tTpetraOperatorWrapper::test_functionality: "
                      << toString(status) << ": "
                      << " checking ThyraIntoTpetra copy "
                      << "( thyra dim = " << tv->range()->dim()
                      << ", global length = " << ev->getGlobalLength() << " )");

    LO numMyElements               = domainMap->getLocalNumElements();
    bool compareThyraToTpetraValue = true;
    ST tval                        = 0.0;
    auto evView                    = ev->get1dView();
    for (LO i = 0; i < numMyElements; i++) {
      GO gid = domainMap->getGlobalElement(i);
      if (gid - off_0 < nx * ny) {
        tval = vv_0[gid - off_0];
      } else {
        tval = vv_1[gid - off_1 - nx * ny];
      }
      compareThyraToTpetraValue &= (evView[i] == tval);
    }
    TEST_ASSERT(compareThyraToTpetraValue, "   tTpetraOperatorWrapper::test_functionality: "
                                               << toString(status) << ": "
                                               << " comparing Thyra to Tpetra values");
  }

  {
    const RCP<Tpetra::Vector<ST, LO, GO, NT> > ev =
        rcp(new Tpetra::Vector<ST, LO, GO, NT>(tpetra_A->getDomainMap()));
    ev->randomize();

    const RCP<MultiVectorBase<ST> > tv = Thyra::createMembers(A->domain(), 1);
    const RCP<const MultiVectorBase<ST> > tv_0 =
        Teuchos::rcp_dynamic_cast<const Thyra::ProductMultiVectorBase<ST> >(tv)
            ->getMultiVectorBlock(0);
    const RCP<const MultiVectorBase<ST> > tv_1 =
        Teuchos::rcp_dynamic_cast<const Thyra::ProductMultiVectorBase<ST> >(tv)
            ->getMultiVectorBlock(1);
    const Thyra::ConstDetachedSpmdVectorView<ST> vv_0(tv_0->col(0));
    const Thyra::ConstDetachedSpmdVectorView<ST> vv_1(tv_1->col(0));

    LO off_0 =
        rcp_dynamic_cast<const Thyra::SpmdVectorSpaceBase<ST> >(tv_0->range())->localOffset();
    LO off_1 =
        rcp_dynamic_cast<const Thyra::SpmdVectorSpaceBase<ST> >(tv_1->range())->localOffset();

    ms->copyTpetraIntoThyra(*ev, tv.ptr());

    TEST_EQUALITY((Tpetra::global_size_t)tv->range()->dim(), ev->getGlobalLength(),
                  "   tTpetraOperatorWrapper::test_functionality: "
                      << toString(status) << ": "
                      << " checking TpetraIntoThyra copy "
                      << "( thyra dim = " << tv->range()->dim()
                      << ", global length = " << ev->getGlobalLength() << " )");

    LO numMyElements               = domainMap->getLocalNumElements();
    bool compareTpetraToThyraValue = true;
    ST tval                        = 0.0;
    auto evView                    = ev->get1dView();
    for (LO i = 0; i < numMyElements; i++) {
      GO gid = domainMap->getGlobalElement(i);
      if (gid - off_0 < nx * ny) {
        tval = vv_0[gid - off_0];
      } else {
        tval = vv_1[gid - off_1 - nx * ny];
      }
      compareTpetraToThyraValue &= (evView[i] == tval);
    }
    TEST_ASSERT(compareTpetraToThyraValue, "   tTpetraOperatorWrapper::test_functionality: "
                                               << toString(status) << ": "
                                               << " comparing Thyra to Tpetra values");
  }

  return allPassed;
}

}  // namespace Teko::Test
