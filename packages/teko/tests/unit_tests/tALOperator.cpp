// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/*
 * Author: Zhen Wang
 * Email: wangz@ornl.gov
 *        zhen.wang@alum.emory.edu
 */

/*
 * This test reads:
 *   data/tOpMat.mm  (saddle point matrix),
 *   data/tOpMp.mm   (pressure mass matrix), and
 *   data/tOpRhs.mm  (reference result),
 * and tests Teko::NS::ALOperator on the Tpetra stack.
 */

#include "Teko_Config.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/types.h>
#include <unistd.h>

// Teuchos
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_RCP.hpp"

// Tpetra
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_Import.hpp"
#include "MatrixMarket_Tpetra.hpp"

// Thyra
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"

// Teko
#include "Teko_ALOperator.hpp"
#include "Teko_ConfigDefs.hpp"

using namespace Teko;

TEUCHOS_UNIT_TEST(tALOperator, test_tpetra) {
  Tpetra::ScopeGuard scopeGuard(Teuchos::UnitTestRepository::getCLP().argc,
                                Teuchos::UnitTestRepository::getCLP().argv);

  using ST = Teko::ST;
  using LO = Teko::LO;
  using GO = Teko::GO;
  using NT = Teko::NT;

  using map_t = Tpetra::Map<LO, GO, NT>;
  using crs_t = Tpetra::CrsMatrix<ST, LO, GO, NT>;
  using vec_t = Tpetra::Vector<ST, LO, GO, NT>;

  auto comm = Tpetra::getDefaultComm();

  const int myPID = comm->getRank();
  out << "MPI_PID = " << myPID << ", UNIX_PID = " << getpid() << std::endl;

  // Maps
  const int dim   = 2;
  const GO numVel = 3;
  const GO numPre = 2;
  int errCode     = 0;

  RCP<const map_t> mapVel = Teuchos::rcp(new map_t(numVel, 0, comm));
  RCP<const map_t> mapPre = Teuchos::rcp(new map_t(numPre, 0, comm));
  RCP<const map_t> mapAll = Teuchos::rcp(new map_t(numVel * dim + numPre, 0, comm));

  // Build reordered global IDs
  std::vector<GO> reorderedVec;
  const auto invalidLO = Teuchos::OrdinalTraits<LO>::invalid();

  for (LO lid = 0; lid < static_cast<LO>(mapVel->getLocalNumElements()); ++lid) {
    GO gid0 = mapVel->getGlobalElement(lid);
    for (int i = 0; i < dim; i++) {
      reorderedVec.push_back(gid0 + numVel * i);
    }
  }

  for (LO lid = 0; lid < static_cast<LO>(mapPre->getLocalNumElements()); ++lid) {
    GO gid = mapPre->getGlobalElement(lid);
    reorderedVec.push_back(gid + numVel * dim);
  }

  RCP<const map_t> mapReorder =
      Teuchos::rcp(new map_t(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                             Teuchos::ArrayView<const GO>(reorderedVec), 0, comm));

  RCP<Tpetra::Import<LO, GO, NT> > importReorder =
      Teuchos::rcp(new Tpetra::Import<LO, GO, NT>(mapAll, mapReorder));

  // Build blocked vector of GIDs
  std::vector<std::vector<GO> > blockedVec;
  for (int i = 0; i < dim; i++) {
    std::vector<GO> blk;
    for (LO lid = 0; lid < static_cast<LO>(mapVel->getLocalNumElements()); ++lid) {
      GO gid = mapVel->getGlobalElement(lid);
      blk.push_back(gid + numVel * i);
    }
    blockedVec.push_back(blk);
  }
  {
    std::vector<GO> blk;
    for (LO lid = 0; lid < static_cast<LO>(mapPre->getLocalNumElements()); ++lid) {
      GO gid = mapPre->getGlobalElement(lid);
      blk.push_back(gid + numVel * dim);
    }
    blockedVec.push_back(blk);
  }

  // Read matrices and vector
  RCP<crs_t> ptrMat = Tpetra::MatrixMarket::Reader<crs_t>::readSparseFile("data/tOpMat.mm", comm);
  TEST_ASSERT(!ptrMat.is_null());

  RCP<crs_t> ptrMp = Tpetra::MatrixMarket::Reader<crs_t>::readSparseFile("data/tOpMp.mm", comm);
  TEST_ASSERT(!ptrMp.is_null());

  LinearOp lpMp = Thyra::tpetraLinearOp<ST, LO, GO, NT>(
      Thyra::tpetraVectorSpace<ST, LO, GO, NT>(ptrMp->getRangeMap()),
      Thyra::tpetraVectorSpace<ST, LO, GO, NT>(ptrMp->getDomainMap()), ptrMp);

  RCP<const map_t> mapAllL = mapAll;
  RCP<vec_t> ptrExact = Tpetra::MatrixMarket::Reader<crs_t>::readVectorFile("data/tOpRhs.mm", comm,
                                                                            mapAllL, false, false);
  TEST_ASSERT(!ptrExact.is_null());

  // Reorder matrix
  RCP<crs_t> mat = Teuchos::rcp(new crs_t(mapReorder, ptrMat->getGlobalMaxNumRowEntries()));
  mat->doImport(*ptrMat, *importReorder, Tpetra::INSERT);
  mat->fillComplete();

  // Build augmented Lagrangian operator
  Teko::NS::ALOperator al(blockedVec, mat, lpMp);

  // Initialize vectors
  vec_t x(mapReorder);
  vec_t b(mapReorder);
  x.putScalar(1.0);
  b.putScalar(0.0);

  // Apply operator
  al.apply(x, b);

  // Compare computed vector and exact vector
  b.update(-1.0, *ptrExact, 1.0);
  ST norm2 = b.norm2();

  if (norm2 < 1.0e-15) {
    out << "Test:ALOperator(Tpetra): Passed." << std::endl;
    errCode = 0;
  } else {
    out << "Test:ALOperator(Tpetra): Failed. norm2 = " << norm2 << std::endl;
    errCode = -1;
  }

  TEST_ASSERT(errCode == 0);
}