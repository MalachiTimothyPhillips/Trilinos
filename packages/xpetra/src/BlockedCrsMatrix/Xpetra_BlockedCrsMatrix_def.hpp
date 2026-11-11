// @HEADER
// *****************************************************************************
//             Xpetra: A linear algebra interface package
//
// Copyright 2012 NTESS and the Xpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef XPETRA_BLOCKEDCRSMATRIX_DEF_HPP
#define XPETRA_BLOCKEDCRSMATRIX_DEF_HPP

#include <Tpetra_KokkosCompat_DefaultNode.hpp>

#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_Hashtable.hpp>

#include "Xpetra_ConfigDefs.hpp"
#include "Xpetra_Exceptions.hpp"

#include "Xpetra_BlockedCrsMatrix_decl.hpp"
#include "Xpetra_MapFactory.hpp"
#include "Xpetra_MultiVector.hpp"
#include "Xpetra_BlockedMultiVector.hpp"
#include "Xpetra_MultiVectorFactory.hpp"
#include "Xpetra_BlockedVector.hpp"
#include "Xpetra_CrsGraph.hpp"
#include "Xpetra_CrsMatrix.hpp"
#include "Xpetra_CrsMatrixFactory.hpp"

#include "Xpetra_MapExtractor.hpp"
#include "Xpetra_MapExtractorFactory.hpp"

#include "Xpetra_Matrix.hpp"
#include "Xpetra_MatrixFactory.hpp"
#include "Xpetra_CrsMatrixWrap.hpp"
#include "Xpetra_TpetraCrsMatrix.hpp"
#include "Xpetra_TpetraMap.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Details_makeColMap.hpp>
#include <KokkosSparse_SortCrs.hpp>
#include <cstdlib>
#include <string>

#ifdef HAVE_XPETRA_THYRA
#include <Thyra_ProductVectorSpaceBase.hpp>
#include <Thyra_VectorSpaceBase.hpp>
#include <Thyra_LinearOpBase.hpp>
#include <Thyra_BlockedLinearOpBase.hpp>
#include <Thyra_PhysicallyBlockedLinearOpBase.hpp>
#include "Xpetra_ThyraUtils.hpp"
#endif

#include "Xpetra_VectorFactory.hpp"

/** \file Xpetra_BlockedCrsMatrix.hpp

  Declarations for the class Xpetra::BlockedCrsMatrix.
*/
namespace Xpetra {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::BlockedCrsMatrix(const Teuchos::RCP<const BlockedMap>& rangeMaps,
                                                                              const Teuchos::RCP<const BlockedMap>& domainMaps,
                                                                              size_t numEntriesPerRow)
  : is_diagonal_(true) {
  domainmaps_       = MapExtractorFactory::Build(domainMaps);
  rangemaps_        = MapExtractorFactory::Build(rangeMaps);
  bRangeThyraMode_  = rangeMaps->getThyraMode();
  bDomainThyraMode_ = domainMaps->getThyraMode();

  blocks_.reserve(Rows() * Cols());

  // add CrsMatrix objects in row,column order
  for (size_t r = 0; r < Rows(); ++r)
    for (size_t c = 0; c < Cols(); ++c)
      blocks_.push_back(MatrixFactory::Build(getRangeMap(r, bRangeThyraMode_), numEntriesPerRow));

  // Default view
  CreateDefaultView();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::BlockedCrsMatrix(Teuchos::RCP<const Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>>& rangeMapExtractor,
                                                                              Teuchos::RCP<const Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>>& domainMapExtractor,
                                                                              size_t numEntriesPerRow)
  : is_diagonal_(true)
  , domainmaps_(domainMapExtractor)
  , rangemaps_(rangeMapExtractor) {
  bRangeThyraMode_  = rangeMapExtractor->getThyraMode();
  bDomainThyraMode_ = domainMapExtractor->getThyraMode();

  blocks_.reserve(Rows() * Cols());

  // add CrsMatrix objects in row,column order
  for (size_t r = 0; r < Rows(); ++r)
    for (size_t c = 0; c < Cols(); ++c)
      blocks_.push_back(MatrixFactory::Build(getRangeMap(r, bRangeThyraMode_), numEntriesPerRow));

  // Default view
  CreateDefaultView();
}

#ifdef HAVE_XPETRA_THYRA
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::BlockedCrsMatrix(const Teuchos::RCP<const Thyra::BlockedLinearOpBase<Scalar>>& thyraOp, const Teuchos::RCP<const Teuchos::Comm<int>>& /* comm */)
  : is_diagonal_(true)
  , thyraOp_(thyraOp) {
  // extract information from Thyra blocked operator and rebuilt information
  const Teuchos::RCP<const Thyra::ProductVectorSpaceBase<Scalar>> productRangeSpace  = thyraOp->productRange();
  const Teuchos::RCP<const Thyra::ProductVectorSpaceBase<Scalar>> productDomainSpace = thyraOp->productDomain();

  int numRangeBlocks  = productRangeSpace->numBlocks();
  int numDomainBlocks = productDomainSpace->numBlocks();

  // build range map extractor from Thyra::BlockedLinearOpBase object
  std::vector<Teuchos::RCP<const Map>> subRangeMaps(numRangeBlocks);
  for (size_t r = 0; r < Teuchos::as<size_t>(numRangeBlocks); ++r) {
    for (size_t c = 0; c < Teuchos::as<size_t>(numDomainBlocks); ++c) {
      if (thyraOp->blockExists(r, c)) {
        // we only need at least one block in each block row to extract the range map
        Teuchos::RCP<const Thyra::LinearOpBase<Scalar>> const_op = thyraOp->getBlock(r, c);  // nonConst access is not allowed.
        Teuchos::RCP<const Xpetra::Matrix<Scalar, LO, GO, Node>> xop =
            Xpetra::ThyraUtils<Scalar, LocalOrdinal, GlobalOrdinal, Node>::toXpetra(const_op);
        subRangeMaps[r] = xop->getRangeMap();
        if (r != c) is_diagonal_ = false;
        break;
      }
    }
  }
  Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> fullRangeMap = mergeMaps(subRangeMaps);

  // check whether the underlying Thyra operator uses Thyra-style numbering (default for most applications) or
  // Xpetra style numbering
  bool bRangeUseThyraStyleNumbering = false;
  size_t numAllElements             = 0;
  for (size_t v = 0; v < subRangeMaps.size(); ++v) {
    numAllElements += subRangeMaps[v]->getGlobalNumElements();
  }
  if (fullRangeMap->getGlobalNumElements() != numAllElements) bRangeUseThyraStyleNumbering = true;
  rangemaps_ = Teuchos::rcp(new Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>(fullRangeMap, subRangeMaps, bRangeUseThyraStyleNumbering));

  // build domain map extractor from Thyra::BlockedLinearOpBase object
  std::vector<Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>> subDomainMaps(numDomainBlocks);
  for (size_t c = 0; c < Teuchos::as<size_t>(numDomainBlocks); ++c) {
    for (size_t r = 0; r < Teuchos::as<size_t>(numRangeBlocks); ++r) {
      if (thyraOp->blockExists(r, c)) {
        // we only need at least one block in each block row to extract the range map
        Teuchos::RCP<const Thyra::LinearOpBase<Scalar>> const_op = thyraOp->getBlock(r, c);  // nonConst access is not allowed.
        Teuchos::RCP<const Xpetra::Matrix<Scalar, LO, GO, Node>> xop =
            Xpetra::ThyraUtils<Scalar, LO, GO, Node>::toXpetra(const_op);
        subDomainMaps[c] = xop->getDomainMap();
        break;
      }
    }
  }
  Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> fullDomainMap = mergeMaps(subDomainMaps);
  // plausibility check for numbering style (Xpetra or Thyra)
  bool bDomainUseThyraStyleNumbering = false;
  numAllElements                     = 0;
  for (size_t v = 0; v < subDomainMaps.size(); ++v) {
    numAllElements += subDomainMaps[v]->getGlobalNumElements();
  }
  if (fullDomainMap->getGlobalNumElements() != numAllElements) bDomainUseThyraStyleNumbering = true;
  domainmaps_ = Teuchos::rcp(new Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>(fullDomainMap, subDomainMaps, bDomainUseThyraStyleNumbering));

  // store numbering mode
  bRangeThyraMode_  = bRangeUseThyraStyleNumbering;
  bDomainThyraMode_ = bDomainUseThyraStyleNumbering;

  // add CrsMatrix objects in row,column order
  blocks_.reserve(Rows() * Cols());
  for (size_t r = 0; r < Rows(); ++r) {
    for (size_t c = 0; c < Cols(); ++c) {
      if (thyraOp->blockExists(r, c)) {
        // TODO we do not support nested Thyra operators here!
        Teuchos::RCP<const Thyra::LinearOpBase<Scalar>> const_op = thyraOp->getBlock(r, c);                                         // nonConst access is not allowed.
        Teuchos::RCP<Thyra::LinearOpBase<Scalar>> op             = Teuchos::rcp_const_cast<Thyra::LinearOpBase<Scalar>>(const_op);  // cast away const
        Teuchos::RCP<Xpetra::Matrix<Scalar, LO, GO, Node>> xop =
            Xpetra::ThyraUtils<Scalar, LO, GO, Node>::toXpetra(op);
        Teuchos::RCP<Xpetra::CrsMatrixWrap<Scalar, LO, GO, Node>> xwrap =
            Teuchos::rcp_dynamic_cast<Xpetra::CrsMatrixWrap<Scalar, LO, GO, Node>>(xop, true);
        blocks_.push_back(xwrap);
      } else {
        // add empty block
        blocks_.push_back(MatrixFactory::Build(getRangeMap(r, bRangeThyraMode_), 0));
      }
    }
  }
  // Default view
  CreateDefaultView();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::mergeMaps(std::vector<Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>>& subMaps) {
  // TODO merging for Thyra mode is missing (similar to what we do in constructor of MapExtractor

  // merge submaps to global map
  std::vector<GlobalOrdinal> gids;
  for (size_t tt = 0; tt < subMaps.size(); ++tt) {
    Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> subMap = subMaps[tt];
#if 1
    Teuchos::ArrayView<const GlobalOrdinal> subMapGids = subMap->getLocalElementList();
    gids.insert(gids.end(), subMapGids.begin(), subMapGids.end());
#else
    size_t myNumElements = subMap->getLocalNumElements();
    for (LocalOrdinal l = 0; l < Teuchos::as<LocalOrdinal>(myNumElements); ++l) {
      GlobalOrdinal gid = subMap->getGlobalElement(l);
      gids.push_back(gid);
    }
#endif
  }

  // we have to sort the matrix entries and get rid of the double entries
  // since we use this to detect Thyra-style numbering or Xpetra-style
  // numbering. In Thyra-style numbering mode, the Xpetra::MapExtractor builds
  // the correct row maps.
  const GO INVALID = Teuchos::OrdinalTraits<Xpetra::global_size_t>::invalid();
  std::sort(gids.begin(), gids.end());
  gids.erase(std::unique(gids.begin(), gids.end()), gids.end());
  Teuchos::ArrayView<GO> gidsView(&gids[0], gids.size());
  Teuchos::RCP<Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> fullMap = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(subMaps[0]->lib(), INVALID, gidsView, subMaps[0]->getIndexBase(), subMaps[0]->getComm());
  return fullMap;
}

#endif

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::~BlockedCrsMatrix() = default;

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::insertGlobalValues(GlobalOrdinal globalRow, const ArrayView<const GlobalOrdinal>& cols, const ArrayView<const Scalar>& vals) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::insertGlobalValues");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->insertGlobalValues(globalRow, cols, vals);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("insertGlobalValues not supported by BlockedCrsMatrix");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::insertLocalValues(LocalOrdinal localRow, const ArrayView<const LocalOrdinal>& cols, const ArrayView<const Scalar>& vals) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::insertLocalValues");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->insertLocalValues(localRow, cols, vals);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("insertLocalValues not supported by BlockedCrsMatrix");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::removeEmptyProcessesInPlace(const Teuchos::RCP<const Map>& newMap) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::removeEmptyProcessesInPlace");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->removeEmptyProcessesInPlace(newMap);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("removeEmptyProcesses not supported by BlockedCrsMatrix");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::replaceGlobalValues(GlobalOrdinal globalRow,
                                                                                      const ArrayView<const GlobalOrdinal>& cols,
                                                                                      const ArrayView<const Scalar>& vals) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::replaceGlobalValues");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->replaceGlobalValues(globalRow, cols, vals);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("replaceGlobalValues not supported by BlockedCrsMatrix");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::replaceLocalValues(LocalOrdinal localRow,
                                                                                     const ArrayView<const LocalOrdinal>& cols,
                                                                                     const ArrayView<const Scalar>& vals) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::replaceLocalValues");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->replaceLocalValues(localRow, cols, vals);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("replaceLocalValues not supported by BlockedCrsMatrix");
}

//! Set all matrix entries equal to scalar
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::setAllToScalar(const Scalar& alpha) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::setAllToScalar");
  for (size_t row = 0; row < Rows(); row++) {
    for (size_t col = 0; col < Cols(); col++) {
      if (!getMatrix(row, col).is_null()) {
        getMatrix(row, col)->setAllToScalar(alpha);
      }
    }
  }
}

//! Scale the current values of a matrix, this = alpha*this.
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::scale(const Scalar& alpha) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::scale");
  for (size_t row = 0; row < Rows(); row++) {
    for (size_t col = 0; col < Cols(); col++) {
      if (!getMatrix(row, col).is_null()) {
        getMatrix(row, col)->scale(alpha);
      }
    }
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::resumeFill(const RCP<ParameterList>& params) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::resumeFill");
  for (size_t row = 0; row < Rows(); row++) {
    for (size_t col = 0; col < Cols(); col++) {
      if (!getMatrix(row, col).is_null()) {
        getMatrix(row, col)->resumeFill(params);
      }
    }
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::fillComplete(const RCP<const Map>& domainMap, const RCP<const Map>& rangeMap, const RCP<ParameterList>& params) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::fillComplete");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->fillComplete(domainMap, rangeMap, params);
    return;
  }
  fillComplete(params);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::fillComplete(const RCP<ParameterList>& params) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::fillComplete");
  TEUCHOS_TEST_FOR_EXCEPTION(rangemaps_ == Teuchos::null, Xpetra::Exceptions::RuntimeError, "BlockedCrsMatrix::fillComplete: rangemaps_ is not set. Error.");

  for (size_t r = 0; r < Rows(); ++r)
    for (size_t c = 0; c < Cols(); ++c) {
      if (getMatrix(r, c) != Teuchos::null) {
        Teuchos::RCP<Matrix> Ablock = getMatrix(r, c);
        if (r != c) is_diagonal_ = false;
        if (Ablock != Teuchos::null && !Ablock->isFillComplete())
          Ablock->fillComplete(getDomainMap(c, bDomainThyraMode_), getRangeMap(r, bRangeThyraMode_), params);
      }
    }

#if 0
      // get full row map
      RCP<const Map> rangeMap = rangemaps_->getFullMap();
      fullrowmap_ = MapFactory::Build(rangeMap()->lib(), rangeMap()->getGlobalNumElements(), rangeMap()->getLocalElementList(), rangeMap()->getIndexBase(), rangeMap()->getComm());

      // build full col map
      fullcolmap_ = Teuchos::null; // delete old full column map

      std::vector<GO> colmapentries;
      for (size_t c = 0; c < Cols(); ++c) {
        // copy all local column lids of all block rows to colset
        std::set<GO> colset;
        for (size_t r = 0; r < Rows(); ++r) {
          Teuchos::RCP<CrsMatrix> Ablock = getMatrix(r,c);

          if (Ablock != Teuchos::null) {
            Teuchos::ArrayView<const GO> colElements = Ablock->getColMap()->getLocalElementList();
            Teuchos::RCP<const Map>      colmap      = Ablock->getColMap();
            copy(colElements.getRawPtr(), colElements.getRawPtr() + colElements.size(), inserter(colset, colset.begin()));
          }
        }

        // remove duplicates (entries which are in column maps of more than one block row)
        colmapentries.reserve(colmapentries.size() + colset.size());
        copy(colset.begin(), colset.end(), back_inserter(colmapentries));
        sort(colmapentries.begin(), colmapentries.end());
        typename std::vector<GO>::iterator gendLocation;
        gendLocation = std::unique(colmapentries.begin(), colmapentries.end());
        colmapentries.erase(gendLocation,colmapentries.end());
      }

      // sum up number of local elements
      size_t numGlobalElements = 0;
      Teuchos::reduceAll(*(rangeMap->getComm()), Teuchos::REDUCE_SUM, colmapentries.size(), Teuchos::outArg(numGlobalElements));

      // store global full column map
      const Teuchos::ArrayView<const GO> aView = Teuchos::ArrayView<const GO>(colmapentries);
      fullcolmap_ = MapFactory::Build(rangeMap->lib(), numGlobalElements, aView, 0, rangeMap->getComm());
#endif
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
global_size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getGlobalNumRows() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getGlobalNumRows");
  global_size_t globalNumRows = 0;

  for (size_t row = 0; row < Rows(); row++)
    for (size_t col = 0; col < Cols(); col++)
      if (!getMatrix(row, col).is_null()) {
        globalNumRows += getMatrix(row, col)->getGlobalNumRows();
        break;  // we need only one non-null matrix in a row
      }

  return globalNumRows;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
global_size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getGlobalNumCols() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getGlobalNumCols");
  global_size_t globalNumCols = 0;

  for (size_t col = 0; col < Cols(); col++)
    for (size_t row = 0; row < Rows(); row++)
      if (!getMatrix(row, col).is_null()) {
        globalNumCols += getMatrix(row, col)->getGlobalNumCols();
        break;  // we need only one non-null matrix in a col
      }

  return globalNumCols;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getLocalNumRows() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getLocalNumRows");
  global_size_t nodeNumRows = 0;

  for (size_t row = 0; row < Rows(); ++row)
    for (size_t col = 0; col < Cols(); col++)
      if (!getMatrix(row, col).is_null()) {
        nodeNumRows += getMatrix(row, col)->getLocalNumRows();
        break;  // we need only one non-null matrix in a row
      }

  return nodeNumRows;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
global_size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getGlobalNumEntries() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getGlobalNumEntries");
  global_size_t globalNumEntries = 0;

  for (size_t row = 0; row < Rows(); ++row)
    for (size_t col = 0; col < Cols(); ++col)
      if (!getMatrix(row, col).is_null())
        globalNumEntries += getMatrix(row, col)->getGlobalNumEntries();

  return globalNumEntries;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getLocalNumEntries() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getLocalNumEntries");
  global_size_t nodeNumEntries = 0;

  for (size_t row = 0; row < Rows(); ++row)
    for (size_t col = 0; col < Cols(); ++col)
      if (!getMatrix(row, col).is_null())
        nodeNumEntries += getMatrix(row, col)->getLocalNumEntries();

  return nodeNumEntries;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getNumEntriesInLocalRow(LocalOrdinal localRow) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getNumEntriesInLocalRow");
  GlobalOrdinal gid           = this->getRowMap()->getGlobalElement(localRow);
  size_t row                  = getBlockedRangeMap()->getMapIndexForGID(gid);
  LocalOrdinal lid            = getBlockedRangeMap()->getMap(row)->getLocalElement(gid);
  size_t numEntriesInLocalRow = 0;
  for (size_t col = 0; col < Cols(); ++col)
    if (!getMatrix(row, col).is_null())
      numEntriesInLocalRow += getMatrix(row, col)->getNumEntriesInLocalRow(lid);
  return numEntriesInLocalRow;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getNumEntriesInGlobalRow");
  size_t row                   = getBlockedRangeMap()->getMapIndexForGID(globalRow);
  size_t numEntriesInGlobalRow = 0;
  for (size_t col = 0; col < Cols(); ++col)
    if (!getMatrix(row, col).is_null())
      numEntriesInGlobalRow += getMatrix(row, col)->getNumEntriesInGlobalRow(globalRow);
  return numEntriesInGlobalRow;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getGlobalMaxNumRowEntries() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getGlobalMaxNumRowEntries");
  global_size_t globalMaxEntries = 0;

  for (size_t row = 0; row < Rows(); row++) {
    global_size_t globalMaxEntriesBlockRows = 0;
    for (size_t col = 0; col < Cols(); col++) {
      if (!getMatrix(row, col).is_null()) {
        globalMaxEntriesBlockRows += getMatrix(row, col)->getGlobalMaxNumRowEntries();
      }
    }
    if (globalMaxEntriesBlockRows > globalMaxEntries)
      globalMaxEntries = globalMaxEntriesBlockRows;
  }
  return globalMaxEntries;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getLocalMaxNumRowEntries() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getLocalMaxNumRowEntries");
  size_t localMaxEntries = 0;

  for (size_t row = 0; row < Rows(); row++) {
    size_t localMaxEntriesBlockRows = 0;
    for (size_t col = 0; col < Cols(); col++) {
      if (!getMatrix(row, col).is_null()) {
        localMaxEntriesBlockRows += getMatrix(row, col)->getLocalMaxNumRowEntries();
      }
    }
    if (localMaxEntriesBlockRows > localMaxEntries)
      localMaxEntries = localMaxEntriesBlockRows;
  }
  return localMaxEntries;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::isLocallyIndexed() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::isLocallyIndexed");
  for (size_t i = 0; i < blocks_.size(); ++i)
    if (blocks_[i] != Teuchos::null && !blocks_[i]->isLocallyIndexed())
      return false;
  return true;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::isGloballyIndexed() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::isGloballyIndexed");
  for (size_t i = 0; i < blocks_.size(); i++)
    if (blocks_[i] != Teuchos::null && !blocks_[i]->isGloballyIndexed())
      return false;
  return true;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::isFillComplete() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::isFillComplete");
  for (size_t i = 0; i < blocks_.size(); i++)
    if (blocks_[i] != Teuchos::null && !blocks_[i]->isFillComplete())
      return false;
  return true;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getLocalRowCopy(LocalOrdinal LocalRow,
                                                                                  const ArrayView<LocalOrdinal>& Indices,
                                                                                  const ArrayView<Scalar>& Values,
                                                                                  size_t& NumEntries) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getLocalRowCopy");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->getLocalRowCopy(LocalRow, Indices, Values, NumEntries);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("getLocalRowCopy not supported by BlockedCrsMatrix");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView<const GlobalOrdinal>& indices, ArrayView<const Scalar>& values) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getGlobalRowView");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->getGlobalRowView(GlobalRow, indices, values);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("getGlobalRowView not supported by BlockedCrsMatrix");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getLocalRowView(LocalOrdinal LocalRow, ArrayView<const LocalOrdinal>& indices, ArrayView<const Scalar>& values) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getLocalRowView");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->getLocalRowView(LocalRow, indices, values);
    return;
  } else if (is_diagonal_) {
    GlobalOrdinal gid = this->getRowMap()->getGlobalElement(LocalRow);
    size_t row        = getBlockedRangeMap()->getMapIndexForGID(gid);
    getMatrix(row, row)->getLocalRowView(getMatrix(row, row)->getRowMap()->getLocalElement(gid), indices, values);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("getLocalRowView not supported by BlockedCrsMatrix");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getLocalDiagCopy(Vector& diag) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getLocalDiagCopy");

  // RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

  Teuchos::RCP<Vector> rcpdiag      = Teuchos::rcpFromRef(diag);
  Teuchos::RCP<BlockedVector> bdiag = Teuchos::rcp_dynamic_cast<BlockedVector>(rcpdiag);

  // special treatment for 1x1 block matrices
  // ReorderedBlockedCrsMatrix object encapsulate single blocks in ReorderedBlocks
  // BlockedVectors have Vector objects as Leaf objects.
  if (Rows() == 1 && Cols() == 1 && bdiag.is_null() == true) {
    Teuchos::RCP<const Matrix> rm = getMatrix(0, 0);
    rm->getLocalDiagCopy(diag);
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(bdiag.is_null() == true, Xpetra::Exceptions::RuntimeError, "BlockedCrsMatrix::getLocalDiagCopy: diag must be a Blocked(Multi)Vector.");
  TEUCHOS_TEST_FOR_EXCEPTION(bdiag->getNumVectors() != 1, Xpetra::Exceptions::RuntimeError, "BlockedCrsMatrix::getLocalDiagCopy: diag must be a Blocked(Multi)Vector with exactly one vector. However, the number of stored vectors is " << bdiag->getNumVectors());
  TEUCHOS_TEST_FOR_EXCEPTION(bdiag->getBlockedMap()->getNumMaps() != this->Rows(), Xpetra::Exceptions::RuntimeError,
                             "BlockedCrsMatrix::getLocalDiagCopy(): the number of blocks in diag differ from the number of blocks in this operator.");
  // XPETRA_TEST_FOR_EXCEPTION(bdiag->getMap()->isSameAs(*(getMap())) == false, Xpetra::Exceptions::RuntimeError,
  //   "BlockedCrsMatrix::getLocalDiagCopy(): the map of the vector diag is not compatible with the map of the blocked operator." );

  for (size_t row = 0; row < Rows(); row++) {
    Teuchos::RCP<const Matrix> rm = getMatrix(row, row);
    if (!rm.is_null()) {
      Teuchos::RCP<Vector> rv = VectorFactory::Build(bdiag->getBlockedMap()->getMap(row, bdiag->getBlockedMap()->getThyraMode()));
      rm->getLocalDiagCopy(*rv);
      bdiag->setMultiVector(row, rv, bdiag->getBlockedMap()->getThyraMode());
    }
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::leftScale(const Vector& x) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::leftScale");

  Teuchos::RCP<const Vector> rcpx      = Teuchos::rcpFromRef(x);
  Teuchos::RCP<const BlockedVector> bx = Teuchos::rcp_dynamic_cast<const BlockedVector>(rcpx);

  // RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

  // special treatment for 1xn block matrices
  // ReorderedBlockedCrsMatrix object encapsulate single blocks in ReorderedBlocks
  // BlockedVectors have Vector objects as Leaf objects.
  if (Rows() == 1 && bx.is_null() == true) {
    for (size_t col = 0; col < Cols(); ++col) {
      Teuchos::RCP<Matrix> rm = getMatrix(0, col);
      rm->leftScale(x);
    }
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(bx.is_null() == true, Xpetra::Exceptions::RuntimeError, "BlockedCrsMatrix::leftScale: x must be a Blocked(Multi)Vector.");
  TEUCHOS_TEST_FOR_EXCEPTION(bx->getNumVectors() != 1, Xpetra::Exceptions::RuntimeError, "BlockedCrsMatrix::leftScale: x must be a Blocked(Multi)Vector with exactly one vector. However, the number of stored vectors is " << bx->getNumVectors());
  TEUCHOS_TEST_FOR_EXCEPTION(bx->getBlockedMap()->getNumMaps() != this->Rows(), Xpetra::Exceptions::RuntimeError,
                             "BlockedCrsMatrix::leftScale(): the number of blocks in diag differ from the number of blocks in this operator.");

  for (size_t row = 0; row < Rows(); row++) {
    Teuchos::RCP<const MultiVector> rmv = bx->getMultiVector(row);
    Teuchos::RCP<const Vector> rscale   = rmv->getVector(0);
    XPETRA_TEST_FOR_EXCEPTION(rscale.is_null() == true, Xpetra::Exceptions::RuntimeError, "BlockedCrsMatrix::leftScale: x must be a Vector.");
    for (size_t col = 0; col < Cols(); ++col) {
      Teuchos::RCP<Matrix> rm = getMatrix(row, col);
      if (!rm.is_null()) {
        rm->leftScale(*rscale);
      }
    }
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::rightScale(const Vector& x) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::rightScale");

  Teuchos::RCP<const Vector> rcpx      = Teuchos::rcpFromRef(x);
  Teuchos::RCP<const BlockedVector> bx = Teuchos::rcp_dynamic_cast<const BlockedVector>(rcpx);

  // RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

  // special treatment for nx1 block matrices
  // ReorderedBlockedCrsMatrix object encapsulate single blocks in ReorderedBlocks
  // BlockedVectors have Vector objects as Leaf objects.
  if (Cols() == 1 && bx.is_null() == true) {
    for (size_t row = 0; row < Rows(); ++row) {
      Teuchos::RCP<Matrix> rm = getMatrix(row, 0);
      rm->rightScale(x);
    }
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(bx.is_null() == true, Xpetra::Exceptions::RuntimeError, "BlockedCrsMatrix::rightScale: x must be a Blocked(Multi)Vector.");
  TEUCHOS_TEST_FOR_EXCEPTION(bx->getNumVectors() != 1, Xpetra::Exceptions::RuntimeError, "BlockedCrsMatrix::rightScale: x must be a Blocked(Multi)Vector with exactly one vector. However, the number of stored vectors is " << bx->getNumVectors());
  TEUCHOS_TEST_FOR_EXCEPTION(bx->getBlockedMap()->getNumMaps() != this->Cols(), Xpetra::Exceptions::RuntimeError,
                             "BlockedCrsMatrix::rightScale(): the number of blocks in diag differ from the number of blocks in this operator.");

  for (size_t col = 0; col < Cols(); ++col) {
    Teuchos::RCP<const MultiVector> rmv = bx->getMultiVector(col);
    Teuchos::RCP<const Vector> rscale   = rmv->getVector(0);
    XPETRA_TEST_FOR_EXCEPTION(rscale.is_null() == true, Xpetra::Exceptions::RuntimeError, "BlockedCrsMatrix::leftScale: x must be a Vector.");
    for (size_t row = 0; row < Rows(); row++) {
      Teuchos::RCP<Matrix> rm = getMatrix(row, col);
      if (!rm.is_null()) {
        rm->rightScale(*rscale);
      }
    }
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
typename ScalarTraits<Scalar>::magnitudeType BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getFrobeniusNorm() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getFrobeniusNorm");
  typename ScalarTraits<Scalar>::magnitudeType ret = Teuchos::ScalarTraits<Scalar>::magnitude(Teuchos::ScalarTraits<Scalar>::zero());
  for (size_t col = 0; col < Cols(); ++col) {
    for (size_t row = 0; row < Rows(); ++row) {
      if (getMatrix(row, col) != Teuchos::null) {
        typename ScalarTraits<Scalar>::magnitudeType n = getMatrix(row, col)->getFrobeniusNorm();
        ret += n * n;
      }
    }
  }
  return Teuchos::ScalarTraits<typename ScalarTraits<Scalar>::magnitudeType>::squareroot(ret);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::haveGlobalConstants() const { return true; }

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::apply(const MultiVector& X, MultiVector& Y, Teuchos::ETransp mode, Scalar alpha, Scalar beta, bool sumInterfaceValues,
                                                                        const RCP<Xpetra::Import<LocalOrdinal, GlobalOrdinal, Node>>& regionInterfaceImporter,
                                                                        const Teuchos::ArrayRCP<LocalOrdinal>& regionInterfaceLIDs) const {}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::applyBlockedInPlace(const MultiVector& X, MultiVector& Y,
                                                                                      Teuchos::ETransp mode,
                                                                                      Scalar alpha,
                                                                                      Scalar beta) const {
  auto bX = dynamic_cast<const BlockedMultiVector*>(&X);
  auto bY = dynamic_cast<BlockedMultiVector*>(&Y);
  if (bX == nullptr || bY == nullptr || static_cast<const MultiVector*>(bX) == static_cast<const MultiVector*>(bY))
    return false;

  // In NO_TRANS, X lives on the domain blocks and Y on the range blocks; TRANS swaps them.
  const bool noTrans          = mode == Teuchos::NO_TRANS;
  const size_t numXBlocks     = noTrans ? Cols() : Rows();
  const size_t numYBlocks     = noTrans ? Rows() : Cols();
  const bool xThyraMode       = noTrans ? bDomainThyraMode_ : bRangeThyraMode_;
  const bool yThyraMode       = noTrans ? bRangeThyraMode_ : bDomainThyraMode_;
  if (bX->getBlockedMap()->getNumMaps() != numXBlocks || bY->getBlockedMap()->getNumMaps() != numYBlocks ||
      bX->getBlockedMap()->getThyraMode() != xThyraMode || bY->getBlockedMap()->getThyraMode() != yThyraMode)
    return false;
  for (size_t r = 0; r < Rows(); ++r)
    for (size_t c = 0; c < Cols(); ++c)
      if (!getMatrix(r, c).is_null() && !Teuchos::rcp_dynamic_cast<const BlockedCrsMatrix>(getMatrix(r, c)).is_null())
        return false;

  const SC zero = ScalarTraits<SC>::zero();
  const SC one  = ScalarTraits<SC>::one();
  for (size_t i = 0; i < numYBlocks; ++i) {
    RCP<MultiVector> Yi = bY->getMultiVector(i, yThyraMode);
    bool contributed    = false;  // the first contribution scales the old Y_i by beta, the rest accumulate
    for (size_t j = 0; j < numXBlocks; ++j) {
      RCP<Matrix> Aij = noTrans ? getMatrix(i, j) : getMatrix(j, i);
      // Skip blocks known to be empty. Global constants (and with them getGlobalNumEntries()) exist on every rank or
      // on none, so every rank skips the same blocks; blocks without them (e.g. transposes) are applied.
      if (Aij.is_null() || (Aij->haveGlobalConstants() && Aij->getGlobalNumEntries() == 0))
        continue;
      Aij->apply(*bX->getMultiVector(j, xThyraMode), *Yi, mode, alpha, contributed ? one : beta);
      contributed = true;
    }
    if (!contributed) {  // no block with entries in this row: Y_i = beta * Y_i
      if (beta == zero)
        Yi->putScalar(zero);
      else if (beta != one)
        Yi->scale(beta);
    }
  }
  return true;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::apply(const MultiVector& X, MultiVector& Y,
                                                                        Teuchos::ETransp mode,
                                                                        Scalar alpha,
                                                                        Scalar beta) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::apply");
  // using Teuchos::RCP;

  TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS && mode != Teuchos::TRANS, Xpetra::Exceptions::RuntimeError,
                             "apply() only supports the following modes: NO_TRANS and TRANS.");

  // check whether input parameters are blocked or not
  RCP<const MultiVector> refX         = rcpFromRef(X);
  RCP<const BlockedMultiVector> refbX = Teuchos::rcp_dynamic_cast<const BlockedMultiVector>(refX);
  // RCP<MultiVector>               tmpY = rcpFromRef(Y);
  // RCP<BlockedMultiVector>       tmpbY = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(tmpY);

  // TODO get rid of me: adapt MapExtractor
  bool bBlockedX = (refbX != Teuchos::null) ? true : false;

  // Fast path for blocked X and Y laid out like this matrix: apply each block straight into the matching block of
  // Y, accumulating in place, and skip blocks without entries. No temporaries, and no Import for empty blocks.
  if (applyBlockedInPlace(X, Y, mode, alpha, beta))
    return;

  // create (temporary) vectors for output
  // In the end we call Y.update(alpha, *tmpY, beta). Therefore we need a new vector storing the temporary results
  RCP<MultiVector> tmpY = MultiVectorFactory::Build(Y.getMap(), Y.getNumVectors(), true);

  // RCP<Teuchos::FancyOStream> out = rcp(new Teuchos::FancyOStream(rcp(&std::cout,false)));

  SC one = ScalarTraits<SC>::one();

  if (mode == Teuchos::NO_TRANS) {
    for (size_t row = 0; row < Rows(); row++) {
      RCP<MultiVector> Yblock = rangemaps_->getVector(row, Y.getNumVectors(), bRangeThyraMode_, true);
      for (size_t col = 0; col < Cols(); col++) {
        // extract matrix block
        RCP<Matrix> Ablock = getMatrix(row, col);

        if (Ablock.is_null())
          continue;

        // check whether Ablock is itself a blocked operator
        // If it is a blocked operator we have to provide Xpetra style GIDs, i.e. we have to transform GIDs
        bool bBlockedSubMatrix = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(Ablock) == Teuchos::null ? false : true;

        // input/output vectors for local block operation
        RCP<const MultiVector> Xblock = Teuchos::null;  // subpart of X vector to be applied to subblock of A

        // extract sub part of X using Xpetra or Thyra GIDs
        // if submatrix is again blocked, we extract it using Xpetra style gids. If it is a single
        // block matrix we use the Thyra or Xpetra style GIDs that are used to store the matrix
        if (bBlockedX)
          Xblock = domainmaps_->ExtractVector(refbX, col, bDomainThyraMode_);
        else
          Xblock = domainmaps_->ExtractVector(refX, col, bBlockedSubMatrix == true ? false : bDomainThyraMode_);

        RCP<MultiVector> tmpYblock = rangemaps_->getVector(row, Y.getNumVectors(), bRangeThyraMode_, false);  // subpart of Y vector containing part of solution of Xblock applied to Ablock
        Ablock->apply(*Xblock, *tmpYblock);

        Yblock->update(one, *tmpYblock, one);
      }
      rangemaps_->InsertVector(Yblock, row, tmpY, bRangeThyraMode_);
    }

  } else if (mode == Teuchos::TRANS) {
    // TODO: test me!
    for (size_t col = 0; col < Cols(); col++) {
      RCP<MultiVector> Yblock = domainmaps_->getVector(col, Y.getNumVectors(), bDomainThyraMode_, true);

      for (size_t row = 0; row < Rows(); row++) {
        RCP<Matrix> Ablock = getMatrix(row, col);

        if (Ablock.is_null())
          continue;

        // check whether Ablock is itself a blocked operator
        // If it is a blocked operator we have to provide Xpetra style GIDs, i.e. we have to transform GIDs
        bool bBlockedSubMatrix = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(Ablock) == Teuchos::null ? false : true;

        RCP<const MultiVector> Xblock = Teuchos::null;

        // extract sub part of X using Xpetra or Thyra GIDs
        if (bBlockedX)
          Xblock = rangemaps_->ExtractVector(refbX, row, bRangeThyraMode_);
        else
          Xblock = rangemaps_->ExtractVector(refX, row, bBlockedSubMatrix == true ? false : bRangeThyraMode_);
        RCP<MultiVector> tmpYblock = domainmaps_->getVector(col, Y.getNumVectors(), bDomainThyraMode_, false);
        Ablock->apply(*Xblock, *tmpYblock, Teuchos::TRANS);

        Yblock->update(one, *tmpYblock, one);
      }
      domainmaps_->InsertVector(Yblock, col, tmpY, bDomainThyraMode_);
    }
  }
  Y.update(alpha, *tmpY, beta);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getFullDomainMap() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getFullDomainMap()");
  return domainmaps_->getFullMap();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::BlockedMap<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getBlockedDomainMap() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getBlockedDomainMap()");
  return domainmaps_->getBlockedMap();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
const RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getDomainMap() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getDomainMap()");
  return domainmaps_->getMap(); /*domainmaps_->getFullMap();*/
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getDomainMap(size_t i) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getDomainMap(size_t)");
  return domainmaps_->getMap(i, bDomainThyraMode_);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getDomainMap(size_t i, bool bThyraMode) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getDomainMap(size_t,bool)");
  return domainmaps_->getMap(i, bThyraMode);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getFullRangeMap() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getRangeMap()");
  return rangemaps_->getFullMap();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::BlockedMap<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getBlockedRangeMap() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getBlockedRangeMap()");
  return rangemaps_->getBlockedMap();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
const RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getRangeMap() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getRangeMap()");
  return rangemaps_->getMap(); /*rangemaps_->getFullMap();*/
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getRangeMap(size_t i) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getRangeMap(size_t)");
  return rangemaps_->getMap(i, bRangeThyraMode_);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getRangeMap(size_t i, bool bThyraMode) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getRangeMap(size_t,bool)");
  return rangemaps_->getMap(i, bThyraMode);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getRangeMapExtractor() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getRangeMapExtractor()");
  return rangemaps_;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getDomainMapExtractor() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getDomainMapExtractor()");
  return domainmaps_;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::bgs_apply(
    const MultiVector& X,   ///< Vector to be multiplied by matrix (input)
    MultiVector& Y,         ///< result vector
    size_t row,             ///< Index of block row to be treated
    Teuchos::ETransp mode,  ///< Transpose mode
    Scalar alpha,           ///< scaling factor for result of matrix-vector product
    Scalar beta             ///< scaling factor for linear combination with result vector
) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::bgs_apply");
  // using Teuchos::RCP;

  TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS && mode != Teuchos::TRANS, Xpetra::Exceptions::RuntimeError,
                             "apply() only supports the following modes: NO_TRANS and TRANS.");

  // check whether input parameters are blocked or not
  RCP<const MultiVector> refX         = rcpFromRef(X);
  RCP<const BlockedMultiVector> refbX = Teuchos::rcp_dynamic_cast<const BlockedMultiVector>(refX);
  // RCP<MultiVector>               tmpY = rcpFromRef(Y);
  // RCP<BlockedMultiVector>       tmpbY = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(tmpY);

  bool bBlockedX = (refbX != Teuchos::null) ? true : false;

  // create (temporary) vectors for output
  // In the end we call Y.update(alpha, *tmpY, beta). Therefore we need a new vector storing the temporary results
  RCP<MultiVector> tmpY = MultiVectorFactory::Build(Y.getMap(), Y.getNumVectors(), true);

  SC one = ScalarTraits<SC>::one();

  if (mode == Teuchos::NO_TRANS) {
    RCP<MultiVector> Yblock = rangemaps_->getVector(row, Y.getNumVectors(), bRangeThyraMode_, true);
    for (size_t col = 0; col < Cols(); col++) {
      // extract matrix block
      RCP<Matrix> Ablock = getMatrix(row, col);

      if (Ablock.is_null())
        continue;

      // check whether Ablock is itself a blocked operator
      // If it is a blocked operator we have to provide Xpetra style GIDs, i.e. we have to transform GIDs
      bool bBlockedSubMatrix = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(Ablock) == Teuchos::null ? false : true;

      // input/output vectors for local block operation
      RCP<const MultiVector> Xblock = Teuchos::null;  // subpart of X vector to be applied to subblock of A

      // extract sub part of X using Xpetra or Thyra GIDs
      // if submatrix is again blocked, we extract it using Xpetra style gids. If it is a single
      // block matrix we use the Thyra or Xpetra style GIDs that are used to store the matrix
      if (bBlockedX)
        Xblock = domainmaps_->ExtractVector(refbX, col, bDomainThyraMode_);
      else
        Xblock = domainmaps_->ExtractVector(refX, col, bBlockedSubMatrix == true ? false : bDomainThyraMode_);

      RCP<MultiVector> tmpYblock = rangemaps_->getVector(row, Y.getNumVectors(), bRangeThyraMode_, false);  // subpart of Y vector containing part of solution of Xblock applied to Ablock
      Ablock->apply(*Xblock, *tmpYblock);

      Yblock->update(one, *tmpYblock, one);
    }
    rangemaps_->InsertVector(Yblock, row, tmpY, bRangeThyraMode_);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Xpetra::Exceptions::NotImplemented, "Xpetar::BlockedCrsMatrix::bgs_apply: not implemented for transpose case.");
  }
  Y.update(alpha, *tmpY, beta);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
const Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getMap() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getMap");
  if (Rows() == 1 && Cols() == 1) {
    return getMatrix(0, 0)->getMap();
  }
  throw Xpetra::Exceptions::RuntimeError("BlockedCrsMatrix::getMap(): operation not supported.");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::doImport(const Matrix& source, const Import& importer, CombineMode CM) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::doImport");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->doImport(source, importer, CM);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("BlockedCrsMatrix::doImport(): operation not supported.");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::doExport(const Matrix& dest, const Import& importer, CombineMode CM) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::doExport");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->doExport(dest, importer, CM);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("BlockedCrsMatrix::doExport(): operation not supported.");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::doImport(const Matrix& source, const Export& exporter, CombineMode CM) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::doImport");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->doImport(source, exporter, CM);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("BlockedCrsMatrix::doImport(): operation not supported.");
}

//! Export (using an Importer).
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::doExport(const Matrix& dest, const Export& exporter, CombineMode CM) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::doExport");
  if (Rows() == 1 && Cols() == 1) {
    getMatrix(0, 0)->doExport(dest, exporter, CM);
    return;
  }
  throw Xpetra::Exceptions::RuntimeError("BlockedCrsMatrix::doExport(): operation not supported.");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
std::string BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::description() const { return "Xpetra_BlockedCrsMatrix.description()"; }

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::describe(Teuchos::FancyOStream& out, const Teuchos::EVerbosityLevel verbLevel) const {
  out << "Xpetra::BlockedCrsMatrix: " << Rows() << " x " << Cols() << std::endl;

  if (isFillComplete()) {
    out << "BlockMatrix is fillComplete" << std::endl;

    /*if(fullrowmap_ != Teuchos::null) {
      out << "fullRowMap" << std::endl;
      fullrowmap_->describe(out,verbLevel);
    } else {
      out << "fullRowMap not set. Check whether block matrix is properly fillCompleted!" << std::endl;
    }*/

    // out << "fullColMap" << std::endl;
    // fullcolmap_->describe(out,verbLevel);

  } else {
    out << "BlockMatrix is NOT fillComplete" << std::endl;
  }

  for (size_t r = 0; r < Rows(); ++r)
    for (size_t c = 0; c < Cols(); ++c) {
      if (getMatrix(r, c) != Teuchos::null) {
        out << "Block(" << r << "," << c << ")" << std::endl;
        getMatrix(r, c)->describe(out, verbLevel);
      } else
        out << "Block(" << r << "," << c << ") = null" << std::endl;
    }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::setObjectLabel(const std::string& objectLabel) {
  XPETRA_MONITOR("TpetraBlockedCrsMatrix::setObjectLabel");
  for (size_t r = 0; r < Rows(); ++r)
    for (size_t c = 0; c < Cols(); ++c) {
      if (getMatrix(r, c) != Teuchos::null) {
        std::ostringstream oss;
        oss << objectLabel << "(" << r << "," << c << ")";
        getMatrix(r, c)->setObjectLabel(oss.str());
      }
    }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::hasCrsGraph() const {
  if (Rows() == 1 && Cols() == 1)
    return true;
  else
    return false;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getCrsGraph() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getCrsGraph");
  if (Rows() == 1 && Cols() == 1) {
    return getMatrix(0, 0)->getCrsGraph();
  }
  throw Xpetra::Exceptions::RuntimeError("getCrsGraph() not supported by BlockedCrsMatrix");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::isDiagonal() const { return is_diagonal_; }

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Rows() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::Rows");
  return rangemaps_->NumMaps();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
size_t BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Cols() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::Cols");
  return domainmaps_->NumMaps();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getCrsMatrix() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getCrsMatrix");
  TEUCHOS_TEST_FOR_EXCEPTION(Rows() != 1, std::out_of_range, "Can only unwrap a 1x1 blocked matrix. The matrix has " << Rows() << " block rows, though.");
  TEUCHOS_TEST_FOR_EXCEPTION(Cols() != 1, std::out_of_range, "Can only unwrap a 1x1 blocked matrix. The matrix has " << Cols() << " block columns, though.");

  RCP<Matrix> mat            = getMatrix(0, 0);
  RCP<BlockedCrsMatrix> bmat = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(mat);
  if (bmat == Teuchos::null) return mat;
  return bmat->getCrsMatrix();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getInnermostCrsMatrix() {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getInnermostCrsMatrix");
  size_t row = Rows() + 1, col = Cols() + 1;
  for (size_t r = 0; r < Rows(); ++r)
    for (size_t c = 0; c < Cols(); ++c)
      if (getMatrix(r, c) != Teuchos::null) {
        row = r;
        col = c;
        break;
      }
  TEUCHOS_TEST_FOR_EXCEPTION(row == Rows() + 1 || col == Cols() + 1, Xpetra::Exceptions::Incompatible, "Xpetra::BlockedCrsMatrix::getInnermostCrsMatrix: Could not find a non-zero sub-block in blocked operator.")
  RCP<Matrix> mm             = getMatrix(row, col);
  RCP<BlockedCrsMatrix> bmat = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(mm);
  if (bmat == Teuchos::null) return mm;
  return bmat->getInnermostCrsMatrix();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getMatrix(size_t r, size_t c) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::getMatrix");
  TEUCHOS_TEST_FOR_EXCEPTION(r > Rows(), std::out_of_range, "Error, r = " << Rows() << " is too big");
  TEUCHOS_TEST_FOR_EXCEPTION(c > Cols(), std::out_of_range, "Error, c = " << Cols() << " is too big");

  // transfer strided/blocked map information
  /*      if (blocks_[r*Cols()+c] != Teuchos::null &&
      blocks_[r*Cols()+c]->IsView("stridedMaps") == false)
      blocks_[r*Cols()+c]->CreateView("stridedMaps", getRangeMap(r,bRangeThyraMode_), getDomainMap(c,bDomainThyraMode_));*/
  return blocks_[r * Cols() + c];
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::setMatrix(size_t r, size_t c, Teuchos::RCP<Matrix> mat) {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::setMatrix");
  // TODO: if filled -> return error

  TEUCHOS_TEST_FOR_EXCEPTION(r > Rows(), std::out_of_range, "Error, r = " << Rows() << " is too big");
  TEUCHOS_TEST_FOR_EXCEPTION(c > Cols(), std::out_of_range, "Error, c = " << Cols() << " is too big");
  if (!mat.is_null() && r != c) is_diagonal_ = false;
  // set matrix
  blocks_[r * Cols() + c] = mat;
}

namespace Details {
//! XPETRA_BLOCKED_MERGE_LOCAL=0 makes BlockedCrsMatrix::Merge use only the general row-by-row merge.
inline bool mergeLocalDisabled() {
  static const bool disabled = [] {
    const char* v = std::getenv("XPETRA_BLOCKED_MERGE_LOCAL");
    return v != nullptr && std::string(v) == "0";
  }();
  return disabled;
}

template <class OffsetView, class LO>
KOKKOS_INLINE_FUNCTION LO mergeLocalBlockRow(const OffsetView& rowOffset, const LO row) {
  LO r = 0;
  while (row >= rowOffset(r + 1)) ++r;
  return r;
}

/// Kokkos part of BlockedCrsMatrix::MergeLocal. blocks[r * nBCols + c] is block (r,c) or null; merged local row
/// rowOffset[r] + i is local row i of block row r; block column c's column GIDs plus colShift[c] are merged GIDs.
template <class SC, class LO, class GO, class NO>
Teuchos::RCP<Tpetra::CrsMatrix<SC, LO, GO, NO>> mergeLocalBlocks(
    const std::vector<Teuchos::RCP<const Tpetra::CrsMatrix<SC, LO, GO, NO>>>& blocks, const LO nBRows, const LO nBCols,
    const std::vector<LO>& rowOffsetHost, const std::vector<GO>& colShiftHost,
    const Teuchos::RCP<const Tpetra::Map<LO, GO, NO>>& domainMap, const Teuchos::RCP<const Tpetra::Map<LO, GO, NO>>& rangeMap) {
  using crs_type          = Tpetra::CrsMatrix<SC, LO, GO, NO>;
  using lcl_matrix_type   = typename crs_type::local_matrix_device_type;
  using lcl_map_type      = typename Tpetra::Map<LO, GO, NO>::local_map_type;
  using row_map_type      = typename lcl_matrix_type::row_map_type::non_const_type;
  using index_type        = typename lcl_matrix_type::index_type::non_const_type;
  using values_type       = typename lcl_matrix_type::values_type::non_const_type;
  using execution_space   = typename lcl_matrix_type::execution_space;
  using memory_space      = typename NO::memory_space;
  using offset_type       = typename row_map_type::non_const_value_type;
  using policy_type       = Kokkos::RangePolicy<execution_space, LO>;

  // Block descriptions, readable on the device.
  Kokkos::View<lcl_matrix_type**, Kokkos::SharedSpace> lclBlocks(
      Kokkos::view_alloc("MergeLocal::blocks", Kokkos::SequentialHostInit), nBRows, nBCols);
  Kokkos::View<lcl_map_type**, Kokkos::SharedSpace> lclColMaps(
      Kokkos::view_alloc("MergeLocal::colMaps", Kokkos::SequentialHostInit), nBRows, nBCols);
  Kokkos::View<int**, memory_space> hasBlock("MergeLocal::hasBlock", nBRows, nBCols);
  Kokkos::View<LO*, memory_space> rowOffset("MergeLocal::rowOffset", nBRows + 1);
  Kokkos::View<GO*, memory_space> colShift("MergeLocal::colShift", nBCols);
  {
    auto hasBlockHost  = Kokkos::create_mirror_view(hasBlock);
    auto rowOffsetHostV = Kokkos::create_mirror_view(rowOffset);
    auto colShiftHostV  = Kokkos::create_mirror_view(colShift);
    for (LO r = 0; r < nBRows; ++r) {
      for (LO c = 0; c < nBCols; ++c) {
        const auto& blk    = blocks[r * nBCols + c];
        hasBlockHost(r, c) = blk.is_null() ? 0 : 1;
        if (!blk.is_null()) {
          lclBlocks(r, c)  = blk->getLocalMatrixDevice();
          lclColMaps(r, c) = blk->getColMap()->getLocalMap();
        }
      }
    }
    for (LO r = 0; r <= nBRows; ++r) rowOffsetHostV(r) = rowOffsetHost[r];
    for (LO c = 0; c < nBCols; ++c) colShiftHostV(c) = colShiftHost[c];
    Kokkos::deep_copy(hasBlock, hasBlockHost);
    Kokkos::deep_copy(rowOffset, rowOffsetHostV);
    Kokkos::deep_copy(colShift, colShiftHostV);
  }
  const LO numRows = rowOffsetHost[nBRows];

  // Row lengths, then their prefix sum.
  row_map_type rowPtr("MergeLocal::rowPtr", numRows + 1);
  Kokkos::parallel_for(
      "MergeLocal::count", policy_type(0, numRows), KOKKOS_LAMBDA(const LO row) {
        const LO r   = mergeLocalBlockRow(rowOffset, row);
        const LO lid = row - rowOffset(r);
        offset_type n = 0;
        for (LO c = 0; c < nBCols; ++c)
          if (hasBlock(r, c)) n += lclBlocks(r, c).graph.row_map(lid + 1) - lclBlocks(r, c).graph.row_map(lid);
        rowPtr(row + 1) = n;
      });
  offset_type nnz = 0;
  Kokkos::parallel_scan(
      "MergeLocal::scan", policy_type(0, numRows + 1), KOKKOS_LAMBDA(const LO i, offset_type& sum, const bool final) {
        sum += rowPtr(i);
        if (final) rowPtr(i) = sum;
      },
      nnz);

  // Global column indices (merged numbering) and values.
  Kokkos::View<GO*, memory_space> gblCols(Kokkos::ViewAllocateWithoutInitializing("MergeLocal::gblCols"), nnz);
  values_type values(Kokkos::ViewAllocateWithoutInitializing("MergeLocal::values"), nnz);
  Kokkos::parallel_for(
      "MergeLocal::fill", policy_type(0, numRows), KOKKOS_LAMBDA(const LO row) {
        const LO r      = mergeLocalBlockRow(rowOffset, row);
        const LO lid    = row - rowOffset(r);
        offset_type pos = rowPtr(row);
        for (LO c = 0; c < nBCols; ++c) {
          if (!hasBlock(r, c)) continue;
          const auto& A = lclBlocks(r, c);
          for (auto k = A.graph.row_map(lid); k < A.graph.row_map(lid + 1); ++k, ++pos) {
            gblCols(pos) = lclColMaps(r, c).getGlobalElement(A.graph.entries(k)) + colShift(c);
            values(pos)  = A.values(k);
          }
        }
      });

  Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> colMap;
  Tpetra::Details::makeColMap<LO, GO, NO>(colMap, domainMap, gblCols);
  TEUCHOS_TEST_FOR_EXCEPTION(colMap.is_null(), Xpetra::Exceptions::RuntimeError, "BlockedCrsMatrix::MergeLocal: makeColMap failed.");

  index_type lclCols(Kokkos::ViewAllocateWithoutInitializing("MergeLocal::lclCols"), nnz);
  const auto lclColMap = colMap->getLocalMap();
  Kokkos::parallel_for(
      "MergeLocal::localize", Kokkos::RangePolicy<execution_space, offset_type>(0, nnz),
      KOKKOS_LAMBDA(const offset_type k) { lclCols(k) = lclColMap.getLocalElement(gblCols(k)); });
  KokkosSparse::sort_crs_matrix<execution_space, row_map_type, index_type, values_type>(rowPtr, lclCols, values);

  lcl_matrix_type lclMerged("MergeLocal::matrix", numRows, static_cast<LO>(colMap->getLocalNumElements()), nnz, values, rowPtr, lclCols);
  return Teuchos::rcp(new crs_type(lclMerged, rangeMap, colMap, domainMap, rangeMap));
}
}  // namespace Details

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::MergeLocal() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::MergeLocal");
  using Teuchos::RCP;
  using Teuchos::rcp_dynamic_cast;
  using tpetra_crs_type     = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  const LocalOrdinal nBRows = static_cast<LocalOrdinal>(Rows());
  const LocalOrdinal nBCols = static_cast<LocalOrdinal>(Cols());
  const auto comm           = getFullRangeMap()->getComm();

  // Every rank must agree on taking this path; each check below only clears `ok`.
  int ok = 1;
  std::vector<RCP<const tpetra_crs_type>> blocks(nBRows * nBCols);
  for (LocalOrdinal r = 0; r < nBRows && ok; ++r) {
    for (LocalOrdinal c = 0; c < nBCols && ok; ++c) {
      RCP<const Matrix> mat = getMatrix(r, c);
      if (mat.is_null()) continue;
      auto wrap = rcp_dynamic_cast<const CrsMatrixWrap>(mat);
      if (wrap.is_null()) {  // nested blocked operator or something else
        ok = 0;
        break;
      }
      auto tmat = rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(wrap->getCrsMatrix());
      if (tmat.is_null()) {
        ok = 0;
        break;
      }
      blocks[r * nBCols + c] = tmat->getTpetra_CrsMatrix();
    }
  }

  // Merged local rows are the block rows in block order: the local layout BlockedMap gives the full range Map.
  // Block columns move from block numbering to merged (Xpetra) numbering by a constant shift per block column.
  // Check both on every local entry; together the ranks cover every GID because the maps are one-to-one.
  std::vector<LocalOrdinal> rowOffsetHost(nBRows + 1, 0);
  std::vector<GlobalOrdinal> colShiftHost(nBCols, 0);
  if (ok) {
    auto fullRange = getFullRangeMap();
    for (LocalOrdinal r = 0; r < nBRows && ok; ++r) {
      auto xRowMap            = getRangeMapExtractor()->getMap(r, false);
      auto bRowMap            = getRangeMapExtractor()->getMap(r, bRangeThyraMode_);
      const LocalOrdinal nLoc = static_cast<LocalOrdinal>(xRowMap->getLocalNumElements());
      rowOffsetHost[r + 1]    = rowOffsetHost[r] + nLoc;
      for (LocalOrdinal c = 0; c < nBCols && ok; ++c) {
        const auto& blk = blocks[r * nBCols + c];
        if (!blk.is_null() && !blk->getRowMap()->isSameAs(*toTpetra(bRowMap))) ok = 0;
      }
      for (LocalOrdinal i = 0; i < nLoc && ok; ++i)
        if (fullRange->getGlobalElement(rowOffsetHost[r] + i) != xRowMap->getGlobalElement(i)) ok = 0;
    }
    if (ok && static_cast<size_t>(rowOffsetHost[nBRows]) != fullRange->getLocalNumElements()) ok = 0;
    for (LocalOrdinal c = 0; c < nBCols && ok; ++c) {
      auto xDomMap    = getDomainMapExtractor()->getMap(c, false);
      auto bDomMap    = getDomainMapExtractor()->getMap(c, bDomainThyraMode_);
      colShiftHost[c] = xDomMap->getMinAllGlobalIndex() - bDomMap->getMinAllGlobalIndex();
      if (xDomMap->getLocalNumElements() != bDomMap->getLocalNumElements()) ok = 0;
      for (size_t i = 0; i < xDomMap->getLocalNumElements() && ok; ++i)
        if (xDomMap->getGlobalElement(i) != bDomMap->getGlobalElement(i) + colShiftHost[c]) ok = 0;
    }
  }
  int allOk = 0;
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, ok, Teuchos::outArg(allOk));
  if (!allOk) return Teuchos::null;

  auto merged = Details::mergeLocalBlocks<Scalar, LocalOrdinal, GlobalOrdinal, Node>(
      blocks, nBRows, nBCols, rowOffsetHost, colShiftHost, toTpetra(getFullDomainMap()), toTpetra(getFullRangeMap()));
  RCP<CrsMatrix> mergedCrs = Teuchos::rcp(new Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(merged));
  RCP<Matrix> sparse       = Teuchos::rcp(new CrsMatrixWrap(mergedCrs));

  TEUCHOS_TEST_FOR_EXCEPTION(sparse->getGlobalNumEntries() != getGlobalNumEntries(), Xpetra::Exceptions::RuntimeError,
                             "BlockedCrsMatrix::MergeLocal: Global number of entries of merged matrix does not coincide with global number of entries of blocked operator.");
  return sparse;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Merge() const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::Merge");
  using Teuchos::RCP;
  using Teuchos::rcp_dynamic_cast;
  Scalar one = ScalarTraits<SC>::one();

  TEUCHOS_TEST_FOR_EXCEPTION(bRangeThyraMode_ != bDomainThyraMode_, Xpetra::Exceptions::RuntimeError,
                             "BlockedCrsMatrix::Merge: only implemented for Xpetra-style or Thyra-style numbering. No mixup allowed!");

  TEUCHOS_TEST_FOR_EXCEPTION(isFillComplete() == false, Xpetra::Exceptions::RuntimeError,
                             "BlockedCrsMatrix::Merge: BlockMatrix must be fill-completed.");

  // Local (Kokkos) merge when the block layout allows it; otherwise the general row-by-row merge below.
  if (!Details::mergeLocalDisabled()) {
    RCP<Matrix> merged = MergeLocal();
    if (!merged.is_null()) return merged;
  }

  LocalOrdinal lclNumRows = getFullRangeMap()->getLocalNumElements();
  Teuchos::ArrayRCP<size_t> numEntPerRow(lclNumRows);
  for (LocalOrdinal lclRow = 0; lclRow < lclNumRows; ++lclRow)
    numEntPerRow[lclRow] = getNumEntriesInLocalRow(lclRow);

  RCP<Matrix> sparse = MatrixFactory::Build(getFullRangeMap(), numEntPerRow);

  if (bRangeThyraMode_ == false) {
    // Xpetra mode
    for (size_t i = 0; i < Rows(); i++) {
      for (size_t j = 0; j < Cols(); j++) {
        if (getMatrix(i, j) != Teuchos::null) {
          RCP<const Matrix> mat = getMatrix(i, j);

          // recursively call Merge routine
          RCP<const BlockedCrsMatrix> bMat = Teuchos::rcp_dynamic_cast<const BlockedCrsMatrix>(mat);
          if (bMat != Teuchos::null) mat = bMat->Merge();

          bMat = Teuchos::rcp_dynamic_cast<const BlockedCrsMatrix>(mat);
          TEUCHOS_TEST_FOR_EXCEPTION(bMat != Teuchos::null, Xpetra::Exceptions::RuntimeError,
                                     "BlockedCrsMatrix::Merge: Merging of blocked sub-operators failed?!");

          // jump over empty blocks
          if (mat->getLocalNumEntries() == 0) continue;

          this->Add(*mat, one, *sparse, one);
        }
      }
    }
  } else {
    // Thyra mode
    for (size_t i = 0; i < Rows(); i++) {
      for (size_t j = 0; j < Cols(); j++) {
        if (getMatrix(i, j) != Teuchos::null) {
          RCP<const Matrix> mat = getMatrix(i, j);
          // recursively call Merge routine
          RCP<const BlockedCrsMatrix> bMat = Teuchos::rcp_dynamic_cast<const BlockedCrsMatrix>(mat);
          if (bMat != Teuchos::null) mat = bMat->Merge();

          bMat = Teuchos::rcp_dynamic_cast<const BlockedCrsMatrix>(mat);
          TEUCHOS_TEST_FOR_EXCEPTION(bMat != Teuchos::null, Xpetra::Exceptions::RuntimeError,
                                     "BlockedCrsMatrix::Merge: Merging of blocked sub-operators failed?!");

          // check whether we have a CrsMatrix block (no blocked operator)
          RCP<const CrsMatrixWrap> crsMat = Teuchos::rcp_dynamic_cast<const CrsMatrixWrap>(mat);
          TEUCHOS_ASSERT(crsMat != Teuchos::null);

          // these are the thyra style maps of the matrix
          RCP<const Map> trowMap = mat->getRowMap();
          RCP<const Map> tcolMap = mat->getColMap();
          RCP<const Map> tdomMap = mat->getDomainMap();

          // get Xpetra maps
          RCP<const Map> xrowMap = getRangeMapExtractor()->getMap(i, false);
          RCP<const Map> xdomMap = getDomainMapExtractor()->getMap(j, false);

          // generate column map with Xpetra GIDs
          // We have to do this separately for each block since the column
          // map of each block might be different in the same block column
          Teuchos::RCP<Map> xcolMap = MapUtils::transformThyra2XpetraGIDs(
              *tcolMap,
              *tdomMap,
              *xdomMap);

          // jump over empty blocks
          if (mat->getLocalNumEntries() == 0) continue;

          size_t maxNumEntries = mat->getLocalMaxNumRowEntries();

          size_t numEntries;
          Array<GO> inds(maxNumEntries);
          Array<GO> inds2(maxNumEntries);
          Array<SC> vals(maxNumEntries);

          // loop over all rows and add entries
          for (size_t k = 0; k < mat->getLocalNumRows(); k++) {
            GlobalOrdinal rowTGID = trowMap->getGlobalElement(k);
            crsMat->getCrsMatrix()->getGlobalRowCopy(rowTGID, inds(), vals(), numEntries);

            // create new indices array
            for (size_t l = 0; l < numEntries; ++l) {
              LocalOrdinal lid = tcolMap->getLocalElement(inds[l]);
              inds2[l]         = xcolMap->getGlobalElement(lid);
            }

            GlobalOrdinal rowXGID = xrowMap->getGlobalElement(k);
            sparse->insertGlobalValues(
                rowXGID, inds2(0, numEntries),
                vals(0, numEntries));
          }
        }
      }
    }
  }

  sparse->fillComplete(getFullDomainMap(), getFullRangeMap());

  TEUCHOS_TEST_FOR_EXCEPTION(sparse->getLocalNumEntries() != getLocalNumEntries(), Xpetra::Exceptions::RuntimeError,
                             "BlockedCrsMatrix::Merge: Local number of entries of merged matrix does not coincide with local number of entries of blocked operator.");

  TEUCHOS_TEST_FOR_EXCEPTION(sparse->getGlobalNumEntries() != getGlobalNumEntries(), Xpetra::Exceptions::RuntimeError,
                             "BlockedCrsMatrix::Merge: Global number of entries of merged matrix does not coincide with global number of entries of blocked operator.");

  return sparse;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getLocalMatrixDevice() const {
  if (Rows() == 1 && Cols() == 1) {
    return getMatrix(0, 0)->getLocalMatrixDevice();
  }
  throw Xpetra::Exceptions::RuntimeError("BlockedCrsMatrix::getLocalMatrix(): operation not supported.");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type::host_mirror_type BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getLocalMatrixHost() const {
  if (Rows() == 1 && Cols() == 1) {
    return getMatrix(0, 0)->getLocalMatrixHost();
  }
  throw Xpetra::Exceptions::RuntimeError("BlockedCrsMatrix::getLocalMatrix(): operation not supported.");
}

#ifdef HAVE_XPETRA_THYRA
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Thyra::BlockedLinearOpBase<Scalar>> BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getThyraOperator() {
  Teuchos::RCP<Thyra::LinearOpBase<Scalar>> thOp =
      Xpetra::ThyraUtils<Scalar, LO, GO, Node>::toThyra(Teuchos::rcpFromRef(*this));
  TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(thOp));

  Teuchos::RCP<Thyra::BlockedLinearOpBase<Scalar>> thbOp =
      Teuchos::rcp_dynamic_cast<Thyra::BlockedLinearOpBase<Scalar>>(thOp);
  TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(thbOp));
  return thbOp;
}
#endif

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
LocalOrdinal BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetStorageBlockSize() const { return 1; }

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::residual(const MultiVector& X,
                                                                           const MultiVector& B,
                                                                           MultiVector& R) const {
  using STS = Teuchos::ScalarTraits<Scalar>;
  R.update(STS::one(), B, STS::zero());
  this->apply(X, R, Teuchos::NO_TRANS, -STS::one(), STS::one());
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Add(const Matrix& A, const Scalar scalarA, Matrix& B, const Scalar scalarB) const {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::Add");
  TEUCHOS_TEST_FOR_EXCEPTION(!A.isFillComplete(), Xpetra::Exceptions::RuntimeError,
                             "Matrix A is not completed");
  using Teuchos::Array;
  using Teuchos::ArrayView;

  B.scale(scalarB);

  Scalar one  = ScalarTraits<SC>::one();
  Scalar zero = ScalarTraits<SC>::zero();

  if (scalarA == zero)
    return;

  Teuchos::RCP<const Matrix> rcpA            = Teuchos::rcpFromRef(A);
  Teuchos::RCP<const CrsMatrixWrap> rcpAwrap = Teuchos::rcp_dynamic_cast<const CrsMatrixWrap>(rcpA);
  TEUCHOS_TEST_FOR_EXCEPTION(rcpAwrap == Teuchos::null, Xpetra::Exceptions::BadCast,
                             "BlockedCrsMatrix::Add: matrix A must be of type CrsMatrixWrap.");
  Teuchos::RCP<const CrsMatrix> crsA = rcpAwrap->getCrsMatrix();

  size_t maxNumEntries = crsA->getLocalMaxNumRowEntries();

  size_t numEntries;
  Array<GO> inds(maxNumEntries);
  Array<SC> vals(maxNumEntries);

  RCP<const Map> rowMap = crsA->getRowMap();
  RCP<const Map> colMap = crsA->getColMap();

  ArrayView<const GO> rowGIDs = crsA->getRowMap()->getLocalElementList();
  for (size_t i = 0; i < crsA->getLocalNumRows(); i++) {
    GO row = rowGIDs[i];
    crsA->getGlobalRowCopy(row, inds(), vals(), numEntries);

    if (scalarA != one)
      for (size_t j = 0; j < numEntries; ++j)
        vals[j] *= scalarA;

    B.insertGlobalValues(row, inds(0, numEntries), vals(0, numEntries));  // insert should be ok, since blocks in BlockedCrsOpeartor do not overlap!
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::CreateDefaultView() {
  XPETRA_MONITOR("XpetraBlockedCrsMatrix::CreateDefaultView");

  // Create default view
  this->defaultViewLabel_ = "point";
  this->CreateView(this->GetDefaultViewLabel(), getRangeMap(), getDomainMap());

  // Set current view
  this->currentViewLabel_ = this->GetDefaultViewLabel();
}

}  // namespace Xpetra

#define XPETRA_BLOCKEDCRSMATRIX_SHORT
#endif /* XPETRA_BLOCKEDCRSMATRIX_DEF_HPP */
