// @HEADER
// *****************************************************************************
//                 Belos: Block Linear Solvers Package
//
// Copyright 2004-2016 NTESS and the Belos contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef BELOS_FGCRODR_ITER_HPP
#define BELOS_FGCRODR_ITER_HPP

#include "BelosGCRODRIter.hpp"

namespace Belos {

class FGCRODRIterInitFailure : public GCRODRIterInitFailure {
public:
  FGCRODRIterInitFailure(const std::string& what_arg) :
    GCRODRIterInitFailure(what_arg) {}
};

class FGCRODRIterOrthoFailure : public GCRODRIterOrthoFailure {
public:
  FGCRODRIterOrthoFailure(const std::string& what_arg) :
    GCRODRIterOrthoFailure(what_arg) {}
};

template<class ScalarType, class MV, class OP>
class FGCRODRIter : virtual public GCRODRIteration<ScalarType,MV,OP> {
public:
  typedef MultiVecTraits<ScalarType,MV> MVT;
  typedef OperatorTraits<ScalarType,MV,OP> OPT;
  typedef Teuchos::ScalarTraits<ScalarType> SCT;
  typedef typename SCT::magnitudeType MagnitudeType;

  FGCRODRIter(
    const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> >& problem,
    const Teuchos::RCP<OutputManager<ScalarType> >& printer,
    const Teuchos::RCP<StatusTest<ScalarType,MV,OP> >& tester,
    const Teuchos::RCP<MatOrthoManager<ScalarType,MV,OP> >& ortho,
    Teuchos::ParameterList& params
  );

  virtual ~FGCRODRIter() {}

  void iterate();
  void initialize(GCRODRIterState<ScalarType,MV>& newstate);

  void initialize() {
    GCRODRIterState<ScalarType,MV> empty;
    initialize(empty);
  }

  GCRODRIterState<ScalarType,MV> getState() const {
    GCRODRIterState<ScalarType,MV> state;
    state.curDim = curDim_;
    state.V = V_;
    state.Z = Z_;
    state.U = U_;
    state.C = C_;
    state.H = H_;
    state.B = B_;
    return state;
  }

  int getNumIters() const { return iter_; }
  void resetNumIters(int iter = 0) { iter_ = iter; }

  Teuchos::RCP<const MV>
  getNativeResiduals(std::vector<MagnitudeType>* norms) const;

  Teuchos::RCP<MV> getCurrentUpdate() const;

  void updateLSQR(int dim = -1);

  int getCurSubspaceDim() const {
    if (!initialized_) return 0;
    return curDim_;
  }

  int getMaxSubspaceDim() const { return numBlocks_; }

  const LinearProblem<ScalarType,MV,OP>& getProblem() const {
    return *lp_;
  }

  int getBlockSize() const { return 1; }

  void setBlockSize(int blockSize) {
    TEUCHOS_TEST_FOR_EXCEPTION(
      blockSize != 1,
      std::invalid_argument,
      "Belos::FGCRODRIter::setBlockSize(): block size must be one.");
  }

  bool isInitialized() { return initialized_; }

  void setSize(int recycledBlocks, int numBlocks) {
    if (recycledBlocks_ != recycledBlocks || numBlocks_ != numBlocks) {
      recycledBlocks_ = recycledBlocks;
      numBlocks_ = numBlocks;
      cs_.sizeUninitialized(numBlocks_ + 1);
      sn_.sizeUninitialized(numBlocks_ + 1);
      z_.sizeUninitialized(numBlocks_ + 1);
      R_.shapeUninitialized(numBlocks_ + 1, numBlocks_);
    }
  }

private:
  const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > lp_;
  const Teuchos::RCP<OutputManager<ScalarType> > om_;
  const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > stest_;
  const Teuchos::RCP<OrthoManager<ScalarType,MV> > ortho_;

  int numBlocks_;
  int recycledBlocks_;

  Teuchos::SerialDenseVector<int,ScalarType> sn_;
  Teuchos::SerialDenseVector<int,MagnitudeType> cs_;

  bool initialized_;
  int curDim_, iter_;

  Teuchos::RCP<MV> V_;
  Teuchos::RCP<MV> Z_;
  Teuchos::RCP<MV> U_, C_;

  Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > H_;
  Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B_;

  Teuchos::SerialDenseMatrix<int,ScalarType> R_;
  Teuchos::SerialDenseVector<int,ScalarType> z_;
};

template<class ScalarType, class MV, class OP>
FGCRODRIter<ScalarType,MV,OP>::
FGCRODRIter(
  const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> >& problem,
  const Teuchos::RCP<OutputManager<ScalarType> >& printer,
  const Teuchos::RCP<StatusTest<ScalarType,MV,OP> >& tester,
  const Teuchos::RCP<MatOrthoManager<ScalarType,MV,OP> >& ortho,
  Teuchos::ParameterList& params
) :
  lp_(problem),
  om_(printer),
  stest_(tester),
  ortho_(ortho),
  numBlocks_(0),
  recycledBlocks_(0),
  initialized_(false),
  curDim_(0),
  iter_(0),
  V_(Teuchos::null),
  Z_(Teuchos::null),
  U_(Teuchos::null),
  C_(Teuchos::null),
  H_(Teuchos::null),
  B_(Teuchos::null)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !params.isParameter("Num Blocks"),
    std::invalid_argument,
    "Belos::FGCRODRIter::constructor: mandatory parameter \"Num Blocks\" is not specified.");

  TEUCHOS_TEST_FOR_EXCEPTION(
    !params.isParameter("Recycled Blocks"),
    std::invalid_argument,
    "Belos::FGCRODRIter::constructor: mandatory parameter \"Recycled Blocks\" is not specified.");

  int nb = Teuchos::getParameter<int>(params, "Num Blocks");
  int rb = Teuchos::getParameter<int>(params, "Recycled Blocks");

  TEUCHOS_TEST_FOR_EXCEPTION(nb <= 0, std::invalid_argument,
    "Belos::FGCRODRIter: \"Num Blocks\" must be positive.");
  TEUCHOS_TEST_FOR_EXCEPTION(rb >= nb, std::invalid_argument,
    "Belos::FGCRODRIter: \"Recycled Blocks\" must be less than \"Num Blocks\".");

  numBlocks_ = nb;
  recycledBlocks_ = rb;

  cs_.sizeUninitialized(numBlocks_ + 1);
  sn_.sizeUninitialized(numBlocks_ + 1);
  z_.sizeUninitialized(numBlocks_ + 1);
  R_.shapeUninitialized(numBlocks_ + 1, numBlocks_);
}

template<class ScalarType, class MV, class OP>
void
FGCRODRIter<ScalarType,MV,OP>::
initialize(GCRODRIterState<ScalarType,MV>& newstate)
{
  TEUCHOS_TEST_FOR_EXCEPTION(newstate.V == Teuchos::null,
    std::invalid_argument,
    "Belos::FGCRODRIter::initialize(): state does not have V initialized.");
  TEUCHOS_TEST_FOR_EXCEPTION(newstate.Z == Teuchos::null,
    std::invalid_argument,
    "Belos::FGCRODRIter::initialize(): state does not have Z initialized.");
  TEUCHOS_TEST_FOR_EXCEPTION(newstate.H == Teuchos::null,
    std::invalid_argument,
    "Belos::FGCRODRIter::initialize(): state does not have H initialized.");

  curDim_ = newstate.curDim;
  V_ = newstate.V;
  Z_ = newstate.Z;
  U_ = newstate.U;
  C_ = newstate.C;
  H_ = newstate.H;
  B_ = newstate.B;

  initialized_ = true;
}

template<class ScalarType, class MV, class OP>
Teuchos::RCP<MV>
FGCRODRIter<ScalarType,MV,OP>::
getCurrentUpdate() const
{
  Teuchos::RCP<MV> currentUpdate = Teuchos::null;
  if (curDim_ == 0) return currentUpdate;

  const ScalarType one = SCT::one();
  const ScalarType zero = SCT::zero();

  Teuchos::BLAS<int,ScalarType> blas;

  Teuchos::SerialDenseMatrix<int,ScalarType> y(
    Teuchos::Copy, z_, curDim_, 1);

  blas.TRSM(
    Teuchos::LEFT_SIDE,
    Teuchos::UPPER_TRI,
    Teuchos::NO_TRANS,
    Teuchos::NON_UNIT_DIAG,
    curDim_,
    1,
    one,
    R_.values(),
    R_.stride(),
    y.values(),
    y.stride());

  currentUpdate = MVT::Clone(*Z_, 1);

  std::vector<int> index(curDim_);
  for (int i = 0; i < curDim_; ++i) index[i] = i;

  Teuchos::RCP<const MV> Zcur = MVT::CloneView(*Z_, index);
  MVT::MvTimesMatAddMv(one, *Zcur, y, zero, *currentUpdate);

  if (U_ != Teuchos::null) {
    Teuchos::SerialDenseMatrix<int,ScalarType> z(recycledBlocks_, 1);
    Teuchos::SerialDenseMatrix<int,ScalarType> subB(
      Teuchos::View, *B_, recycledBlocks_, curDim_);

    z.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
               one, subB, y, zero);

    MVT::MvTimesMatAddMv(-one, *U_, z, one, *currentUpdate);
  }

  return currentUpdate;
}

template<class ScalarType, class MV, class OP>
Teuchos::RCP<const MV>
FGCRODRIter<ScalarType,MV,OP>::
getNativeResiduals(std::vector<MagnitudeType>* norms) const
{
  if (norms && static_cast<int>(norms->size()) == 0) norms->resize(1);

  if (norms) {
    Teuchos::BLAS<int,ScalarType> blas;
    (*norms)[0] = blas.NRM2(1, &z_(curDim_), 1);
  }

  return Teuchos::null;
}

template<class ScalarType, class MV, class OP>
void
FGCRODRIter<ScalarType,MV,OP>::
iterate()
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    initialized_ == false,
    FGCRODRIterInitFailure,
    "Belos::FGCRODRIter::iterate(): iterator is not initialized.");

  setSize(recycledBlocks_, numBlocks_);

  Teuchos::RCP<MV> Vnext;
  Teuchos::RCP<MV> Znext;
  Teuchos::RCP<const MV> Vprev;

  std::vector<int> curind(1);
  std::vector<int> prevind(numBlocks_ + 1);

  z_.putScalar(SCT::zero());

  curind[0] = 0;
  Vnext = MVT::CloneViewNonConst(*V_, curind);

  Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > z0 =
    Teuchos::rcp(new Teuchos::SerialDenseMatrix<int,ScalarType>(1, 1));

  int rank = ortho_->normalize(*Vnext, z0);
  TEUCHOS_TEST_FOR_EXCEPTION(rank != 1, FGCRODRIterOrthoFailure,
    "Belos::FGCRODRIter::iterate(): could not normalize initial vector.");

  z_(0) = (*z0)(0,0);

  while (stest_->checkStatus(this) != Passed &&
         curDim_ + 1 <= numBlocks_) {
    ++iter_;

    const int lclDim = curDim_ + 1;

    curind[0] = lclDim;
    Vnext = MVT::CloneViewNonConst(*V_, curind);

    curind[0] = curDim_;
    Vprev = MVT::CloneView(*V_, curind);
    Znext = MVT::CloneViewNonConst(*Z_, curind);

    lp_->applyRightPrec(*Vprev, *Znext);
    Vprev = Teuchos::null;

    lp_->applyOp(*Znext, *Vnext);
    Znext = Teuchos::null;

    if (C_ != Teuchos::null) {
      Teuchos::Array<Teuchos::RCP<const MV> > Carray(1, C_);

      Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > subB =
        Teuchos::rcp(new Teuchos::SerialDenseMatrix<int,ScalarType>(
          Teuchos::View, *B_, recycledBlocks_, 1, 0, curDim_));

      Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > AsubB;
      AsubB.append(subB);

      ortho_->project(*Vnext, AsubB, Carray);
    }

    prevind.resize(lclDim);
    for (int i = 0; i < lclDim; ++i) prevind[i] = i;

    Vprev = MVT::CloneView(*V_, prevind);
    Teuchos::Array<Teuchos::RCP<const MV> > AVprev(1, Vprev);

    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > subH =
      Teuchos::rcp(new Teuchos::SerialDenseMatrix<int,ScalarType>(
        Teuchos::View, *H_, lclDim, 1, 0, curDim_));

    Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > AsubH;
    AsubH.append(subH);

    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > subR =
      Teuchos::rcp(new Teuchos::SerialDenseMatrix<int,ScalarType>(
        Teuchos::View, *H_, 1, 1, lclDim, curDim_));

    rank = ortho_->projectAndNormalize(*Vnext, AsubH, subR, AVprev);

    Teuchos::SerialDenseMatrix<int,ScalarType> subR2(
      Teuchos::View, R_, lclDim + 1, 1, 0, curDim_);
    Teuchos::SerialDenseMatrix<int,ScalarType> subH2(
      Teuchos::View, *H_, lclDim + 1, 1, 0, curDim_);
    subR2.assign(subH2);

    TEUCHOS_TEST_FOR_EXCEPTION(rank != 1, FGCRODRIterOrthoFailure,
      "Belos::FGCRODRIter::iterate(): could not generate full-rank basis vector.");

    updateLSQR();
    ++curDim_;
  }
}

template<class ScalarType, class MV, class OP>
void
FGCRODRIter<ScalarType,MV,OP>::
updateLSQR(int dim)
{
  const ScalarType zero = SCT::zero();

  int curDim = curDim_;
  if ((dim >= curDim_) && (dim < getMaxSubspaceDim())) curDim = dim;

  Teuchos::BLAS<int,ScalarType> blas;

  for (int i = 0; i < curDim; ++i) {
    blas.ROT(1, &R_(i, curDim), 1, &R_(i+1, curDim), 1,
             &cs_[i], &sn_[i]);
  }

  blas.ROTG(&R_(curDim, curDim), &R_(curDim+1, curDim),
            &cs_[curDim], &sn_[curDim]);

  R_(curDim+1, curDim) = zero;

  blas.ROT(1, &z_(curDim), 1, &z_(curDim+1), 1,
           &cs_[curDim], &sn_[curDim]);
}

} // namespace Belos

#endif // BELOS_FGCRODR_ITER_HPP
