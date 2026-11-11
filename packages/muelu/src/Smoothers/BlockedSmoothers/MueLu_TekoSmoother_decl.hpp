// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_TEKOSMOOTHER_DECL_HPP_
#define MUELU_TEKOSMOOTHER_DECL_HPP_

#ifdef HAVE_MUELU_TEKO

#include "Teko_Utilities.hpp"

#include "Teko_InverseLibrary.hpp"
#include "Teko_InverseFactory.hpp"

#include "MueLu_ConfigDefs.hpp"

#include <Teuchos_ParameterList.hpp>

#include <Xpetra_MapExtractor_fwd.hpp>
#include <Xpetra_BlockedMap.hpp>
#include <Xpetra_BlockedMultiVector.hpp>
#include <Xpetra_ThyraUtils.hpp>
#include <Thyra_MultiVectorStdOps.hpp>
#include <Thyra_DefaultBlockedLinearOp.hpp>

#include "MueLu_TekoSmoother_fwd.hpp"
#include "MueLu_SmootherPrototype.hpp"
#include "MueLu_FactoryBase_fwd.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

/*!
  @class TekoSmoother
  @brief Interface to block smoothers in Teko package

  This is a dummy implementation for arbitrary scalars (different than SC=double).
  We use Teko and the Teko::LinearOp which is declared as
  Teuchos::RCP<const Thyra::LinearOpBase<ST> > with ST=double. See e.g. Teko_ConfigDefs.hpp.
*/

template <class Scalar        = SmootherPrototype<>::scalar_type,
          class LocalOrdinal  = typename SmootherPrototype<Scalar>::local_ordinal_type,
          class GlobalOrdinal = typename SmootherPrototype<Scalar, LocalOrdinal>::global_ordinal_type,
          class Node          = typename SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal>::node_type>
class TekoSmoother : public SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
  typedef Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node> MapExtractorClass;

#undef MUELU_TEKOSMOOTHER_SHORT
#include "MueLu_UseShortNames.hpp"

 public:
  //! @name Constructors / destructors
  //@{

  /*! @brief Constructor
   */
  TekoSmoother()
    : type_("Teko smoother") {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Exceptions::RuntimeError, "MueLu::TekoSmoother: Teko can only be used with SC=double. For more information refer to the doxygen documentation of TekoSmoother.");
  };

  TekoSmoother(const Teuchos::ParameterList & /*paramList*/)
    : type_("Teko smoother") {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Exceptions::RuntimeError, "MueLu::TekoSmoother: Teko can only be used with SC=double. For more information refer to the doxygen documentation of TekoSmoother.");
  }

  //! Destructor
  virtual ~TekoSmoother() {}
  //@}

  //! Input
  //@{
  RCP<const ParameterList> GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());
    return validParamList;
  }

  void DeclareInput(Level &currentLevel) const {}

  void SetTekoParameters(RCP<ParameterList> tekoParams){};
  //@}

  //! @name Setup and Apply methods.
  //@{

  /*! @brief Setup routine
   */
  void Setup(Level &currentLevel) {}

  /*! @brief Apply the Teko smoother.

  @param X initial guess
  @param B right-hand side
  @param InitialGuessIsZero This option has no effect.
  */
  void Apply(MultiVector &X, const MultiVector &B, bool InitialGuessIsZero = false) const {}
  //@}

  RCP<SmootherPrototype> Copy() const { return Teuchos::null; }

  //! @name Overridden from Teuchos::Describable
  //@{

  //! Return a simple one-line description of this object.
  std::string description() const {
    std::ostringstream out;
    out << SmootherPrototype::description();
    out << "{type = " << type_ << "}";
    return out.str();
  }

  //! Print the object with some verbosity level to an FancyOStream object.
  // using MueLu::Describable::describe; // overloading, not hiding
  void print(Teuchos::FancyOStream &out, const VerbLevel verbLevel = Default) const {
    MUELU_DESCRIBE;

    if (verbLevel & Parameters0)
      out0 << "Prec. type: " << type_ << std::endl;

    if (verbLevel & Debug)
      out0 << "IsSetup: " << Teuchos::toString(SmootherPrototype::IsSetup()) << std::endl;
  }

  //! Get a rough estimate of cost per iteration
  size_t getNodeSmootherComplexity() const;

  //@}

 private:
  //! smoother type
  std::string type_;
};  // class TekoSmoother

/*!
  @class TekoSmoother
  @brief Interface to block smoothers in Teko package

  This is the specialization for SC=double and LO=int, which actually implements the Teko smoother interface.
  We use Teko and the Teko::LinearOp which is declared as
  Teuchos::RCP<const Thyra::LinearOpBase<ST> > with ST=double. See e.g. Teko_ConfigDefs.hpp.

  In Teko SC=double and LO=int are hard-coded. The node is Tpetra::Map<>::node_type.
  The global ordinal GO is chosen as a conservative choice from what Tpetra supports
  (see Teko_ConfigDefs.hpp for more details)

  Note, that the underlying Thyra package is only templated on the Scalar (not the node type).
*/
template <class GlobalOrdinal,
          class Node>
class TekoSmoother<double, int, GlobalOrdinal, Node> : public SmootherPrototype<double, int, GlobalOrdinal, Node> {
  typedef int LocalOrdinal;
  typedef double Scalar;
  typedef Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node> MapExtractorClass;

#undef MUELU_TEKOSMOOTHER_SHORT
#include "MueLu_UseShortNames.hpp"

 public:
  //! @name Constructors / destructors
  //@{

  /*! @brief Constructor
   */
  TekoSmoother()
    : type_("Teko smoother")
    , A_(Teuchos::null)
    , bA_(Teuchos::null)
    , bThyOp_(Teuchos::null)
    , tekoParams_(Teuchos::null)
    , inverseOp_(Teuchos::null){};

  /*! @brief Constructor
   */
  TekoSmoother(const Teuchos::ParameterList &paramList)
    : type_("Teko smoother")
    , A_(Teuchos::null)
    , bA_(Teuchos::null)
    , bThyOp_(Teuchos::null)
    , tekoParams_(Teuchos::null)
    , inverseOp_(Teuchos::null) {
    this->SetParameter("Inverse Type", paramList.getEntry("Inverse Type"));
    const auto &tekoSettings = paramList.sublist("Inverse Factory Library");
    tekoParams_              = Teuchos::make_rcp<Teuchos::ParameterList>(tekoSettings);
  };

  //! Destructor
  virtual ~TekoSmoother() {}
  //@}

  //! Input
  //@{
  RCP<const ParameterList> GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

    validParamList->set<RCP<const FactoryBase>>("A", null, "Generating factory of the matrix A");
    validParamList->set<std::string>("Inverse Type", "", "Name of parameter list within 'Teko parameters' containing the Teko smoother parameters.");

    return validParamList;
  }

  void DeclareInput(Level &currentLevel) const {
    this->Input(currentLevel, "A");
  }

  void SetTekoParameters(RCP<ParameterList> tekoParams) { tekoParams_ = tekoParams; };
  //@}

  //! @name Setup and Apply methods.
  //@{

  /*! @brief Setup routine
   */
  void Setup(Level &currentLevel) {
    RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

    FactoryMonitor m(*this, "Setup TekoSmoother", currentLevel);
    if (this->IsSetup() == true)
      this->GetOStream(Warnings0) << "MueLu::TekoSmoother::Setup(): Setup() has already been called";

    // extract blocked operator A from current level
    A_  = Factory::Get<RCP<Matrix>>(currentLevel, "A");  // A needed for extracting map extractors
    bA_ = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(A_);
    TEUCHOS_TEST_FOR_EXCEPTION(bA_.is_null(), Exceptions::BadCast,
                               "MueLu::TekoSmoother::Build: input matrix A is not of type BlockedCrsMatrix.");

    bThyOp_ = bA_->getThyraOperator();
    TEUCHOS_TEST_FOR_EXCEPTION(bThyOp_.is_null(), Exceptions::BadCast,
                               "MueLu::TekoSmoother::Build: Could not extract thyra operator from BlockedCrsMatrix.");

    // Hand Teko the blocked operator without the blocks that have no entries, so its block sweeps do not apply
    // (and communicate for) structurally zero couplings.
    auto prunedThyOp = Thyra::defaultBlockedLinearOp<Scalar>();
    prunedThyOp->beginBlockFill(bThyOp_->productRange(), bThyOp_->productDomain());
    for (size_t r = 0; r < bA_->Rows(); ++r) {
      for (size_t c = 0; c < bA_->Cols(); ++c) {
        auto Arc = bA_->getMatrix(r, c);
        // Drop only blocks known to be empty (global constants, if present, are present on every rank).
        const bool knownEmpty = !Arc.is_null() && Arc->haveGlobalConstants() && Arc->getGlobalNumEntries() == 0;
        if (bThyOp_->blockExists(r, c) && !Arc.is_null() && !knownEmpty)
          prunedThyOp->setBlock(r, c, bThyOp_->getBlock(r, c));
      }
    }
    prunedThyOp->endBlockFill();

    Teuchos::RCP<const Thyra::LinearOpBase<Scalar>> thyOp = prunedThyOp;

    // parameter list contains TekoSmoother parameters but does not handle the Teko parameters itself!
    const ParameterList &pL  = Factory::GetParameterList();
    std::string smootherType = pL.get<std::string>("Inverse Type");
    TEUCHOS_TEST_FOR_EXCEPTION(smootherType.empty(), Exceptions::RuntimeError,
                               "MueLu::TekoSmoother::Build: You must provide a 'Smoother Type' name that is defined in the 'Teko parameters' sublist.");
    type_ = smootherType;

    TEUCHOS_TEST_FOR_EXCEPTION(tekoParams_.is_null(), Exceptions::BadCast,
                               "MueLu::TekoSmoother::Build: No Teko parameters have been set.");

    Teuchos::RCP<Teko::InverseLibrary> invLib  = Teko::InverseLibrary::buildFromParameterList(*tekoParams_);
    Teuchos::RCP<Teko::InverseFactory> inverse = invLib->getInverseFactory(smootherType);

    inverseOp_ = Teko::buildInverse(*inverse, thyOp);
    TEUCHOS_TEST_FOR_EXCEPTION(inverseOp_.is_null(), Exceptions::BadCast,
                               "MueLu::TekoSmoother::Build: Failed to build Teko inverse operator. Probably a problem with the Teko parameters.");

    // Converting the product domain space to an Xpetra::BlockedMap builds one Map per block, the full Map, and one
    // Import per block, all collective. Do it once here rather than in every Apply.
    auto domainMap = Xpetra::ThyraUtils<Scalar, LocalOrdinal, GlobalOrdinal, Node>::toXpetra(
        Teuchos::rcp_dynamic_cast<const Thyra::VectorSpaceBase<Scalar>>(bThyOp_->productDomain()), A_->getRowMap()->getComm());
    domainBlockedMap_ = Teuchos::rcp_dynamic_cast<const BlockedMap>(domainMap);
    TEUCHOS_TEST_FOR_EXCEPTION(domainBlockedMap_.is_null(), Exceptions::BadCast,
                               "MueLu::TekoSmoother::Build: Product domain space did not convert to an Xpetra::BlockedMap.");
    thyB_ = Teuchos::null;
    thyX_ = Teuchos::null;

    this->IsSetup(true);
  }

  /*! @brief Apply the Teko smoother.

  @param X initial guess
  @param B right-hand side
  @param InitialGuessIsZero whether the initial guess is zero
  */
  void Apply(MultiVector &X, const MultiVector &B, bool InitialGuessIsZero = false) const {
    TEUCHOS_TEST_FOR_EXCEPTION(!this->IsSetup(), Exceptions::RuntimeError,
                               "MueLu::TekoSmoother::Apply(): Setup() has not been called");

    auto comm           = X.getMap()->getComm();
    auto rgMapExtractor = bA_->getRangeMapExtractor();
    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(rgMapExtractor));

    using STS = Teuchos::ScalarTraits<Scalar>;

    // Work vectors are kept between applies; allocate (or reallocate) them only when the vector count changes.
    auto productMultiVector = [](Teuchos::RCP<Thyra::ProductMultiVectorBase<Scalar>> &cached,
                                 const Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar>> &space,
                                 int numVecs,
                                 const char *castErrorMsg) {
      if (cached.is_null() || cached->domain()->dim() != numVecs) {
        cached = Teuchos::rcp_dynamic_cast<Thyra::ProductMultiVectorBase<Scalar>>(Thyra::createMembers(space, numVecs));
        TEUCHOS_TEST_FOR_EXCEPTION(cached.is_null(), Exceptions::BadCast, castErrorMsg);
      }
      return cached;
    };

    auto solveWithThyra = [&](const Teuchos::RCP<const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> &rhs,
                              bool initializeFromX,
                              const typename STS::magnitudeType alphaX) {
      auto thyProdB = productMultiVector(
          thyB_, Teuchos::rcp_dynamic_cast<const Thyra::VectorSpaceBase<Scalar>>(bThyOp_->productRange()),
          Teuchos::as<int>(rhs->getNumVectors()),
          "MueLu::TekoSmoother::Apply: Failed to cast range space to product range space.");

      Xpetra::ThyraUtils<Scalar, LocalOrdinal, GlobalOrdinal, Node>::updateThyra(rhs, rgMapExtractor, thyProdB);

      auto thyProdX = productMultiVector(
          thyX_, Teuchos::rcp_dynamic_cast<const Thyra::VectorSpaceBase<Scalar>>(bThyOp_->productDomain()),
          Teuchos::as<int>(X.getNumVectors()),
          "MueLu::TekoSmoother::Apply: Failed to cast domain space to product domain space.");

      if (initializeFromX) {
        Xpetra::ThyraUtils<Scalar, LocalOrdinal, GlobalOrdinal, Node>::updateThyra(Teuchos::rcpFromRef(X), rgMapExtractor, thyProdX);
      } else {
        // A freshly created work vector is zero; keep that for the reused one.
        thyProdX->assign(STS::zero());
      }

      inverseOp_->apply(
          Thyra::NOTRANS,
          *thyProdB,
          thyProdX.ptr(),
          STS::one(),
          STS::zero());

      // Same result as ThyraUtils::toXpetra(thyX, comm), but on the BlockedMap built in Setup: each block is a
      // shallow Xpetra view of the Thyra block, so no Maps, Imports, or vectors are created here.
      const int numBlocks = thyProdX->productSpace()->numBlocks();
      std::vector<RCP<MultiVector>> xBlocks(numBlocks);
      for (int b = 0; b < numBlocks; ++b) {
        xBlocks[b] = Xpetra::ThyraUtils<Scalar, LocalOrdinal, GlobalOrdinal, Node>::toXpetra(
            thyProdX->getNonconstMultiVectorBlock(b), comm);
      }
      BlockedMultiVector XX(domainBlockedMap_, xBlocks);
      X.update(STS::one(), XX, alphaX);
    };

    if (InitialGuessIsZero) {
      solveWithThyra(Teuchos::rcpFromRef(B), /*initializeFromX=*/true, STS::zero());
      return;
    }

    auto residual = Utilities::Residual(*A_, X, B);
    solveWithThyra(residual, /*initializeFromX=*/false, STS::one());
  }
  //@}

  RCP<SmootherPrototype> Copy() const { return Teuchos::rcp(new MueLu::TekoSmoother<double, int, GlobalOrdinal, Node>(*this)); }

  //! @name Overridden from Teuchos::Describable
  //@{

  //! Return a simple one-line description of this object.
  std::string description() const {
    std::ostringstream out;
    out << SmootherPrototype::description();
    out << "{type = " << type_ << "}";
    return out.str();
  }

  //! Print the object with some verbosity level to an FancyOStream object.
  // using MueLu::Describable::describe; // overloading, not hiding
  void print(Teuchos::FancyOStream &out, const VerbLevel verbLevel = Default) const {
    MUELU_DESCRIBE;

    if (verbLevel & Parameters0)
      out0 << "Prec. type: " << type_ << std::endl;

    if (verbLevel & Debug)
      out0 << "IsSetup: " << Teuchos::toString(SmootherPrototype::IsSetup()) << std::endl;
  }

  //! Get a rough estimate of cost per iteration
  size_t getNodeSmootherComplexity() const {
    size_t cplx = 0;
    return cplx;
  }

  //@}

 private:
  //! smoother type
  std::string type_;

  //! block operator
  RCP<Matrix> A_;  // < ! internal blocked operator "A" generated by AFact_
  RCP<BlockedCrsMatrix> bA_;
  RCP<const Thyra::BlockedLinearOpBase<Scalar>> bThyOp_;

  //! Teko parameters
  RCP<ParameterList> tekoParams_;  // < ! parameter list containing Teko parameters. These parameters are not administrated by the factory and not validated.

  Teko::LinearOp inverseOp_;  // < ! Teko inverse operator

  //! Xpetra map of the product domain space, built once in Setup
  RCP<const BlockedMap> domainBlockedMap_;

  //! Thyra work vectors reused by Apply
  mutable RCP<Thyra::ProductMultiVectorBase<Scalar>> thyB_;
  mutable RCP<Thyra::ProductMultiVectorBase<Scalar>> thyX_;
};  // class TekoSmoother (specialization on SC=double)
}  // namespace MueLu

#define MUELU_TEKOSMOOTHER_SHORT

#endif  // HAVE_MUELU_TEKO

#endif /* MUELU_TEKOSMOOTHER_DECL_HPP_ */
