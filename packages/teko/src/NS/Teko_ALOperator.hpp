/*
 * Author: Zhen Wang
 * Email: wangz@ornl.gov
 *        zhen.wang@alum.emory.edu
 */

#ifndef __Teko_ALOperator_hpp__
#define __Teko_ALOperator_hpp__

#include "Teko_BlockedTpetraOperator.hpp"
#include "Teko_Utilities.hpp"
#include "Teko_ConfigDefs.hpp"

namespace Teko::NS {

/** \brief Sparse matrix-vector multiplication
 * for augmented Lagrangian-based preconditioners.
 *
 * This class implements sparse matrix-vector multiplication
 * for augmented Lagrangian-based preconditioners.
 */
class ALOperator : public Teko::TpetraHelpers::BlockedTpetraOperator {
 public:
  /** Build an augmented Lagrangian operator based on a vector of vector
   * of global IDs.
   *
   * \param[in] vars
   *            Vector of vectors of global ids specifying
   *            how the operator is to be blocked.
   * \param[in] content
   *            Operator to be blocked
   * \param[in] pressureMassMatrix
   *            Pressure mass matrix
   * \param[in] gamma
   *            Augmentation parameter
   * \param[in] label
   *            Label for name the operator
   */
  ALOperator(const std::vector<std::vector<GO> >& vars,
             const Teuchos::RCP<Tpetra::Operator<ST, LO, GO, NT> >& content,
             LinearOp pressureMassMatrix, double gamma = 0.05, const std::string& label = "<ANYM>");

  /** Build a modified augmented Lagrangian operator based on a vector of vector
   * of global IDs.
   *
   * \param[in] vars
   *            Vector of vectors of global ids specifying
   *            how the operator is to be blocked.
   * \param[in] content
   *            Operator to be blocked
   * \param[in] gamma
   *            Augmentation parameter
   * \param[in] label
   *            Name of the operator
   */
  ALOperator(const std::vector<std::vector<GO> >& vars,
             const Teuchos::RCP<Tpetra::Operator<ST, LO, GO, NT> >& content, double gamma = 0.05,
             const std::string& label = "<ANYM>");

  virtual ~ALOperator() {}

  /** Set the pressure mass matrix. */
  void setPressureMassMatrix(LinearOp pressureMassMatrix);

  /** Returns pressure mass matrix that can be used to construct preconditioner. */
  const LinearOp& getPressureMassMatrix() const { return pressureMassMatrix_; }

  /** Set gamma. */
  void setGamma(double gamma);

  /** Returns augmentation parameter gamma. */
  const double& getGamma() const { return gamma_; }

  /** Build augmented RHS.
   *
   * \param[in] b
   *            Right-hand side.
   * \param[out] bAugmented
   *             Augmented right-hand side.
   */
  void augmentRHS(const Tpetra::MultiVector<ST, LO, GO, NT>& b,
                  Tpetra::MultiVector<ST, LO, GO, NT>& bAugmented);

  /** Returns number of block rows. */
  int getNumberOfBlockRows() const { return numBlockRows_; }

  /** Force a rebuild of the blocked operator from the stored content operator. */
  virtual void RebuildOps() { BuildALOperator(); }

  /** Get the (i,j) block of the original (non-augmented) operator. */
  const Teuchos::RCP<const Tpetra::Operator<ST, LO, GO, NT> > GetBlock(int i, int j) const;

 protected:
  /** AL operator. */
  Teuchos::RCP<Thyra::LinearOpBase<ST> > alOperator_;

  /** Operator for augmenting the right-hand side. */
  Teuchos::RCP<Thyra::LinearOpBase<ST> > alOperatorRhs_;

  /** Pressure mass matrix and inverse pressure mass matrix. */
  LinearOp pressureMassMatrix_;
  LinearOp invPressureMassMatrix_;

  /** Augmentation parameter. */
  double gamma_;

  /** Dimension of the problem. */
  int dim_;

  /** Number of block rows. */
  int numBlockRows_;

  /** Check dimension. Only implemented for 2D and 3D problems. */
  void checkDim(const std::vector<std::vector<GO> >& vars);

  /** Build AL operator. */
  void BuildALOperator();
};

}  // end namespace Teko::NS

#endif /* __Teko_ALOperator_hpp__ */
