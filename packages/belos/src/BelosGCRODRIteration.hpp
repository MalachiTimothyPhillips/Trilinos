// @HEADER
// *****************************************************************************
//                 Belos: Block Linear Solvers Package
//
// Copyright 2004-2016 NTESS and the Belos contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef BELOS_GCRODR_ITERATION_HPP
#define BELOS_GCRODR_ITERATION_HPP

#include "BelosConfigDefs.hpp"
#include "BelosTypes.hpp"
#include "BelosIteration.hpp"

namespace Belos {

template <class ScalarType, class MV>
struct GCRODRIterState;

template<class ScalarType, class MV, class OP>
class GCRODRIteration : virtual public Iteration<ScalarType,MV,OP> {
public:
  virtual ~GCRODRIteration() {}

  virtual void initialize(GCRODRIterState<ScalarType,MV>& newstate) = 0;
  virtual GCRODRIterState<ScalarType,MV> getState() const = 0;

  virtual void updateLSQR(int dim = -1) = 0;

  virtual int getCurSubspaceDim() const = 0;
  virtual int getMaxSubspaceDim() const = 0;

  virtual void setSize(int recycledBlocks, int numBlocks) = 0;
};

} // namespace Belos

#endif // BELOS_GCRODR_ITERATION_HPP
