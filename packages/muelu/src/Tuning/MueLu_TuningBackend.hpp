// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_TUNINGBACKEND_HPP
#define MUELU_TUNINGBACKEND_HPP

#include <Teuchos_ParameterList.hpp>

#include "MueLu_TuningTypes.hpp"

namespace MueLu {
namespace GPTuneInt {

class CandidateEvaluator {
 public:
  virtual ~CandidateEvaluator() = default;

  virtual void beginEvaluation(const std::string& /* candidateId */, int /* evaluationIndex */) {}

  virtual EvaluationResult evaluate(const Candidate& candidate) = 0;
};

class TuningBackend {
 public:
  virtual ~TuningBackend() = default;

  virtual void initialize(const ParsedSpec& spec,
                          const Teuchos::ParameterList& baseParams) = 0;

  virtual CandidateRequest ask() = 0;

  virtual void tell(const std::string& candidateId,
                    const Candidate& candidate,
                    const EvaluationResult& result) = 0;

  virtual BackendSummary finalize() = 0;

  virtual void shutdown() {}
};

}  // namespace GPTuneInt
}  // namespace MueLu

#endif  // MUELU_TUNINGBACKEND_HPP
