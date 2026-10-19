// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_FIXEDTUNINGBACKEND_HPP
#define MUELU_FIXEDTUNINGBACKEND_HPP

#include <limits>
#include <vector>

#include "MueLu_TuningBackend.hpp"

namespace MueLu {
namespace GPTuneInt {

class FixedTuningBackend : public TuningBackend {
 public:
  void initialize(const ParsedSpec& spec,
                  const Teuchos::ParameterList& baseParams) override;

  CandidateRequest ask() override;

  void tell(const std::string& candidateId,
            const Candidate& candidate,
            const EvaluationResult& result) override;

  BackendSummary finalize() override;

 private:
  Candidate makeBaselineCandidate(const ParsedSpec& spec,
                                  const Teuchos::ParameterList& baseParams) const;
  void addCandidateIfNew(const Candidate& candidate,
                         const ParsedSpec& spec);

  std::vector<Candidate> candidates_;
  std::size_t nextCandidate_ = 0;
  BackendSummary summary_;
  double bestObjective_ = std::numeric_limits<double>::infinity();
};

}  // namespace GPTuneInt
}  // namespace MueLu

#endif  // MUELU_FIXEDTUNINGBACKEND_HPP
