// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "MueLu_FixedTuningBackend.hpp"

#include <cmath>
#include <sstream>

#include <Teuchos_TestForException.hpp>

namespace MueLu {
namespace GPTuneInt {

namespace {

std::string anyToComparableString(const Teuchos::any& value) {
  if (value.type() == typeid(std::string)) return Teuchos::any_cast<std::string>(value);
  if (value.type() == typeid(int)) return std::to_string(Teuchos::any_cast<int>(value));
  if (value.type() == typeid(double)) {
    std::ostringstream os;
    os << Teuchos::any_cast<double>(value);
    return os.str();
  }
  if (value.type() == typeid(bool)) return Teuchos::any_cast<bool>(value) ? "true" : "false";
  return "<unsupported>";
}

bool sameCandidate(const Candidate& lhs, const Candidate& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (const auto& kv : lhs) {
    auto it = rhs.find(kv.first);
    if (it == rhs.end()) return false;
    if (anyToComparableString(kv.second) != anyToComparableString(it->second)) return false;
  }
  return true;
}

void pruneInactive(Candidate& candidate,
                   const ParsedSpec& spec) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& param : spec.search_parameters) {
      auto it = candidate.find(param.muelu_key);
      if (it != candidate.end() && !IsActive(param, candidate)) {
        candidate.erase(it);
        changed = true;
      }
    }
  }
}

}  // namespace

Candidate FixedTuningBackend::makeBaselineCandidate(const ParsedSpec& spec,
                                                    const Teuchos::ParameterList& baseParams) const {
  Candidate candidate;

  for (const auto& param : spec.search_parameters) {
    if (param.has_initial_value) {
      candidate[param.muelu_key] = param.initial_value;
      continue;
    }

    Teuchos::any baseValue;
    if (GetAnyParameterByPath(baseParams, param.muelu_key, baseValue)) {
      candidate[param.muelu_key] = baseValue;
      continue;
    }

    if (param.type == ParamType::Categorical) {
      TEUCHOS_TEST_FOR_EXCEPTION(param.categorical_values.empty(), std::runtime_error,
                                 "Categorical GPTune parameter '" << param.muelu_key << "' has empty values");
      candidate[param.muelu_key] = param.categorical_values.front();
    } else if (param.type == ParamType::Integer) {
      candidate[param.muelu_key] = param.int_min;
    } else {
      candidate[param.muelu_key] = param.real_min;
    }
  }

  pruneInactive(candidate, spec);
  return candidate;
}

void FixedTuningBackend::addCandidateIfNew(const Candidate& candidate,
                                           const ParsedSpec& spec) {
  Candidate activeCandidate(candidate);
  pruneInactive(activeCandidate, spec);

  for (const auto& existing : candidates_) {
    if (sameCandidate(existing, activeCandidate)) return;
  }
  candidates_.push_back(activeCandidate);
}

void FixedTuningBackend::initialize(const ParsedSpec& spec,
                                    const Teuchos::ParameterList& baseParams) {
  candidates_.clear();
  nextCandidate_ = 0;
  summary_ = BackendSummary{};
  summary_.status = "running";
  summary_.message = "Fixed tuning backend generated deterministic candidates.";
  bestObjective_ = std::numeric_limits<double>::infinity();

  Candidate baseline = makeBaselineCandidate(spec, baseParams);
  addCandidateIfNew(baseline, spec);

  for (const auto& param : spec.search_parameters) {
    if (static_cast<int>(candidates_.size()) >= spec.max_evaluations) break;

    if (param.type == ParamType::Categorical) {
      for (const auto& value : param.categorical_values) {
        Candidate c(baseline);
        c[param.muelu_key] = value;
        addCandidateIfNew(c, spec);
        if (static_cast<int>(candidates_.size()) >= spec.max_evaluations) break;
      }
    } else if (param.type == ParamType::Integer) {
      const int values[] = {param.int_min, param.int_max};
      for (const auto value : values) {
        Candidate c(baseline);
        c[param.muelu_key] = value;
        addCandidateIfNew(c, spec);
        if (static_cast<int>(candidates_.size()) >= spec.max_evaluations) break;
      }
    } else {
      const double mid = 0.5 * (param.real_min + param.real_max);
      const double values[] = {param.real_min, mid, param.real_max};
      for (const auto value : values) {
        Candidate c(baseline);
        c[param.muelu_key] = value;
        addCandidateIfNew(c, spec);
        if (static_cast<int>(candidates_.size()) >= spec.max_evaluations) break;
      }
    }
  }
}

CandidateRequest FixedTuningBackend::ask() {
  CandidateRequest request;
  if (nextCandidate_ >= candidates_.size()) {
    request.stop = true;
    return request;
  }

  request.hasCandidate = true;
  request.candidate = candidates_[nextCandidate_];
  request.candidateId = std::to_string(nextCandidate_);
  ++nextCandidate_;
  return request;
}

void FixedTuningBackend::tell(const std::string& /* candidateId */,
                              const Candidate& candidate,
                              const EvaluationResult& result) {
  ++summary_.evaluationsPerformed;

  const bool success = (result.status == "success");
  if (success && result.objective < bestObjective_) {
    bestObjective_ = result.objective;
    summary_.bestObjective = result.objective;
    summary_.bestCandidate = candidate;
    summary_.status = "success";
    summary_.message = "Fixed tuning backend selected the best evaluated candidate.";
  } else if (summary_.bestCandidate.empty()) {
    summary_.bestObjective = result.objective;
    summary_.bestCandidate = candidate;
    summary_.status = success ? "success" : "failed";
    summary_.message = success ? "Fixed tuning backend selected the only evaluated candidate."
                               : result.message;
  }
}

BackendSummary FixedTuningBackend::finalize() {
  if (summary_.evaluationsPerformed == 0) {
    summary_.status = "failed";
    summary_.message = "Fixed tuning backend did not evaluate any candidates.";
  }
  return summary_;
}

}  // namespace GPTuneInt
}  // namespace MueLu
