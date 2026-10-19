// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_TUNINGTYPES_HPP
#define MUELU_TUNINGTYPES_HPP

#include <string>
#include <vector>

#include <Teuchos_ParameterList.hpp>

#include "MueLu_GPTuneParameterList.hpp"

namespace MueLu {
namespace GPTuneInt {

struct EvaluationMetrics {
  double setupTime = 0.0;
  double solveTime = 0.0;
  double totalTime = 0.0;
  int iterationCount = -1;
  bool converged = true;
};

struct EvaluationResult {
  double objective = 0.0;
  EvaluationMetrics metrics;
  std::string status = "success";
  std::string message;
};

struct CandidateRequest {
  bool hasCandidate = false;
  bool stop = false;
  Candidate candidate;
  std::string candidateId;
};

struct CandidateEvaluationRecord {
  int evaluationIndex = -1;
  std::string candidateId;
  Candidate candidate;
  EvaluationResult result;
  bool retainedAsBest = false;
};

struct TuningRunSummary {
  Candidate bestCandidate;
  double bestObjective = 0.0;
  int evaluationsPerformed = 0;
  std::string status = "disabled";
  std::string message;
  std::vector<CandidateEvaluationRecord> history;
  Teuchos::ParameterList bestParameterList;
  bool hasBestCandidate = false;
  bool hasRetainedHierarchy = false;
  bool returnedRetainedHierarchy = false;
};

struct BackendSummary {
  Candidate bestCandidate;
  double bestObjective = 0.0;
  int evaluationsPerformed = 0;
  std::string status = "disabled";
  std::string message;
};

}  // namespace GPTuneInt
}  // namespace MueLu

#endif  // MUELU_TUNINGTYPES_HPP
