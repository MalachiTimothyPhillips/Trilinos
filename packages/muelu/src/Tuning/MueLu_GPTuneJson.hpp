// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_GPTUNEJSON_HPP
#define MUELU_GPTUNEJSON_HPP

#include <string>

#include "MueLu_GPTuneParameterList.hpp"
#include "MueLu_TuningTypes.hpp"

namespace MueLu {
namespace GPTuneInt {

struct WorkerMessage {
  std::string type;
  std::string candidateId;
  Candidate candidate;
  Candidate bestCandidate;
  double bestObjective = 0.0;
  int evaluationsPerformed = 0;
  std::string status;
  std::string message;
};

std::string BuildWorkerInitMessage(const ParsedSpec& spec);

std::string BuildWorkerTellMessage(const std::string& candidateId,
                                   const Candidate& candidate,
                                   const EvaluationResult& result);

std::string BuildWorkerShutdownMessage();

WorkerMessage ParseWorkerMessage(const std::string& line,
                                 const ParsedSpec& spec);

}  // namespace GPTuneInt
}  // namespace MueLu

#endif  // MUELU_GPTUNEJSON_HPP
