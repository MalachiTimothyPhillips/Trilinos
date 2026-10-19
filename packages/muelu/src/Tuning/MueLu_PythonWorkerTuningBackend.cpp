// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "MueLu_PythonWorkerTuningBackend.hpp"

#include <stdexcept>

#include <Teuchos_TestForException.hpp>

namespace MueLu {
namespace GPTuneInt {

void PythonWorkerTuningBackend::initialize(const ParsedSpec& spec,
                                           const Teuchos::ParameterList& /* baseParams */) {
  TEUCHOS_TEST_FOR_EXCEPTION(spec.worker.argv.empty(), std::runtime_error,
                             "MueLu GPTune Python worker backend requires gptune.worker.argv");

  spec_ = spec;
  summary_ = BackendSummary{};
  summary_.status = "running";
  summary_.message = "Python GPTune worker is running.";
  haveSummary_ = false;

  worker_.start(spec_.worker.argv);
  running_ = true;

  worker_.writeLine(BuildWorkerInitMessage(spec_));
  const auto ready = readWorkerMessage();
  TEUCHOS_TEST_FOR_EXCEPTION(ready.type != "ready", std::runtime_error,
                             "Expected GPTune worker 'ready' message but received '" << ready.type << "'");
}

WorkerMessage PythonWorkerTuningBackend::readWorkerMessage() {
  return ParseWorkerMessage(worker_.readLine(spec_.worker.request_timeout_seconds), spec_);
}

CandidateRequest PythonWorkerTuningBackend::ask() {
  CandidateRequest request;
  if (haveSummary_) {
    request.stop = true;
    return request;
  }

  const auto message = readWorkerMessage();
  if (message.type == "candidate") {
    request.hasCandidate = true;
    request.candidateId = message.candidateId;
    request.candidate = message.candidate;
    return request;
  }

  if (message.type == "best") {
    summary_.bestCandidate = message.bestCandidate;
    summary_.bestObjective = message.bestObjective;
    summary_.evaluationsPerformed = message.evaluationsPerformed;
    summary_.status = message.status.empty() ? std::string("success") : message.status;
    summary_.message = message.message;
    haveSummary_ = true;
    request.stop = true;
    return request;
  }

  if (message.type == "stop") {
    summary_.status = message.status.empty() ? std::string("stopped") : message.status;
    summary_.message = message.message;
    haveSummary_ = true;
    request.stop = true;
    return request;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Unexpected GPTune worker message type in ask(): '" << message.type << "'");
  return request;
}

void PythonWorkerTuningBackend::tell(const std::string& candidateId,
                                     const Candidate& candidate,
                                     const EvaluationResult& result) {
  TEUCHOS_TEST_FOR_EXCEPTION(!running_, std::runtime_error,
                             "Cannot tell GPTune worker before it is running");
  worker_.writeLine(BuildWorkerTellMessage(candidateId, candidate, result));
}

BackendSummary PythonWorkerTuningBackend::finalize() {
  if (!haveSummary_) {
    try {
      const auto message = readWorkerMessage();
      if (message.type == "best") {
        summary_.bestCandidate = message.bestCandidate;
        summary_.bestObjective = message.bestObjective;
        summary_.evaluationsPerformed = message.evaluationsPerformed;
        summary_.status = message.status.empty() ? std::string("success") : message.status;
        summary_.message = message.message;
        haveSummary_ = true;
      }
    } catch (const std::exception& exc) {
      summary_.status = "failed";
      summary_.message = std::string("GPTune worker failed to finalize: ") + exc.what();
    }
  }

  return summary_;
}

void PythonWorkerTuningBackend::shutdown() {
  if (running_) {
    try {
      worker_.writeLine(BuildWorkerShutdownMessage());
    } catch (...) {
    }
    worker_.shutdown();
    running_ = false;
  }
}

}  // namespace GPTuneInt
}  // namespace MueLu
