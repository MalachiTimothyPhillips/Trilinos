// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_PYTHONWORKERTUNINGBACKEND_HPP
#define MUELU_PYTHONWORKERTUNINGBACKEND_HPP

#include "MueLu_GPTuneJson.hpp"
#include "MueLu_Process.hpp"
#include "MueLu_TuningBackend.hpp"

namespace MueLu {
namespace GPTuneInt {

class PythonWorkerTuningBackend : public TuningBackend {
 public:
  void initialize(const ParsedSpec& spec,
                  const Teuchos::ParameterList& baseParams) override;

  CandidateRequest ask() override;

  void tell(const std::string& candidateId,
            const Candidate& candidate,
            const EvaluationResult& result) override;

  BackendSummary finalize() override;

  void shutdown() override;

 private:
  WorkerMessage readWorkerMessage();

  ParsedSpec spec_;
  Process worker_;
  bool running_ = false;
  bool haveSummary_ = false;
  BackendSummary summary_;
};

}  // namespace GPTuneInt
}  // namespace MueLu

#endif  // MUELU_PYTHONWORKERTUNINGBACKEND_HPP
