// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_GPTUNEBRIDGE_HPP
#define MUELU_GPTUNEBRIDGE_HPP

#include <memory>

#include <Teuchos_RCP.hpp>
#include <Teuchos_TestForException.hpp>

#include <Xpetra_Matrix.hpp>

#include "MueLu_config.hpp"
#include "MueLu_FixedTuningBackend.hpp"
#include "MueLu_GPTuneParameterList.hpp"
#ifdef HAVE_MUELU_GPTUNE_PYTHON_WORKER
#include "MueLu_PythonWorkerTuningBackend.hpp"
#endif
#include "MueLu_TuningEvaluator.hpp"
#include "MueLu_TuningLoop.hpp"

namespace MueLu {
namespace GPTuneInt {

struct BridgeResult {
  Candidate bestCandidate;
  double bestObjective = 0.0;
  int evaluationsPerformed = 0;
  std::string status = "disabled";
  std::string message;
};

inline std::unique_ptr<TuningBackend> CreateTuningBackend(const ParsedSpec& spec) {
  switch (spec.backend) {
    case Backend::Fixed:
    case Backend::ExternalDriver:
      return std::unique_ptr<TuningBackend>(new FixedTuningBackend());
    case Backend::Python:
#ifdef HAVE_MUELU_GPTUNE_PYTHON_WORKER
      return std::unique_ptr<TuningBackend>(new PythonWorkerTuningBackend());
#else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                 "MueLu GPTune backend 'python' requires MueLu_ENABLE_GPTUNE_PYTHON_WORKER=ON and a gptune.worker.argv command");
      break;
#endif
    case Backend::Embedded:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                 "MueLu GPTune backend 'embedded' is not supported; use the external Python worker backend instead");
      break;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Unknown MueLu GPTune backend");
  return std::unique_ptr<TuningBackend>();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BridgeResult RunGPTuneBridge(const Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>& op,
                             const Teuchos::ParameterList& baseParams,
                             const Teuchos::ParameterList& fullParams,
                             const ParsedSpec& spec) {
  BridgeResult result;

  auto backend = CreateTuningBackend(spec);
  MueLuHierarchyEvaluator<Scalar, LocalOrdinal, GlobalOrdinal, Node> evaluator(op, baseParams, fullParams, spec);
  const auto summary = RunTuningLoop(spec, *backend, evaluator, baseParams);

  result.bestCandidate = summary.bestCandidate;
  result.bestObjective = summary.bestObjective;
  result.evaluationsPerformed = summary.evaluationsPerformed;
  result.status = summary.status;
  result.message = summary.message;
  return result;
}

}  // namespace GPTuneInt
}  // namespace MueLu

#define MUELU_GPTUNEBRIDGE_SHORT

#endif  // MUELU_GPTUNEBRIDGE_HPP
