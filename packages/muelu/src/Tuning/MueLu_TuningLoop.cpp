// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "MueLu_TuningLoop.hpp"

#include <sstream>

namespace MueLu {
namespace GPTuneInt {

BackendSummary RunTuningLoop(const ParsedSpec& spec,
                             TuningBackend& backend,
                             CandidateEvaluator& evaluator,
                             const Teuchos::ParameterList& baseParams) {
  backend.initialize(spec, baseParams);

  int evaluations = 0;
  while (evaluations < spec.max_evaluations) {
    auto request = backend.ask();
    if (request.stop) break;
    if (!request.hasCandidate) {
      break;
    }

    auto result = evaluator.evaluate(request.candidate);
    backend.tell(request.candidateId, request.candidate, result);
    ++evaluations;
  }

  auto summary = backend.finalize();
  if (summary.evaluationsPerformed == 0) {
    summary.evaluationsPerformed = evaluations;
  }
  backend.shutdown();
  return summary;
}

}  // namespace GPTuneInt
}  // namespace MueLu
