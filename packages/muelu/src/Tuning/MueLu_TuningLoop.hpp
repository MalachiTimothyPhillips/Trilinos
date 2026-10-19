// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_TUNINGLOOP_HPP
#define MUELU_TUNINGLOOP_HPP

#include "MueLu_TuningBackend.hpp"

namespace MueLu {
namespace GPTuneInt {

BackendSummary RunTuningLoop(const ParsedSpec& spec,
                             TuningBackend& backend,
                             CandidateEvaluator& evaluator,
                             const Teuchos::ParameterList& baseParams);

}  // namespace GPTuneInt
}  // namespace MueLu

#endif  // MUELU_TUNINGLOOP_HPP
