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

#include <Teuchos_RCP.hpp>
#include <Teuchos_TestForException.hpp>

#include <Xpetra_Matrix.hpp>

#include "MueLu_GPTuneParameterList.hpp"

namespace MueLu {
namespace GPTuneInt {

struct BridgeResult {
  Candidate bestCandidate;
  double bestObjective = 0.0;
  int evaluationsPerformed = 0;
  std::string status = "disabled";
  std::string message;
};

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BridgeResult RunGPTuneBridge(const Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>& /* op */,
                             const Teuchos::ParameterList& baseParams,
                             const Teuchos::ParameterList& /* fullParams */,
                             const ParsedSpec& spec) {
  BridgeResult result;
  result.status = "success";
  result.message = "GPTune bridge placeholder selected baseline parameter values from the base ParameterList.";

  for (const auto& param : spec.search_parameters) {
    if (param.has_initial_value) {
      result.bestCandidate[param.muelu_key] = param.initial_value;
      continue;
    }

    if (baseParams.isParameter(param.muelu_key)) {
      const auto& entry = baseParams.getEntry(param.muelu_key);
      if (entry.isType<std::string>()) {
        result.bestCandidate[param.muelu_key] = baseParams.get<std::string>(param.muelu_key);
      } else if (entry.isType<int>()) {
        result.bestCandidate[param.muelu_key] = baseParams.get<int>(param.muelu_key);
      } else if (entry.isType<double>()) {
        result.bestCandidate[param.muelu_key] = baseParams.get<double>(param.muelu_key);
      } else if (entry.isType<bool>()) {
        result.bestCandidate[param.muelu_key] = baseParams.get<bool>(param.muelu_key);
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                   "Unsupported base ParameterList type for GPTune key '" << param.muelu_key << "'");
      }
      continue;
    }

    if (param.type == ParamType::Categorical) {
      TEUCHOS_TEST_FOR_EXCEPTION(param.categorical_values.empty(), std::runtime_error,
                                 "Categorical GPTune parameter '" << param.muelu_key << "' has empty values");
      result.bestCandidate[param.muelu_key] = param.categorical_values.front();
    } else if (param.type == ParamType::Integer) {
      result.bestCandidate[param.muelu_key] = param.int_min;
    } else {
      result.bestCandidate[param.muelu_key] = param.real_min;
    }
  }

  result.bestObjective = 0.0;
  result.evaluationsPerformed = 0;
  return result;
}

}  // namespace GPTuneInt
}  // namespace MueLu

#define MUELU_GPTUNEBRIDGE_SHORT

#endif  // MUELU_GPTUNEBRIDGE_HPP
