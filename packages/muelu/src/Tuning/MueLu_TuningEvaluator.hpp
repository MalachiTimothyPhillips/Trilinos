// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_TUNINGEVALUATOR_HPP
#define MUELU_TUNINGEVALUATOR_HPP

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_TestForException.hpp>

#include <Xpetra_Matrix.hpp>

#include "MueLu_Exceptions.hpp"
#include "MueLu_Hierarchy.hpp"
#include "MueLu_HierarchyUtils.hpp"
#include "MueLu_ML2MueLuParameterTranslator.hpp"
#include "MueLu_ParameterListInterpreter.hpp"
#include "MueLu_TuningBackend.hpp"
#include "MueLu_Utilities.hpp"

namespace MueLu {
namespace GPTuneInt {

inline double ObjectiveFromMetrics(const ParsedSpec& spec,
                                   const EvaluationMetrics& metrics) {
  switch (spec.objective) {
    case Objective::SetupTime:
      return metrics.setupTime;
    case Objective::TotalTime:
      return metrics.totalTime;
    case Objective::SolveTime:
    case Objective::IterationCount:
    case Objective::AmortizedSolveTime:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                 "MueLu GPTune C++ evaluator does not yet implement solve-dependent objectives");
  }

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Unknown MueLu GPTune objective");
  return spec.failure_penalty;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class MueLuHierarchyEvaluator final : public CandidateEvaluator {
 public:
  using SC = Scalar;
  using LO = LocalOrdinal;
  using GO = GlobalOrdinal;
  using NO = Node;
  using Matrix = Xpetra::Matrix<SC, LO, GO, NO>;
  using Hierarchy = MueLu::Hierarchy<SC, LO, GO, NO>;
  using HierarchyManager = MueLu::HierarchyManager<SC, LO, GO, NO>;
  using HierarchyUtils = MueLu::HierarchyUtils<SC, LO, GO, NO>;
  using ParameterListInterpreter = MueLu::ParameterListInterpreter<SC, LO, GO, NO>;

  MueLuHierarchyEvaluator(const Teuchos::RCP<Matrix>& op,
                          const Teuchos::ParameterList& baseParams,
                          const Teuchos::ParameterList& fullParams,
                          const ParsedSpec& spec)
    : op_(op), baseParams_(baseParams), fullParams_(fullParams), spec_(spec) {}

  EvaluationResult evaluate(const Candidate& candidate) override {
    EvaluationResult result;
    try {
      Teuchos::ParameterList params(baseParams_);
      ApplyCandidateToParameterList(candidate, params);

      std::string label;
      if (params.isParameter("hierarchy label")) {
        label = params.get<std::string>("hierarchy label");
      } else if (!op_.is_null()) {
        label = op_->getObjectLabel();
      }

      Teuchos::ParameterList nonSerialList, serialList;
      MueLu::ExtractNonSerializableData(fullParams_, serialList, nonSerialList);

      if (params.isParameter("parameterlist: syntax") &&
          params.get<std::string>("parameterlist: syntax") == "ml") {
        params.remove("parameterlist: syntax");
        params = *MueLu::ML2MueLuParameterTranslator::translate(params, "");
      }

      Teuchos::Time timer("MueLu GPTune candidate setup");
      timer.start();

      Teuchos::RCP<HierarchyManager> mueLuFactory =
          Teuchos::rcp(new ParameterListInterpreter(params, op_->getDomainMap()->getComm()));
      Teuchos::RCP<Hierarchy> H = mueLuFactory->CreateHierarchy(label);
      H->setlib(op_->getDomainMap()->lib());
      H->GetLevel(0)->Set("A", op_);
      H->SetProcRankVerbose(op_->getDomainMap()->getComm()->getRank());
      HierarchyUtils::AddNonSerializableDataToHierarchy(*mueLuFactory, *H, nonSerialList);
      mueLuFactory->SetupHierarchy(*H);

      timer.stop();
      result.metrics.setupTime = timer.totalElapsedTime();
      result.metrics.totalTime = result.metrics.setupTime;
      result.objective = ObjectiveFromMetrics(spec_, result.metrics);
      result.status = "success";
    } catch (const std::exception& exc) {
      result.objective = spec_.failure_penalty;
      result.status = "failed";
      result.message = exc.what();
      result.metrics.converged = false;
    }
    return result;
  }

 private:
  Teuchos::RCP<Matrix> op_;
  Teuchos::ParameterList baseParams_;
  Teuchos::ParameterList fullParams_;
  ParsedSpec spec_;
};

}  // namespace GPTuneInt
}  // namespace MueLu

#endif  // MUELU_TUNINGEVALUATOR_HPP
