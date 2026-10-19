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

#include <algorithm>
#include <cctype>
#include <limits>
#include <string>
#include <utility>

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVector.hpp>

#ifdef HAVE_MUELU_BELOS
#include <BelosLinearProblem.hpp>
#include <BelosOperatorT.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosXpetraAdapter.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#endif

#include "MueLu_Exceptions.hpp"
#include "MueLu_Hierarchy.hpp"
#include "MueLu_HierarchyUtils.hpp"
#include "MueLu_ML2MueLuParameterTranslator.hpp"
#include "MueLu_ParameterListInterpreter.hpp"
#include "MueLu_TuningBackend.hpp"
#include "MueLu_Utilities.hpp"

namespace MueLu {
namespace GPTuneInt {

inline bool IsSolveDependentObjective(const Objective objective) {
  return objective == Objective::SolveTime ||
         objective == Objective::IterationCount ||
         objective == Objective::AmortizedSolveTime;
}

inline std::string ToLower(const std::string& value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return result;
}

inline std::string BelosSolverName(const std::string& solver) {
  const std::string lower = ToLower(solver);
  if (lower == "cg") {
    return "Pseudo Block CG";
  }
  if (lower == "gmres") {
    return "Pseudo Block GMRES";
  }
  return solver;
}

inline double ObjectiveFromMetrics(const ParsedSpec& spec,
                                   const EvaluationMetrics& metrics) {
  switch (spec.objective) {
    case Objective::SetupTime:
      return metrics.setupTime;
    case Objective::SolveTime:
      return metrics.solveTime;
    case Objective::TotalTime:
      return metrics.totalTime;
    case Objective::IterationCount:
      return static_cast<double>(metrics.iterationCount);
    case Objective::AmortizedSolveTime:
      return metrics.solveTime + metrics.setupTime / std::max(1, spec.reuse_count);
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
  using MultiVector = Xpetra::MultiVector<SC, LO, GO, NO>;
  using Hierarchy = MueLu::Hierarchy<SC, LO, GO, NO>;
  using HierarchyManager = MueLu::HierarchyManager<SC, LO, GO, NO>;
  using HierarchyUtils = MueLu::HierarchyUtils<SC, LO, GO, NO>;
  using ParameterListInterpreter = MueLu::ParameterListInterpreter<SC, LO, GO, NO>;

  MueLuHierarchyEvaluator(const Teuchos::RCP<Matrix>& op,
                          const Teuchos::ParameterList& baseParams,
                          const Teuchos::ParameterList& fullParams,
                          const ParsedSpec& spec)
    : op_(op), baseParams_(baseParams), fullParams_(fullParams), spec_(spec) {}

  void beginEvaluation(const std::string& candidateId, int evaluationIndex) override {
    pendingCandidateId_ = candidateId;
    pendingEvaluationIndex_ = evaluationIndex;
  }

  EvaluationResult evaluate(const Candidate& candidate) override {
    EvaluationResult result;
    Teuchos::RCP<Hierarchy> H;
    Teuchos::ParameterList params(baseParams_);

    try {
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
      H = mueLuFactory->CreateHierarchy(label);
      H->setlib(op_->getDomainMap()->lib());
      H->GetLevel(0)->Set("A", op_);
      H->SetProcRankVerbose(op_->getDomainMap()->getComm()->getRank());
      HierarchyUtils::AddNonSerializableDataToHierarchy(*mueLuFactory, *H, nonSerialList);
      mueLuFactory->SetupHierarchy(*H);

      timer.stop();
      result.metrics.setupTime = timer.totalElapsedTime();

      if (spec_.evaluation.measure_solve) {
        RunSolve(*H, result);
      } else if (IsSolveDependentObjective(spec_.objective)) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                   "Selected MueLu GPTune objective requires solve measurement");
      }

      result.metrics.totalTime = 0.0;
      if (spec_.evaluation.measure_setup) {
        result.metrics.totalTime += result.metrics.setupTime;
      }
      if (spec_.evaluation.measure_solve) {
        result.metrics.totalTime += result.metrics.solveTime;
      }

      if (!result.metrics.converged && spec_.evaluation.fail_on_nonconvergence) {
        result.objective = spec_.failure_penalty;
        result.status = "failed";
      } else {
        result.objective = ObjectiveFromMetrics(spec_, result.metrics);
        result.status = "success";
      }
    } catch (const std::exception& exc) {
      result.objective = spec_.failure_penalty;
      result.status = "failed";
      result.message = exc.what();
      result.metrics.converged = false;
    }

    const bool retainedAsBest = result.status == "success" &&
                                !H.is_null() &&
                                (!hasBest_ || result.objective < bestObjective_);
    if (retainedAsBest) {
      hasBest_ = true;
      bestObjective_ = result.objective;
      bestCandidate_ = candidate;
      bestParams_ = params;
      bestHierarchy_ = H;
    }

    CandidateEvaluationRecord record;
    record.evaluationIndex = pendingEvaluationIndex_ >= 0 ? pendingEvaluationIndex_ : static_cast<int>(history_.size());
    record.candidateId = pendingCandidateId_;
    record.candidate = candidate;
    record.result = result;
    record.retainedAsBest = retainedAsBest;
    history_.push_back(record);

    pendingCandidateId_.clear();
    pendingEvaluationIndex_ = -1;
    return result;
  }

  const std::vector<CandidateEvaluationRecord>& history() const { return history_; }

  bool hasBest() const { return hasBest_; }

  const Candidate& bestCandidate() const { return bestCandidate_; }

  double bestObjective() const { return bestObjective_; }

  const Teuchos::ParameterList& bestParameterList() const { return bestParams_; }

  const Teuchos::RCP<Hierarchy>& bestHierarchy() const { return bestHierarchy_; }

 private:
#ifdef HAVE_MUELU_BELOS
  class HierarchyBelosOperator final : public Belos::OperatorT<MultiVector> {
   public:
    explicit HierarchyBelosOperator(const Teuchos::RCP<Hierarchy>& hierarchy)
      : hierarchy_(hierarchy) {}

    void Apply(const MultiVector& x, MultiVector& y, Belos::ETrans trans = Belos::NOTRANS) const override {
      TEUCHOS_TEST_FOR_EXCEPTION(trans != Belos::NOTRANS, std::runtime_error,
                                 "MueLu GPTune Belos hierarchy operator does not support transpose apply");
      y.putScalar(Teuchos::ScalarTraits<SC>::zero());
      hierarchy_->Iterate(x, y, 1, true);
    }

   private:
    Teuchos::RCP<Hierarchy> hierarchy_;
  };
#endif

  void RunSolve(Hierarchy& H, EvaluationResult& result) const {
#ifdef HAVE_MUELU_BELOS
    TEUCHOS_TEST_FOR_EXCEPTION(ToLower(spec_.evaluation.objective_aggregation) != "mean",
                               std::runtime_error,
                               "MueLu GPTune C++ evaluator currently supports only mean objective aggregation");

    using BelosOperator = Belos::OperatorT<MultiVector>;
    const auto one      = Teuchos::ScalarTraits<SC>::one();
    const auto zero     = Teuchos::ScalarTraits<SC>::zero();

    Teuchos::RCP<MultiVector> xExact =
        Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(op_->getDomainMap(), 1);
    Teuchos::RCP<MultiVector> rhs =
        Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(op_->getRangeMap(), 1);
    xExact->putScalar(one);
    rhs->putScalar(zero);
    op_->apply(*xExact, *rhs, Teuchos::NO_TRANS, one, zero);

    H.IsPreconditioner(true);
    Teuchos::RCP<BelosOperator> belosOp =
        Teuchos::rcp(new Belos::XpetraOp<SC, LO, GO, NO>(op_));
    Teuchos::RCP<BelosOperator> belosPrec =
        Teuchos::rcp(new HierarchyBelosOperator(Teuchos::rcpFromRef(H)));

    const int reruns = std::max(1, spec_.evaluation.reruns);
    double solveTime = 0.0;
    int iterationCount = 0;
    bool converged = true;

    auto runOneSolve = [&]() {
      Teuchos::RCP<MultiVector> x =
          Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(op_->getDomainMap(), 1);
      x->putScalar(zero);

      Teuchos::RCP<Belos::LinearProblem<SC, MultiVector, BelosOperator>> problem =
          Teuchos::rcp(new Belos::LinearProblem<SC, MultiVector, BelosOperator>(belosOp, x, rhs));
      if (spec_.evaluation.solve_preconditioned) {
        problem->setRightPrec(belosPrec);
      }

      const bool set = problem->setProblem();
      TEUCHOS_TEST_FOR_EXCEPTION(!set, std::runtime_error,
                                 "Belos failed to set up the MueLu GPTune evaluation problem");

      Teuchos::ParameterList belosList;
      belosList.set("Maximum Iterations", spec_.evaluation.max_iterations);
      belosList.set("Convergence Tolerance", spec_.evaluation.tolerance);

      Belos::SolverFactory<SC, MultiVector, BelosOperator> factory;
      Teuchos::RCP<Belos::SolverManager<SC, MultiVector, BelosOperator>> solver =
          factory.create(BelosSolverName(spec_.evaluation.solver), Teuchos::rcp(&belosList, false));
      solver->setProblem(problem);

      Teuchos::Time timer("MueLu GPTune candidate solve");
      timer.start();
      const Belos::ReturnType ret = solver->solve();
      timer.stop();

      return std::make_pair(ret, std::make_pair(timer.totalElapsedTime(), solver->getNumIters()));
    };

    if (spec_.evaluation.warmup_solve) {
      runOneSolve();
    }

    for (int i = 0; i < reruns; ++i) {
      const auto solve = runOneSolve();
      solveTime += solve.second.first;
      iterationCount += solve.second.second;
      converged = converged && (solve.first == Belos::Converged);
    }

    result.metrics.solveTime = solveTime / static_cast<double>(reruns);
    result.metrics.iterationCount = iterationCount / reruns;
    result.metrics.converged = converged;
    if (!converged) {
      result.message = "Belos solve did not converge for MueLu GPTune candidate evaluation";
    }
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "MueLu GPTune solve-time evaluation requires MueLu_ENABLE_Belos=ON and Belos_ENABLE_Xpetra=ON");
#endif
  }

  Teuchos::RCP<Matrix> op_;
  Teuchos::ParameterList baseParams_;
  Teuchos::ParameterList fullParams_;
  ParsedSpec spec_;

  std::vector<CandidateEvaluationRecord> history_;
  bool hasBest_ = false;
  double bestObjective_ = std::numeric_limits<double>::max();
  Candidate bestCandidate_;
  Teuchos::ParameterList bestParams_;
  Teuchos::RCP<Hierarchy> bestHierarchy_;
  std::string pendingCandidateId_;
  int pendingEvaluationIndex_ = -1;
};

}  // namespace GPTuneInt
}  // namespace MueLu

#endif  // MUELU_TUNINGEVALUATOR_HPP
