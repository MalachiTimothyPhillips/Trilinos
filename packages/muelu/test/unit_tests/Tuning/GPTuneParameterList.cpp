// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_any.hpp>

#include <Tpetra_Map.hpp>

#include "MueLu_GPTuneBridge.hpp"
#include "MueLu_GPTuneParameterList.hpp"

namespace MueLuTests {

TEUCHOS_UNIT_TEST(GPTuneParameterList, ParseMinimalValidSpec) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("enabled", true);
  gt.set("objective", "setup time");
  auto& search = gt.sublist("search space");
  auto& sweeps = search.sublist("relaxation: sweeps");
  sweeps.set("type", "integer");
  sweeps.set("min", 1);
  sweeps.set("max", 4);

  const auto spec = MueLu::GPTuneInt::ParseAndValidateGPTuneList(params);

  TEST_ASSERT(spec.enabled);
  TEST_EQUALITY(spec.search_parameters.size(), 1);
  TEST_EQUALITY(spec.search_parameters[0].muelu_key, "relaxation: sweeps");
  TEST_EQUALITY(static_cast<int>(spec.search_parameters[0].type), static_cast<int>(MueLu::GPTuneInt::ParamType::Integer));
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, ValidBackendsParse) {
  const Teuchos::Array<std::string> backends{"external-driver", "python", "embedded"};

  for (const auto& backend : backends) {
    Teuchos::ParameterList params;
    auto& gt = params.sublist("gptune");
    gt.set("enabled", true);
    gt.set("backend", backend);
    auto& search = gt.sublist("search space");
    auto& sweeps = search.sublist("relaxation: sweeps");
    sweeps.set("type", "integer");
    sweeps.set("min", 1);
    sweeps.set("max", 4);

    const auto spec = MueLu::GPTuneInt::ParseAndValidateGPTuneList(params);
    TEST_ASSERT(spec.enabled);
  }
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, InvalidModeThrows) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("enabled", true);
  gt.set("mode", "bad-mode");
  auto& search = gt.sublist("search space");
  auto& sweeps = search.sublist("relaxation: sweeps");
  sweeps.set("type", "integer");
  sweeps.set("min", 1);
  sweeps.set("max", 4);

  TEST_THROW(MueLu::GPTuneInt::ParseAndValidateGPTuneList(params), std::runtime_error);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, InvalidBackendThrows) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("enabled", true);
  gt.set("backend", "not-a-backend");
  auto& search = gt.sublist("search space");
  auto& sweeps = search.sublist("relaxation: sweeps");
  sweeps.set("type", "integer");
  sweeps.set("min", 1);
  sweeps.set("max", 4);

  TEST_THROW(MueLu::GPTuneInt::ParseAndValidateGPTuneList(params), std::runtime_error);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, InvalidBoundsThrow) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("enabled", true);
  auto& search = gt.sublist("search space");
  auto& dropTol = search.sublist("aggregation: drop tol");
  dropTol.set("type", "real");
  dropTol.set("min", 1.0e-2);
  dropTol.set("max", 1.0e-4);

  TEST_THROW(MueLu::GPTuneInt::ParseAndValidateGPTuneList(params), std::runtime_error);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, InvalidActiveIfThrows) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("enabled", true);
  auto& search = gt.sublist("search space");
  auto& smootherType = search.sublist("smoother: type");
  smootherType.set("type", "categorical");
  smootherType.set("values", Teuchos::Array<std::string>{"RELAXATION", "CHEBYSHEV"});

  auto& degree = search.sublist("chebyshev: degree");
  degree.set("type", "integer");
  degree.set("min", 2);
  degree.set("max", 6);
  degree.sublist("active if").set("not-a-tuned-key", "CHEBYSHEV");

  TEST_THROW(MueLu::GPTuneInt::ParseAndValidateGPTuneList(params), std::runtime_error);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, LogTransformRequiresPositiveBounds) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("enabled", true);
  auto& search = gt.sublist("search space");
  auto& dropTol = search.sublist("aggregation: drop tol");
  dropTol.set("type", "real");
  dropTol.set("transform", "log");
  dropTol.set("min", 0.0);
  dropTol.set("max", 1.0e-2);

  TEST_THROW(MueLu::GPTuneInt::ParseAndValidateGPTuneList(params), std::runtime_error);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, ObjectiveRequiresSolveMeasurement) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("enabled", true);
  gt.set("objective", "solve time");
  auto& evaluation = gt.sublist("evaluation");
  evaluation.set("measure solve", false);
  auto& search = gt.sublist("search space");
  auto& sweeps = search.sublist("relaxation: sweeps");
  sweeps.set("type", "integer");
  sweeps.set("min", 1);
  sweeps.set("max", 4);

  TEST_THROW(MueLu::GPTuneInt::ParseAndValidateGPTuneList(params), std::runtime_error);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, AmortizedSolveTimeRequiresSetupAndSolveMeasurement) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("enabled", true);
  gt.set("objective", "amortized solve time");
  auto& evaluation = gt.sublist("evaluation");
  evaluation.set("measure setup", true);
  evaluation.set("measure solve", false);
  auto& search = gt.sublist("search space");
  auto& sweeps = search.sublist("relaxation: sweeps");
  sweeps.set("type", "integer");
  sweeps.set("min", 1);
  sweeps.set("max", 4);

  TEST_THROW(MueLu::GPTuneInt::ParseAndValidateGPTuneList(params), std::runtime_error);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, TaskParametersParseTypedValues) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("enabled", true);
  auto& task = gt.sublist("task parameters");
  task.set("matrix family", std::string("laplace1d"));
  task.set("rows", 128);
  task.set("drop tolerance", 1.0e-3);
  task.set("source", std::string("auto"));
  auto& search = gt.sublist("search space");
  auto& sweeps = search.sublist("relaxation: sweeps");
  sweeps.set("type", "integer");
  sweeps.set("min", 1);
  sweeps.set("max", 4);

  const auto spec = MueLu::GPTuneInt::ParseAndValidateGPTuneList(params);

  TEST_ASSERT(spec.task_parameters.at("matrix family").kind == MueLu::GPTuneInt::TaskParameterValue::Kind::String);
  TEST_ASSERT(spec.task_parameters.at("rows").kind == MueLu::GPTuneInt::TaskParameterValue::Kind::Int);
  TEST_ASSERT(spec.task_parameters.at("drop tolerance").kind == MueLu::GPTuneInt::TaskParameterValue::Kind::Double);
  TEST_ASSERT(spec.task_parameters.at("source").kind == MueLu::GPTuneInt::TaskParameterValue::Kind::Auto);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, IsActiveHonorsActiveIfClauses) {
  MueLu::GPTuneInt::SearchParameter degree;
  degree.muelu_key = "chebyshev: degree";
  degree.active_if.push_back({"smoother: type", "CHEBYSHEV"});

  MueLu::GPTuneInt::Candidate activeCandidate;
  activeCandidate["smoother: type"] = std::string("CHEBYSHEV");
  TEST_ASSERT(MueLu::GPTuneInt::IsActive(degree, activeCandidate));

  MueLu::GPTuneInt::Candidate inactiveCandidate;
  inactiveCandidate["smoother: type"] = std::string("RELAXATION");
  TEST_ASSERT(!MueLu::GPTuneInt::IsActive(degree, inactiveCandidate));

  MueLu::GPTuneInt::Candidate missingCandidate;
  TEST_ASSERT(!MueLu::GPTuneInt::IsActive(degree, missingCandidate));
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, StripGPTuneSublistRemovesOnlyGPTune) {
  Teuchos::ParameterList params;
  params.set("max levels", 3);
  params.sublist("gptune").set("enabled", true);

  const auto stripped = MueLu::GPTuneInt::StripGPTuneSublist(params);

  TEST_ASSERT(!stripped.isSublist("gptune"));
  TEST_EQUALITY(stripped.get<int>("max levels"), 3);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, ApplyCandidateWritesValues) {
  Teuchos::ParameterList params;
  params.set("max levels", 3);

  MueLu::GPTuneInt::Candidate candidate;
  candidate["max levels"] = 5;
  candidate["aggregation: drop tol"] = 1.0e-3;
  candidate["smoother: type"] = std::string("RELAXATION");

  MueLu::GPTuneInt::ApplyCandidateToParameterList(candidate, params);

  TEST_EQUALITY(params.get<int>("max levels"), 5);
  TEST_FLOATING_EQUALITY(params.get<double>("aggregation: drop tol"), 1.0e-3, 1.0e-15);
  TEST_EQUALITY(params.get<std::string>("smoother: type"), "RELAXATION");
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, WriteResultsLeavesTopLevelUntouchedWhenDisabled) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("set best parameters on output", false);
  params.set("max levels", 3);

  MueLu::GPTuneInt::Candidate candidate;
  candidate["max levels"] = 7;

  MueLu::GPTuneInt::WriteResultsToParameterList(params, candidate, 12.5, 4, "success", "placeholder");

  const auto& results = params.sublist("gptune").sublist("results");
  TEST_EQUALITY(results.get<std::string>("status"), "success");
  TEST_EQUALITY(results.sublist("best parameters").get<int>("max levels"), 7);
  TEST_EQUALITY(params.get<int>("max levels"), 3);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, WriteResultsPopulatesResultsSublist) {
  Teuchos::ParameterList params;
  auto& gt = params.sublist("gptune");
  gt.set("set best parameters on output", true);

  MueLu::GPTuneInt::Candidate candidate;
  candidate["max levels"] = 7;
  candidate["smoother: type"] = std::string("CHEBYSHEV");

  MueLu::GPTuneInt::WriteResultsToParameterList(params, candidate, 12.5, 4, "success", "placeholder");

  const auto& results = params.sublist("gptune").sublist("results");
  TEST_EQUALITY(results.get<std::string>("status"), "success");
  TEST_EQUALITY(results.get<std::string>("message"), "placeholder");
  TEST_FLOATING_EQUALITY(results.get<double>("best objective"), 12.5, 1.0e-15);
  TEST_EQUALITY(results.get<int>("evaluations performed"), 4);
  TEST_EQUALITY(results.sublist("best parameters").get<int>("max levels"), 7);
  TEST_EQUALITY(results.sublist("best parameters").get<std::string>("smoother: type"), "CHEBYSHEV");
  TEST_EQUALITY(params.get<int>("max levels"), 7);
  TEST_EQUALITY(params.get<std::string>("smoother: type"), "CHEBYSHEV");
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, PlaceholderBridgeUsesInitialValueBeforeBaseValue) {
  Teuchos::ParameterList baseParams;
  baseParams.set("relaxation: sweeps", 2);

  MueLu::GPTuneInt::ParsedSpec spec;
  spec.enabled = true;

  MueLu::GPTuneInt::SearchParameter sweeps;
  sweeps.muelu_key = "relaxation: sweeps";
  sweeps.type = MueLu::GPTuneInt::ParamType::Integer;
  sweeps.has_int_bounds = true;
  sweeps.int_min = 1;
  sweeps.int_max = 4;
  sweeps.has_initial_value = true;
  sweeps.initial_value = 4;
  spec.search_parameters.push_back(sweeps);

  using Node = Tpetra::Map<>::node_type;
  using GO = Tpetra::Map<>::global_ordinal_type;

  const auto result = MueLu::GPTuneInt::RunGPTuneBridge<double, int, GO, Node>(Teuchos::null, baseParams, baseParams, spec);

  TEST_EQUALITY(result.status, "success");
  TEST_EQUALITY(Teuchos::any_cast<int>(result.bestCandidate.at("relaxation: sweeps")), 4);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, PlaceholderBridgeUsesBaseValueWhenNoInitialValue) {
  Teuchos::ParameterList baseParams;
  baseParams.set("relaxation: sweeps", 3);

  MueLu::GPTuneInt::ParsedSpec spec;
  spec.enabled = true;

  MueLu::GPTuneInt::SearchParameter sweeps;
  sweeps.muelu_key = "relaxation: sweeps";
  sweeps.type = MueLu::GPTuneInt::ParamType::Integer;
  sweeps.has_int_bounds = true;
  sweeps.int_min = 1;
  sweeps.int_max = 4;
  spec.search_parameters.push_back(sweeps);

  using Node = Tpetra::Map<>::node_type;
  using GO = Tpetra::Map<>::global_ordinal_type;

  const auto result = MueLu::GPTuneInt::RunGPTuneBridge<double, int, GO, Node>(Teuchos::null, baseParams, baseParams, spec);

  TEST_EQUALITY(Teuchos::any_cast<int>(result.bestCandidate.at("relaxation: sweeps")), 3);
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, PlaceholderBridgeFallsBackToCategoricalFirstValue) {
  Teuchos::ParameterList baseParams;

  MueLu::GPTuneInt::ParsedSpec spec;
  spec.enabled = true;

  MueLu::GPTuneInt::SearchParameter smoother;
  smoother.muelu_key = "smoother: type";
  smoother.type = MueLu::GPTuneInt::ParamType::Categorical;
  smoother.categorical_values = {"RELAXATION", "CHEBYSHEV"};
  spec.search_parameters.push_back(smoother);

  using Node = Tpetra::Map<>::node_type;
  using GO = Tpetra::Map<>::global_ordinal_type;

  const auto result = MueLu::GPTuneInt::RunGPTuneBridge<double, int, GO, Node>(Teuchos::null, baseParams, baseParams, spec);

  TEST_EQUALITY(Teuchos::any_cast<std::string>(result.bestCandidate.at("smoother: type")), "RELAXATION");
}

TEUCHOS_UNIT_TEST(GPTuneParameterList, PlaceholderBridgeFallsBackToNumericMins) {
  Teuchos::ParameterList baseParams;

  MueLu::GPTuneInt::ParsedSpec spec;
  spec.enabled = true;

  MueLu::GPTuneInt::SearchParameter sweeps;
  sweeps.muelu_key = "relaxation: sweeps";
  sweeps.type = MueLu::GPTuneInt::ParamType::Integer;
  sweeps.has_int_bounds = true;
  sweeps.int_min = 2;
  sweeps.int_max = 5;
  spec.search_parameters.push_back(sweeps);

  MueLu::GPTuneInt::SearchParameter dropTol;
  dropTol.muelu_key = "aggregation: drop tol";
  dropTol.type = MueLu::GPTuneInt::ParamType::Real;
  dropTol.has_real_bounds = true;
  dropTol.real_min = 1.0e-4;
  dropTol.real_max = 1.0e-2;
  spec.search_parameters.push_back(dropTol);

  using Node = Tpetra::Map<>::node_type;
  using GO = Tpetra::Map<>::global_ordinal_type;

  const auto result = MueLu::GPTuneInt::RunGPTuneBridge<double, int, GO, Node>(Teuchos::null, baseParams, baseParams, spec);

  TEST_EQUALITY(Teuchos::any_cast<int>(result.bestCandidate.at("relaxation: sweeps")), 2);
  TEST_FLOATING_EQUALITY(Teuchos::any_cast<double>(result.bestCandidate.at("aggregation: drop tol")), 1.0e-4, 1.0e-15);
}

}  // namespace MueLuTests
