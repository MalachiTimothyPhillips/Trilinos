// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_GPTUNEPARAMETERLIST_HPP
#define MUELU_GPTUNEPARAMETERLIST_HPP

#include <map>
#include <string>
#include <vector>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_any.hpp>

namespace MueLu {
namespace GPTuneInt {

enum class Mode {
  TuneAndBuild,
  TuneOnly
};

enum class Backend {
  ExternalDriver,
  Python,
  Embedded
};

enum class Objective {
  SetupTime,
  SolveTime,
  TotalTime,
  IterationCount,
  AmortizedSolveTime
};

enum class ParamType {
  Categorical,
  Integer,
  Real
};

enum class Transform {
  None,
  Log
};

struct ActiveIfClause {
  std::string parameter_name;
  std::string required_value;
};

struct SearchParameter {
  std::string muelu_key;
  ParamType type;
  Transform transform = Transform::None;

  std::vector<std::string> categorical_values;

  bool has_int_bounds = false;
  int int_min         = 0;
  int int_max         = 0;
  int int_step        = 1;

  bool has_real_bounds = false;
  double real_min      = 0.0;
  double real_max      = 0.0;
  double real_step     = 0.0;

  bool has_initial_value = false;
  Teuchos::any initial_value;

  std::vector<ActiveIfClause> active_if;
  std::vector<std::string> depends_on;

  bool required = false;
  std::string description;
};

struct TaskParameterValue {
  enum class Kind {
    String,
    Int,
    Double,
    Auto
  };

  Kind kind = Kind::String;
  std::string string_value;
  int int_value       = 0;
  double double_value = 0.0;
};

struct EvaluationSpec {
  std::string driver    = "belos";
  std::string solver    = "cg";
  double tolerance      = 1e-10;
  int max_iterations    = 200;

  bool measure_setup = true;
  bool measure_solve = true;
  bool enforce_boundary_conditions_on_initial_guess = true;
  bool scaled_residual_history = true;
  bool solve_preconditioned = true;

  bool warmup_setup = false;
  bool warmup_solve = false;

  int reruns = 1;
  std::string objective_aggregation = "mean";
  bool fail_on_nonconvergence = true;
};

struct ParsedSpec {
  bool enabled = false;

  Mode mode = Mode::TuneAndBuild;
  Backend backend = Backend::ExternalDriver;
  Objective objective = Objective::TotalTime;

  int max_evaluations = 20;
  bool reuse_history = true;
  int seed = 0;
  bool verbose = false;
  bool set_best_parameters_on_output = false;
  bool allow_override_of_fixed_params = false;

  int reuse_count = 1;
  double failure_penalty = 1.0e100;
  std::string time_metric = "teuchos timer";
  std::string history_db_path;
  std::string history_key_policy = "auto";

  std::map<std::string, TaskParameterValue> task_parameters;
  EvaluationSpec evaluation;
  std::vector<SearchParameter> search_parameters;
};

using Candidate = std::map<std::string, Teuchos::any>;

ParsedSpec ParseAndValidateGPTuneList(const Teuchos::ParameterList& fullList);

Teuchos::ParameterList StripGPTuneSublist(const Teuchos::ParameterList& fullList);

void ApplyCandidateToParameterList(const Candidate& candidate,
                                   Teuchos::ParameterList& params);

bool IsActive(const SearchParameter& param,
              const Candidate& candidate);

void WriteResultsToParameterList(Teuchos::ParameterList& fullList,
                                 const Candidate& best,
                                 double bestObjective,
                                 int evaluationsPerformed,
                                 const std::string& status,
                                 const std::string& message);

}  // namespace GPTuneInt
}  // namespace MueLu

#endif  // MUELU_GPTUNEPARAMETERLIST_HPP
