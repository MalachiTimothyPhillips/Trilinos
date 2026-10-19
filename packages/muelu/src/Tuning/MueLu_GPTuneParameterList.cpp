// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "MueLu_GPTuneParameterList.hpp"
#include "MueLu_TuningTypes.hpp"

#include <set>
#include <sstream>
#include <stdexcept>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_TestForException.hpp>

#include "MueLu_MasterList.hpp"

namespace MueLu {
namespace GPTuneInt {

namespace {

Mode parseMode(const std::string& s) {
  if (s == "tune-and-build") return Mode::TuneAndBuild;
  if (s == "tune-only") return Mode::TuneOnly;
  throw std::runtime_error("Invalid gptune.mode: '" + s + "'");
}

Backend parseBackend(const std::string& s) {
  if (s == "fixed") return Backend::Fixed;
  if (s == "external-driver") return Backend::ExternalDriver;
  if (s == "python") return Backend::Python;
  if (s == "embedded") return Backend::Embedded;
  throw std::runtime_error("Invalid gptune.backend: '" + s + "'");
}

Objective parseObjective(const std::string& s) {
  if (s == "setup time") return Objective::SetupTime;
  if (s == "solve time") return Objective::SolveTime;
  if (s == "total time") return Objective::TotalTime;
  if (s == "iteration count") return Objective::IterationCount;
  if (s == "amortized solve time") return Objective::AmortizedSolveTime;
  throw std::runtime_error("Invalid gptune.objective: '" + s + "'");
}

ParamType parseParamType(const std::string& s) {
  if (s == "categorical") return ParamType::Categorical;
  if (s == "integer") return ParamType::Integer;
  if (s == "real") return ParamType::Real;
  throw std::runtime_error("Invalid gptune search-space type: '" + s + "'");
}

Transform parseTransform(const std::string& s) {
  if (s == "none") return Transform::None;
  if (s == "log") return Transform::Log;
  throw std::runtime_error("Invalid gptune search-space transform: '" + s + "'");
}

TaskParameterValue parseTaskValue(const Teuchos::ParameterEntry& entry) {
  TaskParameterValue out;

  if (entry.isType<std::string>()) {
    const std::string s = Teuchos::getValue<std::string>(entry);
    if (s == "auto" || s == "operator" || s == "comm" || s == "parameterlist") {
      out.kind = TaskParameterValue::Kind::Auto;
      out.string_value = s;
    } else {
      out.kind = TaskParameterValue::Kind::String;
      out.string_value = s;
    }
  } else if (entry.isType<int>()) {
    out.kind = TaskParameterValue::Kind::Int;
    out.int_value = Teuchos::getValue<int>(entry);
  } else if (entry.isType<double>()) {
    out.kind = TaskParameterValue::Kind::Double;
    out.double_value = Teuchos::getValue<double>(entry);
  } else {
    throw std::runtime_error("Unsupported gptune task-parameter value type");
  }

  return out;
}

SearchParameter parseSearchParameter(const std::string& key,
                                     const Teuchos::ParameterList& pl) {
  SearchParameter p;
  p.muelu_key = key;
  p.type = parseParamType(pl.get<std::string>("type"));

  if (pl.isParameter("transform")) {
    p.transform = parseTransform(pl.get<std::string>("transform"));
  }

  auto getIntOrDefault = [&pl](const std::string& name, int value) {
    return pl.isParameter(name) ? pl.get<int>(name) : value;
  };

  auto getDoubleOrDefault = [&pl](const std::string& name, double value) {
    return pl.isParameter(name) ? pl.get<double>(name) : value;
  };

  auto getBoolOrDefault = [&pl](const std::string& name, bool value) {
    return pl.isParameter(name) ? pl.get<bool>(name) : value;
  };

  auto getStringOrDefault = [&pl](const std::string& name, const std::string& value) {
    return pl.isParameter(name) ? pl.get<std::string>(name) : value;
  };

  if (p.type == ParamType::Categorical) {
    TEUCHOS_TEST_FOR_EXCEPTION(!pl.isParameter("values"), std::runtime_error,
                               "Categorical parameter '" + key + "' is missing 'values'");

    const auto& valuesEntry = pl.getEntry("values");
    if (valuesEntry.isType<Teuchos::Array<std::string>>()) {
      auto vals = pl.get<Teuchos::Array<std::string>>("values");
      p.categorical_values.assign(vals.begin(), vals.end());
    } else if (valuesEntry.isType<Teuchos::Array<int>>()) {
      auto vals = pl.get<Teuchos::Array<int>>("values");
      for (const auto& v : vals) p.categorical_values.push_back(std::to_string(v));
    } else if (valuesEntry.isType<Teuchos::Array<double>>()) {
      auto vals = pl.get<Teuchos::Array<double>>("values");
      for (const auto& v : vals) {
        std::ostringstream os;
        os << v;
        p.categorical_values.push_back(os.str());
      }
    } else {
      throw std::runtime_error("Categorical parameter '" + key + "' has unsupported values type");
    }

    TEUCHOS_TEST_FOR_EXCEPTION(p.categorical_values.empty(), std::runtime_error,
                               "Categorical parameter '" + key + "' has empty 'values'");
  } else if (p.type == ParamType::Integer) {
    p.has_int_bounds = true;
    p.int_min = pl.get<int>("min");
    p.int_max = pl.get<int>("max");
    p.int_step = getIntOrDefault("step", 1);

    TEUCHOS_TEST_FOR_EXCEPTION(p.int_min > p.int_max, std::runtime_error,
                               "Integer parameter '" + key + "' has min > max");
    TEUCHOS_TEST_FOR_EXCEPTION(p.int_step < 1, std::runtime_error,
                               "Integer parameter '" + key + "' must have step >= 1");
  } else {
    p.has_real_bounds = true;
    p.real_min = pl.get<double>("min");
    p.real_max = pl.get<double>("max");
    p.real_step = getDoubleOrDefault("step", 0.0);

    TEUCHOS_TEST_FOR_EXCEPTION(p.real_min > p.real_max, std::runtime_error,
                               "Real parameter '" + key + "' has min > max");
    if (p.transform == Transform::Log) {
      TEUCHOS_TEST_FOR_EXCEPTION(p.real_min <= 0.0 || p.real_max <= 0.0, std::runtime_error,
                                 "Log-transformed parameter '" + key + "' must have positive bounds");
    }
  }

  if (pl.isSublist("active if")) {
    const auto& active = pl.sublist("active if");
    for (auto it = active.begin(); it != active.end(); ++it) {
      ActiveIfClause c;
      c.parameter_name = active.name(it);

      const auto& entry = active.entry(it);
      if (entry.isType<std::string>()) {
        c.required_value = Teuchos::getValue<std::string>(entry);
      } else if (entry.isType<int>()) {
        c.required_value = std::to_string(Teuchos::getValue<int>(entry));
      } else if (entry.isType<double>()) {
        std::ostringstream os;
        os << Teuchos::getValue<double>(entry);
        c.required_value = os.str();
      } else if (entry.isType<bool>()) {
        c.required_value = Teuchos::getValue<bool>(entry) ? "true" : "false";
      } else {
        throw std::runtime_error("Unsupported active-if value type for '" + key + "'");
      }

      p.active_if.push_back(c);
    }
  }

  if (pl.isParameter("depends on")) {
    auto deps = pl.get<Teuchos::Array<std::string>>("depends on");
    p.depends_on.assign(deps.begin(), deps.end());
  }

  p.required = getBoolOrDefault("required", false);
  p.description = getStringOrDefault("description", std::string(""));

  if (pl.isParameter("initial value")) {
    p.has_initial_value = true;
    const auto& entry = pl.getEntry("initial value");
    if (entry.isType<std::string>()) {
      p.initial_value = pl.get<std::string>("initial value");
    } else if (entry.isType<int>()) {
      p.initial_value = pl.get<int>("initial value");
    } else if (entry.isType<double>()) {
      p.initial_value = pl.get<double>("initial value");
    } else if (entry.isType<bool>()) {
      p.initial_value = pl.get<bool>("initial value");
    } else {
      throw std::runtime_error("Unsupported initial value type for '" + key + "'");
    }
  }

  return p;
}

std::string anyToString(const Teuchos::any& value) {
  if (value.type() == typeid(std::string))
    return Teuchos::any_cast<std::string>(value);
  if (value.type() == typeid(int))
    return std::to_string(Teuchos::any_cast<int>(value));
  if (value.type() == typeid(double)) {
    std::ostringstream os;
    os << Teuchos::any_cast<double>(value);
    return os.str();
  }
  if (value.type() == typeid(bool))
    return Teuchos::any_cast<bool>(value) ? "true" : "false";
  throw std::runtime_error("Unsupported candidate value type");
}

std::string leafParameterName(const std::string& key) {
  const std::string delimiter = "||";
  const auto pos = key.rfind(delimiter);
  if (pos == std::string::npos) return key;
  return key.substr(pos + delimiter.size());
}

bool isKnownMueLuKey(const std::string& key) {
  static const std::set<std::string> supportedTunableKeys = {
      "smoother: type",
      "relaxation: type",
      "relaxation: sweeps",
      "relaxation: damping factor",
      "chebyshev: degree",
      "chebyshev: ratio eigenvalue",
      "aggregation: drop tol",
      "coarse: max size",
      "max levels"};

  const auto leafKey = leafParameterName(key);
  if (supportedTunableKeys.find(leafKey) != supportedTunableKeys.end()) return true;

  try {
    (void)MueLu::MasterList::getEntry(leafKey);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

ParsedSpec ParseAndValidateGPTuneList(const Teuchos::ParameterList& fullList) {
  TEUCHOS_TEST_FOR_EXCEPTION(!fullList.isSublist("gptune"), std::runtime_error,
                             "Missing required sublist 'gptune'");

  const auto& gt = fullList.sublist("gptune");

  ParsedSpec spec;

  auto getIntOrDefault = [&gt](const std::string& name, int value) {
    return gt.isParameter(name) ? gt.get<int>(name) : value;
  };

  auto getDoubleOrDefault = [&gt](const std::string& name, double value) {
    return gt.isParameter(name) ? gt.get<double>(name) : value;
  };

  auto getBoolOrDefault = [&gt](const std::string& name, bool value) {
    return gt.isParameter(name) ? gt.get<bool>(name) : value;
  };

  auto getStringOrDefault = [&gt](const std::string& name, const std::string& value) {
    return gt.isParameter(name) ? gt.get<std::string>(name) : value;
  };

  spec.enabled = getBoolOrDefault("enabled", false);
  if (!spec.enabled) return spec;

  spec.mode = parseMode(getStringOrDefault("mode", std::string("tune-and-build")));
  spec.backend = parseBackend(getStringOrDefault("backend", std::string("external-driver")));
  spec.objective = parseObjective(getStringOrDefault("objective", std::string("total time")));

  spec.max_evaluations = getIntOrDefault("max evaluations", 20);
  spec.reuse_history = getBoolOrDefault("reuse history", true);
  spec.seed = getIntOrDefault("seed", 0);
  spec.verbose = getBoolOrDefault("verbose", false);
  spec.set_best_parameters_on_output = getBoolOrDefault("set best parameters on output", false);
  spec.allow_override_of_fixed_params = getBoolOrDefault("allow override of fixed params", false);

  spec.reuse_count = getIntOrDefault("reuse count", 1);
  spec.failure_penalty = getDoubleOrDefault("failure penalty", 1.0e100);
  spec.time_metric = getStringOrDefault("time metric", std::string("teuchos timer"));
  spec.history_db_path = getStringOrDefault("history db path", std::string(""));
  spec.history_key_policy = getStringOrDefault("history key policy", std::string("auto"));

  TEUCHOS_TEST_FOR_EXCEPTION(spec.max_evaluations < 1, std::runtime_error,
                             "gptune.max evaluations must be >= 1");
  TEUCHOS_TEST_FOR_EXCEPTION(spec.reuse_count < 1, std::runtime_error,
                             "gptune.reuse count must be >= 1");
  TEUCHOS_TEST_FOR_EXCEPTION(spec.failure_penalty <= 0.0, std::runtime_error,
                             "gptune.failure penalty must be > 0");

  if (gt.isSublist("task parameters")) {
    const auto& tp = gt.sublist("task parameters");
    for (auto it = tp.begin(); it != tp.end(); ++it) {
      spec.task_parameters[tp.name(it)] = parseTaskValue(tp.entry(it));
    }
  }

  if (gt.isSublist("evaluation")) {
    const auto& ev = gt.sublist("evaluation");

    auto getEvalIntOrDefault = [&ev](const std::string& name, int value) {
      return ev.isParameter(name) ? ev.get<int>(name) : value;
    };

    auto getEvalDoubleOrDefault = [&ev](const std::string& name, double value) {
      return ev.isParameter(name) ? ev.get<double>(name) : value;
    };

    auto getEvalBoolOrDefault = [&ev](const std::string& name, bool value) {
      return ev.isParameter(name) ? ev.get<bool>(name) : value;
    };

    auto getEvalStringOrDefault = [&ev](const std::string& name, const std::string& value) {
      return ev.isParameter(name) ? ev.get<std::string>(name) : value;
    };

    spec.evaluation.driver = getEvalStringOrDefault("driver", std::string("belos"));
    spec.evaluation.solver = getEvalStringOrDefault("solver", std::string("cg"));
    spec.evaluation.tolerance = getEvalDoubleOrDefault("tolerance", 1e-10);
    spec.evaluation.max_iterations = getEvalIntOrDefault("max iterations", 200);
    spec.evaluation.measure_setup = getEvalBoolOrDefault("measure setup", true);
    spec.evaluation.measure_solve = getEvalBoolOrDefault("measure solve", true);
    spec.evaluation.enforce_boundary_conditions_on_initial_guess =
        getEvalBoolOrDefault("enforce boundary conditions on initial guess", true);
    spec.evaluation.scaled_residual_history =
        getEvalBoolOrDefault("scaled residual history", true);
    spec.evaluation.solve_preconditioned =
        getEvalBoolOrDefault("solve preconditioned", true);
    spec.evaluation.warmup_setup = getEvalBoolOrDefault("warmup setup", false);
    spec.evaluation.warmup_solve = getEvalBoolOrDefault("warmup solve", false);
    spec.evaluation.reruns = getEvalIntOrDefault("reruns", 1);
    spec.evaluation.objective_aggregation =
        getEvalStringOrDefault("objective aggregation", std::string("mean"));
    spec.evaluation.fail_on_nonconvergence =
        getEvalBoolOrDefault("fail on nonconvergence", true);

    TEUCHOS_TEST_FOR_EXCEPTION(spec.evaluation.tolerance <= 0.0, std::runtime_error,
                               "gptune.evaluation.tolerance must be > 0");
    TEUCHOS_TEST_FOR_EXCEPTION(spec.evaluation.max_iterations < 1, std::runtime_error,
                               "gptune.evaluation.max iterations must be >= 1");
    TEUCHOS_TEST_FOR_EXCEPTION(spec.evaluation.reruns < 1, std::runtime_error,
                               "gptune.evaluation.reruns must be >= 1");
  }

  if (gt.isSublist("worker")) {
    const auto& worker = gt.sublist("worker");
    if (worker.isParameter("argv")) {
      const auto argv = worker.get<Teuchos::Array<std::string>>("argv");
      spec.worker.argv.assign(argv.begin(), argv.end());
    }
    if (worker.isParameter("startup timeout seconds")) {
      spec.worker.startup_timeout_seconds = worker.get<int>("startup timeout seconds");
    }
    if (worker.isParameter("request timeout seconds")) {
      spec.worker.request_timeout_seconds = worker.get<int>("request timeout seconds");
    }

    TEUCHOS_TEST_FOR_EXCEPTION(spec.worker.startup_timeout_seconds < 1, std::runtime_error,
                               "gptune.worker.startup timeout seconds must be >= 1");
    TEUCHOS_TEST_FOR_EXCEPTION(spec.worker.request_timeout_seconds < 1, std::runtime_error,
                               "gptune.worker.request timeout seconds must be >= 1");
  }

  TEUCHOS_TEST_FOR_EXCEPTION(!gt.isSublist("search space"), std::runtime_error,
                             "gptune requires sublist 'search space'");

  const auto& ss = gt.sublist("search space");
  std::set<std::string> seen;

  for (auto it = ss.begin(); it != ss.end(); ++it) {
    const std::string key = ss.name(it);

    TEUCHOS_TEST_FOR_EXCEPTION(!ss.isSublist(key), std::runtime_error,
                               "Each gptune.search space entry must be a sublist: '" + key + "'");
    TEUCHOS_TEST_FOR_EXCEPTION(!seen.insert(key).second, std::runtime_error,
                               "Duplicate search-space key '" + key + "'");
    TEUCHOS_TEST_FOR_EXCEPTION(!isKnownMueLuKey(key), std::runtime_error,
                               "Unknown MueLu parameter in gptune.search space: '" + key + "'");

    spec.search_parameters.push_back(parseSearchParameter(key, ss.sublist(key)));
  }

  TEUCHOS_TEST_FOR_EXCEPTION(spec.search_parameters.empty(), std::runtime_error,
                             "gptune.search space must contain at least one parameter");

  std::set<std::string> tunedNames;
  for (const auto& p : spec.search_parameters)
    tunedNames.insert(p.muelu_key);

  for (const auto& p : spec.search_parameters) {
    for (const auto& c : p.active_if) {
      TEUCHOS_TEST_FOR_EXCEPTION(!tunedNames.count(c.parameter_name), std::runtime_error,
                                 "Parameter '" + p.muelu_key +
                                     "' has active-if reference to unknown tuned parameter '" +
                                     c.parameter_name + "'");
    }
  }

  if (spec.objective == Objective::SolveTime ||
      spec.objective == Objective::TotalTime ||
      spec.objective == Objective::IterationCount ||
      spec.objective == Objective::AmortizedSolveTime) {
    TEUCHOS_TEST_FOR_EXCEPTION(!spec.evaluation.measure_solve, std::runtime_error,
                               "Selected gptune.objective requires evaluation.measure solve = true");
  }

  if (spec.objective == Objective::AmortizedSolveTime) {
    TEUCHOS_TEST_FOR_EXCEPTION(!(spec.evaluation.measure_setup && spec.evaluation.measure_solve),
                               std::runtime_error,
                               "Objective 'amortized solve time' requires both setup and solve measurement");
  }

  return spec;
}

Teuchos::ParameterList StripGPTuneSublist(const Teuchos::ParameterList& fullList) {
  Teuchos::ParameterList out(fullList);
  if (out.isSublist("gptune")) {
    out.remove("gptune");
  }
  return out;
}

std::vector<std::string> SplitParameterPath(const std::string& key,
                                            const std::string& delimiter) {
  std::vector<std::string> parts;
  std::string::size_type begin = 0;
  while (true) {
    const auto end = key.find(delimiter, begin);
    if (end == std::string::npos) {
      parts.push_back(key.substr(begin));
      break;
    }
    parts.push_back(key.substr(begin, end - begin));
    begin = end + delimiter.size();
  }
  return parts;
}

bool GetAnyParameterByPath(const Teuchos::ParameterList& params,
                           const std::string& key,
                           Teuchos::any& value) {
  const auto parts = SplitParameterPath(key);
  if (parts.empty()) return false;

  const Teuchos::ParameterList* activeList = &params;
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    if (!activeList->isSublist(parts[i])) return false;
    activeList = &activeList->sublist(parts[i]);
  }

  const auto& leaf = parts.back();
  if (!activeList->isParameter(leaf)) return false;

  const auto& entry = activeList->getEntry(leaf);
  if (entry.isType<std::string>()) {
    value = activeList->get<std::string>(leaf);
  } else if (entry.isType<int>()) {
    value = activeList->get<int>(leaf);
  } else if (entry.isType<double>()) {
    value = activeList->get<double>(leaf);
  } else if (entry.isType<bool>()) {
    value = activeList->get<bool>(leaf);
  } else {
    throw std::runtime_error("Unsupported base ParameterList type for GPTune key '" + key + "'");
  }

  return true;
}

void SetAnyParameterByPath(Teuchos::ParameterList& params,
                           const std::string& key,
                           const Teuchos::any& value) {
  const auto parts = SplitParameterPath(key);
  TEUCHOS_TEST_FOR_EXCEPTION(parts.empty(), std::runtime_error,
                             "Empty GPTune candidate parameter key");

  Teuchos::ParameterList* activeList = &params;
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    activeList = &activeList->sublist(parts[i]);
  }

  const auto& leaf = parts.back();
  if (value.type() == typeid(std::string)) {
    activeList->set(leaf, Teuchos::any_cast<std::string>(value));
  } else if (value.type() == typeid(int)) {
    activeList->set(leaf, Teuchos::any_cast<int>(value));
  } else if (value.type() == typeid(double)) {
    activeList->set(leaf, Teuchos::any_cast<double>(value));
  } else if (value.type() == typeid(bool)) {
    activeList->set(leaf, Teuchos::any_cast<bool>(value));
  } else {
    throw std::runtime_error("Unsupported candidate value type for '" + key + "'");
  }
}

void ApplyCandidateToParameterList(const Candidate& candidate,
                                   Teuchos::ParameterList& params) {
  for (const auto& kv : candidate) {
    SetAnyParameterByPath(params, kv.first, kv.second);
  }
}

bool IsActive(const SearchParameter& param,
              const Candidate& candidate) {
  for (const auto& clause : param.active_if) {
    auto it = candidate.find(clause.parameter_name);
    if (it == candidate.end()) return false;
    if (anyToString(it->second) != clause.required_value) return false;
  }
  return true;
}

void WriteCandidateToParameterList(const Candidate& candidate,
                                   Teuchos::ParameterList& out) {
  for (const auto& kv : candidate) {
    const auto& key = kv.first;
    const auto& value = kv.second;

    if (value.type() == typeid(std::string)) {
      out.set(key, Teuchos::any_cast<std::string>(value));
    } else if (value.type() == typeid(int)) {
      out.set(key, Teuchos::any_cast<int>(value));
    } else if (value.type() == typeid(double)) {
      out.set(key, Teuchos::any_cast<double>(value));
    } else if (value.type() == typeid(bool)) {
      out.set(key, Teuchos::any_cast<bool>(value));
    }
  }
}

void WriteResultsToParameterList(Teuchos::ParameterList& fullList,
                                 const Candidate& best,
                                 double bestObjective,
                                 int evaluationsPerformed,
                                 const std::string& status,
                                 const std::string& message) {
  auto& gt = fullList.sublist("gptune");
  auto& results = gt.sublist("results");

  results.set("status", status);
  results.set("message", message);
  results.set("best objective", bestObjective);
  results.set("evaluations performed", evaluationsPerformed);

  WriteCandidateToParameterList(best, results.sublist("best parameters"));

  if (gt.get("set best parameters on output", false)) {
    ApplyCandidateToParameterList(best, fullList);
  }
}

void WriteResultsToParameterList(Teuchos::ParameterList& fullList,
                                 const TuningRunSummary& summary,
                                 bool writeHistory) {
  WriteResultsToParameterList(fullList,
                              summary.bestCandidate,
                              summary.bestObjective,
                              summary.evaluationsPerformed,
                              summary.status,
                              summary.message);

  auto& results = fullList.sublist("gptune").sublist("results");
  results.set("has best candidate", summary.hasBestCandidate);
  results.set("retained best hierarchy", summary.hasRetainedHierarchy);
  results.set("returned retained hierarchy", summary.returnedRetainedHierarchy);

  if (!summary.bestParameterList.name().empty() || summary.bestParameterList.numParams() > 0) {
    results.sublist("best parameter list") = summary.bestParameterList;
  }

  if (writeHistory) {
    auto& history = results.sublist("history");
    int index = 0;
    for (const auto& record : summary.history) {
      auto& entry = history.sublist(std::to_string(index));
      entry.set("evaluation index", record.evaluationIndex);
      entry.set("candidate id", record.candidateId);
      entry.set("objective", record.result.objective);
      entry.set("status", record.result.status);
      entry.set("message", record.result.message);
      entry.set("retained as best", record.retainedAsBest);

      auto& metrics = entry.sublist("metrics");
      metrics.set("setup time", record.result.metrics.setupTime);
      metrics.set("solve time", record.result.metrics.solveTime);
      metrics.set("total time", record.result.metrics.totalTime);
      metrics.set("iteration count", record.result.metrics.iterationCount);
      metrics.set("converged", record.result.metrics.converged);

      WriteCandidateToParameterList(record.candidate, entry.sublist("candidate"));
      ++index;
    }
  }
}

}  // namespace GPTuneInt
}  // namespace MueLu
