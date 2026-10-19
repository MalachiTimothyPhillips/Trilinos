// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "MueLu_GPTuneJson.hpp"

#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Teuchos_any.hpp>

namespace MueLu {
namespace GPTuneInt {

namespace {

struct JsonValue {
  enum class Kind { Null, Bool, Number, String, Array, Object };

  Kind kind = Kind::Null;
  bool boolValue = false;
  double numberValue = 0.0;
  std::string stringValue;
  std::vector<JsonValue> arrayValue;
  std::map<std::string, JsonValue> objectValue;
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  JsonValue parse() {
    auto value = parseValue();
    skipWhitespace();
    if (pos_ != text_.size()) {
      throw std::runtime_error("Trailing characters in GPTune worker JSON message");
    }
    return value;
  }

 private:
  void skipWhitespace() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
  }

  char peek() {
    skipWhitespace();
    if (pos_ >= text_.size()) throw std::runtime_error("Unexpected end of GPTune worker JSON message");
    return text_[pos_];
  }

  char consume() {
    if (pos_ >= text_.size()) throw std::runtime_error("Unexpected end of GPTune worker JSON message");
    return text_[pos_++];
  }

  void expect(char expected) {
    skipWhitespace();
    const char actual = consume();
    if (actual != expected) {
      throw std::runtime_error("Malformed GPTune worker JSON message");
    }
  }

  JsonValue parseValue() {
    const char c = peek();
    if (c == '{') return parseObject();
    if (c == '[') return parseArray();
    if (c == '"') return parseStringValue();
    if (c == 't' || c == 'f') return parseBool();
    if (c == 'n') return parseNull();
    return parseNumber();
  }

  JsonValue parseObject() {
    JsonValue value;
    value.kind = JsonValue::Kind::Object;
    expect('{');
    skipWhitespace();
    if (peek() == '}') {
      consume();
      return value;
    }
    while (true) {
      auto key = parseString();
      expect(':');
      value.objectValue[key] = parseValue();
      skipWhitespace();
      const char c = consume();
      if (c == '}') break;
      if (c != ',') throw std::runtime_error("Malformed GPTune worker JSON object");
    }
    return value;
  }

  JsonValue parseArray() {
    JsonValue value;
    value.kind = JsonValue::Kind::Array;
    expect('[');
    skipWhitespace();
    if (peek() == ']') {
      consume();
      return value;
    }
    while (true) {
      value.arrayValue.push_back(parseValue());
      skipWhitespace();
      const char c = consume();
      if (c == ']') break;
      if (c != ',') throw std::runtime_error("Malformed GPTune worker JSON array");
    }
    return value;
  }

  std::string parseString() {
    expect('"');
    std::string out;
    while (true) {
      const char c = consume();
      if (c == '"') return out;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }

      const char esc = consume();
      switch (esc) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        default:
          throw std::runtime_error("Unsupported escape sequence in GPTune worker JSON string");
      }
    }
  }

  JsonValue parseStringValue() {
    JsonValue value;
    value.kind = JsonValue::Kind::String;
    value.stringValue = parseString();
    return value;
  }

  JsonValue parseBool() {
    JsonValue value;
    value.kind = JsonValue::Kind::Bool;
    if (text_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      value.boolValue = true;
      return value;
    }
    if (text_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      value.boolValue = false;
      return value;
    }
    throw std::runtime_error("Malformed boolean in GPTune worker JSON message");
  }

  JsonValue parseNull() {
    JsonValue value;
    if (text_.compare(pos_, 4, "null") != 0) {
      throw std::runtime_error("Malformed null in GPTune worker JSON message");
    }
    pos_ += 4;
    return value;
  }

  JsonValue parseNumber() {
    skipWhitespace();
    const std::size_t start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    if (start == pos_) throw std::runtime_error("Malformed number in GPTune worker JSON message");

    JsonValue value;
    value.kind = JsonValue::Kind::Number;
    value.numberValue = std::strtod(text_.substr(start, pos_ - start).c_str(), nullptr);
    return value;
  }

  const std::string& text_;
  std::size_t pos_ = 0;
};

std::string quote(const std::string& s) {
  std::ostringstream os;
  os << '"';
  for (const char c : s) {
    switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\n': os << "\\n"; break;
      case '\r': os << "\\r"; break;
      case '\t': os << "\\t"; break;
      default: os << c; break;
    }
  }
  os << '"';
  return os.str();
}

std::string anyToJson(const Teuchos::any& value) {
  std::ostringstream os;
  os << std::setprecision(17);
  if (value.type() == typeid(std::string)) return quote(Teuchos::any_cast<std::string>(value));
  if (value.type() == typeid(int)) {
    os << Teuchos::any_cast<int>(value);
    return os.str();
  }
  if (value.type() == typeid(double)) {
    os << Teuchos::any_cast<double>(value);
    return os.str();
  }
  if (value.type() == typeid(bool)) return Teuchos::any_cast<bool>(value) ? "true" : "false";
  throw std::runtime_error("Unsupported GPTune worker JSON value type");
}

std::string paramTypeName(ParamType type) {
  if (type == ParamType::Categorical) return "categorical";
  if (type == ParamType::Integer) return "integer";
  return "real";
}

std::string objectiveName(Objective objective) {
  switch (objective) {
    case Objective::SetupTime: return "setup time";
    case Objective::SolveTime: return "solve time";
    case Objective::TotalTime: return "total time";
    case Objective::IterationCount: return "iteration count";
    case Objective::AmortizedSolveTime: return "amortized solve time";
  }
  return "unknown";
}

std::string candidateToJson(const Candidate& candidate) {
  std::ostringstream os;
  os << '{';
  bool first = true;
  for (const auto& kv : candidate) {
    if (!first) os << ',';
    first = false;
    os << quote(kv.first) << ':' << anyToJson(kv.second);
  }
  os << '}';
  return os.str();
}

const JsonValue* findMember(const JsonValue& value, const std::string& key) {
  if (value.kind != JsonValue::Kind::Object) return nullptr;
  const auto it = value.objectValue.find(key);
  if (it == value.objectValue.end()) return nullptr;
  return &it->second;
}

std::string getStringMember(const JsonValue& value, const std::string& key, const std::string& defaultValue = std::string()) {
  const auto* member = findMember(value, key);
  if (member == nullptr) return defaultValue;
  if (member->kind != JsonValue::Kind::String) throw std::runtime_error("Expected string field '" + key + "' in GPTune worker message");
  return member->stringValue;
}

double getNumberMember(const JsonValue& value, const std::string& key, double defaultValue = 0.0) {
  const auto* member = findMember(value, key);
  if (member == nullptr) return defaultValue;
  if (member->kind != JsonValue::Kind::Number) throw std::runtime_error("Expected number field '" + key + "' in GPTune worker message");
  return member->numberValue;
}

int getIntMember(const JsonValue& value, const std::string& key, int defaultValue = 0) {
  return static_cast<int>(getNumberMember(value, key, static_cast<double>(defaultValue)));
}

Teuchos::any jsonToAny(const JsonValue& value, const SearchParameter& param) {
  if (param.type == ParamType::Categorical) {
    if (value.kind == JsonValue::Kind::String) return Teuchos::any(value.stringValue);
    if (value.kind == JsonValue::Kind::Number) {
      std::ostringstream os;
      os << std::setprecision(17) << value.numberValue;
      return Teuchos::any(os.str());
    }
  } else if (param.type == ParamType::Integer) {
    if (value.kind == JsonValue::Kind::Number) return Teuchos::any(static_cast<int>(value.numberValue));
  } else {
    if (value.kind == JsonValue::Kind::Number) return Teuchos::any(value.numberValue);
  }
  if (value.kind == JsonValue::Kind::Bool) return Teuchos::any(value.boolValue);
  throw std::runtime_error("GPTune worker candidate field has an incompatible type for '" + param.muelu_key + "'");
}

Candidate parseCandidateObject(const JsonValue& object, const ParsedSpec& spec) {
  if (object.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("GPTune worker candidate must be a JSON object");
  }

  Candidate candidate;
  for (const auto& param : spec.search_parameters) {
    const auto it = object.objectValue.find(param.muelu_key);
    if (it != object.objectValue.end()) {
      candidate[param.muelu_key] = jsonToAny(it->second, param);
    }
  }
  return candidate;
}

}  // namespace

std::string BuildWorkerInitMessage(const ParsedSpec& spec) {
  std::ostringstream os;
  os << std::setprecision(17);
  os << "{\"type\":\"init\",\"protocol\":\"muelu-gptune-worker\",\"version\":1";
  os << ",\"max_evaluations\":" << spec.max_evaluations;
  os << ",\"seed\":" << spec.seed;
  os << ",\"objective\":" << quote(objectiveName(spec.objective));
  os << ",\"search_space\":[";
  for (std::size_t i = 0; i < spec.search_parameters.size(); ++i) {
    const auto& param = spec.search_parameters[i];
    if (i != 0) os << ',';
    os << "{\"name\":" << quote(param.muelu_key)
       << ",\"type\":" << quote(paramTypeName(param.type));
    if (param.type == ParamType::Categorical) {
      os << ",\"values\":[";
      for (std::size_t j = 0; j < param.categorical_values.size(); ++j) {
        if (j != 0) os << ',';
        os << quote(param.categorical_values[j]);
      }
      os << ']';
    } else if (param.type == ParamType::Integer) {
      os << ",\"min\":" << param.int_min << ",\"max\":" << param.int_max << ",\"step\":" << param.int_step;
    } else {
      os << ",\"min\":" << param.real_min << ",\"max\":" << param.real_max << ",\"step\":" << param.real_step;
    }
    if (param.has_initial_value) {
      os << ",\"initial_value\":" << anyToJson(param.initial_value);
    }
    os << '}';
  }
  os << "]}";
  return os.str();
}

std::string BuildWorkerTellMessage(const std::string& candidateId,
                                   const Candidate& candidate,
                                   const EvaluationResult& result) {
  std::ostringstream os;
  os << std::setprecision(17);
  os << "{\"type\":\"tell\",\"candidate_id\":" << quote(candidateId)
     << ",\"candidate\":" << candidateToJson(candidate)
     << ",\"objective\":" << result.objective
     << ",\"status\":" << quote(result.status)
     << ",\"message\":" << quote(result.message)
     << ",\"metrics\":{\"setup_time\":" << result.metrics.setupTime
     << ",\"solve_time\":" << result.metrics.solveTime
     << ",\"total_time\":" << result.metrics.totalTime
     << ",\"iteration_count\":" << result.metrics.iterationCount
     << ",\"converged\":" << (result.metrics.converged ? "true" : "false") << "}}";
  return os.str();
}

std::string BuildWorkerShutdownMessage() {
  return "{\"type\":\"shutdown\"}";
}

WorkerMessage ParseWorkerMessage(const std::string& line,
                                 const ParsedSpec& spec) {
  JsonParser parser(line);
  const auto root = parser.parse();
  if (root.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("GPTune worker message must be a JSON object");
  }

  WorkerMessage message;
  message.type = getStringMember(root, "type");
  message.candidateId = getStringMember(root, "candidate_id");
  message.status = getStringMember(root, "status");
  message.message = getStringMember(root, "message");
  message.bestObjective = getNumberMember(root, "best_objective", 0.0);
  message.evaluationsPerformed = getIntMember(root, "evaluations_performed", 0);

  if (const auto* candidate = findMember(root, "candidate")) {
    message.candidate = parseCandidateObject(*candidate, spec);
  }
  if (const auto* best = findMember(root, "best_candidate")) {
    message.bestCandidate = parseCandidateObject(*best, spec);
  }

  return message;
}

}  // namespace GPTuneInt
}  // namespace MueLu
