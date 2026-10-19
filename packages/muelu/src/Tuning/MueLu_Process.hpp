// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_PROCESS_HPP
#define MUELU_PROCESS_HPP

#include <string>
#include <vector>

namespace MueLu {
namespace GPTuneInt {

class Process {
 public:
  Process() = default;
  ~Process();

  Process(const Process&) = delete;
  Process& operator=(const Process&) = delete;

  void start(const std::vector<std::string>& argv);
  void writeLine(const std::string& line);
  std::string readLine(int timeoutSeconds);
  void shutdown();

 private:
  int stdinFd_ = -1;
  int stdoutFd_ = -1;
  int pid_ = -1;
};

}  // namespace GPTuneInt
}  // namespace MueLu

#endif  // MUELU_PROCESS_HPP
