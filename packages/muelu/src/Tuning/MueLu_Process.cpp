// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "MueLu_Process.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace MueLu {
namespace GPTuneInt {

namespace {

void closeIfOpen(int& fd) {
  if (fd >= 0) {
    ::close(fd);
    fd = -1;
  }
}

std::string errnoMessage(const std::string& context) {
  return context + ": " + std::strerror(errno);
}

}  // namespace

Process::~Process() {
  try {
    shutdown();
  } catch (...) {
  }
}

void Process::start(const std::vector<std::string>& argv) {
  if (argv.empty()) {
    throw std::runtime_error("Cannot start GPTune worker without an argv[0]");
  }
  if (pid_ >= 0) {
    throw std::runtime_error("GPTune worker process is already running");
  }

  int childStdin[2] = {-1, -1};
  int childStdout[2] = {-1, -1};

  if (::pipe(childStdin) != 0) {
    throw std::runtime_error(errnoMessage("Failed to create GPTune worker stdin pipe"));
  }
  if (::pipe(childStdout) != 0) {
    closeIfOpen(childStdin[0]);
    closeIfOpen(childStdin[1]);
    throw std::runtime_error(errnoMessage("Failed to create GPTune worker stdout pipe"));
  }

  const pid_t pid = ::fork();
  if (pid < 0) {
    closeIfOpen(childStdin[0]);
    closeIfOpen(childStdin[1]);
    closeIfOpen(childStdout[0]);
    closeIfOpen(childStdout[1]);
    throw std::runtime_error(errnoMessage("Failed to fork GPTune worker"));
  }

  if (pid == 0) {
    ::dup2(childStdin[0], STDIN_FILENO);
    ::dup2(childStdout[1], STDOUT_FILENO);

    ::close(childStdin[0]);
    ::close(childStdin[1]);
    ::close(childStdout[0]);
    ::close(childStdout[1]);

    std::vector<char*> execArgv;
    execArgv.reserve(argv.size() + 1);
    for (const auto& arg : argv) {
      execArgv.push_back(const_cast<char*>(arg.c_str()));
    }
    execArgv.push_back(nullptr);

    ::execvp(execArgv[0], execArgv.data());
    _exit(127);
  }

  pid_ = static_cast<int>(pid);
  stdinFd_ = childStdin[1];
  stdoutFd_ = childStdout[0];

  closeIfOpen(childStdin[0]);
  closeIfOpen(childStdout[1]);
}

void Process::writeLine(const std::string& line) {
  if (stdinFd_ < 0) {
    throw std::runtime_error("GPTune worker stdin is closed");
  }

  std::string data = line;
  data.push_back('\n');

  const char* ptr = data.data();
  std::size_t remaining = data.size();
  while (remaining > 0) {
    const ssize_t written = ::write(stdinFd_, ptr, remaining);
    if (written < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(errnoMessage("Failed to write to GPTune worker"));
    }
    ptr += written;
    remaining -= static_cast<std::size_t>(written);
  }
}

std::string Process::readLine(int timeoutSeconds) {
  if (stdoutFd_ < 0) {
    throw std::runtime_error("GPTune worker stdout is closed");
  }

  std::string line;
  const int timeoutMs = timeoutSeconds < 0 ? -1 : timeoutSeconds * 1000;

  while (true) {
    pollfd fd;
    fd.fd = stdoutFd_;
    fd.events = POLLIN;
    fd.revents = 0;

    const int rc = ::poll(&fd, 1, timeoutMs);
    if (rc < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(errnoMessage("Failed while waiting for GPTune worker output"));
    }
    if (rc == 0) {
      throw std::runtime_error("Timed out waiting for GPTune worker output");
    }

    char c = '\0';
    const ssize_t nread = ::read(stdoutFd_, &c, 1);
    if (nread < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(errnoMessage("Failed to read from GPTune worker"));
    }
    if (nread == 0) {
      throw std::runtime_error("GPTune worker closed stdout before sending a complete message");
    }
    if (c == '\n') return line;
    line.push_back(c);
  }
}

void Process::shutdown() {
  closeIfOpen(stdinFd_);
  closeIfOpen(stdoutFd_);

  if (pid_ >= 0) {
    int status = 0;
    while (::waitpid(static_cast<pid_t>(pid_), &status, 0) < 0) {
      if (errno == EINTR) continue;
      break;
    }
    pid_ = -1;
  }
}

}  // namespace GPTuneInt
}  // namespace MueLu
