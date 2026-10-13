// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "MueLu_GPTuneBridge.hpp"

namespace MueLu {
namespace GPTuneInt {

#define MUELU_ETI_GROUP(Scalar, LocalOrdinal, GlobalOrdinal, Node)                                            \
  template BridgeResult RunGPTuneBridge<Scalar, LocalOrdinal, GlobalOrdinal, Node>(                          \
      const Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>& op,                    \
      const Teuchos::ParameterList& baseParams,                                                              \
      const Teuchos::ParameterList& fullParams,                                                              \
      const ParsedSpec& spec);

#include <MueLu_ETI_4arg.hpp>

}  // namespace GPTuneInt
}  // namespace MueLu
