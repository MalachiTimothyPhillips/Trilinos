// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_MULTIPHYS_DEF_HPP
#define MUELU_MULTIPHYS_DEF_HPP

#include <sstream>
#include "MueLu_ConfigDefs.hpp"

#include "Xpetra_Map.hpp"
#include "MueLu_CrsMatrixUtils.hpp"
#include "Xpetra_MatrixUtils.hpp"

#include "MueLu_MultiPhys_decl.hpp"
#include "MueLu_SaPFactory.hpp"
#include "MueLu_AggregationExportFactory.hpp"
#include "MueLu_Utilities.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_Hierarchy.hpp"
#include "MueLu_RAPFactory.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_PerfUtils.hpp"
#include "MueLu_ParameterListInterpreter.hpp"
#include "MueLu_HierarchyManager.hpp"
#include "MueLu_MasterList.hpp"
#include <MueLu_HierarchyUtils.hpp>
#include "MueLu_VerbosityLevel.hpp"
#include <MueLu_CreateXpetraPreconditioner.hpp>
#include <MueLu_ML2MueLuParameterTranslator.hpp>

#ifdef HAVE_MUELU_CUDA
#include "cuda_profiler_api.h"
#endif

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
const Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getDomainMap() const {
  return AmatMultiphysics_->getDomainMap();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
const Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getRangeMap() const {
  return AmatMultiphysics_->getRangeMap();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::setParameters(Teuchos::ParameterList& list) {
  // this operator only makes sense for the Combo when using TransP for R

  list.set("multigrid algorithm", "combine");
  list.set("combine: numBlks", nBlks_);

  // Make sure verbosity gets passed to the sublists
  std::string verbosity = list.get("verbosity", "high");
  VerboseObject::SetDefaultVerbLevel(toVerbLevel(verbosity));

  arrayOfParamLists_.resize(nBlks_);
  for (int i = 0; i < nBlks_; i++) {
    std::string listName = "subblockList" + Teuchos::toString(i);
    if (list.isSublist(listName)) {
      arrayOfParamLists_[i] = Teuchos::rcpFromRef(list.sublist(listName));
    } else
      TEUCHOS_TEST_FOR_EXCEPTION(true, Exceptions::RuntimeError, "Must provide sublist " + listName);

    arrayOfParamLists_[i]->set("verbosity", arrayOfParamLists_[i]->get("verbosity", verbosity));
    if (OmitSubblockSmoother_) {
      arrayOfParamLists_[i]->set("smoother: pre or post", "none");
      arrayOfParamLists_[i]->set("smoother: type", "none");
    }
  }

  // Handled by MultiPhys::compute, not by the ParameterListInterpreter.
  matchLevels_ = list.get("combine: match levels", false);
  list.remove("combine: match levels", false);
  coordinateRepartitioning_ = list.get("combine: coordinate repartitioning", false);
  list.remove("combine: coordinate repartitioning", false);

  // Are we using Kokkos?
  useKokkos_ = !Node::is_serial;
  useKokkos_ = list.get("use kokkos refactor", useKokkos_);

  paramListMultiphysics_ = Teuchos::rcpFromRef(list);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::compute(bool reuse) {
  /*

     Create a set of AMG hierarchies whose interpolation matrices are used to build on combined
     AMG hierarchy for a multiphysics problem

   */

  //#ifdef HAVE_MUELU_CUDA
  //   if (paramListMultiphysics_.get<bool>("multiphysics: cuda profile setup", false)) cudaProfilerStart();
  //#endif

  std::string timerLabel;
  if (reuse)
    timerLabel = "MueLu MultiPhys: compute (reuse)";
  else
    timerLabel = "MueLu MultiPhys: compute";
  RCP<Teuchos::TimeMonitor> tmCompute = getTimer(timerLabel);

  ////////////////////////////////////////////////////////////////////////////////
  // Generate the (iii,iii) Hierarchy

  // "combine: coordinate repartitioning": the combined levels' block smoother works through the blocks one at a time
  // over the same ranks, and a combined row carries the coupling entries too. So each subblock's rows-per-process
  // targets are divided by (number of blocks) x (nnz of the combined operator / nnz of its diagonal blocks).
  double repartitionDivisor = 1.0;
  if (coordinateRepartitioning_ && !coordinatedRepartitioningApplied_) {
    double diagonalNnz = 0.0;
    for (int i = 0; i < nBlks_; i++) diagonalNnz += static_cast<double>(arrayOfAuxMatrices_[i]->getGlobalNumEntries());
    const double couplingWeight = diagonalNnz > 0.0 ? std::max(1.0, static_cast<double>(AmatMultiphysics_->getGlobalNumEntries()) / diagonalNnz) : 1.0;
    repartitionDivisor          = nBlks_ * couplingWeight;
    GetOStream(Runtime0) << "MultiPhys: coordinated repartitioning divides subblock rows-per-process targets by " << repartitionDivisor
                         << " (" << nBlks_ << " blocks x coupling weight " << couplingWeight << ")" << std::endl;
  }

  for (int iii = 0; iii < nBlks_; iii++) {
    if (arrayOfCoords_ != Teuchos::null) {
      if (arrayOfCoords_[iii] != Teuchos::null) {
        arrayOfParamLists_[iii]->sublist("user data").set("Coordinates", arrayOfCoords_[iii]);
      }
    }

    if (arrayOfMaterials_ != Teuchos::null) {
      if (arrayOfMaterials_[iii] != Teuchos::null) {
        arrayOfParamLists_[iii]->sublist("user data").set("Material", arrayOfMaterials_[iii]);
      }
    }

    if (arrayOfNullspaces_ != Teuchos::null) {
      if (arrayOfNullspaces_[iii] != Teuchos::null) {
        arrayOfParamLists_[iii]->sublist("user data").set("Nullspace", arrayOfNullspaces_[iii]);
      }
    }

    bool wantToRepartition = false;
    if (paramListMultiphysics_->isParameter("repartition: enable"))
      wantToRepartition = paramListMultiphysics_->get<bool>("repartition: enable");

    arrayOfParamLists_[iii]->set("repartition: enable", wantToRepartition);
    arrayOfParamLists_[iii]->set("repartition: rebalance P and R", true);
    arrayOfParamLists_[iii]->set("repartition: explicit via new copy rebalance P and R", true);

    // "combine: coordinate repartitioning" (applied once: compute() runs again on reuse).
    if (wantToRepartition && coordinateRepartitioning_ && !coordinatedRepartitioningApplied_) {
      auto& sub = *arrayOfParamLists_[iii];
      sub.set("repartition: remap parts", false);
      for (const char* name : {"repartition: target rows per proc", "repartition: min rows per proc"}) {
        const int value = sub.isParameter(name) ? sub.get<int>(name) : MasterList::getDefault<int>(name);
        if (value > 0) sub.set(name, std::max(1, static_cast<int>(value / repartitionDivisor)));
      }
    }

    if (paramListMultiphysics_->isParameter("repartition: use subcommunicators"))
      arrayOfParamLists_[iii]->set("repartition: use subcommunicators", paramListMultiphysics_->get<bool>("repartition: use subcommunicators"));
    else
      arrayOfParamLists_[iii]->set("repartition: use subcommunicators", true);

    // Only the subblocks' prolongators and operators are used (the combined hierarchy has its own coarse solve),
    // so skip their coarse solvers. Their smoothers stay unless omitted: their Chebyshev setup provides the
    // lambda_max estimate that SA prolongator damping reads from the level operator.
    arrayOfParamLists_[iii]->set("coarse: type", "none");
    arrayOfParamLists_[iii]->remove("coarse: params", false);

    // "combine: match levels" continues coarsening from a subblock's coarsest level, so keep what that needs.
    if (matchLevels_ && paramListMultiphysics_->isParameter("combine: useMaxLevels") && paramListMultiphysics_->get<bool>("combine: useMaxLevels")) {
      std::string keep = arrayOfParamLists_[iii]->get<std::string>("keep data", "{}");
      keep             = keep.substr(keep.find('{') + 1, keep.rfind('}') - keep.find('{') - 1);
      std::vector<std::string> wanted{"Nullspace"};
      if (arrayOfCoords_ != Teuchos::null && arrayOfCoords_[iii] != Teuchos::null) wanted.push_back("Coordinates");
      if (arrayOfMaterials_ != Teuchos::null && arrayOfMaterials_[iii] != Teuchos::null) wanted.push_back("Material");
      for (const auto& name : wanted)
        if (keep.find(name) == std::string::npos) keep += (keep.empty() ? "" : ",") + name;
      arrayOfParamLists_[iii]->set("keep data", "{" + keep + "}");
    }
  }
  // repartitioning should only happen when createing the individual P's , not
  // when combiing them

  paramListMultiphysics_->set<bool>("repartition: enable", false);

  bool useMaxLevels = false;
  if (paramListMultiphysics_->isParameter("combine: useMaxLevels"))
    useMaxLevels = paramListMultiphysics_->get<bool>("combine: useMaxLevels");

  if (coordinateRepartitioning_) coordinatedRepartitioningApplied_ = true;

  const bool extendShortSubblocks = useMaxLevels && matchLevels_;
  std::vector<RCP<HierarchyManager<SC, LO, GO, NO>>> subblockManagers(nBlks_);

  LO maxLevels = useMaxLevels ? 0 : std::numeric_limits<LO>::max();
  for (int i = 0; i < nBlks_; i++) {
    std::string operatorLabel = "MultiPhys (" + Teuchos::toString(i) + "," + Teuchos::toString(i) + ")";
    arrayOfAuxMatrices_[i]->setObjectLabel(operatorLabel);
    arrayOfHierarchies_[i] = MueLu::CreateXpetraPreconditioner(arrayOfAuxMatrices_[i], *arrayOfParamLists_[i], subblockManagers[i]);
    LO tempNlevels         = arrayOfHierarchies_[i]->GetGlobalNumLevels();
    if (useMaxLevels) {
      if (tempNlevels > maxLevels) maxLevels = tempNlevels;
    } else {
      if (tempNlevels < maxLevels) maxLevels = tempNlevels;
    }
  }

  // "combine: match levels": a subblock that stopped coarsening early continues from its coarsest level down to
  // maxLevels. Only the missing levels are built, in a hierarchy whose fine level is that coarsest level; their P
  // and A are spliced in below. Levels it still cannot reach (coarsening stagnates) get identity prolongators.
  std::vector<RCP<Hierarchy>> subblockExtensions(nBlks_);
  if (extendShortSubblocks) {
    for (int i = 0; i < nBlks_; i++) {
      auto H         = arrayOfHierarchies_[i];
      const LO nLevs = H->GetGlobalNumLevels();
      if (nLevs >= maxLevels) continue;

      // With subcommunicators some ranks hold fewer levels; continuing would need the coarse communicator.
      int haveAllLevels = H->GetNumLevels() == nLevs ? 1 : 0, allHaveAllLevels = 0;
      Teuchos::reduceAll(*AmatMultiphysics_->getDomainMap()->getComm(), Teuchos::REDUCE_MIN, haveAllLevels, Teuchos::outArg(allHaveAllLevels));
      if (!allHaveAllLevels) {
        GetOStream(Warnings0) << "MultiPhys: subblock " << i << " stopped at " << nLevs << " of " << maxLevels
                              << " levels but is not on every rank (subcommunicators); using identity prolongators below it" << std::endl;
        continue;
      }

      auto coarsest       = H->GetLevel(nLevs - 1);
      auto coarsestFactMg = subblockManagers[i]->GetFactoryManager(nLevs - 1);
      Teuchos::ParameterList extensionList(*arrayOfParamLists_[i]);
      extensionList.set("max levels", maxLevels - nLevs + 1);
      extensionList.set("coarse: max size", 1);
      extensionList.remove("keep data", false);
      if (extensionList.isParameter("repartition: start level"))
        extensionList.set("repartition: start level", std::max(0, extensionList.get<int>("repartition: start level") - (nLevs - 1)));
      extensionList.set("hierarchy label", "MultiPhys (" + Teuchos::toString(i) + "," + Teuchos::toString(i) + ") continued");

      auto& userData = extensionList.sublist("user data");
      userData.remove("Nullspace", false);
      userData.remove("Coordinates", false);
      userData.remove("Material", false);
      auto keptFactory = [&](const std::string& name) { return coarsestFactMg->GetFactory(name).get(); };
      if (coarsest->IsAvailable("Nullspace", keptFactory("Nullspace")))
        userData.set("Nullspace", coarsest->template Get<RCP<MultiVector>>("Nullspace", keptFactory("Nullspace")));
      if (arrayOfCoords_ != Teuchos::null && arrayOfCoords_[i] != Teuchos::null && coarsest->IsAvailable("Coordinates", keptFactory("Coordinates")))
        userData.set("Coordinates", coarsest->template Get<RCP<RealValuedMultiVector>>("Coordinates", keptFactory("Coordinates")));
      if (arrayOfMaterials_ != Teuchos::null && arrayOfMaterials_[i] != Teuchos::null && coarsest->IsAvailable("Material", keptFactory("Material")))
        userData.set("Material", coarsest->template Get<RCP<MultiVector>>("Material", keptFactory("Material")));

      subblockExtensions[i]  = MueLu::CreateXpetraPreconditioner(coarsest->template Get<RCP<Matrix>>("A"), extensionList);
      const LO reachedLevels = nLevs - 1 + subblockExtensions[i]->GetGlobalNumLevels();
      GetOStream(Runtime0) << "MultiPhys: subblock " << i << " stopped at " << nLevs << " levels; continued coarsening to "
                           << reachedLevels << " of " << maxLevels << " levels" << std::endl;
    }
  }

  hierarchyMultiphysics_ = rcp(new Hierarchy("Combo"));
  for (LO i = 0; i < maxLevels; i++) {
    hierarchyMultiphysics_->AddNewLevel();
  }
  for (int i = 0; i < nBlks_; i++) {
    std::string subblkName = "Psubblock" + Teuchos::toString(i);
    MueLu::HierarchyUtils<SC, LO, GO, NO>::CopyBetweenHierarchies(*(arrayOfHierarchies_[i]), *(hierarchyMultiphysics_), "P", subblkName, "RCP<Matrix>");

    std::string subblkOpName = "Operatorsubblock" + Teuchos::toString(i);
    MueLu::HierarchyUtils<SC, LO, GO, NO>::CopyBetweenHierarchies(*(arrayOfHierarchies_[i]), *(hierarchyMultiphysics_), "A", subblkOpName, "RCP<Matrix>");

    // Copy remaining levels, if needed
    if (useMaxLevels) {
      auto numLevelsBlk             = arrayOfHierarchies_[i]->GetNumLevels();
      const auto numGlobalLevelsBlk = arrayOfHierarchies_[i]->GetGlobalNumLevels();
      if (numLevelsBlk == numGlobalLevelsBlk) {
        auto crsLevel = arrayOfHierarchies_[i]->GetLevel(numLevelsBlk - 1);
        if (!subblockExtensions[i].is_null()) {
          // Continued levels: extension level e is Combo level (numLevelsBlk - 1 + e); its level 0 is crsLevel.
          const auto& ext      = subblockExtensions[i];
          const LO firstCombo = numLevelsBlk - 1;
          for (LO e = 1; e < ext->GetNumLevels() && firstCombo + e < maxLevels; ++e) {
            auto level = hierarchyMultiphysics_->GetLevel(firstCombo + e);
            MueLu::HierarchyUtils<SC, LO, GO, NO>::CopyBetweenLevels(*ext->GetLevel(e), *level, "P", subblkName, "RCP<Matrix>");
            MueLu::HierarchyUtils<SC, LO, GO, NO>::CopyBetweenLevels(*ext->GetLevel(e), *level, "A", subblkOpName, "RCP<Matrix>");
          }
          const LO reached = std::min<LO>(maxLevels, firstCombo + ext->GetNumLevels());
          crsLevel         = ext->GetLevel(reached - 1 - firstCombo);
          numLevelsBlk     = reached;
        }
        TEUCHOS_ASSERT(crsLevel->IsAvailable("A"));
        for (int levelId = numLevelsBlk; levelId < maxLevels; ++levelId) {
          auto level = hierarchyMultiphysics_->GetLevel(levelId);
          MueLu::HierarchyUtils<SC, LO, GO, NO>::CopyBetweenLevels(*crsLevel, *level, "A", subblkOpName, "RCP<Matrix>");
        }
      }
    }
  }
  paramListMultiphysics_->set("coarse: max size", 1);
  paramListMultiphysics_->set("max levels", maxLevels);

  AmatMultiphysics_->setObjectLabel("A: block " + Teuchos::toString(nBlks_) + " x " + Teuchos::toString(nBlks_) + "multiphysics matrix");

  // Rip off non-serializable data before validation
  Teuchos::ParameterList nonSerialListMultiphysics, processedListMultiphysics;
  MueLu::ExtractNonSerializableData(*paramListMultiphysics_, processedListMultiphysics, nonSerialListMultiphysics);

  // Rip off the subblock List stuff as we  don't need it any more and I think it messes up validator

  Teuchos::ParameterList stripped;
  for (ParameterList::ConstIterator inListEntry = processedListMultiphysics.begin(); inListEntry != processedListMultiphysics.end(); inListEntry++) {
    const std::string& levelName = inListEntry->first;
    if (levelName.find("subblockList") != 0) stripped.setEntry(inListEntry->first, inListEntry->second);
  }

  RCP<HierarchyManager<SC, LO, GO, NO>> mueLuFactory = rcp(new ParameterListInterpreter<SC, LO, GO, NO>(stripped, AmatMultiphysics_->getDomainMap()->getComm()));
  hierarchyMultiphysics_->setlib(AmatMultiphysics_->getDomainMap()->lib());
  hierarchyMultiphysics_->SetProcRankVerbose(AmatMultiphysics_->getDomainMap()->getComm()->getRank());

  // We don't need nullspace or coordinates, since we don't use them when just combining prolongators that have been already created
  hierarchyMultiphysics_->GetLevel(0)->Set("A", AmatMultiphysics_);

  // Stick the non-serializible data on the hierarchy.
  // Not sure that we need this, since we don't use it in building the multiphysics hierarchy
  HierarchyUtils<SC, LO, GO, NO>::AddNonSerializableDataToHierarchy(*mueLuFactory, *hierarchyMultiphysics_, nonSerialListMultiphysics);
  mueLuFactory->SetupHierarchy(*hierarchyMultiphysics_);

  describe(GetOStream(Runtime0));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Teuchos::TimeMonitor> MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getTimer(std::string name, RCP<const Teuchos::Comm<int>> comm) const {
  if (IsPrint(Timings)) {
    if (!syncTimers_)
      return Teuchos::rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(name)));
    else {
      if (comm.is_null())
        return Teuchos::rcp(new Teuchos::SyncTimeMonitor(*Teuchos::TimeMonitor::getNewTimer(name), AmatMultiphysics_->getRowMap()->getComm().ptr()));
      else
        return Teuchos::rcp(new Teuchos::SyncTimeMonitor(*Teuchos::TimeMonitor::getNewTimer(name), comm.ptr()));
    }
  } else
    return Teuchos::null;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::resetMatrix(RCP<Matrix> AmatMultiphysics_new, bool ComputePrec) {
  bool reuse        = !AmatMultiphysics_.is_null();
  AmatMultiphysics_ = AmatMultiphysics_new;
  if (ComputePrec) compute(reuse);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::applyInverse(const MultiVector& RHS, MultiVector& X) const {
  hierarchyMultiphysics_->Iterate(RHS, X, 1, true);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::apply(const MultiVector& RHS, MultiVector& X,
                                                                 Teuchos::ETransp /* mode */,
                                                                 Scalar /* alpha */,
                                                                 Scalar /* beta */) const {
  RCP<Teuchos::TimeMonitor> tm = getTimer("MueLu MultiPhys: solve");
  hierarchyMultiphysics_->Iterate(RHS, X, 1, true);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::hasTransposeApply() const {
  return false;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    initialize(const Teuchos::RCP<Matrix>& /*AmatMultiPhysics*/,
               const Teuchos::ArrayRCP<RCP<Matrix>> /*arrayOfAuxMatrices*/,
               const Teuchos::ArrayRCP<Teuchos::RCP<MultiVector>> /*arrayOfNullspaces*/,
               const Teuchos::ArrayRCP<Teuchos::RCP<RealValuedMultiVector>> /*arrayOfCoords*/,
               const int /*nBlks*/,
               Teuchos::ParameterList& List,
               const Teuchos::ArrayRCP<Teuchos::RCP<MultiVector>> /*arrayOfMaterials*/) {
  arrayOfHierarchies_.resize(nBlks_);
  for (int i = 0; i < nBlks_; i++) arrayOfHierarchies_[i] = Teuchos::null;

  // Default settings
  useKokkos_    = false;
  enable_reuse_ = false;
  syncTimers_   = false;

  // set parameters
  setParameters(List);

}  // initialize

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MultiPhys<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    describe(Teuchos::FancyOStream& out, const Teuchos::EVerbosityLevel /* verbLevel */) const {
  std::ostringstream oss;

  RCP<const Teuchos::Comm<int>> comm = AmatMultiphysics_->getDomainMap()->getComm();

  oss << "\n--------------------------------------------------------------------------------\n"
      << "---                            MultiPhysics Summary                            ---\n"
         "--------------------------------------------------------------------------------"
      << std::endl;
  oss << std::endl;

  GlobalOrdinal numRows;
  GlobalOrdinal nnz;

  AmatMultiphysics_->getRowMap()->getComm()->barrier();

  for (int i = 0; i < nBlks_; i++) {
    numRows                  = arrayOfAuxMatrices_[i]->getGlobalNumRows();
    nnz                      = arrayOfAuxMatrices_[i]->getGlobalNumEntries();
    Xpetra::global_size_t tt = numRows;
    int rowspacer            = 3;
    while (tt != 0) {
      tt /= 10;
      rowspacer++;
    }
    tt            = nnz;
    int nnzspacer = 2;
    while (tt != 0) {
      tt /= 10;
      nnzspacer++;
    }

    oss << "block " << std::setw(rowspacer) << " rows " << std::setw(nnzspacer) << " nnz " << std::setw(9) << " nnz/row" << std::endl;
    oss << "(" << Teuchos::toString(i) << ", " << Teuchos::toString(i) << ")" << std::setw(rowspacer) << numRows << std::setw(nnzspacer) << nnz << std::setw(9) << as<double>(nnz) / numRows << std::endl;
  }
  oss << std::endl;

  out << oss.str();

  for (int i = 0; i < nBlks_; i++) {
    arrayOfHierarchies_[i]->describe(out, GetVerbLevel());
  }

}  // describe

}  // namespace MueLu

#define MUELU_MULTIPHYS_SHORT
#endif  // ifdef MUELU_MULTIPHYS_DEF_HPP
