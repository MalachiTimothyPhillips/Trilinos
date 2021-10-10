// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>

#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

#include "MueLu_UseDefaultTypes.hpp"

#include "MueLu_TestHelpers_kokkos.hpp"
#include "MueLu_Utilities_kokkos.hpp"
#include "MueLu_Version.hpp"

namespace MueLuTests
{

  template <typename SC, typename LO, typename GO, typename NO>
  void CreateDirchletRow(Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> &A, LO localRowToZero, SC value = Teuchos::ScalarTraits<SC>::zero())
  {
    Teuchos::ArrayView<const LO> indices;
    Teuchos::ArrayView<const SC> values;

    A->resumeFill();
    A->getLocalRowView(localRowToZero, indices, values);
    Array<SC> newValues(values.size(), value);

    for (int j = 0; j < indices.size(); j++)
    {
      //keep diagonal
      if (indices[j] == localRowToZero)
      {
        newValues[j] = values[j];
      }
    }

    A->replaceLocalValues(localRowToZero, indices, newValues);
    A->fillComplete();
  }

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, CuthillMcKee, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include "MueLu_UseShortNames.hpp"
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
    out << "version: " << MueLu::Version() << std::endl;

    using Teuchos::RCP;
    using Teuchos::rcp;
    using real_type = typename Teuchos::ScalarTraits<SC>::coordinateType;
    using RealValuedMultiVector = Xpetra::MultiVector<real_type, LO, GO, NO>;

    // Build the problem
    RCP<Matrix> A = MueLuTests::TestHelpers_kokkos::TestFactory<SC, LO, GO, NO>::Build1DPoisson(2001);
    RCP<const Map> map = A->getMap();
    RCP<RealValuedMultiVector> coordinates = Xpetra::MultiVectorFactory<real_type, LO, GO, NO>::Build(map, 1);
    RCP<MultiVector> nullspace = MultiVectorFactory::Build(map, 1);
    nullspace->putScalar(Teuchos::ScalarTraits<SC>::one());

    MueLu::Utilities_kokkos<SC, LO, GO, NO>::Transpose(*A); // compile test

    // CM Test
    RCP<Xpetra::Vector<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node>> ordering = MueLu::Utilities_kokkos<SC, LO, GO, NO>::CuthillMcKee(*A);

    RCP<Xpetra::Vector<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node>> ordering2 = MueLu::Utilities_kokkos<SC, LO, GO, NO>::ReverseCuthillMcKee(*A);

    TEST_EQUALITY(1, 1);
  }

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, DetectDirichletRows, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);
    LocalOrdinal localRowToZero = 5;

    CreateDirchletRow<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A, localRowToZero);

    auto drows = Utils_Kokkos::DetectDirichletRows(*A);
    TEST_EQUALITY(drows[localRowToZero], true);
    TEST_EQUALITY(drows[localRowToZero - 1], false);

    CreateDirchletRow<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A, localRowToZero, Teuchos::as<Scalar>(0.25));

    //row 5 should not be Dirichlet
    drows = Utils_Kokkos::DetectDirichletRows(*A, TST::magnitude(0.24));
    TEST_EQUALITY(drows[localRowToZero], false);
    TEST_EQUALITY(drows[localRowToZero - 1], false);

    //row 5 should be Dirichlet
    drows = Utils_Kokkos::DetectDirichletRows(*A, TST::magnitude(0.26));
    TEST_EQUALITY(drows[localRowToZero], true);
    TEST_EQUALITY(drows[localRowToZero - 1], false);
  } //DetectDirichletRows

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, DetectDirichletCols, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);

    LocalOrdinal localRowToZero = 5;
    CreateDirchletRow<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A, localRowToZero);

    auto drows = Utils_Kokkos::DetectDirichletRows(*A);
    auto dcols = Utils_Kokkos::DetectDirichletCols(*A, drows);

    for (size_t col = 0; col < dcols.extent(0); ++col)
    {
      const auto isDirchletCol = col >= 4 and col <= 6;
      TEST_EQUALITY(dcols(col), isDirchletCol);
    }
  } //DetectDirichletCols

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, GetMatrixDiagonal, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);
    auto diag = Utils_Kokkos::GetMatrixDiagonal(*A)->getHostLocalView(Xpetra::Access::ReadOnly);

    TEST_EQUALITY(diag.extent(0), A->getNodeNumRows());

    for (size_t idx = 0; idx < diag.extent(0); ++idx)
    {
      TEST_EQUALITY(diag(idx, 0), Scalar(2));
    }

  } //GetMatrixDiagonal

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, GetMatrixDiagonalInverse, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);
    auto diag = Utils_Kokkos::GetMatrixDiagonalInverse(*A)->getHostLocalView(Xpetra::Access::ReadOnly);

    TEST_EQUALITY(diag.extent(0), A->getNodeNumRows());

    for (size_t idx = 0; idx < diag.extent(0); ++idx)
    {
      TEST_EQUALITY(diag(idx, 0), Kokkos::ArithTraits<Scalar>::one() / Scalar(2));
    }
  } //GetMatrixDiagonalInverse

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, GetMatrixOverlappedDiagonal, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);
    auto diag = Utils_Kokkos::GetMatrixOverlappedDiagonal(*A)->getHostLocalView(Xpetra::Access::ReadOnly);

    for (size_t idx = 0; idx < diag.extent(0); ++idx)
    {
      TEST_EQUALITY(diag(idx, 0), Scalar(2));
    }
  } //GetMatrixOverlappedDiagonal

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, FindNonZeros, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);
    auto map = A->getMap();
    auto vector = Xpetra::MultiVectorFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(map, 1);
    auto vectorView = vector->getDeviceLocalView(Xpetra::Access::ReadOnly);

    Kokkos::View<bool *> nonZeros("", vectorView.extent(0));
    unsigned int result = 0;

    // Zero-ed out Vector
    {
      Utils_Kokkos::FindNonZeros(vector->getDeviceLocalView(Xpetra::Access::ReadOnly), nonZeros);

      Kokkos::parallel_reduce(
          "", nonZeros.extent(0),
          KOKKOS_LAMBDA(const int i, unsigned int &r) { r += static_cast<unsigned int>(nonZeros(i)); }, result);

      TEST_EQUALITY(result, 0);
    }

    // Vector filled with ones
    {
      vector->putScalar(TST::one());

      Utils_Kokkos::FindNonZeros(vector->getDeviceLocalView(Xpetra::Access::ReadOnly), nonZeros);

      Kokkos::parallel_reduce(
          "", nonZeros.extent(0),
          KOKKOS_LAMBDA(const int i, unsigned int &r) { r += static_cast<unsigned int>(nonZeros(i)); }, result);

      TEST_EQUALITY(result, nonZeros.extent(0));
    }

  } //FindNonZeros

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, DetectDirichletColsAndDomains, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);

    LocalOrdinal localRowToZero = 5;
    CreateDirchletRow<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A, localRowToZero);

    auto drows = Utils_Kokkos::DetectDirichletRows(*A);
    Kokkos::View<bool *, typename Node::device_type> dirichletCols("dirichletCols", A->getColMap()->getNodeNumElements());
    Kokkos::View<bool *, typename Node::device_type> dirichletDomain("dirichletDomain", A->getDomainMap()->getNodeNumElements());

    Utils_Kokkos::DetectDirichletColsAndDomains(*A, drows, dirichletCols, dirichletDomain);

    auto dCol_host = Kokkos::create_mirror_view(dirichletCols);
    Kokkos::deep_copy(dCol_host, dirichletCols);

    for (size_t col = 0; col < dCol_host.extent(0); ++col)
    {
      const auto isDirchletCol = col >= 4 and col <= 6;
      TEST_EQUALITY(dCol_host(col), isDirchletCol);
    }

  } //DetectDirichletColsAndDomains

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, ZeroDirichletRows, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);

    LocalOrdinal localRowToZero = 5;
    CreateDirchletRow<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A, localRowToZero);

    auto drows = Utils_Kokkos::DetectDirichletRows(*A);
    auto dcols = Utils_Kokkos::DetectDirichletCols(*A, drows);

    // Matrix version
    {
      const auto zeroVal = Scalar(0.35);
      Utils_Kokkos::ZeroDirichletRows(A, dcols, zeroVal);

      Teuchos::ArrayView<const LocalOrdinal> indices;
      Teuchos::ArrayView<const Scalar> values;
      A->getLocalRowView(localRowToZero, indices, values);

      for (int idx = 0; idx < indices.size(); ++idx)
      {
        TEST_EQUALITY(values[idx], zeroVal);
      }
    }

    // Multi-vector version
    {
      auto map = A->getMap();
      auto vector = Xpetra::MultiVectorFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(map, A->getGlobalNumCols());

      const auto zeroVal = Scalar(0.25);
      Utils_Kokkos::ZeroDirichletRows(vector, dcols, zeroVal);

      auto vecView = vector->getHostLocalView(Xpetra::Access::ReadOnly);

      for (size_t idx = 0; idx < vecView.extent(0); ++idx)
      {
        TEST_EQUALITY(vecView(localRowToZero, idx), zeroVal);
      }
    }

  } //ZeroDirichletRows

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, ZeroDirichletCols, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);

    const auto numCols = A->getColMap()->getNodeNumElements();
    const auto colToZero = 2;
    Kokkos::View<bool *, typename Node::device_type> dCols("", numCols);
    Kokkos::parallel_for(
        numCols, KOKKOS_LAMBDA(const int colIdx) {
          dCols(colIdx) = colIdx == colToZero;
        });

    const auto zeroVal = Scalar(0.25);
    Utils_Kokkos::ZeroDirichletCols(A, dCols, zeroVal);

    const auto localMatrix = A->getLocalMatrixHost();
    const auto numRows = A->getNodeNumRows();
    for (size_t row = 0; row < numRows; ++row)
    {
      auto rowView = localMatrix.row(row);
      auto length = rowView.length;

      for (int colID = 0; colID < length; colID++)
        if (rowView.colidx(colID) == colToZero)
        {
          TEST_EQUALITY(rowView.value(colID), zeroVal);
        }
    }

  } //ZeroDirichletCols

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, ApplyRowSumCriterion, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);

    const auto numRows = A->getNodeNumRows();
    Kokkos::View<bool *, typename Node::device_type> dRows("", numRows);

    Utils_Kokkos::ApplyRowSumCriterion(*A, Scalar(1.0), dRows);

    for (size_t idx = 0; idx < dRows.extent(0); ++idx)
    {
      TEST_EQUALITY(dRows(idx), false);
    }

    Utils_Kokkos::ApplyRowSumCriterion(*A, Scalar(-1.0), dRows);

    for (size_t idx = 0; idx < dRows.extent(0); ++idx)
    {
      TEST_EQUALITY(dRows(idx), true);
    }
  } //ApplyRowSumCriterion

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, MyOldScaleMatrix, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);
    auto B = MueLu_TestHelper_Factory::Build1DPoisson(100);

    Teuchos::ArrayRCP<Scalar> arr(100, Scalar(2.0));

    Utils_Kokkos::MyOldScaleMatrix(*A, arr, false);

    const auto localMatrixScaled = A->getLocalMatrixHost();
    const auto localMatrixOriginal = B->getLocalMatrixHost();
    const auto numRows = A->getNodeNumRows();
    for (size_t row = 0; row < numRows; ++row)
    {
      auto scaledRowView = localMatrixScaled.row(row);
      auto origRowView = localMatrixOriginal.row(row);
      auto length = scaledRowView.length;

      for (int colID = 0; colID < length; colID++)
      {
        TEST_EQUALITY(scaledRowView.value(colID), Scalar(2.0) * origRowView.value(colID));
      }
    }

  } //MyOldScaleMatrix

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities_kokkos, ApplyOAZToMatrixRows, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

    using TST = Teuchos::ScalarTraits<Scalar>;
    using Utils_Kokkos = MueLu::Utilities_kokkos<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MueLu_TestHelper_Factory = MueLuTests::TestHelpers_kokkos::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto A = MueLu_TestHelper_Factory::Build1DPoisson(100);
    Kokkos::View<bool *> dRowsIn("", A->getNodeNumRows());

    Kokkos::parallel_for(
        "", dRowsIn.extent(0), KOKKOS_LAMBDA(const int index) {
          dRowsIn(index) = index % 2;
        });

    Utils_Kokkos::ApplyOAZToMatrixRows(A, dRowsIn);

    auto dRowsOut = Utils_Kokkos::DetectDirichletRows(*A);
    for (size_t row = 0; row < dRowsOut.extent(0); ++row)
    {
      TEST_EQUALITY(dRowsOut(row), row % 2);
    }
  } //MyOldScaleMatrix

#define MUELU_ETI_GROUP(SC, LO, GO, NO)                                                                 \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, CuthillMcKee, SC, LO, GO, NO)                  \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, DetectDirichletRows, SC, LO, GO, NO)           \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, DetectDirichletCols, SC, LO, GO, NO)           \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, GetMatrixDiagonal, SC, LO, GO, NO)             \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, GetMatrixDiagonalInverse, SC, LO, GO, NO)      \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, GetMatrixOverlappedDiagonal, SC, LO, GO, NO)   \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, FindNonZeros, SC, LO, GO, NO)                  \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, DetectDirichletColsAndDomains, SC, LO, GO, NO) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, ZeroDirichletRows, SC, LO, GO, NO)             \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, ZeroDirichletCols, SC, LO, GO, NO)             \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, ApplyRowSumCriterion, SC, LO, GO, NO)          \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, MyOldScaleMatrix, SC, LO, GO, NO)              \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities_kokkos, ApplyOAZToMatrixRows, SC, LO, GO, NO)

#include <MueLu_ETI_4arg.hpp>

} //namespace MueLuTests
