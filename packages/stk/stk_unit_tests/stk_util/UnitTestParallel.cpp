// Copyright 2002 - 2008, 2010, 2011 National Technology Engineering
// Solutions of Sandia, LLC (NTESS). Under the terms of Contract
// DE-NA0003525 with NTESS, the U.S. Government retains certain rights
// in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
// 
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
// 
//     * Neither the name of NTESS nor the names of its contributors
//       may be used to endorse or promote products derived from this
//       software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

#include "gtest/gtest.h"                         // for Test, AssertionResult, SuiteApiResolver
#include "stk_util/parallel/Parallel.hpp"        // for MPI_COMM_WORLD, parallel_machine_rank
#include "stk_util/parallel/ParallelReduce.hpp"  // for all_write_string
#include "stk_util/parallel/mpi_filebuf.hpp"     // for add_aprepro_defines
#include <cstddef>                               // for size_t
#include <iostream>                              // for ostringstream, operator<<, flush, basic_...
#include <string>                                // for string

#if !defined(NOT_HAVE_STK_SEACASAPREPRO_LIB)
#include "aprepro.h"                             // for Aprepro
#endif

TEST(UnitTestParallel, testUnit)
{
  int mpi_rank = stk::parallel_machine_rank(MPI_COMM_WORLD);
  int mpi_size = stk::parallel_machine_size(MPI_COMM_WORLD);

  std::string s;
  std::ostringstream strout;

  std::cout << "all_write_string " << std::flush;
  
//  for (size_t i = 0; i < 250000; ++i) {
  for (size_t i = 0; i < 100; ++i) {
    if (mpi_rank == 0 && i%1000 == 0)
      std::cout << "." << std::flush;
    
    stk::all_write_string(MPI_COMM_WORLD, strout, s);
  }
  
  ASSERT_LT(mpi_rank, mpi_size);
}

#if !defined(NOT_HAVE_STK_SEACASAPREPRO_LIB)
TEST(UnitTestParallel, apreproDefines_singleSubstitution)
{
    const std::string defines = "a=b";

    SEAMS::Aprepro aprepro;
    stk::add_aprepro_defines(aprepro, defines);

    const std::string inputString = "{a}";

    std::string errorString;
    bool success = aprepro.parse_string(inputString, errorString);
    ASSERT_TRUE(success);

    ASSERT_EQ(aprepro.parsing_results().str(), "b");
}

TEST(UnitTestParallel, apreproDefines_twoSubstitutions)
{
    const std::string defines = "a=b, c=d";

    SEAMS::Aprepro aprepro;
    stk::add_aprepro_defines(aprepro, defines);

    const std::string inputString = "{a}={c}";

    std::string errorString;
    bool success = aprepro.parse_string(inputString, errorString);
    ASSERT_TRUE(success);

    ASSERT_EQ(aprepro.parsing_results().str(), "b=d");
}

TEST(UnitTestParallel, apreproDefines_numericSubstitution)
{
    const std::string defines = "a=1.23";

    SEAMS::Aprepro aprepro;
    stk::add_aprepro_defines(aprepro, defines);

    const std::string inputString = "{a}";

    std::string errorString;
    bool success = aprepro.parse_string(inputString, errorString);
    ASSERT_TRUE(success);

    ASSERT_EQ(aprepro.parsing_results().str(), "1.23");
}

TEST(UnitTestParallel, apreproDefines_badNumericSubstitution)
{
    const std::string defines = "a=1_23";

    SEAMS::Aprepro aprepro;
    stk::add_aprepro_defines(aprepro, defines);

    const std::string inputString = "{a}";

    std::string errorString;
    bool success = aprepro.parse_string(inputString, errorString);
    ASSERT_TRUE(success);

    ASSERT_EQ(aprepro.parsing_results().str(), "1_23");
}

TEST(UnitTestParallel, apreproDefines_mathSubstitution)
{
    const std::string defines = "a=1.23";

    SEAMS::Aprepro aprepro;
    stk::add_aprepro_defines(aprepro, defines);

    const std::string inputString = "{a*2}";

    std::string errorString;
    bool success = aprepro.parse_string(inputString, errorString);
    ASSERT_TRUE(success);

    ASSERT_EQ(aprepro.parsing_results().str(), "2.46");
}


TEST(UnitTestParallel, apreproDefines_badMathSubstitution)
{
    const std::string defines = "a=1_23";

    SEAMS::Aprepro aprepro;
    stk::add_aprepro_defines(aprepro, defines);

    const std::string inputString = "{a*2}";

    std::string errorString;
    bool success = aprepro.parse_string(inputString, errorString);
    ASSERT_TRUE(success);

    ASSERT_EQ(aprepro.parsing_results().str(), "");
}
#endif

