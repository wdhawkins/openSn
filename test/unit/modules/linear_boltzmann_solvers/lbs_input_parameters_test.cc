// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/parameters/parameter_block.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <stdexcept>
#include <string>

using namespace opensn;

TEST(LBSInputParametersTest, OptionsRejectUnknownParameter)
{
  auto params = LBSProblem::GetOptionsBlock();
  ParameterBlock options;
  options.AddParameter("wrong-parameter", true);

  try
  {
    params.AssignParameters(options);
    FAIL() << "Expected an unknown LBS option to be rejected";
  }
  catch (const std::invalid_argument& error)
  {
    EXPECT_THAT(std::string(error.what()), testing::HasSubstr("\"wrong-parameter\""));
  }
}

TEST(LBSInputParametersTest, OptionsRejectIncorrectParameterType)
{
  auto params = LBSProblem::GetOptionsBlock();
  ParameterBlock options;
  options.AddParameter("max_mpi_message_size", true);

  try
  {
    params.AssignParameters(options);
    FAIL() << "Expected an incorrectly typed LBS option to be rejected";
  }
  catch (const std::invalid_argument& error)
  {
    const std::string message(error.what());
    EXPECT_THAT(message, testing::HasSubstr("BOOLEAN"));
    EXPECT_THAT(message, testing::HasSubstr("\"max_mpi_message_size\""));
  }
}
