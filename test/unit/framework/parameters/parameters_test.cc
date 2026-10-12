// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/utils/utils.h"
#include "framework/parameters/input_parameters.h"
#include "framework/data_types/allowable_range.h"
#include <gmock/gmock.h>

using namespace opensn;

namespace
{

InputParameters
GetInputParameters()
{
  InputParameters params;
  params.AddRequiredParameter<std::string>("name", "Name");
  return params;
}

} // namespace

TEST(ParametersTest, AssignAParameter)
{
  ParameterBlock name("name", "this_is_my_name");

  ParameterBlock input("params");
  input.AddParameter(name);

  auto params = GetInputParameters();
  params.AssignParameters(input);

  EXPECT_EQ(params.GetParamValue<std::string>("name"), "this_is_my_name");
}

TEST(ParametersTest, AssignNonexistentParameter)
{
  ParameterBlock non("non-existent", 42.);

  ParameterBlock input("params");
  input.AddParameter(non);

  auto params = GetInputParameters();
  EXPECT_THROW(params.AssignParameters(input), std::invalid_argument);
}

TEST(ParametersTest, AssignIncorrectlyTypedParameter)
{
  ParameterBlock name("name", 42.);

  ParameterBlock input("params");
  input.AddParameter(name);

  auto params = GetInputParameters();
  EXPECT_THROW(params.AssignParameters(input), std::invalid_argument);
}

TEST(ParametersTest, RequiredMissingAndRenamed)
{
  InputParameters params;
  params.SetObjectType("UnitTest");
  params.AddRequiredParameter<int>("count", "Count");

  ParameterBlock input_missing("params");
  EXPECT_THROW(params.AssignParameters(input_missing), std::invalid_argument);

  params.MarkParameterRenamed("count", "Use new_count instead");
  ParameterBlock input_with_old("params");
  input_with_old.AddParameter(ParameterBlock("count", 3));
  EXPECT_THROW(params.AssignParameters(input_with_old), std::invalid_argument);
}

TEST(ParametersTest, DeprecationAndConstraints)
{
  InputParameters params;
  params.SetObjectType("UnitTest");
  params.AddOptionalParameter<int>("mode", 1, "Mode");
  params.AddOptionalParameter<double>("scale", 1.0, "Scale");

  params.MarkParameterDeprecatedError("mode", "Deprecated mode");
  params.ConstrainParameterRange("scale", AllowableRangeLowHighLimit::New(0.0, 2.0));

  ParameterBlock input("params");
  input.AddParameter(ParameterBlock("mode", 2));
  input.AddParameter(ParameterBlock("scale", 3.0));

  EXPECT_THROW(params.AssignParameters(input), std::runtime_error);

  ParameterBlock input_bad_scale("params");
  input_bad_scale.AddParameter(ParameterBlock("scale", 3.0));
  EXPECT_THROW(params.AssignParameters(input_bad_scale), std::invalid_argument);
}

TEST(ParametersTest, ParameterBlockTypesValuesAndNumericArrayOrder)
{
  EXPECT_EQ(ParameterBlockTypeName(ParameterBlockType::BOOLEAN), "BOOLEAN");
  EXPECT_EQ(ParameterBlockTypeName(ParameterBlockType::FLOAT), "FLOAT");
  EXPECT_EQ(ParameterBlockTypeName(ParameterBlockType::STRING), "STRING");
  EXPECT_EQ(ParameterBlockTypeName(ParameterBlockType::INTEGER), "INTEGER");
  EXPECT_EQ(ParameterBlockTypeName(ParameterBlockType::ARRAY), "ARRAY");
  EXPECT_EQ(ParameterBlockTypeName(ParameterBlockType::BLOCK), "BLOCK");
  EXPECT_EQ(ParameterBlockTypeName(ParameterBlockType::USER_DATA), "USER_DATA");
  EXPECT_THROW(ParameterBlockTypeName(ParameterBlockType::INVALID_VALUE), std::logic_error);

  ParameterBlock block("root");
  block.AddParameter("enabled", true);
  block.AddParameter("iterations", 4);
  block.AddParameter("scale", 2.5);
  block.AddParameter("name", std::string("sample"));
  block.AddParameter("values", std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

  EXPECT_EQ(block.GetParamValue<bool>("enabled"), true);
  EXPECT_EQ(block.GetParamValue<int>("iterations"), 4);
  EXPECT_DOUBLE_EQ(block.GetParamValue<double>("scale"), 2.5);
  EXPECT_EQ(block.GetParamValue<std::string>("name"), "sample");
  EXPECT_THAT(block.GetParamVectorValue<int>("values"),
              testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11));
  EXPECT_TRUE(block.GetParam("enabled").IsScalar());
  EXPECT_FALSE(block.GetParam("values").IsScalar());
}

TEST(ParametersTest, ParameterBlockValidationAndLookupFailures)
{
  ParameterBlock block("root");
  block.SetErrorOriginScope("unit-test");
  block.AddParameter("value", 1);

  EXPECT_EQ(block.GetErrorOriginScope(), "unit-test");
  EXPECT_TRUE(block.Has("value"));
  EXPECT_FALSE(block.Has("missing"));
  EXPECT_THROW(block.AddParameter("value", 2), std::invalid_argument);
  EXPECT_THROW(block.GetParam("missing"), std::out_of_range);
  EXPECT_THROW(block.GetParam(2), std::out_of_range);
  EXPECT_THROW(block.GetParamValue<double>("value"), std::logic_error);
  EXPECT_NO_THROW(block.RequireBlockTypeIs(ParameterBlockType::BLOCK));
  EXPECT_THROW(block.RequireBlockTypeIs(ParameterBlockType::ARRAY), std::logic_error);
  EXPECT_NO_THROW(block.RequireParameter("value"));
  EXPECT_THROW(block.RequireParameter("missing"), std::logic_error);

  ParameterBlock mixed("mixed");
  mixed.AddParameter("0", 1);
  mixed.AddParameter("1", 2.0);
  EXPECT_THROW(mixed.ChangeToArray(), std::logic_error);
  EXPECT_THROW(block.GetVectorValue<int>(), std::logic_error);
  EXPECT_THROW(block.GetValue(), std::logic_error);
}

TEST(ParametersTest, ParameterBlockCopyMoveIterationAndDump)
{
  ParameterBlock original("root");
  original.AddParameter("beta", 2);
  original.AddParameter("alpha", 1);

  ParameterBlock copy(original);
  copy.GetParam("alpha") = ParameterBlock("alpha", 10);
  EXPECT_EQ(original.GetParamValue<int>("alpha"), 1);
  EXPECT_EQ(copy.GetParamValue<int>("alpha"), 10);

  ParameterBlock assigned("assigned");
  assigned = original;
  EXPECT_EQ(assigned.GetParamValue<int>("beta"), 2);

  ParameterBlock scalar("scalar", 3);
  scalar = original;
  EXPECT_FALSE(scalar.HasValue());
  EXPECT_EQ(scalar.GetType(), ParameterBlockType::BLOCK);

  ParameterBlock moved("moved");
  moved = std::move(assigned);
  EXPECT_EQ(moved.GetName(), "root");
  EXPECT_EQ(moved.GetNumParameters(), 2);

  std::vector<std::string> names;
  for (const auto& parameter : std::as_const(moved))
    names.push_back(parameter.GetName());
  EXPECT_THAT(names, testing::ElementsAre("alpha", "beta"));

  std::string text;
  moved.RecursiveDumpToString(text);
  EXPECT_THAT(text, testing::HasSubstr("alpha"));
  EXPECT_THAT(text, testing::HasSubstr("beta"));

  std::string json;
  moved.RecursiveDumpToJSON(json);
  EXPECT_THAT(json, testing::HasSubstr("\"alpha\""));
  EXPECT_THAT(json, testing::HasSubstr("\"beta\""));
}

TEST(ParametersTest, InputParameterBlocksArraysMergingAndMetadata)
{
  ParameterBlock default_block("unused");
  default_block.AddParameter("x", 1);

  InputParameters first;
  first.SetObjectType("First");
  first.AddOptionalParameterBlock("settings", default_block, "Nested settings");
  first.AddOptionalParameterArray<int>("values", {1, 2}, "Integer values");
  first.AddOptionalParameter<int>("mode", 1, "Mode");

  InputParameters second;
  second.AddRequiredParameter<std::string>("name", "Name");
  first += second;

  EXPECT_EQ(first.GetObjectType(), "First");
  EXPECT_EQ(first.GetParameterDocString("settings"), "Nested settings");
  EXPECT_THROW(first.GetParameterDocString("missing"), std::invalid_argument);

  ParameterBlock settings("settings");
  settings.AddParameter("x", 5);
  ParameterBlock supplied("input");
  supplied.AddParameter(settings);
  supplied.AddParameter("values", std::vector<int>{3, 4, 5});
  supplied.AddParameter("mode", std::string("legacy"));
  supplied.AddParameter("name", std::string("case"));

  first.SetParameterTypeMismatchAllowed("mode");
  first.AssignParameters(supplied);
  EXPECT_TRUE(first.IsParameterValid("settings"));
  EXPECT_TRUE(first.IsParameterValid("values"));
  EXPECT_TRUE(first.IsParameterValid("mode"));
  EXPECT_FALSE(first.IsParameterValid("missing"));
  EXPECT_EQ(first.GetParam("settings").GetParamValue<int>("x"), 5);
  EXPECT_THAT(first.GetParamVectorValue<int>("values"), testing::ElementsAre(3, 4, 5));
  EXPECT_EQ(first.GetParamValue<std::string>("mode"), "legacy");
  EXPECT_EQ(first.GetParametersAtAssignment().GetParamValue<std::string>("name"), "case");
}

TEST(ParametersTest, InputParameterTagOperationsRejectUnknownOrCompositeParameters)
{
  InputParameters params;
  params.AddOptionalParameter<int>("value", 1, "Value");
  params.AddOptionalParameterBlock("block", ParameterBlock("default"), "Block");

  EXPECT_THROW(params.MarkParameterDeprecatedWarning("missing"), std::logic_error);
  EXPECT_THROW(params.MarkParameterDeprecatedError("missing"), std::logic_error);
  EXPECT_THROW(params.MarkParameterRenamed("missing", "renamed"), std::logic_error);
  EXPECT_THROW(params.ConstrainParameterRange("missing", AllowableRangeLowLimit::New(0.0)),
               std::logic_error);
  EXPECT_THROW(params.ConstrainParameterRange("block", AllowableRangeLowLimit::New(0.0)),
               std::invalid_argument);
  EXPECT_THROW(params.SetParameterTypeMismatchAllowed("missing"), std::invalid_argument);
}
