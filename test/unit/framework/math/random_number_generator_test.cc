// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/random_number_generation/random_number_generator.h"
#include "gtest/gtest.h"

using namespace opensn;

TEST(RandomNumberGeneratorTest, SeededGeneratorsProduceRepeatableInRangeSequences)
{
  RandomNumberGenerator first(12345);
  RandomNumberGenerator second(12345);

  for (int sample = 0; sample < 20; ++sample)
  {
    const double first_value = first.Rand();
    const double second_value = second.Rand();
    EXPECT_DOUBLE_EQ(first_value, second_value);
    EXPECT_GE(first_value, 0.0);
    EXPECT_LT(first_value, 1.0);
  }
}

TEST(RandomNumberGeneratorTest, DefaultGeneratorProducesValuesInDocumentedRange)
{
  RandomNumberGenerator generator;
  for (int sample = 0; sample < 20; ++sample)
  {
    const double value = generator.Rand();
    EXPECT_GE(value, 0.0);
    EXPECT_LT(value, 1.0);
  }
}
