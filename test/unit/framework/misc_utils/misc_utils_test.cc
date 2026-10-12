#include "framework/utils/utils.h"
#include <gmock/gmock.h>

using namespace opensn;

TEST(MiscUtilsTest, Progress)
{
  const unsigned int I = 4;
  const size_t N = 39;

  std::stringstream progress;
  for (size_t i = 0; i < N; ++i)
  {
    progress << PrintIterationProgress(i, N, I);
  }

  EXPECT_EQ(progress.str(), "  25.00  50.00  75.00 100.00");
}

TEST(MiscUtilsTest, StringOperationsHandleBoundaries)
{
  EXPECT_EQ(StringLTrim(" \t value \n"), "value \n");
  EXPECT_EQ(StringRTrim(" \t value \n"), " \t value");
  EXPECT_EQ(StringTrim(" \t value \n"), "value");
  EXPECT_EQ(StringTrim(" \t\n"), "");

  EXPECT_TRUE(StartsWith("transport", "trans"));
  EXPECT_TRUE(StartsWith("transport", ""));
  EXPECT_FALSE(StartsWith("transport", "sport"));
  EXPECT_FALSE(StartsWith("short", "longer"));

  EXPECT_THAT(StringSplit("alpha,beta,gamma", ","), testing::ElementsAre("alpha", "beta", "gamma"));
  EXPECT_THAT(StringSplit("single", ","), testing::ElementsAre("single"));
  EXPECT_EQ(StringUpToFirstReverse("path/to/file", "/"), "file");
  EXPECT_EQ(StringUpToFirstReverse("file", "/"), "file");
  EXPECT_EQ(LowerCase("MiXeD-123"), "mixed-123");
  EXPECT_EQ(UpperCase("MiXeD-123"), "MIXED-123");
}

TEST(MiscUtilsTest, MakeSubSetsCoversEveryItemContiguously)
{
  const auto subsets = MakeSubSets(7, 3);
  ASSERT_EQ(subsets.size(), 3);
  EXPECT_EQ(subsets[0].ss_begin, 0);
  EXPECT_EQ(subsets[0].ss_end, 2);
  EXPECT_EQ(subsets[0].ss_size, 3);
  EXPECT_EQ(subsets[1].ss_begin, 3);
  EXPECT_EQ(subsets[1].ss_end, 4);
  EXPECT_EQ(subsets[1].ss_size, 2);
  EXPECT_EQ(subsets[2].ss_begin, 5);
  EXPECT_EQ(subsets[2].ss_end, 6);
  EXPECT_EQ(subsets[2].ss_size, 2);

  const auto sparse_subsets = MakeSubSets(2, 4);
  ASSERT_EQ(sparse_subsets.size(), 4);
  EXPECT_EQ(sparse_subsets[0].ss_size, 1);
  EXPECT_EQ(sparse_subsets[1].ss_size, 1);
  EXPECT_EQ(sparse_subsets[2].ss_size, 0);
  EXPECT_EQ(sparse_subsets[3].ss_size, 0);
}

TEST(MiscUtilsTest, ReadableFileValidation)
{
  EXPECT_NO_THROW(AssertReadableFile(__FILE__));
  EXPECT_THROW(AssertReadableFile("file-that-does-not-exist"), std::logic_error);
}
