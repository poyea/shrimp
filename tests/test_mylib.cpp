#include "shrimp/package.hpp"

#include <gtest/gtest.h>

TEST(PackageSmokeTest, AddProducesExpectedValue) {
    EXPECT_EQ(poyea::shrimp::add(20, 22), 42);
}

TEST(PackageSmokeTest, GreetIncludesName) {
    EXPECT_NE(poyea::shrimp::greet("developer").find("developer"), std::string::npos);
}
