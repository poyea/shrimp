#include "cpp_package_boilerplate/package.hpp"

#include <gtest/gtest.h>

TEST(PackageSmokeTest, AddProducesExpectedValue) {
    EXPECT_EQ(cpp_package_boilerplate::add(20, 22), 42);
}

TEST(PackageSmokeTest, GreetIncludesName) {
    EXPECT_NE(cpp_package_boilerplate::greet("developer").find("developer"), std::string::npos);
}
