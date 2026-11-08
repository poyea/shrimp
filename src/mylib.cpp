#include "shrimp/package.hpp"

#include <string>

namespace poyea::shrimp {

auto greet(std::string_view name) -> std::string {
    return "hello, " + std::string(name) + " from shrimp";
}

} // namespace poyea::shrimp
