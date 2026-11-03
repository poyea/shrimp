#include "cpp_package_boilerplate/package.hpp"

#include <string>

namespace cpp_package_boilerplate {

auto greet(std::string_view name) -> std::string {
    return "hello, " + std::string(name) + " from cpp-package-boilerplate";
}

} // namespace cpp_package_boilerplate
