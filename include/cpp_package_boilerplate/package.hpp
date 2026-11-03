#pragma once

#include <string>
#include <string_view>

namespace cpp_package_boilerplate {

[[nodiscard]] constexpr auto add(int left, int right) noexcept -> int {
    return left + right;
}

[[nodiscard]] auto greet(std::string_view name) -> std::string;

} // namespace cpp_package_boilerplate

#ifdef CPP_PACKAGE_BOILERPLATE_HEADER_ONLY
namespace cpp_package_boilerplate {

inline auto greet(std::string_view name) -> std::string {
    return "hello, " + std::string(name) + " from cpp-package-boilerplate";
}

} // namespace cpp_package_boilerplate
#endif
