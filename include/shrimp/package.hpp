#pragma once

#include <string>
#include <string_view>

namespace poyea::shrimp {

[[nodiscard]] constexpr auto add(int left, int right) noexcept -> int {
    return left + right;
}

[[nodiscard]] auto greet(std::string_view name) -> std::string;

} // namespace poyea::shrimp

#ifdef SHRIMP_HEADER_ONLY
namespace poyea::shrimp {

inline auto greet(std::string_view name) -> std::string {
    return "hello, " + std::string(name) + " from shrimp";
}

} // namespace poyea::shrimp
#endif
