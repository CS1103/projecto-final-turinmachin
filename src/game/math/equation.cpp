#include "game/math/equation.h"
#include <format>

namespace math {

    AddEquation::AddEquation(const int lhs, const int rhs)
        : lhs(lhs),
          rhs(rhs) {}

    auto AddEquation::answer() const -> int {
        return lhs + rhs;
    }

    auto AddEquation::display() const -> std::string {
        return std::format("{} + {} = ?", lhs, rhs);
    }

    auto AddEquation::display_solved() const -> std::string {
        return std::format("{} + {} = {}", lhs, rhs, answer());
    }

    SubtractEquation::SubtractEquation(const int lhs, const int rhs)
        : lhs(lhs),
          rhs(rhs) {}

    auto SubtractEquation::answer() const -> int {
        return lhs - rhs;
    }

    auto SubtractEquation::display() const -> std::string {
        return std::format("{} - {} = ?", lhs, rhs);
    }

    auto SubtractEquation::display_solved() const -> std::string {
        return std::format("{} - {} = {}", lhs, rhs, answer());
    }

    DivideEquation::DivideEquation(const int lhs, const int rhs)
        : lhs(lhs),
          rhs(rhs) {}

    auto DivideEquation::answer() const -> int {
        return lhs / rhs;
    }

    auto DivideEquation::display() const -> std::string {
        return std::format("{} / {} = ?", lhs, rhs);
    }

    auto DivideEquation::display_solved() const -> std::string {
        return std::format("{} / {} = {}", lhs, rhs, answer());
    }

}  // namespace math
