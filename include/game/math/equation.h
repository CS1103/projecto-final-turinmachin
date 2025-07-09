#ifndef INCLUDE_GAME_MATH_EQUATION_H
#define INCLUDE_GAME_MATH_EQUATION_H

#include <string>
#include "game/math/interfaces.h"

namespace game::math {

    class AddEquation final : public IEquation {
        int lhs;
        int rhs;

    public:
        AddEquation(int lhs, int rhs);

        [[nodiscard]] auto answer() const -> int override;

        [[nodiscard]] auto display() const -> std::string override;

        [[nodiscard]] auto display_solved() const -> std::string override;
    };

    class SubtractEquation final : public IEquation {
        int lhs;
        int rhs;

    public:
        SubtractEquation(int lhs, int rhs);

        [[nodiscard]] auto answer() const -> int override;

        [[nodiscard]] auto display() const -> std::string override;

        [[nodiscard]] auto display_solved() const -> std::string override;
    };

    class DivideEquation final : public IEquation {
        int lhs;
        int rhs;

    public:
        DivideEquation(int lhs, int rhs);

        [[nodiscard]] auto answer() const -> int override;

        [[nodiscard]] auto display() const -> std::string override;

        [[nodiscard]] auto display_solved() const -> std::string override;
    };

}  // namespace game::math

#endif
