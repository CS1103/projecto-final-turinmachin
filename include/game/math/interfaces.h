#ifndef INCLUDE_GAME_MATH_INTERFACES_H
#define INCLUDE_GAME_MATH_INTERFACES_H

#include <memory>
#include <string>

namespace game::math {

    struct IEquation {
        virtual ~IEquation() = default;

        [[nodiscard]] virtual auto answer() const -> int = 0;

        [[nodiscard]] virtual auto display() const -> std::string = 0;

        [[nodiscard]] virtual auto display_solved() const -> std::string = 0;
    };

    struct IEquationFactory {
        virtual ~IEquationFactory() = default;

        [[nodiscard]] virtual auto create() -> std::unique_ptr<IEquation> = 0;
    };
}  // namespace game::math

#endif
