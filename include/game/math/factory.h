#ifndef INCLUDE_GAME_MATH_FACTORY_H
#define INCLUDE_GAME_MATH_FACTORY_H

#include <memory>
#include <random>
#include "game/math/interfaces.h"

namespace game::math {

    class AddEquationFactory final : public IEquationFactory {
        std::mt19937& rng;
        std::uniform_int_distribution<int> dist_answer{0, 9};
        std::uniform_int_distribution<int> dist_rhs{0, 11};

    public:
        explicit AddEquationFactory(std::mt19937& rng);

        [[nodiscard]] auto create() -> std::unique_ptr<IEquation> override;
    };

    class SubtractEquationFactory final : public IEquationFactory {
        std::mt19937& rng;
        std::uniform_int_distribution<int> dist_answer{0, 9};
        std::uniform_int_distribution<int> dist_rhs{0, 11};

    public:
        explicit SubtractEquationFactory(std::mt19937& rng);

        [[nodiscard]] auto create() -> std::unique_ptr<IEquation> override;
    };

    class DivideEquationFactory final : public IEquationFactory {
        std::mt19937& rng;
        std::uniform_int_distribution<int> dist_answer{0, 9};
        std::uniform_int_distribution<int> dist_rhs{1, 8};

    public:
        explicit DivideEquationFactory(std::mt19937& rng);

        [[nodiscard]] auto create() -> std::unique_ptr<IEquation> override;
    };

}  // namespace game::math

#endif
