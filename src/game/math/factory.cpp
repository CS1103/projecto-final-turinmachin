#include "game/math/factory.h"
#include <memory>
#include "game/math/equation.h"

namespace math {

    AddEquationFactory::AddEquationFactory(std::mt19937& rng)
        : rng(rng) {}

    auto AddEquationFactory::create() -> std::unique_ptr<IEquation> {
        const int answer = dist_answer(rng);
        const int rhs = dist_rhs(rng);
        const int lhs = answer - rhs;

        return std::make_unique<AddEquation>(lhs, rhs);
    }

    SubtractEquationFactory::SubtractEquationFactory(std::mt19937& rng)
        : rng(rng) {}

    auto SubtractEquationFactory::create() -> std::unique_ptr<IEquation> {
        const int answer = dist_answer(rng);
        const int rhs = dist_rhs(rng);
        const int lhs = answer + rhs;

        return std::make_unique<SubtractEquation>(lhs, rhs);
    }

    DivideEquationFactory::DivideEquationFactory(std::mt19937& rng)
        : rng(rng) {}

    auto DivideEquationFactory::create() -> std::unique_ptr<IEquation> {
        const int answer = dist_answer(rng);
        const int rhs = dist_rhs(rng);
        const int lhs = answer * rhs;

        return std::make_unique<DivideEquation>(lhs, rhs);
    }

}  // namespace math
