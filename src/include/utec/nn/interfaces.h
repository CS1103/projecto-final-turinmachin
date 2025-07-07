#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H

#include "utec/algebra/tensor.h"

namespace utec::neural_network {

    enum class LayerId : uint8_t {
        ReLU = 0,
        Sigmoid = 1,
        Dense = 2,
        Softmax = 3,
        KAN = 4,
    };

    template <typename T>
    struct IOptimizer {
        virtual ~IOptimizer() = default;
        virtual void update(algebra::Tensor<T, 2>& params,
                            const algebra::Tensor<T, 2>& gradients) = 0;
        virtual void step() {}
    };

    template <typename T>
    struct ILayer {
        virtual ~ILayer() = default;
        virtual auto forward(const algebra::Tensor<T, 2>& x) -> algebra::Tensor<T, 2> = 0;
        virtual auto backward(const algebra::Tensor<T, 2>& gradients) -> algebra::Tensor<T, 2> = 0;

        virtual void update_params([[maybe_unused]] IOptimizer<T>& optimizer) {}

        [[nodiscard]] virtual auto id() const -> LayerId = 0;
        virtual void save([[maybe_unused]] std::ostream& out) const {};
    };

    template <typename T, size_t Dims>
    struct ILoss {
        virtual ~ILoss() = default;
        virtual auto loss() const -> T = 0;
        virtual auto loss_gradient() const -> algebra::Tensor<T, Dims> = 0;
    };

}  // namespace utec::neural_network

#endif
