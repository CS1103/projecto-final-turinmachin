#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include <cmath>
#include "interfaces.h"

namespace utec::neural_network {
    template <typename T>
    class SGD final : public IOptimizer<T> {
        T learning_rate;

    public:
        explicit SGD(T learning_rate = 0.01)
            : learning_rate(learning_rate) {}

        void update(algebra::Tensor<T, 2>& params, const algebra::Tensor<T, 2>& grads) override {
            params = params - grads * learning_rate;
        }
    };

    template <typename T>
    class Adam final : public IOptimizer<T> {
        T learning_rate;
        T beta1;
        T beta2;
        T epsilon;

        algebra::Tensor<T, 2> m;
        algebra::Tensor<T, 2> v;
        std::size_t t = 0;

    public:
        explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
            : learning_rate(learning_rate),
              beta1(beta1),
              beta2(beta2),
              epsilon(epsilon) {}

        void update(algebra::Tensor<T, 2>& params, const algebra::Tensor<T, 2>& grads) override {
            if (m.shape() != grads.shape()) {
                m.reshape(grads.shape());
                v.reshape(grads.shape());
            }

            step();
            m = beta1 * m + (T{1} - beta1) * grads;
            v = beta2 * v + (T{1} - beta2) * grads * grads;

            const algebra::Tensor<T, 2> m_hat = m / (T{1} - std::pow(beta1, t));
            const algebra::Tensor<T, 2> v_hat = v / (T{1} - std::pow(beta2, t));

            params -= m_hat * learning_rate /
                      (std::apply(v_hat, [](const T x) { return std::sqrt(x); }) + epsilon);
        }

        void step() override {
            t += 1;
        }
    };
}  // namespace utec::neural_network

#endif
