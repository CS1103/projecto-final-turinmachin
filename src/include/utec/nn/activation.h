#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include <algorithm>
#include <cmath>
#include "interfaces.h"

namespace utec::neural_network {
    template <typename T>
    class ReLU final : public ILayer<T> {
        algebra::Tensor<T, 2> input{0, 0};

    public:
        auto forward(const algebra::Tensor<T, 2>& z) -> algebra::Tensor<T, 2> override {
            input = z;
            return algebra::apply(z, [](const T t) { return std::max(static_cast<T>(0), t); });
        }

        auto backward(const algebra::Tensor<T, 2>& g) -> algebra::Tensor<T, 2> override {
            return algebra::apply(input, [](const T t) { return T(t > 0); }) * g;
        }

        [[nodiscard]] auto id() const -> LayerId override {
            return LayerId::ReLU;
        }
    };

    template <typename T>
    class Sigmoid final : public ILayer<T> {
        algebra::Tensor<T, 2> output{0, 0};

    public:
        auto forward(const algebra::Tensor<T, 2>& z) -> algebra::Tensor<T, 2> override {
            output = algebra::apply(z, [](const T t) { return 1 / (1 + std::exp(-t)); });
            return output;
        }

        auto backward(const algebra::Tensor<T, 2>& g) -> algebra::Tensor<T, 2> override {
            return algebra::apply(output, [](const T t) { return t * (1 - t); }) * g;
        }

        [[nodiscard]] auto id() const -> LayerId override {
            return LayerId::Sigmoid;
        }
    };

    template <typename T>
    class Softmax final : public ILayer<T> {
        algebra::Tensor<T, 2> output{0, 0};

    public:
        auto forward(const algebra::Tensor<T, 2>& z) -> algebra::Tensor<T, 2> override {
            output = algebra::Tensor<T, 2>(z.shape());

            for (std::size_t i = 0; i < z.shape()[0]; ++i) {
                // T max_val = *std::max_element(z.row(i).begin(), z.row(i).end());

                T sum_exp = 0;
                for (std::size_t j = 0; j < z.shape()[1]; ++j) {
                    output(i, j) = std::exp(z(i, j));
                    sum_exp += output(i, j);
                }

                for (std::size_t j = 0; j < z.shape()[1]; ++j) {
                    output(i, j) /= sum_exp;
                }
            }

            return output;
        }

        auto backward(const algebra::Tensor<T, 2>& g) -> algebra::Tensor<T, 2> override {
            algebra::Tensor<T, 2> grad(output.shape());

            for (std::size_t i = 0; i < output.shape()[0]; ++i) {
                for (std::size_t j = 0; j < output.shape()[1]; ++j) {
                    T sum = 0;
                    for (std::size_t k = 0; k < output.shape()[1]; ++k) {
                        const T delta = (j == k) ? 1 : 0;
                        sum += g(i, k) * output(i, j) * (delta - output(i, k));
                    }
                    grad(i, j) = sum;
                }
            }

            return grad;
        }

        [[nodiscard]] auto id() const -> LayerId override {
            return LayerId::Softmax;
        }
    };

}  // namespace utec::neural_network

#endif
