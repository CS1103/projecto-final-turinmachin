//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"

namespace utec::neural_network {
    template<typename T>
    class Dense final : public ILayer<T> {
        Tensor<T, 2> weights;
        Tensor<T, 2> biases;
        Tensor<T, 2> activations;

        Tensor<T, 2> gradient_weights;
        Tensor<T, 2> gradient_biases;
    public:
        Dense(const size_t in_f, const size_t out_f): weights(std::array{in_f, out_f}), biases(std::array<size_t, 2>{1, out_f}), activations(std::array<size_t, 2> {1, out_f}), gradient_weights(std::array{in_f, out_f}), gradient_biases(std::array<size_t, 2> {1, out_f}) {}

        template<typename InitWFun, typename InitBFun>
        Dense(const size_t in_f, const size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun): Dense(in_f, out_f) {
            init_w_fun(weights);
            init_b_fun(biases);
        }

        Tensor<T,2> forward(const Tensor<T,2>& x) override {
            activations = x;
            return algebra::matrix_product(x, weights) + biases;
        }

        Tensor<T,2> backward(const Tensor<T,2>& dZ) override {
            gradient_weights = algebra::matrix_product(algebra::transpose_2d(activations), dZ);
            auto dZ_shape = dZ.shape();
            gradient_biases.fill(static_cast<T>(0));
            for (size_t i = 0; i < dZ_shape[0]; i++) {
                for (size_t j = 0; j < dZ_shape[1]; j++) {
                    gradient_biases(0, j) += dZ(i, j);
                }
            }
            return algebra::matrix_product(dZ, algebra::transpose_2d(weights));
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(weights, gradient_weights);
            optimizer.update(biases, gradient_biases);
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
