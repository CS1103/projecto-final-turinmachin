//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "nn_interfaces.h"
#include <algorithm>
#include <cmath>

namespace utec::neural_network {
    template<typename T>
    class ReLU final : public ILayer<T> {
        Tensor<T, 2> input{0, 0};
    public:
        Tensor<T,2> forward(const Tensor<T,2>& z) override {
            input = z;
            return algebra::apply(z, [](const T t) { return std::max(static_cast<T>(0), t); });
        }

        Tensor<T,2> backward(const Tensor<T,2>& g) override {
            return algebra::apply(input, [](const T t) { return T(t > 0); }) * g;
        }
    };

    template<typename T>
    class Sigmoid final : public ILayer<T> {
        Tensor<T, 2> output{0, 0};
    public:
        Tensor<T,2> forward(const Tensor<T,2>& z) override {
            output = algebra::apply(z, [](const T t) { return 1/(1+std::exp(-t)); });
            return output;
        }

        Tensor<T,2> backward(const Tensor<T,2>& g) override {
            return algebra::apply(output, [](const T t ) { return t * (1- t); }) * g;
        }
    };

    template<typename T>
        class Tanh final : public ILayer<T> {
        Tensor<T, 2> output{0, 0};
    public:
        Tensor<T,2> forward(const Tensor<T,2>& z) override {
            output = algebra::apply(z, [](const T t) { return std::tanh(t); });
            return output;
        }

        Tensor<T,2> backward(const Tensor<T,2>& g) override {
            return algebra::apply(output, [](const T t ) { return 1 - t * t; }) * g;
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
