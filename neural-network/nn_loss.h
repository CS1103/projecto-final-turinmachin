//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include <cmath>

#include "nn_interfaces.h"

namespace utec::neural_network {
    template<typename T>
    class MSELoss final: public ILoss<T, 2> {
        Tensor<T, 2> y_prediction;
        Tensor<T, 2> y_true;

    public:
        template<typename Prediction, typename Expected>
        MSELoss(Prediction&& y_prediction, Expected&& y_true): y_prediction(std::forward<Prediction>(y_prediction)), y_true(std::forward<Expected>(y_true)) {
            if (y_prediction.shape() != y_true.shape()) { throw std::invalid_argument("Tensors have incompatible shapes"); }
        }

        T loss() const override {
            T sum = 0;
            const size_t num_elements = y_true.get_size();
            for (size_t i = 0; i < num_elements; ++i) {
                T difference = y_prediction[i] - y_true[i];
                sum += difference * difference;
            }
            return sum / static_cast<T>(num_elements);
        }

        Tensor<T,2> loss_gradient() const override {
            return T(2) / y_true.get_size() * (y_prediction - y_true);
        }
    };

    template<typename T>
    class BCELoss final: public ILoss<T, 2> {
        Tensor<T, 2> y_prediction;
        Tensor<T, 2> y_true;
    public:
        template<typename Prediction, typename Expected>
        BCELoss(Prediction&& y_prediction, Expected&& y_true): y_prediction(std::forward<Prediction>(y_prediction)), y_true(std::forward<Expected>(y_true)) {
            if (y_prediction.shape() != y_true.shape()) { throw std::invalid_argument("Tensors have incompatible shapes"); }
        }

        T loss() const override {
            T sum = 0;
            const size_t num_elements = y_true.get_size();
            for (size_t i = 0; i < num_elements; ++i) {
                sum += y_true[i] * std::log(y_prediction[i]) + (1 - y_true[i]) * std::log(1 - y_prediction[i]);
            }
            return -sum / static_cast<T>(num_elements);
        }

        Tensor<T,2> loss_gradient() const override {
            return -(y_true / y_prediction - (static_cast<T>(1) - y_true) / (static_cast<T>(1) - y_prediction)) / y_true.get_size();
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
