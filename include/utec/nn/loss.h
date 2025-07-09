#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include <cmath>

#include "interfaces.h"

namespace utec::neural_network {

    template <typename T>
    class MSELoss final : public ILoss<T, 2> {
        algebra::Tensor<T, 2> y_prediction;
        algebra::Tensor<T, 2> y_true;

    public:
        template <typename Prediction, typename Expected>
        MSELoss(Prediction&& y_prediction, Expected&& y_true)
            : y_prediction(std::forward<Prediction>(y_prediction)),
              y_true(std::forward<Expected>(y_true)) {
            if (y_prediction.shape() != y_true.shape()) {
                throw std::invalid_argument("algebra::Tensors have incompatible shapes");
            }
        }

        auto loss() const -> T override {
            T sum = 0;
            const size_t num_elements = y_true.size();
            for (size_t i = 0; i < num_elements; ++i) {
                T difference = y_prediction[i] - y_true[i];
                sum += difference * difference;
            }
            return sum / static_cast<T>(num_elements);
        }

        auto loss_gradient() const -> algebra::Tensor<T, 2> override {
            return T(2) / y_true.size() * (y_prediction - y_true);
        }
    };

    template <typename T>
    class BCELoss final : public ILoss<T, 2> {
        algebra::Tensor<T, 2> y_prediction;
        algebra::Tensor<T, 2> y_true;

    public:
        template <typename Prediction, typename Expected>
        BCELoss(Prediction&& y_prediction, Expected&& y_true)
            : y_prediction(std::forward<Prediction>(y_prediction)),
              y_true(std::forward<Expected>(y_true)) {
            if (y_prediction.shape() != y_true.shape()) {
                throw std::invalid_argument("algebra::Tensors have incompatible shapes");
            }
        }

        auto loss() const -> T override {
            T sum = 0;
            const size_t num_elements = y_true.size();
            for (size_t i = 0; i < num_elements; ++i) {
                sum += y_true[i] * std::log(y_prediction[i]) +
                       (1 - y_true[i]) * std::log(1 - y_prediction[i]);
            }
            return -sum / static_cast<T>(num_elements);
        }

        auto loss_gradient() const -> algebra::Tensor<T, 2> override {
            return -((y_true / y_prediction) -
                     ((static_cast<T>(1) - y_true) / (static_cast<T>(1) - y_prediction))) /
                   y_true.size();
        }
    };

    template <typename T>
    class CrossEntropyLoss final : public ILoss<T, 2> {
        algebra::Tensor<T, 2> y_prediction;
        algebra::Tensor<T, 2> y_true;
        T epsilon;

    public:
        template <typename Prediction, typename Expected>
        CrossEntropyLoss(Prediction&& y_prediction, Expected&& y_true, const T epsilon = 1e-7)
            : y_prediction(std::forward<Prediction>(y_prediction)),
              y_true(std::forward<Expected>(y_true)),
              epsilon(epsilon) {
            if (y_prediction.shape() != y_true.shape()) {
                throw std::invalid_argument("algebra::Tensors have incompatible shapes");
            }
        }

        auto loss() const -> T override {
            T sum = 0;
            const std::size_t num_samples = y_true.shape()[0];
            const std::size_t num_classes = y_true.shape()[1];

            for (std::size_t i = 0; i < num_samples; ++i) {
                for (std::size_t j = 0; j < num_classes; ++j) {
                    const T pred = std::clamp(y_prediction(i, j), epsilon, 1 - epsilon);
                    sum += y_true(i, j) * std::log(pred);
                }
            }

            return -sum / num_samples;
        }

        auto loss_gradient() const -> algebra::Tensor<T, 2> override {
            const std::size_t num_samples = y_true.shape()[0];
            algebra::Tensor<T, 2> grad = y_prediction - y_true;
            return grad / num_samples;
        }
    };

}  // namespace utec::neural_network

#endif
