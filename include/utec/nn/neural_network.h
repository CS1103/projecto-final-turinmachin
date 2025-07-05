//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_optimizer.h"
#include "nn_dense.h"
#include <memory>
#include <random>

namespace utec::neural_network {
    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers;

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers.emplace_back(std::move(layer));
        }

        template <template <typename ...> class LossType, template <typename ...> class OptimizerType = SGD>
        void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, const size_t epochs, const size_t batch_size, T learning_rate) {
            OptimizerType<T> optimizer(learning_rate);
            const size_t num_samples = X.shape()[0];

            std::vector<size_t> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0);

            std::mt19937 rng(std::random_device{}());

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                std::ranges::shuffle(indices, rng);

                for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
                    size_t current_batch_size = std::min(batch_size, num_samples - batch_start);

                    Tensor<T, 2> batch_X({current_batch_size, X.shape()[1]});
                    Tensor<T, 2> batch_Y({current_batch_size, Y.shape()[1]});

                    for (size_t i = 0; i < current_batch_size; ++i) {
                        batch_X.set_row(i, X.row(indices[batch_start + i]));
                        batch_Y.set_row(i, Y.row(indices[batch_start + i]));
                    }

                    Tensor<T, 2> output = batch_X;
                    for (auto& layer : layers) {
                        output = layer->forward(output);
                    }

                    LossType<T> loss_function(output, batch_Y);
                    Tensor<T, 2> grad = loss_function.loss_gradient();

                    for (size_t i = layers.size(); i-- > 0;) {
                        grad = layers[i]->backward(grad);
                        if (auto* dense = dynamic_cast<Dense<T>*>(layers[i].get())) {
                            dense->update_params(optimizer);
                        }
                    }
                }
            }
        }

        Tensor<T,2> predict(const Tensor<T,2>& X) {
            Tensor<T,2> output = X;
            for (auto& layer : layers) {
                output = layer->forward(output);
            }
            return output;
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
