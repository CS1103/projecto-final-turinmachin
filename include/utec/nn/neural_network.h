#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include <istream>
#include <memory>
#include <numeric>
#include <print>
#include <random>
#include "interfaces.h"
#include "optimizer.h"
#include "utec/nn/layer_registry.h"
#include "utec/utils/serialization.h"

namespace utec::neural_network {
    constexpr std::uint8_t FORMAT_CURRENT_VERSION = 1;

    template <typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers;

    public:
        NeuralNetwork() {
            register_all_layers<T>();
        }

        template <typename L, typename... Args>
        void add_layer(Args&&... args) {
            layers.emplace_back(std::make_unique<L>(std::forward<Args>(args)...));
        }

        template <template <typename...> class LossType,
                  template <typename...> class OptimizerType = SGD>
        void train(const algebra::Tensor<T, 2>& x,
                   const algebra::Tensor<T, 2>& y,
                   const size_t epochs,
                   const size_t batch_size,
                   T learning_rate,
                   std::mt19937& rng) {
            OptimizerType<T> optimizer(learning_rate);
            const size_t num_samples = x.shape()[0];

            std::vector<size_t> indices(num_samples);
            std::ranges::iota(indices, 0);

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                if (epoch % 100 == 0) {
                    std::println("Epoch {}", epoch);
                }

                std::ranges::shuffle(indices, rng);

                for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
                    size_t current_batch_size = std::min(batch_size, num_samples - batch_start);

                    algebra::Tensor<T, 2> batch_x(current_batch_size, x.shape()[1]);
                    algebra::Tensor<T, 2> batch_y(current_batch_size, y.shape()[1]);

                    for (size_t i = 0; i < current_batch_size; ++i) {
                        batch_x.set_row(i, x.row(indices[batch_start + i]));
                        batch_y.set_row(i, y.row(indices[batch_start + i]));
                    }

                    algebra::Tensor<T, 2> output = batch_x;
                    for (auto& layer : layers) {
                        output = layer->forward(output);
                    }

                    LossType<T> loss_function(output, batch_y);
                    algebra::Tensor<T, 2> grad = loss_function.loss_gradient();

                    for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer) {
                        grad = (*layer)->backward(grad);
                        (*layer)->update_params(optimizer);
                    }
                }
            }
        }

        auto predict(const algebra::Tensor<T, 2>& X) -> algebra::Tensor<T, 2> {
            algebra::Tensor<T, 2> output = X;
            for (auto& layer : layers) {
                output = layer->forward(output);
            }
            return output;
        }

        void save(std::ostream& out) const {
            out.put(FORMAT_CURRENT_VERSION);
            out.put(static_cast<std::uint8_t>(sizeof(T)));

            serialization::write_numeric(out, static_cast<std::uint64_t>(layers.size()));

            for (const auto& layer : layers) {
                out.put(static_cast<std::uint8_t>(layer->id()));
                layer->save(out);
            }
        }

        static auto load(std::istream& in) -> NeuralNetwork<T> {
            const int version = in.get();
            if (version != FORMAT_CURRENT_VERSION) {
                throw std::runtime_error("Invalid file format version: " + std::to_string(version));
            }

            const std::size_t t_size = in.get();
            if (t_size != sizeof(T)) {
                throw std::runtime_error(
                    "Stored data size does not match this platform's data size.");
            }

            const auto layers_size = serialization::read_numeric<std::uint64_t>(in);

            NeuralNetwork<T> net;
            net.layers.reserve(layers_size);

            for (std::size_t i = 0; i < layers_size; ++i) {
                const int id_raw = in.get();
                const auto id = static_cast<LayerId>(id_raw);

                net.layers.push_back(LayerRegistry<T>::create(id, in));
            }

            return net;
        }
    };
}  // namespace utec::neural_network

#endif
