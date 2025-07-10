#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include "interfaces.h"
#include "optimizer.h"
#include "utec/nn/layer_registry.h"
#include "utec/utils/serialization.h"

namespace utec::neural_network {
    /// Versión actual del formato .pp20 (Profe Pónganos 20).
    constexpr std::uint8_t FORMAT_CURRENT_VERSION = 1;

    /**
     * @brief Clase que representa una red neuronal completamente conectada.
     * @tparam T Tipo de dato para los pesos y cálculos (float, double, etc.).
     */
    template <typename T>
    class NeuralNetwork {
        /// Capas que componen la red.
        std::vector<std::unique_ptr<ILayer<T>>> layers;

    public:
        /// Constructor por defecto.
        NeuralNetwork() {
            register_all_layers<T>();
        }

        /**
         * @brief Agrega una nueva capa a la red.
         * @tparam L Tipo de capa (e.g., Dense, ReLU).
         * @tparam Args Tipos de argumentos para el constructor de la capa.
         * @param args Argumentos para inicializar la capa.
         * @complexity O(1).
         */
        template <typename L, typename... Args>
        void add_layer(Args&&... args) {
            layers.emplace_back(std::make_unique<L>(std::forward<Args>(args)...));
        }

        /**
         * @brief Entrena la red neuronal usando descenso por lotes.
         * @tparam LossType Tipo de función de pérdida (ej. MSELoss, BCELoss, ...).
         * @tparam OptimizerType Tipo de optimizador (ej. SGD, Adam, ...). Por defecto es SGD.
         * @param x Datos de entrada.
         * @param y Etiquetas esperadas.
         * @param epochs Número de épocas de entrenamiento.
         * @param batch_size Tamaño del batch.
         * @param learning_rate Tasa de aprendizaje.
         * @param rng Generador aleatorio para mezclar los datos.
         * @complexity O(e*(n/b)*L*(f+b+u))),
         * donde:
         * - e: número de épocas
         * - n: número total de muestras
         * - b: batch size
         * - L: número de capas
         * - f: costo forward de una capa
         * - b: costo backward de una capa
         * - u: costo de actualización de parámetros por capa
         */
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
                    std::cout << "Epoch " << epoch << '\n';
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

        /**
         * @brief Realiza una predicción sobre un conjunto de datos.
         * @param X Datos de entrada.
         * @return Tensor con la predicción de salida.
         * @complexity O(L*f), donde L es el número de capas y f es el costo del forward.
         */
        auto predict(const algebra::Tensor<T, 2>& X) -> algebra::Tensor<T, 2> {
            algebra::Tensor<T, 2> output = X;
            for (auto& layer : layers) {
                output = layer->forward(output);
            }
            return output;
        }

        /**
         * @brief Guarda el modelo en un flujo de salida binario.
         * @param out Flujo de salida donde se guarda la red.
         * @complexity O(L*s), donde s es el tamaño serializado de cada capa.
         */
        void save(std::ostream& out) const {
            out.put(FORMAT_CURRENT_VERSION);
            out.put(static_cast<std::uint8_t>(sizeof(T)));

            serialization::write_numeric(out, static_cast<std::uint64_t>(layers.size()));

            for (const auto& layer : layers) {
                out.put(static_cast<std::uint8_t>(layer->id()));
                layer->save(out);
            }
        }

        /**
         * @brief Carga una red neuronal desde un flujo de entrada binario.
         * @param in Flujo de entrada desde el cual se carga la red.
         * @return Instancia de NeuralNetwork cargada con sus capas (y pesos si es que
         * la capa lo permite).
         * @complexity O(L*s), donde L es el número de capas y s el tamaño serializado de cada una.
         */
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
