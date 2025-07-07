#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "interfaces.h"
#include "utec/utils/serialization.h"

namespace utec::neural_network {

    /**
     * @brief Capa totalmente conectada (o Dense Layer) de una red neuronal.
     * Cada neurona de esta capa se conecta con todas las salidas de la capa anterior.
     * Esta capa tiene pesos y biases como parámetros entrenables.
     * @tparam T Tipo de dato (usualmente float o double).
     */
    template <typename T>
    class Dense final : public ILayer<T> {
        /// Pesos del modelo, tamaño: [input_features x output_features]
        algebra::Tensor<T, 2> weights;

        /// Biases del modelo, tamaño: [1 x output_features]
        algebra::Tensor<T, 2> biases;

        /// Entrada almacenada para usar durante la propagación hacia atrás
        algebra::Tensor<T, 2> activations;

        /// Gradientes calculados para actualizar los pesos
        algebra::Tensor<T, 2> gradient_weights;

        /// Gradientes calculados para actualizar los biases
        algebra::Tensor<T, 2> gradient_biases;

    public:
        /**
         * @brief Constructor que inicializa los tensores internos con ceros.
         * @param in_f Número de neuronas de entrada.
         * @param out_f Número de neuronas de salida.
         * @complexity O(1), inicialización directa.
         */
        Dense(const size_t in_f, const size_t out_f)
            : weights(std::array{in_f, out_f}),
              biases(std::array<size_t, 2>{1, out_f}),
              activations(std::array<size_t, 2>{1, out_f}),
              gradient_weights(std::array{in_f, out_f}),
              gradient_biases(std::array<size_t, 2>{1, out_f}) {}

        /**
         * @brief Constructor que permite inicializar pesos y biases con funciones personalizadas.
         * @param in_f Número de neuronas de entrada.
         * @param out_f Número de neuronas de salida.
         * @param init_w_fun Función para inicializar los pesos.
         * @param init_b_fun Función para inicializar los biases.
         * @complexity O(n*m),
         * donde n = in_f, m = out_f (por recorrer los tensores).
         */
        Dense(const size_t in_f, const size_t out_f, auto init_w_fun, auto init_b_fun)
            : Dense(in_f, out_f) {
            init_w_fun(weights);
            init_b_fun(biases);
        }

        /**
         * @brief Propagación hacia adelante de la capa.
         * Calcula: output = x * weights + biases
         * @param x Tensor de entrada.
         * @return Tensor de salida de la capa.
         * @complexity O(b*n*m), donde:
         * b es el número de muestras (filas de x),
         * n es input_features, y
         * m es output_features.
         * Se debe a la multiplicación matricial (x * weights) y la suma con biases.
         */
        auto forward(const algebra::Tensor<T, 2>& x) -> algebra::Tensor<T, 2> override {
            activations = x;
            return algebra::matrix_product(x, weights) + biases;
        }

        /**
         * @brief Propagación hacia atrás (backpropagation).
         * Calcula los gradientes con respecto a pesos, biases y entrada.
         * @param dZ Gradiente de la pérdida respecto a la salida de esta capa.
         * @return Gradiente de la pérdida respecto a la entrada de esta capa.
         * @complexity O(b*n*m), por
         * multiplicación activaciones^T * dZ → O(b*n*m) +
         * suma fila a fila de dZ para biases → O(b*m) +
         * multiplicación dZ * weights^t → O(b*m*n)
         */
        auto backward(const algebra::Tensor<T, 2>& dZ) -> algebra::Tensor<T, 2> override {
            gradient_weights = algebra::matrix_product(algebra::transpose_2d(activations), dZ);
            auto dZ_shape = dZ.shape();
            gradient_biases.fill(T(0));
            for (size_t i = 0; i < dZ_shape[0]; i++) {
                for (size_t j = 0; j < dZ_shape[1]; j++) {
                    gradient_biases(0, j) += dZ(i, j);
                }
            }
            return algebra::matrix_product(dZ, algebra::transpose_2d(weights));
        }

        /**
         * @brief Actualiza los parámetros (pesos y biases) usando un optimizador.
         * @param optimizer Referencia al optimizador (SGD, Adam, etc.).
         * @complexity O(n*m), donde n*m es el tamaño total de parámetros a actualizar.
         * Depende de la implementación del optimizador.
         */
        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(weights, gradient_weights);
            optimizer.update(biases, gradient_biases);
        }

        /**
         * @brief Devuelve el tipo identificador de esta capa.
         * @return LayerID::Dense (o bien 2)
         * @complexity O(1).
         */
        [[nodiscard]] auto id() const -> LayerID override {
            return LayerID::Dense;
        }

        void save(std::ostream& out) const override {
            serialization::write_numeric<std::uint64_t>(out, weights.shape()[0]);
            serialization::write_numeric<std::uint64_t>(out, weights.shape()[1]);

            for (const auto& x : weights) {
                serialization::write_numeric<T>(out, x);
            }

            for (const auto& x : biases) {
                serialization::write_numeric<T>(out, x);
            }
        }

        static auto load(std::istream& in) -> Dense<T> {
            const auto in_feats = serialization::read_numeric<std::uint64_t>(in);
            const auto out_feats = serialization::read_numeric<std::uint64_t>(in);

            Dense<T> result(in_feats, out_feats);

            for (auto& x : result.weights) {
                x = serialization::read_numeric<T>(in);
            }

            for (auto& x : result.biases) {
                x = serialization::read_numeric<T>(in);
            }

            return result;
        }
    };
}  // namespace utec::neural_network

#endif
