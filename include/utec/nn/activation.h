#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include <algorithm>
#include <cmath>
#include "interfaces.h"

    /**
     * @brief Capa de activación de Rectified Linear Unit (ReLU).
     * Los valores negativos del input se convierten en 0.
     * Los valores no negativos permanecen igual.
     * @tparam T Tipo de dato (usualmente float o double).
     */
namespace utec::neural_network {
    template <typename T>
    class ReLU final : public ILayer<T> {
        algebra::Tensor<T, 2> input{0, 0};

    public:
        /**
         * @brief Propagación hacia adelante aplicando la función ReLU a cada elemento.
         * @param z Tensor de entrada.
         * @return Tensor con valores negativos reemplazados por 0.
         * @complexity O(n*m),
         * donde n y m son las dimensiones del tensor.
         */
        auto forward(const algebra::Tensor<T, 2>& z) -> algebra::Tensor<T, 2> override {
            input = z;
            return z.apply([](const T t) { return std::max(static_cast<T>(0), t); });
        }

        /**
          * @brief Propagación hacia atrás del gradiente de ReLU.
          * Devuelve 1 si el valor de entrada fue positivo, 0 si fue negativo.
          * @param g Gradiente de la siguiente capa.
          * @return Gradiente respecto a la entrada de esta capa.
          * @complexity O(n*m*r), donde:
          *  n*m es el tamaño del tensor,
          *  r es el número de dimensiones del resultado (por broadcasting).
          *  En la práctica, si no hay broadcasting, se reduce a O(n*m).
         */
        auto backward(const algebra::Tensor<T, 2>& g) -> algebra::Tensor<T, 2> override {
            return input.apply([](const T t) { return T(t > 0); }) * g;
        }

        /**
         * @brief Identificador único de la capa ReLU.
         * @return LayerID::ReLU (o bien 0)
         * @complexity O(1).
         */
        [[nodiscard]] auto id() const -> LayerId override {
            return LayerId::ReLU;
        }
    };

    /**
     * @brief Capa de activación Sigmoid.
     * Convierte cada valor en el rango (0, 1) abierto usando la función logística.
     * No tiene parámetros entrenables.
     * @tparam T Tipo de dato (usualmente float o double).
     */
    template <typename T>
    class Sigmoid final : public ILayer<T> {
        /// Salida almacenada para el cálculo del gradiente durante la propagación hacia atrás.
        algebra::Tensor<T, 2> output{0, 0};

    public:
        /**
         * @brief Propagación hacia adelante aplicando sigmoide.
         * @param z Tensor de entrada.
         * @return Tensor con valores entre 0 y 1.
         * @complexity O(n*m), dado un tensor de entrada z de tamaño n*m.
         */
        auto forward(const algebra::Tensor<T, 2>& z) -> algebra::Tensor<T, 2> override {
            output = z.apply([](const T t) { return 1 / (1 + std::exp(-t)); });
            return output;
        }

        /**
         * @brief Propagación hacia atrás: gradiente de la función sigmoide.
         * @param g Gradiente de la siguiente capa.
         * @return Gradiente respecto a la entrada de esta capa.
         * @complexity O(n*m*r), donde:
         * n*m es el tamaño del tensor de salida,
         * r es el número de dimensiones del resultado (por broadcasting).
         * En la práctica, si no hay broadcasting, se reduce a O(n*m).
         */
        auto backward(const algebra::Tensor<T, 2>& g) -> algebra::Tensor<T, 2> override {
            return output.apply([](const T t) { return t * (1 - t); }) * g;
        }

        /**
         * @brief Identificador único de la capa Sigmoid.
         * @return LayerID::Sigmoid (o bien 1)
         * @complexity O(1).
         */
        [[nodiscard]] auto id() const -> LayerId override {
            return LayerId::Sigmoid;
        }
    };

    /**
     * @brief Capa de activación Softmax.
     * Convierte un vector de valores en probabilidades
     * las cualesal ser sumadas resultan en 1.
     * Se aplica por fila (una muestra a la vez).
     * No tiene parámetros entrenables.
     * @tparam T Tipo de dato (usualmente float o double).
     */
    template <typename T>
    class Softmax final : public ILayer<T> {
        /// Salida softmax para almacenar y usar en backpropagation.
        algebra::Tensor<T, 2> output{0, 0};

    public:
        /**
         * @brief Propagación hacia adelante: aplica Softmax por fila.
         * @param z Tensor de entrada.
         * @return Tensor con probabilidades (por fila).
         * @complexity O(n*m), dado el tensor z de tamaño n*m,
         * se recorren todas las filas una vez,
         * y por cada fila se recorren sus columnas dos veces no anidadas.
         */
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

        /**
         * @brief Propagación hacia atrás con el gradiente de Softmax.
         * Usa la derivada de Softmax con respecto a su entrada.
         * @param g Gradiente de la siguiente capa.
         * @return Gradiente respecto a la entrada de esta capa.
         * @complexity O(n*m*m), dado el tensor z de tamaño n*m,
         * se recorren todas las filas una vez,
         * y por cada fila se recorren sus columnas,
         * y por cada una de estas columnas se recorren las columnas otra vez.
         */
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

        /**
         * @brief Identificador único de la capa Softmax.
         * @return LayerID::Softmax (o bien 3)
         * @complexity O(1).
         */
        [[nodiscard]] auto id() const -> LayerId override {
            return LayerId::Softmax;
        }
    };

}  // namespace utec::neural_network

#endif
