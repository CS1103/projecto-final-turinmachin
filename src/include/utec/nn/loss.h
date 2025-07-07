#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include <cmath>

#include "interfaces.h"

namespace utec::neural_network {

    /**
     * @brief Función de pérdida MSE (Error Cuadrático Medio).
     * Mide el promedio de los cuadrados de las diferencias entre las predicciones y los valores reales.
     * Es común en tareas de regresión.
     *
     * @tparam T Tipo de dato (float, double, etc.).
     */
    template <typename T>
    class MSELoss final : public ILoss<T, 2> {
        /// Predicciones del modelo
        algebra::Tensor<T, 2> y_prediction;
        /// Valores reales esperados
        algebra::Tensor<T, 2> y_true;

    public:
        /**
         * @brief Constructor que recibe predicciones y valores reales.
         * @param y_prediction Tensores con las predicciones del modelo.
         * @param y_true Tensores con los valores verdaderos (etiquetas).
         * @throws std::invalid_argument si las dimensiones no coinciden.
         */
        template <typename Prediction, typename Expected>
        MSELoss(Prediction&& y_prediction, Expected&& y_true)
            : y_prediction(std::forward<Prediction>(y_prediction)),
              y_true(std::forward<Expected>(y_true)) {
            if (y_prediction.shape() != y_true.shape()) {
                throw std::invalid_argument("algebra::Tensors have incompatible shapes");
            }
        }

        /**
         * @brief Devuelve el valor de la pérdida MSE.
         * @return Escalar con el valor medio del error cuadrático.
         */
        auto loss() const -> T override {
            T sum = 0;
            const size_t num_elements = y_true.size();
            for (size_t i = 0; i < num_elements; ++i) {
                T difference = y_prediction[i] - y_true[i];
                sum += difference * difference;
            }
            return sum / static_cast<T>(num_elements);
        }

        /**
         * @brief Gradiente de la pérdida con respecto a las predicciones.
         * @return Tensor con el mismo shape que y_prediction.
         */
        auto loss_gradient() const -> algebra::Tensor<T, 2> override {
            return T(2) / y_true.size() * (y_prediction - y_true);
        }
    };

    /**
     * @brief Función de pérdida Binary Cross Entropy (entropía cruzada binaria).
     * Usada típicamente en clasificación binaria.
     * @tparam T Tipo de dato (float, double, etc.).
     */
    template <typename T>
    class BCELoss final : public ILoss<T, 2> {
        algebra::Tensor<T, 2> y_prediction;
        algebra::Tensor<T, 2> y_true;

    public:
        /**
         * @brief Constructor con predicciones y etiquetas verdaderas.
         * @param y_prediction Tensores con probabilidades predichas.
         * @param y_true Tensores con valores binarios verdaderos.
         * @throws std::invalid_argument si las dimensiones no coinciden.
         */
        template <typename Prediction, typename Expected>
        BCELoss(Prediction&& y_prediction, Expected&& y_true)
            : y_prediction(std::forward<Prediction>(y_prediction)),
              y_true(std::forward<Expected>(y_true)) {
            if (y_prediction.shape() != y_true.shape()) {
                throw std::invalid_argument("algebra::Tensors have incompatible shapes");
            }
        }

        /**
         * @brief Devuelve el valor de la pérdida BCE.
         * @return Valor escalar negativo de la entropía cruzada binaria.
         */
        auto loss() const -> T override {
            T sum = 0;
            const size_t num_elements = y_true.size();
            for (size_t i = 0; i < num_elements; ++i) {
                sum += y_true[i] * std::log(y_prediction[i]) +
                       (1 - y_true[i]) * std::log(1 - y_prediction[i]);
            }
            return -sum / static_cast<T>(num_elements);
        }

        /**
         * @brief Gradiente de la pérdida BCE con respecto a las predicciones.
         * @return Tensor del mismo tamaño que y_prediction.
         */
        auto loss_gradient() const -> algebra::Tensor<T, 2> override {
            return -((y_true / y_prediction) -
                     ((static_cast<T>(1) - y_true) / (static_cast<T>(1) - y_prediction))) /
                   y_true.size();
        }
    };

    /**
     * @brief Función de pérdida Cross Entropy para clasificación multiclase.
     * Requiere que las etiquetas verdaderas estén codificadas one-hot.
     * @tparam T Tipo de dato (float, double, etc.).
     */
    template <typename T>
    class CrossEntropyLoss final : public ILoss<T, 2> {
        algebra::Tensor<T, 2> y_prediction;
        algebra::Tensor<T, 2> y_true;

    public:
        /**
         * @brief Constructor que recibe tensores de predicciones y etiquetas reales.
         * @param y_prediction Tensores con probabilidades predichas.
         * @param y_true Tensores one-hot con etiquetas verdaderas.
         * @throws std::invalid_argument si las dimensiones no coinciden.
         */
        template <typename Prediction, typename Expected>
        CrossEntropyLoss(Prediction&& y_prediction, Expected&& y_true)
            : y_prediction(std::forward<Prediction>(y_prediction)),
              y_true(std::forward<Expected>(y_true)) {
            if (y_prediction.shape() != y_true.shape()) {
                throw std::invalid_argument("algebra::Tensors have incompatible shapes");
            }
        }

        /**
         * @brief Valor de la pérdida Cross Entropy.
         * Aplica logaritmo y protección contra valores extremos.
         * @return Promedio negativo del logaritmo de las probabilidades verdaderas.
         */
        auto loss() const -> T override {
            T sum = 0;
            const std::size_t num_samples = y_true.shape()[0];
            const std::size_t num_classes = y_true.shape()[1];

            for (std::size_t i = 0; i < num_samples; ++i) {
                for (std::size_t j = 0; j < num_classes; ++j) {
                    const T pred = std::clamp(y_prediction(i, j), T(1e-7), T(1.0 - 1e-7));
                    sum += y_true(i, j) * std::log(pred);
                }
            }

            return -sum / num_samples;
        }

        /**
         * @brief Gradiente de la pérdida Cross Entropy.
         * Simplemente calcula la diferencia entre predicción y etiqueta.
         * @return Tensor del mismo shape que la entrada.
         */
        auto loss_gradient() const -> algebra::Tensor<T, 2> override {
            const std::size_t num_samples = y_true.shape()[0];
            algebra::Tensor<T, 2> grad = y_prediction - y_true;
            return grad / num_samples;
        }
    };

}  // namespace utec::neural_network

#endif
