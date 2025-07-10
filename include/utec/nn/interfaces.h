#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H

#include "utec/algebra/tensor.h"

namespace utec::neural_network {
    /**
     * @brief Identificador para los diferentes tipos de capas en la red neuronal.
     * Se emplea uint8_t (unsigned 8-bit int, valores de 0-255) para ahorrar memoria
     */
    enum class LayerId : uint8_t {
        ReLU = 0,
        Sigmoid = 1,
        Dense = 2,
        Softmax = 3,
        Kan = 4,
    };

    /**
     * @brief Interfaz para definir un optimizador (ej. SGD, Adam, ...).
     * Un optimizador se encarga de actualizar los parámetros entrenables
     * (como pesos y biases) en base a los gradientes.
     * @tparam T Tipo de dato usado para los cálculos (usualmente float o double).
     */
    template <typename T>
    struct IOptimizer {
        /// Destructor virtual.
        virtual ~IOptimizer() = default;

        /**
         * @brief Actualiza los parámetros del modelo usando los gradientes.
         * @param params Parámetros del modelo que se desean actualizar.
         * @param gradients Gradientes calculados con respecto a dichos parámetros.
         */
        virtual void update(algebra::Tensor<T, 2>& params,
                            const algebra::Tensor<T, 2>& gradients) = 0;

        /**
         * @brief Avanza el estado interno del optimizador si este lo permite.
         * Sólo aplica a aquellos optimizadores que llevan estados (ej. Adam).
         * Para otros (ej. SGD), no hace nada.
         */
        virtual void step() {}
    };

    /**
     * @brief Interfaz para una capa de la red neuronal.
     * Permite que distintas capas se conecten entre sí con polimorfismo.
     * @tparam T Tipo de dato usado en los tensores.
     */
    template <typename T>
    struct ILayer {
        /// Destructor virtual.
        virtual ~ILayer() = default;

        /**
         * @brief Propagación hacia adelante de la capa.
         * @param x Tensor2D input de la capa.
         * @return Salida producida dependiendo de lo que haga la capa.
         */
        virtual auto forward(const algebra::Tensor<T, 2>& x) -> algebra::Tensor<T, 2> = 0;

        /**
         * @brief Propagación hacia atrás de la capa.
         * @param gradients Gradiente de la pérdida respecto a la salida de esta capa.
         * @return Gradiente de la pérdida respecto a la entrada de esta capa.
         */
        virtual auto backward(const algebra::Tensor<T, 2>& gradients) -> algebra::Tensor<T, 2> = 0;

        /**
         * @brief Actualiza los parámetros internos de la capa (si tiene).
         * @param optimizer Instancia del optimizador a usar.
         * @note Si la capa no tiene parámetros entrenables (como ReLU o Softmax),
         * no necesita hacer nada en esta función.
         */
        virtual void update_params([[maybe_unused]] IOptimizer<T>& optimizer) {}

        /**
         * @brief Devuelve el tipo de la capa.
         * Sirve para serialización o reconstrucción de la red.
         * @return Identificador único de tipo LayerID.
         */
        [[nodiscard]] virtual auto id() const -> LayerId = 0;

        /**
         * @brief Guarda los parámetros internos de la capa en un flujo binario.
         * @param out Flujo de salida (por ejemplo, un archivo binario).
         * @note Capas sin parámetros (ReLu, Sigmoid, Softmax) pueden dejar esta función vacía.
         * Se usa principalmente para guardar el modelo entrenado.
         */
        virtual void save([[maybe_unused]] std::ostream& out) const {};
    };

    /**
     * @brief Interfaz para una función de pérdida (loss).
     * Se encarga de calcular qué tan mal lo hizo la red con respecto a los
     * resultados reales y obtener el gradiente necesario para ajustar los parámetros.
     * @tparam T Tipo de dato (usualmente float o double).
     * @tparam Dims Número de dimensiones del tensor de entrada (usualmente dos).
     */
    template <typename T, size_t Dims>
    struct ILoss {
        /// Destructor virtual.
        virtual ~ILoss() = default;

        /**
         * @brief Devuelve el valor escalar de la pérdida.
         * @return Valor numérico que representa el "error" actual de la red.
         */
        virtual auto loss() const -> T = 0;

        /**
         * @brief Devuelve el gradiente de la pérdida respecto a la predicción.
         * @return Tensor con el gradiente (mismo shape que la predicción).
         */
        virtual auto loss_gradient() const -> algebra::Tensor<T, Dims> = 0;
    };

}  // namespace utec::neural_network

#endif
