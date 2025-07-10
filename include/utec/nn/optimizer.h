#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include <cmath>
#include "interfaces.h"

namespace utec::neural_network {
    /**
     * @brief Optimizador Stochastic Gradient Descent (SGD).
     * Actualiza los parámetros en dirección opuesta al gradiente multiplicado por el learning rate.
     * @tparam T Tipo de dato (usualmente float o double).
     */
    template <typename T>
    class SGD final : public IOptimizer<T> {
        /// Tasa de aprendizaje del optimizador.
        T learning_rate;

    public:
        /**
         * @brief Constructor del optimizador.
         * @param learning_rate Valor de la tasa de aprendizaje (por defecto 0.01).
         * @complexity O(1).
         */
        explicit SGD(T learning_rate = 0.01)
            : learning_rate(learning_rate) {}

        /**
         * @brief Actualiza los parámetros en función del gradiente.
         * @param params Parámetros actuales del modelo.
         * @param grads Gradientes calculados.
         * @complexity O(n*m), donde n*m es el tamaño del tensor de parámetros.
         */
        void update(algebra::Tensor<T, 2>& params, const algebra::Tensor<T, 2>& grads) override {
            params = params - grads * learning_rate;
        }
    };

    /**
     * @brief Optimizador Adam (Adaptive Moment Estimation).
     * Combina momentum y RMSProp para un aprendizaje más estable y rápido.
     * @tparam T Tipo de dato (float, double, etc.).
     */
    template <typename T>
    class Adam final : public IOptimizer<T> {
        /// Tasa de aprendizaje
        T learning_rate;

        /// Parámetro beta1 para el promedio exponencial del primer momento (momentum)
        T beta1;

        /// Parámetro beta2 para el promedio exponencial del segundo momento (aceleración)
        T beta2;

        /// Valor pequeño para evitar divisiones por cero
        T epsilon;

        /// Primer momento (media del gradiente)
        algebra::Tensor<T, 2> m;

        /// Segundo momento (media del cuadrado del gradiente)
        algebra::Tensor<T, 2> v;

        /// Paso de actualización actual
        std::size_t t = 0;

    public:
        /**
         * @brief Constructor de Adam con parámetros configurables.
         * @param learning_rate Tasa de aprendizaje.
         * @param beta1 Coeficiente para el promedio del primer momento.
         * @param beta2 Coeficiente para el promedio del segundo momento.
         * @param epsilon Valor pequeño para estabilizar la división.
         * @complexity O(1)
         */
        explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
            : learning_rate(learning_rate),
              beta1(beta1),
              beta2(beta2),
              epsilon(epsilon) {}

        /**
         * @brief Actualiza los parámetros del modelo usando el algoritmo de Adam.
         * @param params Parámetros actuales.
         * @param grads Gradientes correspondientes.
         * @complexity O(n*m), donde n*m es el tamaño de los tensores.
         * Incluye: operaciones de actualización para m, v, normalización con m̂ y v̂,
         * y la división final.
         */
        void update(algebra::Tensor<T, 2>& params, const algebra::Tensor<T, 2>& grads) override {
            if (m.shape() != grads.shape()) {
                m.reshape(grads.shape());
                v.reshape(grads.shape());
            }

            step();
            m = beta1 * m + (T{1} - beta1) * grads;
            v = beta2 * v + (T{1} - beta2) * grads * grads;

            const algebra::Tensor<T, 2> m_hat = m / (T{1} - std::pow(beta1, t));
            const algebra::Tensor<T, 2> v_hat = v / (T{1} - std::pow(beta2, t));

            params -= m_hat * learning_rate /
                      (v_hat.apply([](const T x) { return std::sqrt(x); }) + epsilon);
        }

        /**
         * @brief Incrementa el contador de pasos.
         * Es importante para las correcciones de sesgo de Adam.
         * @complexity O(1)
         */
        void step() override {
            t += 1;
        }
    };
}  // namespace utec::neural_network

#endif
