//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H

#include "tensor.h"

namespace utec::neural_network {

    template<typename T, size_t DIMS>
    using Tensor = algebra::Tensor<T, DIMS>;

    // Interfaz del optimizador (SGD o Adam)
    template<typename T>
    struct IOptimizer {
        virtual ~IOptimizer() = default;
        virtual void update(Tensor<T,2>& params, const Tensor<T,2>& gradients) = 0;
        virtual void step() {}
    };

    // Interfaz de las capas (Dense y los diferentes tipos de activación)
    template<typename T>
    struct ILayer {
        virtual ~ILayer() = default;
        virtual Tensor<T,2> forward(const Tensor<T,2>& x) = 0;
        virtual Tensor<T,2> backward(const Tensor<T,2>& gradients) = 0;

        // Se utiliza para actualizar los parameters a través el optimizador
        // Se puede llamar tanto el método update y step si es requerido
        virtual void update_params(IOptimizer<T>& optimizer) {}
    };

    // Interfaz de las perdidas (MSE o BCE)
    template<typename T, size_t DIMS>
    struct ILoss {
        virtual ~ILoss() = default;
        virtual T loss() const = 0;
        virtual Tensor<T,DIMS> loss_gradient() const = 0;
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
