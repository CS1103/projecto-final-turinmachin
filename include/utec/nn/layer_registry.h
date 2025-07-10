#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_REGISTRY_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_REGISTRY_H

#include <algorithm>
#include <functional>
#include <istream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "utec/nn/activation.h"
#include "utec/nn/dense.h"
#include "utec/nn/interfaces.h"

namespace utec::neural_network {

    /**
     * @brief Registro de tipos de capas para deserialización.
     * @tparam T Tipo de dato utilizado por las capas (por ejemplo, float o double).
     */
    template <typename T>
    class LayerRegistry {
        /// Tipo alias para una función que crea una capa leyendo desde un flujo de entrada.
        using LayerCreator = std::function<std::unique_ptr<ILayer<T>>(std::istream&)>;

    public:
        /**
         * @brief Registra un tipo de capa junto con su función de construcción.
         * Se asocia el identificador con una función que sabe cómo crear la capa
         * correspondiente.
         * @param id Identificador único del tipo de capa.
         * @param creator Función que crea una instancia de esa capa leyendo desde un flujo.
         */
        static void register_layer(LayerId id, LayerCreator creator) {
            get_map()[id] = std::move(creator);
        }

        /**
         * @brief Crea una instancia de una capa registrada a partir de su ID.
         * Busca la función asociada al ID proporcionado y la invoca con el flujo de entrada.
         * @param id Identificador del tipo de capa a crear.
         * @param in Flujo binario de entrada desde el cual leer los datos de la capa.
         * @return Puntero a la capa recién creada.
         * @throws std::invalid_argument Si no se ha registrado una capa con el ID dado.
         */
        static auto create(LayerId id, std::istream& in) -> std::unique_ptr<ILayer<T>> {
            auto it = get_map().find(id);
            if (it == get_map().end()) {
                throw std::invalid_argument("Invalid layer ID");
            }

            return it->second(in);
        }

    private:
        /**
         * @brief Obtiene el mapa que almacena las funciones de creación de capas.
         * @return Referencia al mapa que asocia el identificador con sus constructores.
         */
        static auto get_map() -> auto& {
            static std::unordered_map<LayerId, LayerCreator> map;
            return map;
        }
    };

    /**
     * @brief Registra todas las capas disponibles en el sistema.
     * Llamado al inicio del programa para asegurar que todos los tipos
     * de capas estén disponibles al deserializar una red neuronal.
     * Capas sin parámetros (ej. ReLU, Softmax) se crean directamente. Capas parametrizadas
     * (ej. Dense) leen sus parámetros del flujo de entrada.
     * @tparam T Tipo de dato utilizado por las capas (float, double, etc.).
     */
    template <typename T>
    void register_all_layers() {
        LayerRegistry<T>::register_layer(LayerId::ReLU,
                                         [](std::istream&) { return std::make_unique<ReLU<T>>(); });
        LayerRegistry<T>::register_layer(
            LayerId::Sigmoid, [](std::istream&) { return std::make_unique<Sigmoid<T>>(); });
        LayerRegistry<T>::register_layer(
            LayerId::Softmax, [](std::istream&) { return std::make_unique<Softmax<T>>(); });
        LayerRegistry<T>::register_layer(LayerId::Dense, [](std::istream& in) {
            return std::make_unique<Dense<T>>(Dense<T>::load(in));
        });
    }

}  // namespace utec::neural_network

#endif
