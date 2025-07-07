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
#include "utec/nn/kan.h"

namespace utec::neural_network {

    template <typename T>
    class LayerRegistry {
        using LayerCreator = std::function<std::unique_ptr<ILayer<T>>(std::istream&)>;

    public:
        static void register_layer(LayerId id, LayerCreator creator) {
            get_map()[id] = std::move(creator);
        }

        static auto create(LayerId id, std::istream& in) -> std::unique_ptr<ILayer<T>> {
            auto it = get_map().find(id);
            if (it == get_map().end()) {
                throw std::invalid_argument("Invalid layer ID");
            }

            return it->second(in);
        }

    private:
        static auto get_map() -> auto& {
            static std::unordered_map<LayerId, LayerCreator> map;
            return map;
        }
    };

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
        LayerRegistry<T>::register_layer(LayerId::Kan, [](std::istream& in) {
            return std::make_unique<Kan<T>>(Kan<T>::load(in));
        });
    }

}  // namespace utec::neural_network

#endif
