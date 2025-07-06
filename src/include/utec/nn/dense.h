#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "interfaces.h"
#include "utec/utils/serialization.h"

namespace utec::neural_network {
    template <typename T>
    class Dense final : public ILayer<T> {
        algebra::Tensor<T, 2> weights;
        algebra::Tensor<T, 2> biases;
        algebra::Tensor<T, 2> activations;

        algebra::Tensor<T, 2> gradient_weights;
        algebra::Tensor<T, 2> gradient_biases;

    public:
        Dense(const size_t in_f, const size_t out_f)
            : weights(std::array{in_f, out_f}),
              biases(std::array<size_t, 2>{1, out_f}),
              activations(std::array<size_t, 2>{1, out_f}),
              gradient_weights(std::array{in_f, out_f}),
              gradient_biases(std::array<size_t, 2>{1, out_f}) {}

        Dense(const size_t in_f, const size_t out_f, auto init_w_fun, auto init_b_fun)
            : Dense(in_f, out_f) {
            init_w_fun(weights);
            init_b_fun(biases);
        }

        auto forward(const algebra::Tensor<T, 2>& x) -> algebra::Tensor<T, 2> override {
            activations = x;
            return algebra::matrix_product(x, weights) + biases;
        }

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

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(weights, gradient_weights);
            optimizer.update(biases, gradient_biases);
        }

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
