#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_KAN_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_KAN_H

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include "interfaces.h"
#include "utec/algebra/tensor.h"
#include "utec/utils/serialization.h"

namespace utec::neural_network {
    template <typename T>
    struct BSpline {
        size_t knots;

        T x_min;
        T x_max;

        T step;

        BSpline(size_t knots, T a, T b)
            : knots(knots),
              x_min(a),
              x_max(b),
              step((b - a) / (knots - 1)) {}

        auto eval(T x) const -> std::vector<T> {
            std::vector<T> B;
            B.resize(knots, T{0});

            if (x <= x_min) {
                B.front() = 1;
                return B;
            }
            if (x >= x_max) {
                B.back() = 1;
                return B;
            }

            T pos = (x - x_min) / step;
            auto i = static_cast<size_t>(pos);
            T t = pos - 1;

            B[i] = 1 - t;
            B[i + 1] = t;
            return B;
        }
    };

    template <typename T>
    class KAN final : public ILayer<T> {
        size_t in_f;
        size_t out_f;
        size_t width;
        size_t knots;

        T x_min;
        T x_max;

        algebra::Tensor<T, 3> psi_weights;
        algebra::Tensor<T, 2> phi_weights;
        algebra::Tensor<T, 2> phi_biases;

        algebra::Tensor<T, 3> gradient_psi_weights;
        algebra::Tensor<T, 2> gradient_phi_weights;
        algebra::Tensor<T, 2> gradient_phi_biases;

        algebra::Tensor<T, 2> input;
        algebra::Tensor<T, 3> psi_output;
        algebra::Tensor<T, 2> psi_sum;

        BSpline<T> basis;

    public:
        KAN(size_t in_f, size_t out_f, size_t knots, T x_min = -1, T x_max = +1)
            : in_f(in_f),
              out_f(out_f),
              width((2 * in_f) + 1),
              knots(knots),
              x_min(x_min),
              x_max(x_max),
              psi_weights(width, in_f, knots),
              phi_weights(out_f, width),
              phi_biases(1, out_f),
              gradient_psi_weights(width, in_f, knots),
              gradient_phi_weights(out_f, width),
              gradient_phi_biases(1, out_f),
              input(0, 0),
              psi_output(0, 0, 0),
              psi_sum(0, 0),
              basis(knots, x_min, x_max) {}

        KAN(size_t in_f,
            size_t out_f,
            size_t knots,
            auto init_psi_w_fun,
            auto init_phi_w_fun,
            auto init_phi_b_fun)
            : KAN(in_f, out_f, knots) {
            init_psi_w_fun(psi_weights);
            init_phi_w_fun(phi_weights);
            init_phi_b_fun(phi_biases);
        }

        auto forward(const algebra::Tensor<T, 2>& x) -> algebra::Tensor<T, 2> override {
            input = x;
            size_t B = x.shape()[0];

            psi_output = algebra::Tensor<T, 3>(B, width, in_f);
            psi_sum = algebra::Tensor<T, 2>(B, width);

            for (size_t b = 0; b < B; ++b) {
                for (size_t q = 0; q < width; ++q) {
                    T sum_q = 0;
                    for (size_t p = 0; p < in_f; ++p) {
                        auto Bk = basis.eval(x(b, p));
                        T tmp = 0;
                        for (size_t k = 0; k < knots; ++k) {
                            tmp += psi_weights(q, p, k) * Bk[k];
                        }
                        psi_output(b, q, p) = tmp;
                        sum_q += tmp;
                    }
                    psi_sum(b, q) = sum_q;
                }
            }

            algebra::Tensor<T, 2> result(B, out_f);
            result.fill(0);
            auto phi_w_T = algebra::transpose_2d(phi_weights);

            auto tmp = algebra::matrix_product(psi_sum, phi_w_T);
            for (size_t b = 0; b < B; ++b) {
                for (size_t j = 0; j < out_f; ++j) {
                    result(b, j) = tmp(b, j) + phi_biases(0, j);
                }
            }

            return result;
        }

        auto backward(const algebra::Tensor<T, 2>& dZ) -> algebra::Tensor<T, 2> override {
            size_t B = input.shape()[0];

            gradient_phi_weights = algebra::matrix_product(algebra::transpose_2d(dZ), psi_sum);
            // gradient_phi_weights = algebra::transpose_2d(gradient_phi_weights);

            gradient_phi_biases.fill(0);
            for (size_t b = 0; b < B; ++b) {
                for (size_t j = 0; j < out_f; ++j) {
                    gradient_phi_biases(0, j) += dZ(b, j);
                }
            }

            auto DPsiSum = algebra::matrix_product(dZ, phi_weights);

            gradient_psi_weights.fill(0);

            algebra::Tensor<T, 2> dInput(B, in_f);
            dInput.fill(0);

            for (size_t b = 0; b < B; ++b) {
                for (size_t q = 0; q < width; ++q) {
                    T dPs = DPsiSum(b, q);
                    for (size_t p = 0; p < in_f; ++p) {
                        auto Bk = basis.eval(input(b, p));
                        for (size_t k = 0; k < knots; ++k) {
                            gradient_psi_weights(q, p, k) += dPs * Bk[k];
                        }
                        T wqpk_sum = 0;
                        for (size_t k = 0; k < knots; ++k) {
                            wqpk_sum += psi_weights(q, p, k) * Bk[k];
                        }
                        dInput(b, p) += dPs * wqpk_sum;
                    }
                }
            }

            return dInput;
        }

        void update_params(IOptimizer<T>& optimizer) override {
            for (size_t q = 0; q < width; ++q) {
                auto psi_slice = psi_weights.slice(q);
                auto grad_slice = gradient_psi_weights.slice(q);
                optimizer.update(psi_slice, grad_slice);
                psi_weights.set_slice(q, psi_slice);
            }
            optimizer.update(phi_weights, gradient_phi_weights);
            optimizer.update(phi_biases, gradient_phi_biases);
        }

        [[nodiscard]] auto id() const -> LayerID override {
            return LayerID::KAN;
        }

        void save(std::ostream& out) const override {
            serialization::write_numeric<std::uint64_t>(out, in_f);
            serialization::write_numeric<std::uint64_t>(out, out_f);
            serialization::write_numeric<std::uint64_t>(out, knots);
            serialization::write_numeric<double>(out, x_min);
            serialization::write_numeric<double>(out, x_max);

            for (const auto& x : psi_weights) {
                serialization::write_numeric<T>(out, x);
            }

            for (const auto& x : phi_weights) {
                serialization::write_numeric<T>(out, x);
            }

            for (const auto& x : phi_biases) {
                serialization::write_numeric<T>(out, x);
            }
        }

        static auto load(std::istream& in) -> KAN<T> {
            auto in_feat = serialization::read_numeric<uint64_t>(in);
            auto out_feat = serialization::read_numeric<uint64_t>(in);
            auto knots = serialization::read_numeric<uint64_t>(in);
            T a = serialization::read_numeric<T>(in);
            T b = serialization::read_numeric<T>(in);

            KAN<T> layer(in_feat, out_feat, knots, a, b);
            for (auto& x : layer.psi_weights) {
                x = serialization::read_numeric<T>(in);
            }

            for (auto& x : layer.phi_weights) {
                x = serialization::read_numeric<T>(in);
            }

            for (auto& x : layer.phi_biases) {
                x = serialization::read_numeric<T>(in);
            }
            return layer;
        }
    };

}  // namespace utec::neural_network

#endif
