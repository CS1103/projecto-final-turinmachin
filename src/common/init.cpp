#include "common/init.h"

namespace init {
    using utec::algebra::Tensor;

    void he_init(Tensor<double, 2>& tensor, std::mt19937& rng) {
        const double scale = std::sqrt(2.0 / static_cast<double>(tensor.shape()[1]));
        std::normal_distribution<double> dist(0.0, scale);

        for (auto& v : tensor) {
            v = dist(rng);
        }
    };

}  // namespace init
