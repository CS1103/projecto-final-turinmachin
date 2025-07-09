#ifndef INCLUDE_TRAINER_INIT_H
#define INCLUDE_TRAINER_INIT_H

#include <random>
#include "utec/algebra/tensor.h"

namespace common {

    void he_init(utec::algebra::Tensor<double, 2>& tensor, std::mt19937& rng);

}  // namespace common

#endif
