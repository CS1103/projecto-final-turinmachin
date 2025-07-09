#ifndef INCLUDE_TRAINER_INIT_H
#define INCLUDE_TRAINER_INIT_H

#include <random>
#include "utec/algebra/tensor.h"
namespace init {

    void he_init(utec::algebra::Tensor<double, 2>& tensor, std::mt19937& rng);

}  // namespace init

#endif
