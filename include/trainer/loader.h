#ifndef INCLUDE_TRAINER_LOADER_H
#define INCLUDE_TRAINER_LOADER_H

#include <fstream>
#include <vector>
#include "common/agent.h"

namespace trainer {

    auto load_digit_samples(std::ifstream& infile) -> std::vector<common::DigitSample>;

}  // namespace trainer

#endif
