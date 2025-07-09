#ifndef INCLUDE_TRAINER_LOADER_H
#define INCLUDE_TRAINER_LOADER_H

#include <fstream>
#include <vector>
#include "common/agent.h"
namespace loader {

    using DigitSample = agent::Sample<std::vector<double>, int>;

    auto load_digit_samples(std::ifstream& infile) -> std::vector<DigitSample>;

}  // namespace loader

#endif
