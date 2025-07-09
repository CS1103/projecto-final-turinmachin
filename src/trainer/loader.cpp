#include "trainer/loader.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "common/agent.h"
#include "common/constants.h"

namespace loader {

    auto load_digit_samples(std::ifstream& infile) -> std::vector<DigitSample> {
        using Sample = agent::Sample<std::vector<double>, int>;

        std::vector<Sample> samples;

        std::string line;

        while (std::getline(infile, line)) {
            std::stringstream line_ss(line);
            std::string buf;
            Sample sample;

            for (std::size_t i = 0; i < IMAGE_SIZE; ++i) {
                std::getline(line_ss, buf, ',');
                sample.input.push_back(std::stoi(buf) / IMAGE_DIMENSION);
            }

            std::getline(line_ss, buf, ',');
            sample.output = std::stoi(buf);

            samples.push_back(sample);
        }

        return samples;
    }

}  // namespace loader
