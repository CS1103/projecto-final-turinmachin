#include "common/agent.h"
#include <iostream>
#include <ranges>
#include "common/constants.h"
#include "common/init.h"
#include "utec/algebra/tensor.h"
#include "utec/nn/loss.h"

using utec::algebra::Tensor;
using namespace utec::neural_network;

namespace common {

    DigitReader::DigitReader(std::mt19937& rng) {
        std::cout << "No existing network data file found. Initializing DigitReader..." << "\n";

        const auto he_init_t = [&](Tensor<double, 2>& tensor) { he_init(tensor, rng); };

        net.add_layer<Dense<double>>(IMAGE_SIZE, 64, he_init_t, he_init_t);
        net.add_layer<ReLU<double>>();
        net.add_layer<Dense<double>>(64, 32, he_init_t, he_init_t);
        net.add_layer<ReLU<double>>();
        net.add_layer<Dense<double>>(32, 10, he_init_t, he_init_t);
        net.add_layer<Softmax<double>>();
    }

    DigitReader::DigitReader(std::istream& net_in)
        : net(NeuralNetwork<double>::load(net_in)) {
        std::cout << "DigitReader loaded from network data file" << "\n";
    }

    auto DigitReader::predict(const Input& features) -> int {
        Tensor<double, 2> input(1, IMAGE_SIZE);
        std::ranges::copy(features, input.begin());

        const Tensor<double, 2> output = net.predict(input);

        int highest_score_index = 0;
        for (int j = 1; j < 10; ++j) {
            if (output[j] > output[highest_score_index]) {
                highest_score_index = j;
            }
        }

        return highest_score_index;
    }

    void DigitReader::train(const std::vector<DigitSample>& samples,
                            const std::size_t epochs,
                            const double learning_rate,
                            std::mt19937& rng) {
        Tensor<double, 2> inputs(samples.size(), IMAGE_SIZE);

        for (auto const& [i, sample] : std::views::enumerate(samples)) {
            for (std::size_t j = 0; j < IMAGE_SIZE; ++j) {
                inputs(i, j) = sample.input.at(j);
            }
        }

        Tensor<double, 2> outputs(samples.size(), 10);

        for (auto const& [i, sample] : std::views::enumerate(samples)) {
            outputs(i, sample.output) = 1.0;
        }

        net.train<CrossEntropyLoss>(inputs, outputs, epochs, samples.size(), learning_rate, rng);
    }

    auto DigitReader::test_accuracy(const std::vector<DigitSample>& samples) -> double {
        std::size_t success_count = 0;

        for (const auto& sample : samples) {
            const int predicted = predict(sample.input);
            if (predicted == sample.output) {
                ++success_count;
            }
        }

        return static_cast<double>(success_count) / static_cast<double>(samples.size());
    }

    void DigitReader::load_net(std::istream& in) {
        net = NeuralNetwork<double>::load(in);
    }

    void DigitReader::save_net(std::ostream& out) const {
        net.save(out);
    }

}  // namespace common
