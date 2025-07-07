#include <array>
#include <cstddef>
#include <fstream>
#include <print>
#include <random>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>
#include "utec/algebra/tensor.h"
#include "utec/nn/activation.h"
#include "utec/nn/loss.h"
#include "utec/nn/neural_network.h"

using namespace utec::neural_network;
using namespace utec::algebra;

constexpr std::size_t IMAGE_WIDTH = 8;
constexpr std::size_t IMAGE_HEIGHT = 8;
constexpr std::size_t IMAGE_LENGTH = IMAGE_WIDTH * IMAGE_HEIGHT;

struct Sample {
    std::array<double, IMAGE_LENGTH> features;
    int label;
};

class Dataset {
    std::vector<Sample> samples;

public:
    explicit Dataset(std::ifstream& infile) {
        constexpr double DIMENSION_SIZE = 16.0;

        std::string line;

        while (std::getline(infile, line)) {
            std::stringstream line_ss(line);
            std::string buf;
            Sample sample{};

            for (std::size_t i = 0; i < IMAGE_LENGTH; ++i) {
                std::getline(line_ss, buf, ',');
                sample.features.at(i) = std::stoi(buf) / DIMENSION_SIZE;
            }

            std::getline(line_ss, buf, ',');
            sample.label = std::stoi(buf);

            samples.push_back(sample);
        }
    }

    [[nodiscard]] auto get_input_tensor() const -> Tensor<double, 2> {
        Tensor<double, 2> inputs(samples.size(), IMAGE_LENGTH);

        for (auto const& [i, sample] : std::views::enumerate(samples)) {
            for (std::size_t j = 0; j < IMAGE_LENGTH; ++j) {
                inputs(i, j) = sample.features.at(j);
            }
        }

        return inputs;
    }

    [[nodiscard]] auto get_output_tensor() const -> Tensor<double, 2> {
        Tensor<double, 2> outputs(samples.size(), 10);

        for (auto const& [i, sample] : std::views::enumerate(samples)) {
            outputs(i, sample.label) = 1.0;
        }

        return outputs;
    }

    void train(NeuralNetwork<double>& net, std::mt19937& rng) const {
        const Tensor<double, 2> inputs = get_input_tensor();
        const Tensor<double, 2> outputs = get_output_tensor();

        net.train<CrossEntropyLoss>(inputs, outputs, 1000, samples.size(), 0.01, rng);
    }

    auto test_accuracy(NeuralNetwork<double>& net) const -> double {
        std::size_t success_count = 0;

        for (const auto& sample : samples) {
            Tensor<double, 2> input(1, IMAGE_LENGTH);
            input = sample.features;

            const Tensor<double, 2> output = net.predict(input);

            int max_index = 0;
            for (int j = 1; j < 10; ++j) {
                if (output[j] > output[max_index]) {
                    max_index = j;
                }
            }

            if (max_index == sample.label) {
                ++success_count;
            }
        }

        return static_cast<double>(success_count) / static_cast<double>(samples.size());
    }
};

namespace {

    auto load_neural_network(std::mt19937& rng) -> NeuralNetwork<double> {
        std::ifstream infile("net_kan.pp20");
        if (infile) {
            std::println("Loading existing network...");
            return NeuralNetwork<double>::load(infile);
        }

        const auto kan_init = [&](auto& tensor) {
            std::normal_distribution<double> dist(0.0, 0.01);
            for (auto& v : tensor) {
                v = dist(rng);
            }
        };
        const auto zero_init = [&](auto& tensor) {
            for (auto& v : tensor) {
                v = 0.0;
            }
        };

        std::println("No existing network data detected.");

        NeuralNetwork<double> net;
        net.add_layer<KAN<double>>(IMAGE_LENGTH, 64, 8, kan_init, kan_init, zero_init);
        net.add_layer<KAN<double>>(64, 32, 8, kan_init, kan_init, zero_init);
        net.add_layer<KAN<double>>(32, 10, 8, kan_init, kan_init, zero_init);
        net.add_layer<Softmax<double>>();
        return net;
    }

}  // namespace

auto main() -> int {
    std::random_device rd{};
    std::mt19937 rng(rd());
    NeuralNetwork<double> net = load_neural_network(rng);

    std::ifstream training_file("res/optdigits.tra");
    std::ifstream test_file("res/optdigits.tes");

    const Dataset training_data(training_file);
    const Dataset test_data(test_file);

    std::println("Training...");
    training_data.train(net, rng);

    std::println("Saving...");
    {
        std::ofstream outfile("net_kan.pp20");
        net.save(outfile);
    }

    const double accuracy_tra = training_data.test_accuracy(net);
    const double accuracy_tes = test_data.test_accuracy(net);
    std::println("Accuracy (training): {:.2f}%", 100 * accuracy_tra);
    std::println("Accuracy (test):     {:.2f}%", 100 * accuracy_tes);
    return 0;
}
