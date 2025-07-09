#include <cstddef>
#include <fstream>
#include <print>
#include <random>
#include <string>
#include "common/agent.h"
#include "common/data.h"
#include "trainer/loader.h"

auto main() -> int {
    using common::DigitReader;

    std::random_device rd{};
    std::mt19937 rng(rd());

    std::ifstream training_file = common::get_data_file("optdigits.tra");
    std::ifstream test_file = common::get_data_file("optdigits.tes");

    const auto training_samples = trainer::load_digit_samples(training_file);
    const auto test_samples = trainer::load_digit_samples(test_file);

    training_file.close();
    test_file.close();

    std::ifstream existing_net(common::get_data_file_path("net.pp20"));
    DigitReader agent = existing_net ? DigitReader(existing_net) : DigitReader(rng);

    constexpr std::size_t EPOCHS = 1000;
    constexpr double LEARNING_RATE = 0.3;

    std::println("Training for {} epochs at learning rate {}", EPOCHS, LEARNING_RATE);
    agent.train(training_samples, EPOCHS, LEARNING_RATE, rng);
    std::println("Training done!");

    std::println("Saving network data to net.pp20...");
    {
        std::ofstream outfile("net.pp20");
        agent.save_net(outfile);
    }

    const double accuracy_training = agent.test_accuracy(training_samples);
    const double accuracy_test = agent.test_accuracy(test_samples);
    std::println("Accuracy (training): {:.2f}%", 100 * accuracy_training);
    std::println("Accuracy (test):     {:.2f}%", 100 * accuracy_test);

    return 0;
}
