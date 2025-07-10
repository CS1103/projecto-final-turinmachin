#include <cstddef>
#include <fstream>
#include <iostream>
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

    std::ifstream existing_net(common::get_data_file_path_unchecked("net.pp20"));
    DigitReader agent = existing_net ? DigitReader(existing_net) : DigitReader(rng);

    std::size_t epochs = 0;
    double learning_rate = 0.0;

    std::cout << "Epochs to train: ";
    std::cin >> epochs;
    std::cout << "Learning rate: ";
    std::cin >> learning_rate;

    std::cout << "Training for " << epochs << " epochs at learning rate " << learning_rate << "\n";
    agent.train(training_samples, epochs, learning_rate, rng);
    std::cout << "Training done!" << "\n";

    std::cout << "Saving network data to net.pp20..." << "\n";
    {
        std::ofstream outfile("net.pp20");
        agent.save_net(outfile);
    }

    const double accuracy_training = agent.test_accuracy(training_samples);
    const double accuracy_test = agent.test_accuracy(test_samples);
    std::cout << "Accuracy (training): " << 100 * accuracy_training << "%" << "\n";
    std::cout << "Accuracy (test):     " << 100 * accuracy_test << "%" << "\n";

    return 0;
}
