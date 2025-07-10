#include <utec/algebra/tensor.h>
#include <utec/nn/activation.h>
#include <utec/nn/dense.h>
#include <utec/nn/loss.h>
#include <utec/nn/neural_network.h>
#include <utec/nn/optimizer.h>
#include <catch_amalgamated.hpp>
#include <random>
#include <sstream>

using utec::algebra::Tensor;
using utec::neural_network::Dense;
using utec::neural_network::NeuralNetwork;
using utec::neural_network::Sigmoid;

TEST_CASE("neural network predict simple", "[neural]") {
    NeuralNetwork<float> net;
    SECTION("forward with Dense + Sigmoid") {
        net.add_layer<Dense<float>>(2, 3);
        net.add_layer<Sigmoid<float>>();

        Tensor<float, 2> input(1, 2);
        input(0, 0) = 0.5;
        input(0, 1) = -0.3;
        auto output = net.predict(input);
        REQUIRE(output.shape() == std::array<size_t, 2>{1, 3});
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(output(0, i) >= 0.0F);
            REQUIRE(output(0, i) <= 1.0F);
        }
    }
}

TEST_CASE("Dense + Sigmoid with batch size 2", "[neural]") {
    NeuralNetwork<float> net;
    net.add_layer<Dense<float>>(2, 2);
    net.add_layer<Sigmoid<float>>();

    Tensor<float, 2> input(2, 2);
    input(0, 0) = 0.5F;
    input(0, 1) = -0.3F;
    input(1, 0) = 1.0F;
    input(1, 1) = 0.2F;

    auto output = net.predict(input);
    REQUIRE(output.shape() == std::array<size_t, 2>{2, 2});
    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 2; ++i) {
            REQUIRE(output(b, i) >= 0.0F);
        }
    }
}

TEST_CASE("Dense layer with constant input", "[neural]") {
    NeuralNetwork<float> net;
    net.add_layer<Dense<float>>(1, 1);

    Tensor<float, 2> input(3, 1);
    input.fill(1.0F);

    auto output = net.predict(input);
    REQUIRE(output.shape() == std::array<size_t, 2>{3, 1});
}

TEST_CASE("Empty network returns input", "[neural]") {
    NeuralNetwork<float> net;
    Tensor<float, 2> input(1, 4);
    input.fill(2.0F);
    auto output = net.predict(input);
    REQUIRE(output == input);
}

#include <utec/nn/kan.h>
using utec::neural_network::Kan;

TEST_CASE("Kan layer basic forward", "[kan]") {
    NeuralNetwork<float> net;
    net.add_layer<Kan<float>>(2, 2, 5);
    Tensor<float, 2> input(1, 2);
    input(0, 0) = 0.2F;
    input(0, 1) = 0.8F;
    auto output = net.predict(input);
    REQUIRE(output.shape() == std::array<size_t, 2>{1, 2});
}

TEST_CASE("neural network trains and reduces loss", "[neural][train]") {
    using namespace utec::neural_network;
    NeuralNetwork<float> net;
    std::mt19937 rng(42);  // NOLINT

    const auto he_init = [&](Tensor<float, 2>& tensor) {
        const float scale = std::sqrt(2.0F / static_cast<float>(tensor.shape()[1]));
        std::normal_distribution<float> dist(0.0, scale);
        for (auto& v : tensor) {
            v = dist(rng);
        }
    };

    Tensor<float, 2> x(4, 2);
    x = {0, 0, 0, 1, 1, 0, 1, 1};

    Tensor<float, 2> y(4, 1);
    y = {0, 1, 1, 0};

    net.add_layer<Dense<float>>(2, 4, he_init, he_init);
    net.add_layer<ReLU<float>>();
    net.add_layer<Dense<float>>(4, 1, he_init, he_init);
    constexpr size_t epochs = 3000;
    constexpr float learning_rate = 0.1F;
    net.train<MSELoss>(x, y, epochs, 4, learning_rate, rng);
    auto pred = net.predict(x);
    for (size_t i = 0; i < 4; ++i) {
        float output = pred(i, 0);
        if (y(i, 0) == 1) {
            REQUIRE(output > 0.8F);
        } else {
            REQUIRE(output < 0.2F);
        }
    }
}

TEST_CASE("neural network save/load consistency", "[neural][save]") {
    NeuralNetwork<float> net1;
    net1.add_layer<Dense<float>>(2, 2);
    net1.add_layer<Sigmoid<float>>();

    Tensor<float, 2> x(1, 2);
    x(0, 0) = 0.5;
    x(0, 1) = -0.3;

    auto out1 = net1.predict(x);

    std::stringstream buffer;
    net1.save(buffer);

    auto net2 = NeuralNetwork<float>::load(buffer);
    auto out2 = net2.predict(x);

    REQUIRE(out1.shape() == out2.shape());
    for (size_t i = 0; i < out1.size(); ++i) {
        REQUIRE_THAT(out1[i], Catch::Matchers::WithinAbs(out2[i], 0.001));
    }
}

TEST_CASE("neural network with Kan layer", "[neural][kan]") {
    NeuralNetwork<float> net;
    net.add_layer<Kan<float>>(2, 1, 5);

    Tensor<float, 2> input(1, 2);
    input(0, 0) = 0.1;
    input(0, 1) = -0.5;

    auto output = net.predict(input);
    REQUIRE(output.shape() == std::array<size_t, 2>{1, 1});
}

TEST_CASE("neural network backward call", "[neural][backward]") {
    NeuralNetwork<float> net;
    net.add_layer<Dense<float>>(2, 3);
    net.add_layer<Sigmoid<float>>();
    net.add_layer<Dense<float>>(3, 2);

    Tensor<float, 2> x(2, 2);
    x.fill(0.5);

    Tensor<float, 2> y(2, 2);
    y.fill(1.0);

    std::mt19937 rng(0);  // NOLINT
    net.train<utec::neural_network::MSELoss>(x, y, 1, 2, 0.01, rng);

    SUCCEED();
}

TEST_CASE("neural network learns linear function f(x) = 2x", "[neural][train]") {
    using namespace utec::neural_network;
    NeuralNetwork<float> net;
    std::mt19937 rng(42);  // NOLINT

    const auto he_init = [&](Tensor<float, 2>& tensor) {
        const float scale = std::sqrt(2.0F / static_cast<float>(tensor.shape()[0]));
        std::normal_distribution<float> dist(0.0F, scale);
        for (auto& v : tensor) {
            v = dist(rng);
        }
    };

    Tensor<float, 2> x(7, 1);
    Tensor<float, 2> y(7, 1);
    x = {1, 2, 3, 4, 7, 9, 14};
    y = {2, 4, 6, 8, 14, 18, 28};

    net.add_layer<Dense<float>>(1, 1, he_init, he_init);

    constexpr size_t epochs = 1000;
    constexpr float learning_rate = 0.01F;
    net.train<MSELoss>(x, y, epochs, 7, learning_rate, rng);

    auto pred = net.predict(x);
    for (size_t i = 0; i < 7; ++i) {
        float expected = y(i, 0);
        float output = pred(i, 0);
        REQUIRE(std::abs(output - expected) < 0.5F);
    }
}

TEST_CASE("neural network learns nonlinear function f(x) = x^2", "[neural][train]") {
    using namespace utec::neural_network;
    NeuralNetwork<float> net;
    std::mt19937 rng(42);  // NOLINT

    const auto he_init = [&](Tensor<float, 2>& tensor) {
        const float scale = std::sqrt(2.0F / static_cast<float>(tensor.shape()[0]));
        std::normal_distribution<float> dist(0.0F, scale);
        for (auto& v : tensor) {
            v = dist(rng);
        }
    };

    std::vector<float> inputs = {1, 2, 3, 4, 5, 6, 7};
    std::vector<float> outputs;

    Tensor<float, 2> x(inputs.size(), 1);
    for (size_t i = 0; i < inputs.size(); ++i) {
        x(i, 0) = inputs[i] / 7.0F;
    }

    Tensor<float, 2> y(inputs.size(), 1);
    for (size_t i = 0; i < inputs.size(); ++i) {
        y(i, 0) = (inputs[i] * inputs[i]) / 49.0F;
    }

    net.add_layer<Dense<float>>(1, 8, he_init, he_init);
    net.add_layer<ReLU<float>>();
    net.add_layer<Dense<float>>(8, 1, he_init, he_init);
    net.train<MSELoss>(x, y, 5000, 7, 0.01F, rng);
    auto pred = net.predict(x);
    for (size_t i = 0; i < inputs.size(); ++i) {
        float expected = y(i, 0) * 49.0F;
        float output = pred(i, 0) * 49.0F;
        REQUIRE(std::abs(output - expected) < 2.0F);
    }
}

TEST_CASE("neural network learns sin(x) using ReLU", "[neural][train]") {
    using namespace utec::neural_network;
    NeuralNetwork<float> net;

    std::mt19937 rng(42);  // NOLINT
    const auto he_init = [&](Tensor<float, 2>& tensor) {
        const float scale = std::sqrt(2.0F / static_cast<float>(tensor.shape()[0]));
        std::normal_distribution<float> dist(0.0F, scale);
        for (auto& v : tensor) {
            v = dist(rng);
        }
    };
    std::vector<float> inputs = {0, M_PI / 6, M_PI / 4, M_PI / 2, M_PI, 3 * M_PI / 2, 2 * M_PI};
    Tensor<float, 2> x(inputs.size(), 1);
    Tensor<float, 2> y(inputs.size(), 1);
    for (size_t i = 0; i < inputs.size(); ++i) {
        x(i, 0) = inputs[i];
        y(i, 0) = std::sin(inputs[i]);
    }
    net.add_layer<Dense<float>>(1, 30, he_init, he_init);
    net.add_layer<ReLU<float>>();
    net.add_layer<Dense<float>>(30, 15, he_init, he_init);
    net.add_layer<ReLU<float>>();
    net.add_layer<Dense<float>>(15, 1, he_init, he_init);
    constexpr size_t epochs = 5000;
    constexpr float learning_rate = 0.01F;
    net.train<MSELoss>(x, y, epochs, inputs.size(), learning_rate, rng);
    auto pred = net.predict(x);

    for (size_t i = 0; i < inputs.size(); ++i) {
        float expected = y(i, 0);
        float output = pred(i, 0);
        REQUIRE(std::abs(output - expected) < 0.2F);
    }
}
