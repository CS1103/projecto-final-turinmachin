#ifndef INCLUDE_COMMON_AGENT_H
#define INCLUDE_COMMON_AGENT_H

#include <random>
#include <vector>
#include "utec/nn/neural_network.h"

namespace common {

    template <typename Input, typename Output>
    struct Sample {
        Input input;
        Output output;
    };

    template <typename Input, typename Output>
    class IAgent {
    public:
        virtual ~IAgent() = default;

        virtual auto predict(const Input& input) -> Output = 0;

        virtual void train(const std::vector<Sample<Input, Output>>& samples,
                           std::size_t epochs,
                           double learning_rate,
                           std::mt19937& rng) = 0;
    };

    using IDigitAgent = IAgent<std::vector<double>, int>;
    using DigitSample = Sample<std::vector<double>, int>;

    class DigitReader final : public IDigitAgent {
        utec::neural_network::NeuralNetwork<double> net;

    public:
        using Input = std::vector<double>;
        using Output = int;

        explicit DigitReader(std::mt19937& rng);

        explicit DigitReader(std::istream& net_in);

        auto predict(const Input& features) -> int override;

        void train(const std::vector<DigitSample>& samples,
                   std::size_t epochs,
                   double learning_rate,
                   std::mt19937& rng) override;

        auto test_accuracy(const std::vector<DigitSample>& samples) -> double;

        void load_net(std::istream& in);

        void save_net(std::ostream& out) const;
    };

}  // namespace common

#endif
