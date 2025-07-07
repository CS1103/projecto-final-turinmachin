#ifndef INCLUDE_COMMON_AGENT_H
#define INCLUDE_COMMON_AGENT_H

template <typename Input, typename Output>
class IAgent {
public:
    virtual ~IAgent() = default;

    virtual auto predict(const Input& input) -> Output = 0;
};

#endif
