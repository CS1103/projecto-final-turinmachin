#include <cstddef>
#include <stdexcept>
#include <vector>

namespace utils {

    template <typename T>
    auto unflatten(const std::vector<T>& vec, const std::size_t width)
        -> std::vector<std::vector<T>> {
        if (vec.size() % width != 0) {
            throw std::runtime_error("Vector size must be divisible by unflattened width");
        }

        const std::size_t height = vec.size() / width;
        std::vector<std::vector<T>> result(height, std::vector<T>(width));

        for (std::size_t i = 0; i < width; ++i) {
            for (std::size_t j = 0; j < height; ++j) {
                result.at(i).at(j) = vec.at((i * width) + j);
            }
        }

        return result;
    }

    template <typename T>
    auto flatten(const std::vector<std::vector<T>>& vec) -> std::vector<T> {
        if (vec.empty()) {
            return std::vector<T>();
        }

        std::vector<T> result(vec.size() * vec.at(0).size());

        for (const auto& row : vec) {
            for (const auto& el : row) {
                result.push_back(el);
            }
        }

        return result;
    }

}  // namespace utils
