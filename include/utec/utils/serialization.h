#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <istream>
#include <ostream>

namespace serialization {
    template <typename T>
    void write_numeric(std::ostream& out, const T n) {
        constexpr std::size_t N = sizeof(T);
        const auto data = std::bit_cast<std::array<std::uint8_t, N>>(n);
        // NOLINTNEXTLINE
        out.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    template <typename T>
    auto read_numeric(std::istream& in) -> T {
        constexpr std::size_t N = sizeof(T);
        std::array<std::uint8_t, N> data{};

        // NOLINTNEXTLINE
        in.read(reinterpret_cast<char*>(data.data()), data.size());
        return std::bit_cast<T>(data);
    }

}  // namespace serialization

#endif
