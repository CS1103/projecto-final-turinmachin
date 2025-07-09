#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
    template <std::size_t Size>
    constexpr void apply_with_counter(const auto fn, const std::array<std::size_t, Size>& size) {
        std::array<std::size_t, Size> index{};

        const std::size_t total_size = std::ranges::fold_left(size, 1, std::multiplies());

        for (std::size_t i = 0; i < total_size; ++i) {
            fn(index);

            index[0]++;

            for (std::size_t j = 0; j < Size; ++j) {
                if (index[j] < size[j]) {
                    // Carry stops here
                    break;
                }

                index[j] = 0;
                if (j < Size - 1) {
                    index[j + 1]++;
                }
            }
        }
    }
}  // namespace

namespace utec::algebra {

    template <typename T, size_t Rank>
    class Tensor {
        std::array<size_t, Rank> m_shape;
        std::array<size_t, Rank> m_steps;
        std::vector<T> m_data;

        void update_steps() {
            std::size_t current_step = 1;

            for (std::size_t i = Rank - 1; i != static_cast<std::size_t>(-1); --i) {
                m_steps[i] = current_step;
                current_step *= m_shape[i];
            }
        }

        template <typename... Idxs>
            requires(sizeof...(Idxs) == Rank)
        constexpr auto physical_index(const Idxs... idxs) const -> std::size_t {
            const std::array<std::size_t, Rank> idxs_arr{static_cast<std::size_t>(idxs)...};

            std::size_t physical_index = 0;

            for (std::size_t i = 0; i < Rank; ++i) {
                if (idxs_arr[i] >= m_shape[i]) {
                    throw std::out_of_range("Tensor index out of bounds");
                }
                physical_index += m_steps[i] * idxs_arr[i];
            }

            return physical_index;
        }

    public:
        explicit Tensor(const std::array<size_t, Rank>& shape)
            : m_shape(shape),
              m_data(std::accumulate(shape.begin(),
                                     shape.end(),
                                     static_cast<size_t>(1),
                                     std::multiplies())) {
            update_steps();
        }

        template <typename... Dims>
            requires(sizeof...(Dims) == Rank)
        explicit Tensor(const Dims... dims)
            : m_shape{static_cast<size_t>(dims)...},
              m_data(std::ranges::fold_left(m_shape, 1, std::multiplies())) {
            update_steps();
        }

        [[nodiscard]] constexpr auto operator==(const Tensor<T, Rank>& other) const -> bool {
            return m_shape == other.m_shape && m_data == other.m_data;
        }

        auto operator=(std::initializer_list<T> list) -> Tensor<T, Rank>& {
            if (list.size() != m_data.size()) {
                throw std::invalid_argument("Data size does not match tensor size");
            }

            std::copy(list.begin(), list.end(), m_data.begin());
            return *this;
        }

        constexpr auto operator()(const auto... idxs) -> T& {
            return m_data[physical_index(idxs...)];
        }

        [[nodiscard]] constexpr auto operator()(const auto... idxs) const -> const T& {
            return m_data[physical_index(idxs...)];
        }

        auto operator()(const std::array<size_t, Rank>& idxs) -> T& {
            size_t idx = 0;
            for (size_t dim = 0; dim < Rank; dim++) {
                idxs[dim] < m_shape[dim] ? idx += m_steps[dim] * idxs[dim]
                                         : throw std::out_of_range("Index out of bounds");
            }
            return m_data[idx];
        }

        auto operator()(const std::array<size_t, Rank>& idxs) const -> const T& {
            size_t idx = 0;
            for (size_t dim = 0; dim < Rank; dim++) {
                idxs[dim] < m_shape[dim] ? idx += m_steps[dim] * idxs[dim]
                                         : throw std::out_of_range("Index out of bounds");
            }
            return m_data[idx];
        }

        auto operator[](const size_t idx) -> T& {
            return m_data.at(idx);
        }

        auto operator[](const size_t idx) const -> const T& {
            return m_data.at(idx);
        }

        [[nodiscard]] auto size() const -> size_t {
            return m_data.size();
        }

        auto shape() const noexcept -> const std::array<size_t, Rank>& {
            return m_shape;
        }

        void reshape(const std::array<size_t, Rank>& new_shape) {
            m_data.resize(std::ranges::fold_left(new_shape, 1, std::multiplies()));
            m_shape = new_shape;
            update_steps();
        }

        template <typename... Dims>
            requires(sizeof...(Dims) == Rank)
        void reshape(const Dims... dims) {
            std::array<size_t, Rank> new_shape{static_cast<size_t>(dims)...};
            m_data.resize(std::ranges::fold_left(new_shape, 1, std::multiplies()));
            m_shape = new_shape;
            update_steps();
        }

        void fill(const T& value) noexcept {
            std::ranges::fill(m_data, value);
        }

        auto row(const size_t index) const -> Tensor<T, 2>
            requires(Rank == 2)
        {
            if (index >= m_shape[0]) {
                throw std::out_of_range("Row index out of bounds");
            }

            Tensor<T, 2> result(1, m_shape[1]);
            for (size_t j = 0; j < m_shape[1]; ++j) {
                result(0, j) = (*this)(index, j);
            }
            return result;
        }

        void set_row(const size_t index, const Tensor<T, 2>& row_tensor)
            requires(Rank == 2)
        {
            if (row_tensor.shape()[0] != 1 || row_tensor.shape()[1] != m_shape[1]) {
                throw std::invalid_argument("Row shape does not match");
            }

            for (size_t j = 0; j < m_shape[1]; ++j) {
                (*this)(index, j) = row_tensor(0, j);
            }
        }

        auto slice(const size_t index) const -> Tensor<T, 2>
            requires(Rank == 3)
        {
            if (index >= m_shape[0]) {
                throw std::out_of_range("Index out of bounds");
            }

            Tensor<T, 2> result(m_shape[1], m_shape[2]);
            for (size_t j = 0; j < m_shape[1]; ++j) {
                for (size_t k = 0; k < m_shape[2]; ++k) {
                    result(j, k) = (*this)(index, j, k);
                }
            }

            return result;
        }

        void set_slice(const size_t index, const Tensor<T, 2>& slice)
            requires(Rank == 3)
        {
            if (index >= m_shape[0]) {
                throw std::out_of_range("Index out of bounds");
            }

            if (slice.shape()[0] != m_shape[1] || slice.shape()[1] != m_shape[2]) {
                throw std::invalid_argument("Slice shape does not match");
            }

            for (size_t j = 0; j < m_shape[1]; ++j) {
                for (size_t k = 0; k < m_shape[2]; ++k) {
                    (*this)(index, j, k) = slice(j, k);
                }
            }
        }

        auto broadcast(const Tensor<T, Rank>& rhs, auto fn) const -> Tensor<T, Rank> {
            if (m_shape == rhs.m_shape) {
                // Element-wise
                Tensor<T, Rank> result{m_shape};
                for (std::size_t i = 0; i < m_data.size(); ++i) {
                    result[i] = fn(m_data[i], rhs[i]);
                }
                return result;
            }

            std::array<std::size_t, Rank> result_shape;

            for (std::size_t i = 0; i < Rank; ++i) {
                if (m_shape[i] == rhs.m_shape[i]) {
                    result_shape[i] = m_shape[i];
                } else if (m_shape[i] == 1 || rhs.m_shape[i] == 1) {
                    result_shape[i] = std::max(m_shape[i], rhs.m_shape[i]);
                } else {
                    throw std::invalid_argument(
                        "Shapes do not match and they are not compatible for broadcasting");
                }
            }

            Tensor<T, Rank> result{result_shape};

            apply_with_counter(
                [&](const auto& result_index) {
                    std::array<std::size_t, Rank> lhs_index{result_index};
                    std::array<std::size_t, Rank> rhs_index{result_index};

                    for (std::size_t i = 0; i < Rank; ++i) {
                        lhs_index[i] %= m_shape[i];
                    }

                    for (std::size_t i = 0; i < Rank; ++i) {
                        rhs_index[i] %= rhs.m_shape[i];
                    }

                    std::apply(result, result_index) =
                        fn(std::apply(*this, lhs_index), std::apply(rhs, rhs_index));
                },
                result_shape);

            return result;
        }

        auto operator+(const Tensor<T, Rank>& other) const -> Tensor<T, Rank> {
            return broadcast(other, std::plus());
        }

        auto operator-(const Tensor<T, Rank>& other) const -> Tensor<T, Rank> {
            return broadcast(other, std::minus());
        }

        auto operator*(const Tensor<T, Rank>& other) const -> Tensor<T, Rank> {
            return broadcast(other, std::multiplies());
        }

        auto operator/(const Tensor<T, Rank>& other) const -> Tensor<T, Rank> {
            return broadcast(other, std::divides());
        }

        auto operator+(const T& scalar) const -> Tensor<T, Rank> {
            Tensor<T, Rank> result(m_shape);
            std::ranges::transform(m_data, result.m_data.begin(),
                                   [&](const T& value) { return value + scalar; });
            return result;
        }

        auto operator-(const T& scalar) const -> Tensor<T, Rank> {
            Tensor<T, Rank> result(m_shape);
            std::ranges::transform(m_data, result.m_data.begin(),
                                   [&](const T& value) { return value - scalar; });
            return result;
        }

        auto operator*(const T& scalar) const -> Tensor<T, Rank> {
            Tensor<T, Rank> result(m_shape);
            std::ranges::transform(m_data, result.m_data.begin(),
                                   [&](const T& value) { return value * scalar; });
            return result;
        }

        auto operator/(const T& scalar) const -> Tensor<T, Rank> {
            Tensor<T, Rank> result(m_shape);
            std::ranges::transform(m_data, result.m_data.begin(),
                                   [&](const T& value) { return value / scalar; });
            return result;
        }

        friend auto operator+(const T& scalar, const Tensor& tensor) -> Tensor<T, Rank> {
            Tensor<T, Rank> result(tensor.m_shape);
            std::ranges::transform(tensor.m_data, result.m_data.begin(),
                                   [&](const T& value) { return scalar + value; });
            return result;
        }

        friend auto operator-(const T& scalar, const Tensor& tensor) -> Tensor<T, Rank> {
            Tensor<T, Rank> result(tensor.m_shape);
            std::ranges::transform(tensor.m_data, result.m_data.begin(),
                                   [&](const T& value) { return scalar - value; });
            return result;
        }

        friend auto operator*(const T& scalar, const Tensor& tensor) -> Tensor<T, Rank> {
            Tensor<T, Rank> result(tensor.m_shape);
            std::ranges::transform(tensor.m_data, result.m_data.begin(),
                                   [&](const T& value) { return scalar * value; });
            return result;
        }

        friend auto operator/(const T& scalar, const Tensor& tensor) -> Tensor<T, Rank> {
            Tensor<T, Rank> result(tensor.m_shape);
            std::ranges::transform(tensor.m_data, result.m_data.begin(),
                                   [&](const T& value) { return scalar / value; });
            return result;
        }

        auto operator-() const -> Tensor<T, Rank> {
            Tensor<T, Rank> result(m_shape);
            std::ranges::transform(m_data, result.m_data.begin(), std::negate());
            return result;
        }

        friend auto operator<<(std::ostream& out, const Tensor<T, Rank>& tensor)
            -> std::ostream& requires(Rank > 1) {
                const auto& shape = tensor.shape();
                std::array<size_t, Rank> index{};

                std::function<void(size_t, size_t)> print_recursive = [&](size_t dim,
                                                                          const size_t indent) {
                    out << std::string(indent, ' ') << "{\n";

                    for (size_t i = 0; i < shape[dim]; ++i) {
                        index[dim] = i;

                        if (dim == Rank - 2) {
                            out << std::string(indent + 2, ' ');
                            for (size_t j = 0; j < shape[Rank - 1]; ++j) {
                                index[Rank - 1] = j;
                                out << tensor(index) << " ";
                            }
                            out << "\n";
                        } else {
                            print_recursive(dim + 1, indent + 2);
                        }
                    }

                    out << std::string(indent, ' ') << "}\n";
                };

                print_recursive(0, 0);
                return out;
            }

        friend auto operator<<(std::ostream& out, const Tensor<T, Rank>& tensor)
            -> std::ostream& requires(Rank == 1) {
                std::ranges::copy(tensor, std::ostream_iterator<T>(out, " "));
                return out;
            }

        auto begin() noexcept {
            return m_data.begin();
        }

        auto end() noexcept {
            return m_data.end();
        }

        [[nodiscard]] auto begin() const noexcept {
            return m_data.begin();
        }

        [[nodiscard]] auto end() const noexcept {
            return m_data.end();
        }

        [[nodiscard]] constexpr auto transpose_2d() const -> Tensor<T, 2> {
            Tensor<T, 2> result(m_shape[1], m_shape[0]);

            for (std::size_t i = 0; i < m_shape[0]; ++i) {
                for (std::size_t j = 0; j < m_shape[1]; ++j) {
                    result(j, i) = (*this)(i, j);
                }
            }

            return result;
        }

        [[nodiscard]] constexpr auto transpose_2d() const -> Tensor<T, Rank>
            requires(Rank > 2)
        {
            std::array<std::size_t, Rank> new_shape{m_shape};
            std::swap(new_shape[Rank - 2], new_shape[Rank - 1]);

            Tensor<T, Rank> result{new_shape};
            std::array<std::size_t, Rank - 2> size{};

            std::copy(m_shape.begin(), m_shape.end() - 2, size.begin());

            apply_with_counter(
                [&](const auto& index) {
                    std::array<std::size_t, Rank> full_index;
                    std::copy(index.begin(), index.end(), full_index.begin());

                    for (std::size_t i = 0; i < m_shape[Rank - 2]; ++i) {
                        for (std::size_t j = 0; j < m_shape[Rank - 1]; ++j) {
                            full_index[Rank - 2] = i;
                            full_index[Rank - 1] = j;
                            const T src = std::apply(*this, full_index);

                            full_index[Rank - 2] = j;
                            full_index[Rank - 1] = i;
                            T& dest = std::apply(result, full_index);

                            dest = src;
                        }
                    }
                },
                size);

            return result;
        }

        constexpr auto apply(auto fn) const -> Tensor<T, Rank> {
            Tensor<T, Rank> result(m_shape);
            std::ranges::transform(m_data, result.m_data.begin(), fn);
            return result;
        }
    };

    template <typename T>
    [[nodiscard]] constexpr auto matrix_product(const Tensor<T, 2>& lhs, const Tensor<T, 2>& rhs)
        -> Tensor<T, 2> {
        if (lhs.shape()[1] != rhs.shape()[0]) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }

        // Simple matrix multiplication
        Tensor<T, 2> result(lhs.shape()[0], rhs.shape()[1]);

        for (std::size_t i = 0; i < result.shape()[0]; ++i) {
            for (std::size_t j = 0; j < result.shape()[1]; ++j) {
                for (std::size_t k = 0; k < lhs.shape()[1]; ++k) {
                    result(i, j) += lhs(i, k) * rhs(k, j);
                }
            }
        }

        return result;
    }

    template <typename T, std::size_t Rank>
        requires(Rank > 2)
    [[nodiscard]] constexpr auto matrix_product(const Tensor<T, Rank>& lhs,
                                                const Tensor<T, Rank>& rhs) -> Tensor<T, Rank> {
        for (std::size_t i = 0; i < Rank - 2; ++i) {
            if (lhs.shape()[i] != rhs.shape()[i]) {
                throw std::invalid_argument("Incompatible batch dimensions for multiplication");
            }
        }

        std::array<std::size_t, Rank> new_shape{lhs.shape()};
        new_shape[Rank - 1] = rhs.shape()[Rank - 1];

        Tensor<T, Rank> result(new_shape);
        std::array<std::size_t, Rank - 2> size{};
        std::copy(lhs.shape().begin(), lhs.shape().end() - 2, size.begin());

        apply_with_counter(
            [&](const auto& index) {
                std::array<std::size_t, Rank> full_index;
                std::ranges::copy(index, full_index.begin());

                for (std::size_t i = 0; i < result.shape()[Rank - 2]; ++i) {
                    for (std::size_t j = 0; j < result.shape()[Rank - 1]; ++j) {
                        for (std::size_t k = 0; k < lhs.shape()[Rank - 1]; ++k) {
                            full_index[Rank - 2] = i;
                            full_index[Rank - 1] = k;
                            const T& src1 = std::apply(lhs, full_index);

                            full_index[Rank - 2] = k;
                            full_index[Rank - 1] = j;
                            const T& src2 = std::apply(rhs, full_index);

                            full_index[Rank - 2] = i;
                            full_index[Rank - 1] = j;
                            T& dest = std::apply(result, full_index);

                            dest += src1 * src2;
                        }
                    }
                }
            },
            size);

        return result;
    }

}  // namespace utec::algebra

#endif
