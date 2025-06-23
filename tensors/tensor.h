#ifndef TENSOR_H
#define TENSOR_H
#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace utec::algebra {
    template<typename T, size_t Rank>
    class Tensor {

        std::array<size_t, Rank> dims_array;
        std::array<size_t, Rank> steps;
        std::vector<T> data;

        void update_steps() {
            size_t current_step = 1;
            for (size_t i = Rank; i-- > 0;) {
                steps[i] = current_step;
                current_step *= dims_array[i];
            }
        }

    public:
        explicit Tensor(const std::array<size_t, Rank>& shape): dims_array(shape), data(std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies())) {
            update_steps();
        }

        template<typename... Dims>
        requires(sizeof...(Dims) == Rank)
        explicit Tensor(Dims... dims): dims_array{static_cast<size_t>(dims)...}, data(std::accumulate(dims_array.begin(), dims_array.end(), static_cast<size_t>(1), std::multiplies())) {
            update_steps();
        }

        Tensor(const Tensor& other) = default;
        Tensor(Tensor&& other) noexcept = default;
        Tensor& operator=(const Tensor& other) = default;
        Tensor& operator=(Tensor&& other) noexcept = default;
        ~Tensor() = default;

        template<std::ranges::sized_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, T>
        Tensor& operator=(const Range& range) {
            if (std::ranges::size(range) != data.size()) throw std::invalid_argument("Data size does not match tensor size");
            std::ranges::copy(range, data.begin());
            return *this;
        }

        template<std::ranges::input_range Range>
        requires (!std::ranges::sized_range<Range> && std::convertible_to<std::ranges::range_value_t<Range>, T>)
        Tensor& operator=(const Range& range) {
            if (std::ranges::distance(range) != data.size()) throw std::invalid_argument("Data size does not match tensor size");
            std::ranges::copy(range, data.begin());
            return *this;
        }

        Tensor& operator=(std::initializer_list<T> list) {
            if (list.size() != data.size()) throw std::invalid_argument("Data size does not match tensor size");
            std::copy(list.begin(), list.end(), data.begin());
            return *this;
        }

        template<typename... Idxs>
        requires(sizeof...(Idxs) == Rank)
        T& operator()(Idxs... idxs) {
            size_t idx = 0;
            size_t dim = 0;
            ((static_cast<size_t>(idxs) < dims_array[dim] ? idx += steps[dim++] * idxs : throw std::out_of_range("Index out of bounds")), ...);
            return data[idx];
        }

        template<typename... Idxs>
        requires(sizeof...(Idxs) == Rank)
        const T& operator()(Idxs... idxs) const {
            size_t idx = 0;
            size_t dim = 0;
            ((static_cast<size_t>(idxs) < dims_array[dim] ? idx += steps[dim++] * idxs : throw std::out_of_range("Index out of bounds")), ...);
            return data[idx];
        }

        T& operator()(const std::array<size_t, Rank>& idxs) {
            size_t idx = 0;
            for (size_t dim = 0; dim < Rank; dim++) {
                idxs[dim] < dims_array[dim] ? idx += steps[dim] * idxs[dim] : throw std::out_of_range("Index out of bounds");
            }
            return data[idx];
        }

        const T& operator()(const std::array<size_t, Rank>& idxs) const {
            size_t idx = 0;
            for (size_t dim = 0; dim < Rank; dim++) {
                idxs[dim] < dims_array[dim] ? idx += steps[dim] * idxs[dim] : throw std::out_of_range("Index out of bounds");
            }
            return data[idx];
        }

        T& operator[](const size_t idx) {
            if (idx >= data.size()) throw std::out_of_range("Index out of bounds");
            return data[idx];
        }

        const T& operator[](const size_t idx) const {
            if (idx >= data.size()) throw std::out_of_range("Index out of bounds");
            return data[idx];
        }

        [[nodiscard]] size_t get_size() const {
            return data.size();
        }

        const std::array<size_t, Rank>& shape() const noexcept {
            return dims_array;
        }

        const std::array<size_t, Rank>& get_steps() const noexcept {
            return steps;
        }

        void reshape(const std::array<size_t, Rank>& new_shape) {
            data.resize(std::accumulate(new_shape.begin(), new_shape.end(), static_cast<size_t>(1), std::multiplies()));
            dims_array = new_shape;
            update_steps();
        }

        template<typename... Dims>
        requires(sizeof...(Dims) == Rank)
        void reshape(const Dims... dims) {
            std::array<size_t, Rank> new_shape{static_cast<size_t>(dims)...};
            data.resize(std::accumulate(new_shape.begin(), new_shape.end(), static_cast<size_t>(1), std::multiplies()));
            dims_array = new_shape;
            update_steps();
        }

        void fill(const T& value) noexcept {
            std::fill(data.begin(), data.end(), value);
        }

        template <typename BinaryOperator, size_t OtherRank>
        Tensor<T, std::max(Rank, OtherRank)> apply_elementwise(const Tensor<T, OtherRank>& other, BinaryOperator op) const {

            constexpr size_t ResultRank = std::max(Rank, OtherRank);
            std::array<size_t, ResultRank> result_shape{};

            const auto& other_shape = other.shape();
            const auto& other_steps = other.get_steps();

            for (size_t i = 0; i < ResultRank; ++i) {
                size_t dim_this = i < ResultRank - Rank ? 1 : dims_array[i - (ResultRank - Rank)];
                size_t dim_other = i < ResultRank - OtherRank ? 1 : other_shape[i - (ResultRank - OtherRank)];

                if (dim_this != dim_other && dim_this != 1 && dim_other != 1) throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");

                result_shape[i] = std::max(dim_this, dim_other);
            }

            Tensor<T, ResultRank> result(result_shape);
            const auto result_steps = result.get_steps();
            const size_t total = get_size();

            for (size_t idx = 0; idx < total; ++idx) {
                size_t idx_this = 0;
                size_t idx_other = 0;
                size_t temp = idx;

                for (size_t i = 0; i < ResultRank; ++i) {
                    const size_t coord = temp / result_steps[i];
                    temp %= result_steps[i];

                    if (i >= ResultRank - Rank) {
                        const size_t dim = i - (ResultRank - Rank);
                        const size_t step = dims_array[dim] == 1 ? 0 : steps[dim];
                        idx_this += coord * step;
                    }

                    if (i >= ResultRank - OtherRank) {
                        const size_t dim = i - (ResultRank - OtherRank);
                        const size_t step = other_shape[dim] == 1 ? 0 : other_steps[dim];
                        idx_other += coord * step;
                    }
                }

                result[idx] = op(data[idx_this], other[idx_other]);
            }

            return result;
        }

        template<size_t OtherRank>
        Tensor<T, std::max(Rank, OtherRank)> operator+(const Tensor<T, OtherRank>& other) const {
            return apply_elementwise(other, std::plus());
        }

        template<size_t OtherRank>
        Tensor<T, std::max(Rank, OtherRank)> operator-(const Tensor<T, OtherRank>& other) const {
            return apply_elementwise(other, std::minus());
        }

        template<size_t OtherRank>
        Tensor<T, std::max(Rank, OtherRank)> operator*(const Tensor<T, OtherRank>& other) const {
            return apply_elementwise(other, std::multiplies());
        }

        template<size_t OtherRank>
        Tensor<T, std::max(Rank, OtherRank)> operator/(const Tensor<T, OtherRank>& other) const {
            return apply_elementwise(other, std::divides());
        }

        Tensor operator+(const T& scalar) const {
            Tensor result(dims_array);
            std::transform(data.begin(), data.end(), result.data.begin(), [&](const T& value) { return value + scalar; });
            return result;
        }

        friend Tensor operator+(const T& scalar, const Tensor& tensor) {
            Tensor result(tensor.dims_array);
            std::transform(tensor.data.begin(), tensor.data.end(), result.data.begin(), [&](const T& value) { return scalar + value; });
            return result;
        }

        Tensor operator-(const T& scalar) const {
            Tensor result(dims_array);
            std::transform(data.begin(), data.end(), result.data.begin(), [&](const T& value) { return value - scalar; });
            return result;
        }

        friend Tensor operator-(const T& scalar, const Tensor& tensor) {
            Tensor result(tensor.dims_array);
            std::transform(tensor.data.begin(), tensor.data.end(), result.data.begin(), [&](const T& value) { return scalar - value; });
            return result;
        }

        Tensor operator*(const T& scalar) const {
            Tensor result(dims_array);
            std::transform(data.begin(), data.end(), result.data.begin(), [&](const T& value) { return value * scalar; });
            return result;
        }

        friend Tensor operator*(const T& scalar, const Tensor& tensor) {
            Tensor result(tensor.dims_array);
            std::transform(tensor.data.begin(), tensor.data.end(), result.data.begin(), [&](const T& value) { return scalar * value; });
            return result;
        }

        Tensor operator/(const T& scalar) const {
            Tensor result(dims_array);
            std::transform(data.begin(), data.end(), result.data.begin(), [&](const T& value) { return value / scalar; });
            return result;
        }

        friend Tensor operator/(const T& scalar, const Tensor& tensor) {
            Tensor result(tensor.dims_array);
            std::transform(tensor.data.begin(), tensor.data.end(), result.data.begin(), [&](const T& value) { return scalar / value; });
            return result;
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) requires (Rank > 1) {

            const auto& shape = tensor.shape();
            std::array<size_t, Rank> index{};

            std::function<void(size_t, size_t)> print_recursive = [&](size_t dim, const size_t indent) {
                os << std::string(indent, ' ') << "{\n";

                for (size_t i = 0; i < shape[dim]; ++i) {
                    index[dim] = i;

                    if (dim == Rank - 2) {
                        os << std::string(indent + 2, ' ');
                        for (size_t j = 0; j < shape[Rank - 1]; ++j) {
                            index[Rank - 1] = j;
                            os << tensor(index) << " ";
                        }
                        os << "\n";
                    } else {
                        print_recursive(dim + 1, indent + 2);
                    }
                }

                os << std::string(indent, ' ') << "}\n";
            };

            print_recursive(0, 0);
            return os;
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) requires (Rank == 1) {
            std::ranges::copy(tensor, std::ostream_iterator<T>(os, " "));
            os << "\n";
            return os;
        }

        auto begin() noexcept {
            return data.begin();
        }

        auto end() noexcept {
            return data.end();
        }

        auto begin() const noexcept {
            return data.begin();
        }

        auto end() const noexcept {
            return data.end();
        }

        auto cbegin() const noexcept {
            return data.cbegin();
        }

        auto cend() const noexcept {
            return data.cend();
        }

        auto rbegin() noexcept {
            return data.rbegin();
        }

        auto rend() noexcept {
            return data.rend();
        }

        auto rbegin() const noexcept {
            return data.rbegin();
        }

        auto rend() const noexcept {
            return data.rend();
        }

        auto crbegin() const noexcept {
            return data.crbegin();
        }

        auto crend() const noexcept {
            return data.crend();
        }

        template<typename Function>
        auto apply(Function function) const {
            Tensor result(dims_array);
            std::transform(data.begin(), data.end(), result.begin(), function);
            return result;
        }

    };

    template <typename T, size_t Rank>
    Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& tensor) requires (Rank > 1) {
        std::array<size_t, Rank> new_shape = tensor.shape();
        std::swap(new_shape[Rank - 1], new_shape[Rank - 2]);

        Tensor<T, Rank> result(new_shape);
        const auto& result_steps = result.get_steps();
        const auto& input_steps = tensor.get_steps();
        const size_t total = tensor.get_size();

        for (size_t idx = 0; idx < total; ++idx) {
            size_t temp = idx;
            size_t out_idx = 0;

            for (size_t i = 0; i < Rank; ++i) {
                const size_t coord = temp / input_steps[i];
                temp %= input_steps[i];
                out_idx += coord * result_steps[i == Rank - 2 ? Rank - 1 : i == Rank - 1 ? Rank - 2 : i];
            }

            result[out_idx] = tensor[idx];
        }

        return result;
    }

    template <typename T, size_t Rank>
    auto matrix_product(const Tensor<T, Rank>& tensor1, const Tensor<T, Rank>& tensor2) -> Tensor<T, Rank> {
        const auto& t1_shape = tensor1.shape();
        const auto& t2_shape = tensor2.shape();

        if (t1_shape[Rank - 1] != t2_shape[Rank - 2]) {
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
        }

        for (size_t i = 0; i < Rank - 2; ++i) {
            if (t1_shape[i] != t2_shape[i]) {
                throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
            }
        }

        std::array<size_t, Rank> result_shape = t1_shape;
        result_shape[Rank - 1] = t2_shape[Rank - 1];
        Tensor<T, Rank> result(result_shape);

        const auto& t1_steps = tensor1.get_steps();
        const auto& t2_steps = tensor2.get_steps();
        const auto& result_steps = result.get_steps();

        size_t batch_size = 1;
        for (size_t i = 0; i < Rank - 2; ++i) {
            batch_size *= t1_shape[i];
        }

        const size_t M = t1_shape[Rank - 2];
        const size_t N = t1_shape[Rank - 1];
        const size_t P = t2_shape[Rank - 1];

        for (size_t batch = 0; batch < batch_size; ++batch) {
            size_t t1_offset = 0;
            size_t t2_offset = 0;
            size_t result_offset = 0;

            size_t temp = batch;
            for (size_t i = Rank - 2; i-- > 0;) {
                size_t idx = temp % t1_shape[i];
                temp /= t1_shape[i];
                t1_offset += idx * t1_steps[i];
                t2_offset += idx * t2_steps[i];
                result_offset += idx * result_steps[i];
            }

            for (size_t i = 0; i < M; ++i) {
                size_t t1_row_offset = t1_offset + i * t1_steps[Rank - 2];
                size_t result_row_offset = result_offset + i * result_steps[Rank - 2];
                for (size_t j = 0; j < P; ++j) {
                    T sum{};
                    for (size_t k = 0; k < N; ++k) {
                        sum += tensor1[t1_row_offset + k * t1_steps[Rank - 1]] * tensor2[t2_offset + k * t2_steps[Rank - 2] + j * t2_steps[Rank - 1]];
                    }
                    result[result_row_offset + j * result_steps[Rank - 1]] = sum;
                }
            }
        }

        return result;
    }

}

#endif //TENSOR_H