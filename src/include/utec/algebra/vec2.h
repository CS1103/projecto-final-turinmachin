#ifndef INCLUDE_UTEC_ALGEBRA_VEC2_H
#define INCLUDE_UTEC_ALGEBRA_VEC2_H

#include <cmath>
#include <type_traits>

namespace utec::algebra {

    template <typename T>
    struct Vec2 {
        T x;
        T y;

        constexpr Vec2() = default;

        template <typename U>
            requires(std::is_convertible_v<U, T>)
        constexpr explicit Vec2(const Vec2<U>& other)
            : x(other.x),
              y(other.y) {}

        constexpr Vec2(T x, T y)
            : x(std::move(x)),
              y(std::move(y)) {}

        static constexpr auto Unit() -> Vec2<T> {
            return {T{1}, T{1}};
        }

        [[nodiscard]] constexpr auto operator+(const Vec2<T>& rhs) const -> Vec2<T> {
            return {x + rhs.x, y + rhs.y};
        }

        constexpr auto operator+=(const Vec2<T>& rhs) -> Vec2<T>& {
            x += rhs.x;
            y += rhs.y;
            return *this;
        }

        [[nodiscard]] constexpr auto operator-(const Vec2<T>& rhs) const -> Vec2<T> {
            return {x - rhs.x, y - rhs.y};
        }

        constexpr auto operator-=(const Vec2<T>& rhs) -> Vec2<T>& {
            x -= rhs.x;
            y -= rhs.y;
            return *this;
        }

        [[nodiscard]] constexpr auto operator*(const T& scalar) const -> Vec2<T> {
            return {x * scalar, y * scalar};
        }

        constexpr auto operator*=(const Vec2<T>& rhs) -> Vec2<T>& {
            x *= rhs.x;
            y *= rhs.y;
            return *this;
        }

        [[nodiscard]] constexpr auto operator/(const T& scalar) const -> Vec2<T> {
            return {x / scalar, y / scalar};
        }

        constexpr auto operator/=(const Vec2<T>& rhs) -> Vec2<T>& {
            x /= rhs.x;
            y /= rhs.y;
            return *this;
        }

        [[nodiscard]] constexpr auto dot(const Vec2<T>& rhs) const -> T {
            return (x * rhs.x) + (y * rhs.y);
        }

        [[nodiscard]] constexpr auto norm_sq() const -> T {
            return (x * x) + (y * y);
        }

        [[nodiscard]] constexpr auto norm() const -> T {
            return std::sqrt(norm_sq());
        }

        [[nodiscard]] constexpr auto normalized() const -> Vec2<T> {
            return *this / norm();
        }
    };

}  // namespace utec::algebra

#endif
