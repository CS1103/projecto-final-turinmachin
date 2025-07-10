#include <utec/algebra/tensor.h>
#include <array>
#include <catch/catch_amalgamated.hpp>
#include <stdexcept>

using utec::algebra::Tensor;

TEST_CASE("2d tensor basic operations", "[tensor]") {
    Tensor<int, 2> tensor(2, 3);
    tensor = {1, 2, 3, 4, 5, 6};

    SECTION("fill", "[tensor]") {
        tensor.fill(7);

        REQUIRE(tensor[0] == 7);
        REQUIRE(tensor[1] == 7);
        REQUIRE(tensor[2] == 7);
        REQUIRE(tensor[3] == 7);
        REQUIRE(tensor[4] == 7);
        REQUIRE(tensor[5] == 7);
    }

    SECTION("valid indexing", "[tensor]") {
        REQUIRE(tensor(0, 0) == 1);
        REQUIRE(tensor(0, 1) == 2);
        REQUIRE(tensor(0, 2) == 3);
        REQUIRE(tensor(1, 0) == 4);
        REQUIRE(tensor(1, 1) == 5);
        REQUIRE(tensor(1, 2) == 6);
    }

    SECTION("invalid indexing", "[tensor]") {
        REQUIRE_THROWS_AS(tensor(3, 0), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(2, 42), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(0, 3), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(-1, 0), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(2, 3), std::out_of_range);
    }

    SECTION("exact reshape", "[tensor]") {
        tensor.reshape(3, 2);

        REQUIRE(tensor.shape() == std::array<size_t, 2>{3, 2});
        REQUIRE(tensor(0, 0) == 1);
        REQUIRE(tensor(0, 1) == 2);
        REQUIRE(tensor(1, 0) == 3);
        REQUIRE(tensor(1, 1) == 4);
        REQUIRE(tensor(2, 0) == 5);
        REQUIRE(tensor(2, 1) == 6);
    }
    SECTION("non-exact reshape", "[tensor]") {
        tensor.reshape(3, 4);
        REQUIRE_NOTHROW(tensor(2, 2));
        REQUIRE_NOTHROW(tensor(2, 3));

        tensor.reshape(6, 7);
        REQUIRE_NOTHROW(tensor(5, 3));
    }

    SECTION("scalar multiplication", "[tensor]") {
        Tensor<int, 2> prod = tensor * 3;

        REQUIRE(prod(0, 0) == 3);
        REQUIRE(prod(0, 1) == 6);
        REQUIRE(prod(0, 2) == 9);
        REQUIRE(prod(1, 0) == 12);
        REQUIRE(prod(1, 1) == 15);
        REQUIRE(prod(1, 2) == 18);
    }

    // SECTION("transpose", "[tensor]") {
    //     Tensor<int, 2> mat(3, 2);
    //     mat(0, 0) = 1;
    //     mat(0, 1) = 2;
    //     mat(1, 0) = 3;
    //     mat(1, 1) = 4;
    //     mat(2, 0) = 5;
    //     mat(2, 1) = 6;
    //
    //     Tensor<int, 2> mat_t = mat.transpose_2d();
    //
    //     REQUIRE(mat_t.shape() == std::array<size_t, 2>{2, 3});
    //     REQUIRE(mat_t(0, 0) == 1);
    //     REQUIRE(mat_t(0, 1) == 3);
    //     REQUIRE(mat_t(0, 2) == 5);
    //     REQUIRE(mat_t(1, 0) == 2);
    //     REQUIRE(mat_t(1, 1) == 4);
    //     REQUIRE(mat_t(1, 2) == 6);
    // }
}

TEST_CASE("2d tensor binary operations", "[tensor]") {
    Tensor<int, 2> tensor_a(2, 3);
    tensor_a[0] = 7;
    tensor_a[1] = 17;
    tensor_a[2] = 31;
    tensor_a[3] = 0;
    tensor_a[4] = 63;
    tensor_a[5] = 102;

    Tensor<int, 2> tensor_b(2, 3);
    tensor_b[0] = 3;
    tensor_b[1] = 5;
    tensor_b[2] = -2;
    tensor_b[3] = 5;
    tensor_b[4] = 6;
    tensor_b[5] = 42;

    SECTION("addition", "[tensor]") {
        Tensor<int, 2> sum = tensor_a + tensor_b;

        REQUIRE(sum(0, 0) == 10);
        REQUIRE(sum(0, 1) == 22);
        REQUIRE(sum(0, 2) == 29);
        REQUIRE(sum(1, 0) == 5);
        REQUIRE(sum(1, 1) == 69);
        REQUIRE(sum(1, 2) == 144);
    }

    SECTION("subtraction", "[tensor]") {
        Tensor<int, 2> diff = tensor_a - tensor_b;

        REQUIRE(diff(0, 0) == 4);
        REQUIRE(diff(0, 1) == 12);
        REQUIRE(diff(0, 2) == 33);
        REQUIRE(diff(1, 0) == -5);
        REQUIRE(diff(1, 1) == 57);
        REQUIRE(diff(1, 2) == 60);
    }
}

TEST_CASE("2d tensor multiplication", "[tensor]") {
    Tensor<int, 2> tensor_a(2, 3);
    tensor_a(0, 0) = 7;
    tensor_a(0, 1) = -2;
    tensor_a(0, 2) = 4;
    tensor_a(1, 0) = 8;
    tensor_a(1, 1) = 7;
    tensor_a(1, 2) = 0;

    SECTION("multiplication with broadcasting", "[tensor]") {
        Tensor<int, 2> tensor_b(2, 1);
        tensor_b(0, 0) = 3;
        tensor_b(1, 0) = 4;

        const Tensor<int, 2> prod_a = tensor_a * tensor_b;

        REQUIRE(prod_a(0, 0) == 21);
        REQUIRE(prod_a(0, 1) == -6);
        REQUIRE(prod_a(0, 2) == 12);
        REQUIRE(prod_a(1, 0) == 32);
        REQUIRE(prod_a(1, 1) == 28);
        REQUIRE(prod_a(1, 2) == 0);

        const Tensor<int, 2> prod_b = tensor_b * tensor_a;
        REQUIRE(prod_a == prod_b);
    }

    SECTION("element-wise multiplication", "[tensor]") {
        Tensor<int, 2> tensor_b(2, 3);
        tensor_b(0, 0) = 2;
        tensor_b(0, 1) = 3;
        tensor_b(0, 2) = -2;
        tensor_b(1, 0) = 0;
        tensor_b(1, 1) = 5;
        tensor_b(1, 2) = 4;

        const Tensor<int, 2> prod_a = tensor_a * tensor_b;

        REQUIRE(prod_a(0, 0) == 14);
        REQUIRE(prod_a(0, 1) == -6);
        REQUIRE(prod_a(0, 2) == -8);
        REQUIRE(prod_a(1, 0) == 0);
        REQUIRE(prod_a(1, 1) == 35);
        REQUIRE(prod_a(1, 2) == 0);

        const Tensor<int, 2> prod_b = tensor_b * tensor_a;
        REQUIRE(prod_a == prod_b);
    }
}

TEST_CASE("3d tensor", "[tensor]") {
    Tensor<int, 3> tensor(3, 5, 2);
    tensor[7] = 42;
    tensor[12] = 38;
    tensor[20] = 256;
    tensor[21] = 100;

    SECTION("valid indexing", "[tensor]") {
        REQUIRE(tensor(0, 3, 1) == 42);
        REQUIRE(tensor(1, 1, 0) == 38);
        REQUIRE(tensor(2, 0, 0) == 256);
        REQUIRE(tensor(2, 0, 1) == 100);
    }

    SECTION("invalid indexing", "[tensor]") {
        REQUIRE_THROWS_AS(tensor(4, 0, 0), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(1, 2, -1), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(1, 5, 29), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(2, -4, 2), std::out_of_range);
    }

    SECTION("valid assigning", "[tensor]") {
        tensor(0, 3, 1) = 1;
        tensor(1, 1, 0) = 2;
        tensor(2, 0, 0) = 3;
        tensor(2, 0, 1) = 4;

        REQUIRE(tensor[7] == 1);
        REQUIRE(tensor[12] == 2);
        REQUIRE(tensor[20] == 3);
        REQUIRE(tensor[21] == 4);
    }

    SECTION("invalid assigning", "[tensor]") {
        REQUIRE_THROWS_AS(tensor(4, 0, 0) = 1, std::out_of_range);
        REQUIRE_THROWS_AS(tensor(1, 2, -1) = 2, std::out_of_range);
        REQUIRE_THROWS_AS(tensor(1, 5, 29) = 3, std::out_of_range);
        REQUIRE_THROWS_AS(tensor(2, -4, 2) = 4, std::out_of_range);
    }
}
