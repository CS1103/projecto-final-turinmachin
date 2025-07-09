#include "game/pixels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace game {

    auto make_grayscale(const std::vector<std::uint8_t>& pixel_data, const std::size_t width)
        -> std::vector<std::uint8_t> {
        const std::size_t height = pixel_data.size() / (4 * width);
        std::vector<std::uint8_t> result(pixel_data.size() / 4);

        for (std::size_t i = 0; i < height; ++i) {
            for (std::size_t j = 0; j < width; ++j) {
                const int r = pixel_data.at((((j * width) + i) * 4) + 1);
                const int g = pixel_data.at((((j * width) + i) * 4) + 2);
                const int b = pixel_data.at((((j * width) + i) * 4) + 3);

                result.at((j * width) + i) = static_cast<std::uint8_t>((r + g + b) / 3.0);
            }
        }

        return result;
    }

    auto downscale_to_8x8(const std::vector<std::uint8_t>& grayscale_data,
                          const SDL_Rect& src_rect,
                          const int width) -> std::vector<std::uint8_t> {
        std::vector<std::uint8_t> downscaled_data(64, 0);

        const double block_w = static_cast<double>(src_rect.w) / 8.0;
        const double block_h = static_cast<double>(src_rect.h) / 8.0;

        for (int by = 0; by < 8; ++by) {
            for (int bx = 0; bx < 8; ++bx) {
                double sum = 0.0;
                int count = 0;

                const int x_start = static_cast<int>(std::floor(bx * block_w));
                const int x_end = static_cast<int>(std::floor((bx + 1) * block_w));

                const int y_start = static_cast<int>(std::floor(by * block_h));
                const int y_end = static_cast<int>(std::floor((by + 1) * block_h));

                for (int y = y_start; y < y_end; ++y) {
                    for (int x = x_start; x < x_end; ++x) {
                        const int global_x = src_rect.x + x;
                        const int global_y = src_rect.y + y;

                        const std::size_t index = (global_y * width) + global_x;
                        if (index < grayscale_data.size()) {
                            sum += grayscale_data.at(index);
                            ++count;
                        }
                    }
                }

                const double average = (count > 0) ? (sum / count) : 0.0;
                downscaled_data.at((by * 8) + bx) =
                    static_cast<std::uint8_t>(std::clamp(average, 0.0, 255.0));
            }
        }

        return downscaled_data;
    }

}  // namespace game
