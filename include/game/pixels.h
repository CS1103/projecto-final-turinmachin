#ifndef INCLUDE_GAME_PIXELS_H
#define INCLUDE_GAME_PIXELS_H

#include <SDL3/SDL_rect.h>
#include <cstdint>
#include <vector>

namespace game {

    auto make_grayscale(const std::vector<std::uint8_t>& pixel_data, std::size_t width)
        -> std::vector<std::uint8_t>;

    auto downscale_to_8x8(const std::vector<std::uint8_t>& grayscale_data,
                          const SDL_Rect& src_rect,
                          int width) -> std::vector<std::uint8_t>;

}  // namespace game

#endif
