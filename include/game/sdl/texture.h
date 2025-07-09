#ifndef INCLUDE_GAME_SDL_TEXTURE_H
#define INCLUDE_GAME_SDL_TEXTURE_H

#include <SDL_pixels.h>
#include <SDL_render.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace game::sdl {

    auto extract_pixel_data(SDL_Renderer* renderer,
                            SDL_Texture* texture,
                            std::uint32_t format,
                            std::size_t width,
                            std::size_t height,
                            std::size_t bytes_per_pixel) -> std::vector<std::uint8_t>;

}  // namespace game::sdl

#endif
