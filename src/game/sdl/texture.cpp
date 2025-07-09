#include "game/sdl/texture.h"
#include <SDL_pixels.h>
#include <SDL_render.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sdl::texture {

    auto extract_pixel_data(SDL_Renderer* const renderer,
                            SDL_Texture* const texture,
                            const std::uint32_t format,
                            const std::size_t width,
                            const std::size_t height,
                            const std::size_t bytes_per_pixel) -> std::vector<std::uint8_t> {
        std::vector<std::uint8_t> pixel_data(width * height * bytes_per_pixel);

        SDL_Texture* const cur_target = SDL_GetRenderTarget(renderer);
        SDL_SetRenderTarget(renderer, texture);

        if (SDL_RenderReadPixels(renderer, nullptr, format, pixel_data.data(),
                                 (int)(width * bytes_per_pixel)) != 0) {
            throw std::runtime_error("Could not read texture pixels: " +
                                     std::string(SDL_GetError()));
        }

        SDL_SetRenderTarget(renderer, cur_target);
        return pixel_data;
    }

}  // namespace sdl::texture
