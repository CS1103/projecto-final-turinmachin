#include "game/sdl/texture.h"
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_surface.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace game::sdl {

    auto extract_pixel_data(SDL_Renderer* const renderer,
                            SDL_Texture* const texture,
                            const std::size_t width,
                            const std::size_t height,
                            const std::size_t bytes_per_pixel) -> std::vector<std::uint8_t> {
        SDL_Texture* const cur_target = SDL_GetRenderTarget(renderer);

        SDL_SetRenderTarget(renderer, texture);
        SDL_Surface* surface = SDL_RenderReadPixels(renderer, nullptr);
        SDL_SetRenderTarget(renderer, cur_target);

        if (surface == nullptr) {
            throw std::runtime_error("Could not read texture pixels: " +
                                     std::string(SDL_GetError()));
        }

        std::vector<std::uint8_t> pixel_data(width * height * bytes_per_pixel);
        std::memcpy(pixel_data.data(), surface->pixels, pixel_data.size());

        SDL_DestroySurface(surface);

        return pixel_data;
    }

}  // namespace game::sdl
