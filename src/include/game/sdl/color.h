#ifndef INCLUDE_GAME_COLOR_H
#define INCLUDE_GAME_COLOR_H

#include <SDL_pixels.h>

namespace sdl::color {

    constexpr SDL_Color WHITE = {255, 255, 255, SDL_ALPHA_OPAQUE};
    constexpr SDL_Color BLACK = {0, 0, 0, SDL_ALPHA_OPAQUE};
    constexpr SDL_Color GRAY = {30, 30, 30, SDL_ALPHA_OPAQUE};
    constexpr SDL_Color LIGHT_GRAY = {120, 120, 120, SDL_ALPHA_OPAQUE};
    constexpr SDL_Color GREEN = {50, 255, 50, SDL_ALPHA_OPAQUE};
    constexpr SDL_Color RED = {255, 50, 50, SDL_ALPHA_OPAQUE};

}  // namespace sdl::color

#endif
