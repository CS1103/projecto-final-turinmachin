#ifndef INCLUDE_GAME_SDL_TIME_H
#define INCLUDE_GAME_SDL_TIME_H

#include <SDL_render.h>

namespace game::sdl {

    auto get_performance_time() -> long double;

}  // namespace game::sdl

#endif
