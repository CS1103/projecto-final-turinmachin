#include "game/sdl_extra.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL_render.h>
#include <SDL_timer.h>
#include <cmath>

namespace sdl_extra {

    auto get_performance_time() -> long double {
        return static_cast<long double>(SDL_GetPerformanceCounter()) /
               SDL_GetPerformanceFrequency();
    }

}  // namespace sdl_extra
