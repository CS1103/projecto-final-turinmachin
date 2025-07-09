#include "game/sdl/time.h"
#include <SDL_timer.h>

namespace game::sdl {

    auto get_performance_time() -> long double {
        return static_cast<long double>(SDL_GetPerformanceCounter()) /
               SDL_GetPerformanceFrequency();
    }

}  // namespace game::sdl
