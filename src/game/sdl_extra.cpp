#include "game/sdl_extra.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL_render.h>
#include <SDL_timer.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sdl_extra {

    auto get_performance_time() -> long double {
        return static_cast<long double>(SDL_GetPerformanceCounter()) /
               SDL_GetPerformanceFrequency();
    }

    void draw_better_thick_line(SDL_Renderer* renderer,
                                const float x1,
                                const float y1,
                                const float x2,
                                const float y2,
                                const int thickness) {
        constexpr float STEP = 1.0;

        const float dx = static_cast<float>(x2 - x1);
        const float dy = static_cast<float>(y2 - y1);
        const float len = std::sqrt((dx * dx) + (dy * dy));

        const float step_x = -dy / len;
        const float step_y = dx / len;

        for (int i = 0; i < thickness; ++i) {
            SDL_RenderDrawLine(
                renderer, static_cast<int>(x1 + (i * step_x)), static_cast<int>(y1 + (i * step_y)),
                static_cast<int>(x2 + (i * step_x)), static_cast<int>(y2 + (i * step_y)));
            SDL_RenderDrawLine(
                renderer, static_cast<int>(x1 - (i * step_x)), static_cast<int>(y1 - (i * step_y)),
                static_cast<int>(x2 - (i * step_x)), static_cast<int>(y2 - (i * step_y)));
        }
    }

}  // namespace sdl_extra
