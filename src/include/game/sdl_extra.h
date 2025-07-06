#ifndef INCLUDE_GAME_SDL_EXTRA_H
#define INCLUDE_GAME_SDL_EXTRA_H

#include <SDL_render.h>
#include <cstdint>
namespace sdl_extra {

    auto get_performance_time() -> long double;

    void draw_better_thick_line(SDL_Renderer* renderer,
                                float x1,
                                float y1,
                                float x2,
                                float y2,
                                int thickness);

}  // namespace sdl_extra

#endif
