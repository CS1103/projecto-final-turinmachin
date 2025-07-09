#ifndef INCLUDE_GAME_SDL_DRAW_H
#define INCLUDE_GAME_SDL_DRAW_H

#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_render.h>

namespace game::sdl {

    void draw_line(SDL_Renderer* renderer,
                   float x1,
                   float y1,
                   float x2,
                   float y2,
                   float thickness,
                   SDL_FColor color);

    void draw_filled_circle(SDL_Renderer* renderer,
                            float cx,
                            float cy,
                            float radius,
                            SDL_FColor color,
                            int segments = 64);

}  // namespace game::sdl

#endif
