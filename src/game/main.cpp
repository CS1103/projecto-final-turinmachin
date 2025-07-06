#include <SDL2/SDL_render.h>
#include <SDL2/SDL_video.h>
#include <SDL_ttf.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <ostream>
#include <print>
#include "game/constants.h"
#include "game/font_data.h"
#include "game/sdl_extra.h"
#include "game/state.h"

auto main() -> int {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::println(std::cerr, "Could not initialize video: {}", SDL_GetError());
        return 1;
    }

    if (TTF_Init() != 0) {
        std::println(std::cerr, "Could not initialize TTF: {}", SDL_GetError());
        return 1;
    }

    TTF_Font* display_font =
        TTF_OpenFontRW(SDL_RWFromConstMem(static_cast<const void*>(DISPLAY_FONT_DATA),
                                          static_cast<int>(DISPLAY_FONT_DATA_LEN)),
                       1, 15);

    if (display_font == nullptr) {
        std::println(std::cerr, "Could not load font: {}", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Brain Age", SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    if (window == nullptr || renderer == nullptr) {
        std::println(std::cerr, "Could not initialize window or renderer: {}", SDL_GetError());
        return 1;
    }

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    SDL_RenderPresent(renderer);

    long double last_time = sdl_extra::get_performance_time();
    long double time_accumulator = 0.0;

    State state(renderer);

    while (!state.quit) {
        const long double frame_start = sdl_extra::get_performance_time();

        SDL_Event event;
        while (SDL_PollEvent(&event) != 0) {
            state.handle_event(event);
        }

        const long double new_time = sdl_extra::get_performance_time();
        time_accumulator = std::min(time_accumulator + new_time - last_time, MAX_TIME_ACCUMULATOR);
        last_time = new_time;

        while (time_accumulator >= DELTA) {
            state.update(DELTA);
            time_accumulator -= DELTA;
        }

        state.render();

        const long double offset = sdl_extra::get_performance_time() - frame_start;
        const long double delay = DELTA - offset;
        if (delay > 0) {
            SDL_Delay(static_cast<std::uint64_t>(delay * 1000));
        }
    }

    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);
    SDL_Quit();

    return 0;
}
