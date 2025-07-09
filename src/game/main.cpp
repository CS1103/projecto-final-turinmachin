#include <SDL2/SDL_render.h>
#include <SDL2/SDL_video.h>
#include <SDL_ttf.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <print>
#include <random>
#include <string>
#include "common/agent.h"
#include "common/data.h"
#include "game/constants.h"
#include "game/game.h"
#include "game/render.h"
#include "game/sdl/time.h"

auto main() -> int {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::println(std::cerr, "Could not initialize video: {}", SDL_GetError());
        return 1;
    }

    if (TTF_Init() != 0) {
        std::println(std::cerr, "Could not initialize TTF: {}", SDL_GetError());
        return 1;
    }

    SDL_Window* const window =
        SDL_CreateWindow(GAME_TITLE.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                         WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    SDL_Renderer* const renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    if (window == nullptr || renderer == nullptr) {
        std::println(std::cerr, "Could not initialize window or renderer: {}", SDL_GetError());
        return 1;
    }

    if (TTF_Init() != 0) {
        std::println(std::cerr, "Could not initialize TTF: {}", SDL_GetError());
        return 1;
    }

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    SDL_RenderPresent(renderer);

    long double last_time = sdl::time::get_performance_time();
    long double time_accumulator = 0.0;

    std::random_device rd{};

    using namespace agent;

    std::ifstream net_file = data::get_data_file("net.pp20");
    std::unique_ptr<IDigitAgent> agent = std::make_unique<DigitReader>(net_file);
    net_file.close();

    Game game(renderer, std::move(agent), std::mt19937(rd()));

    const std::string font_path = data::get_data_file_path("ComicShannsMono-Regular.ttf");

    render::ResourceManager resources{};
    resources.font = TTF_OpenFont(font_path.c_str(), 96);

    while (!game.should_quit()) {
        const long double frame_start = sdl::time::get_performance_time();

        SDL_Event event{};
        while (SDL_PollEvent(&event) != 0) {
            game.handle_event(event);
        }

        const long double new_time = sdl::time::get_performance_time();
        time_accumulator = std::min(time_accumulator + new_time - last_time, MAX_TIME_ACCUMULATOR);
        last_time = new_time;

        while (time_accumulator >= DELTA) {
            game.update(DELTA);
            time_accumulator -= DELTA;
        }

        game.render(resources);

        const long double offset = sdl::time::get_performance_time() - frame_start;
        const long double delay = DELTA - offset;
        if (delay > 0) {
            SDL_Delay(static_cast<std::uint64_t>(delay * 1000));
        }
    }

    TTF_CloseFont(resources.font);
    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);
    SDL_Quit();

    return 0;
}
