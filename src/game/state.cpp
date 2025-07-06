#include "game/state.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_mouse.h>
#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_render.h>
#include <SDL_error.h>
#include <SDL_keycode.h>
#include <SDL_surface.h>
#include <SDL_ttf.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <print>
#include <stdexcept>
#include "game/constants.h"
#include "utec/algebra/tensor.h"
#include "utec/nn/neural_network.h"

using utec::algebra::Tensor;

State::State(SDL_Renderer* const renderer)
    : renderer(renderer),

      canvas_texture(SDL_CreateTexture(renderer,
                                       SDL_PIXELFORMAT_RGBA32,
                                       SDL_TEXTUREACCESS_TARGET,
                                       CANVAS_WIDTH,
                                       CANVAS_HEIGHT)) {
    if (canvas_texture == nullptr) {
        throw std::runtime_error("Could not create canvas texture: " + std::string(SDL_GetError()));
    }

    SDL_SetTextureScaleMode(canvas_texture, SDL_ScaleModeLinear);
    clear_canvas();

    std::ifstream infile("net.pp20");
    net = utec::neural_network::NeuralNetwork<double>::load(infile);
}

void State::handle_event(const SDL_Event& event) {
    switch (event.type) {
        case SDL_QUIT:
            quit = true;
            break;
        case SDL_KEYDOWN:
            keyboard.insert(event.key.keysym.sym);
            break;
        case SDL_KEYUP:
            keyboard.erase(event.key.keysym.sym);
            break;
        case SDL_MOUSEBUTTONDOWN:
            mouse_down = true;
            break;
        case SDL_MOUSEBUTTONUP: {
            constexpr int WIDTH = 8;
            constexpr int HEIGHT = 8;

            SDL_Texture* const texture_8x8 = SDL_CreateTexture(
                renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, WIDTH, HEIGHT);

            SDL_SetTextureScaleMode(texture_8x8, SDL_ScaleModeLinear);
            SDL_SetRenderTarget(renderer, texture_8x8);
            SDL_RenderCopy(renderer, canvas_texture, nullptr, nullptr);

            std::array<std::uint8_t, static_cast<std::size_t>(WIDTH * HEIGHT * 4)> pixel_data{};

            if (SDL_RenderReadPixels(renderer, nullptr, SDL_PIXELFORMAT_RGBA8888, pixel_data.data(),
                                     WIDTH * 4) != 0) {
                throw std::runtime_error("Could not read texture pixels: " +
                                         std::string(SDL_GetError()));
            }

            SDL_DestroyTexture(texture_8x8);
            SDL_SetRenderTarget(renderer, nullptr);

            std::array<std::uint8_t, static_cast<std::size_t>(WIDTH * HEIGHT)> pixels{};

            for (std::size_t i = 0; i < pixels.size(); ++i) {
                const int r = pixel_data.at((i * 4) + 1);
                const int g = pixel_data.at((i * 4) + 2);
                const int b = pixel_data.at((i * 4) + 3);
                pixels.at(i) = (r + g + b) / 3;
            }

            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    std::print("{:3} ", pixels.at((i * 8) + j));
                }
                std::println();
            }

            Tensor<double, 2> input(1, WIDTH * HEIGHT);
            std::ranges::transform(pixels, input.begin(),
                                   [](const auto pixel) { return (255 - pixel) / 255.0; });

            const Tensor<double, 2> output = net.predict(input);

            int max_index = 0;
            for (int j = 1; j < 10; ++j) {
                if (output[j] > output[max_index]) {
                    max_index = j;
                }
            }

            std::println("this is a {}", max_index);

            mouse_down = false;
            break;
        }

        default: {
        }
    }
}

void State::update(const double delta) {
    if (keyboard.contains(SDLK_r)) {
        clear_canvas();
    }

    int mouse_x = 0;
    int mouse_y = 0;
    SDL_GetMouseState(&mouse_x, &mouse_y);

    if (mouse_down && (mouse_x != last_mouse_x || mouse_y != last_mouse_y)) {
        constexpr int THICKNESS = 80;

        const auto x1 = static_cast<short>(mouse_x - CANVAS_X);
        const auto y1 = static_cast<short>(mouse_y - CANVAS_Y);
        const auto x2 = static_cast<short>(last_mouse_x - CANVAS_X);
        const auto y2 = static_cast<short>(last_mouse_y - CANVAS_Y);

        SDL_SetRenderTarget(renderer, canvas_texture);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
        thickLineRGBA(renderer, x1, y1, x2, y2, THICKNESS, 0, 0, 0, SDL_ALPHA_OPAQUE);

        filledCircleRGBA(renderer, x1, y1, THICKNESS / 2, 0, 0, 0, SDL_ALPHA_OPAQUE);
        filledCircleRGBA(renderer, x2, y2, THICKNESS / 2, 0, 0, 0, SDL_ALPHA_OPAQUE);
        SDL_SetRenderTarget(renderer, nullptr);
    }

    last_mouse_x = mouse_x;
    last_mouse_y = mouse_y;
}

void State::clear_canvas() const {
    SDL_SetRenderTarget(renderer, canvas_texture);

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);

    SDL_SetRenderTarget(renderer, nullptr);
}

void State::render() const {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);

    SDL_Rect canvas_rect = {CANVAS_X, CANVAS_Y, CANVAS_WIDTH, CANVAS_HEIGHT};
    SDL_RenderCopy(renderer, canvas_texture, nullptr, &canvas_rect);

    SDL_RenderPresent(renderer);
}

State::~State() {
    if (canvas_texture != nullptr) {
        SDL_DestroyTexture(canvas_texture);
    }
}
