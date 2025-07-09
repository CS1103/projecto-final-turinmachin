#include "game/game.h"
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
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/agent.h"
#include "game/constants.h"
#include "game/math/factory.h"
#include "game/pixels.h"
#include "game/render.h"
#include "game/sdl/color.h"
#include "game/sdl/text.h"
#include "game/sdl/texture.h"

using namespace agent;

Game::Game(SDL_Renderer* const renderer, std::unique_ptr<IDigitAgent> agent, std::mt19937 rng)
    : agent(std::move(agent)),
      rng(rng),
      renderer(renderer),
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

    std::ifstream infile("../share/net.pp20");
    net = utec::neural_network::NeuralNetwork<double>::load(infile);
}

Game::~Game() {
    if (canvas_texture != nullptr) {
        SDL_DestroyTexture(canvas_texture);
    }
}

auto Game::should_quit() const -> bool {
    return quit;
}

void Game::handle_event(const SDL_Event& event) {
    switch (event.type) {
        case SDL_QUIT:
            quit = true;
            break;
        case SDL_KEYDOWN: {
            const SDL_Keycode key = event.key.keysym.sym;

            if (state == State::Playing && key == SDLK_r) {
                clear_canvas();
                current_guess = std::nullopt;
            }
            break;
        }
        case SDL_MOUSEBUTTONDOWN:
            mouse_down = true;
            break;
        case SDL_MOUSEBUTTONUP: {
            mouse_down = false;

            if (state == State::Playing && current_equation) {
                process_drawing();
            }
            break;
        }
        default: {
        }
    }
}

void Game::process_drawing() {
    const std::vector<std::uint8_t> pixel_data = sdl::texture::extract_pixel_data(
        renderer, canvas_texture, SDL_PIXELFORMAT_RGBA8888, CANVAS_WIDTH, CANVAS_HEIGHT, 4);

    const std::vector<std::uint8_t> grayscale_data =
        pixels::make_grayscale(pixel_data, CANVAS_WIDTH);

    const SDL_Rect src_rect = {0, 0, CANVAS_WIDTH, CANVAS_HEIGHT};

    const std::vector<std::uint8_t> downscaled =
        pixels::downscale_to_8x8(grayscale_data, src_rect, CANVAS_WIDTH);

    std::vector<double> features(downscaled.size());
    std::ranges::transform(downscaled, features.begin(),
                           [](const auto pixel) { return (255 - pixel) / 255.0; });

    current_guess = agent->predict(features);

    if (current_guess == (*current_equation)->answer()) {
        state = State::WinTransition;
        wait_delay = 60;
    }
}

void Game::transition() {
    static const std::array<std::unique_ptr<math::IEquationFactory>, 3> EQUATION_FACTORIES{
        std::make_unique<math::AddEquationFactory>(rng),
        std::make_unique<math::SubtractEquationFactory>(rng),
        std::make_unique<math::DivideEquationFactory>(rng)};

    if (current_equation) {
        solved_equations.push_back(std::move(*current_equation));
        current_equation = std::nullopt;
    }

    std::uniform_int_distribution<std::size_t> dist(0, EQUATION_FACTORIES.size() - 1);
    const std::size_t factory_index = dist(rng);

    current_guess = std::nullopt;
    current_equation = EQUATION_FACTORIES.at(factory_index)->create();
    state = State::Playing;
    clear_canvas();
}

void Game::update(const double delta) {
    if (state == State::WinTransition) {
        wait_delay -= 60.0 * delta;

        if (wait_delay <= 0.0) {
            transition();
        }
    }

    int mouse_x = 0;
    int mouse_y = 0;
    SDL_GetMouseState(&mouse_x, &mouse_y);

    if (state == State::Playing && mouse_down &&
        (mouse_x != last_mouse_x || mouse_y != last_mouse_y)) {
        constexpr int LINE_THICKNESS = 80;

        const auto x1 = static_cast<short>(mouse_x - CANVAS_X);
        const auto y1 = static_cast<short>(mouse_y - CANVAS_Y);
        const auto x2 = static_cast<short>(last_mouse_x - CANVAS_X);
        const auto y2 = static_cast<short>(last_mouse_y - CANVAS_Y);

        SDL_SetRenderTarget(renderer, canvas_texture);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

        thickLineRGBA(renderer, x1, y1, x2, y2, LINE_THICKNESS, 0, 0, 0, SDL_ALPHA_OPAQUE);
        filledCircleRGBA(renderer, x1, y1, LINE_THICKNESS / 2, 0, 0, 0, SDL_ALPHA_OPAQUE);
        filledCircleRGBA(renderer, x2, y2, LINE_THICKNESS / 2, 0, 0, 0, SDL_ALPHA_OPAQUE);

        SDL_SetRenderTarget(renderer, nullptr);
    }

    last_mouse_x = mouse_x;
    last_mouse_y = mouse_y;
}

void Game::clear_canvas() const {
    using sdl::color::WHITE;

    SDL_SetRenderTarget(renderer, canvas_texture);

    SDL_SetRenderDrawColor(renderer, WHITE.r, WHITE.g, WHITE.b, WHITE.a);
    SDL_RenderClear(renderer);

    SDL_SetRenderTarget(renderer, nullptr);
}

void Game::render(const render::ResourceManager& resources) const {
    using sdl::color::GRAY;

    // Clear window
    SDL_SetRenderDrawColor(renderer, GRAY.r, GRAY.g, GRAY.b, GRAY.a);
    SDL_RenderClear(renderer);

    // Game title
    sdl::text::draw_text(renderer, resources.font, GAME_TITLE, CANVAS_X / 2, 20, 60,
                         sdl::text::HAlign::Center, sdl::text::VAlign::Top, sdl::color::WHITE);

    // Draw texture
    const SDL_Rect canvas_rect = {CANVAS_X, CANVAS_Y, CANVAS_WIDTH, CANVAS_HEIGHT};
    SDL_RenderCopy(renderer, canvas_texture, nullptr, &canvas_rect);

    const bool solved = state == State::WinTransition;

    // Current equation
    if (current_equation) {
        const SDL_Color eq_color = solved ? sdl::color::GREEN : sdl::color::WHITE;
        const std::string eq_text =
            solved ? (*current_equation)->display_solved() : (*current_equation)->display();

        sdl::text::draw_text(renderer, resources.font, eq_text, CANVAS_X / 2, WINDOW_HEIGHT - 86,
                             48, sdl::text::HAlign::Center, sdl::text::VAlign::Bottom, eq_color);
    }

    // Guess hint
    if (!solved && current_guess) {
        const SDL_Color color = solved ? sdl::color::GREEN : sdl::color::RED;
        const std::string guess_text = std::format("{}?", *current_guess);

        sdl::text::draw_text(renderer, resources.font, guess_text, (CANVAS_X / 2),
                             WINDOW_HEIGHT - 24, 42, sdl::text::HAlign::Center,
                             sdl::text::VAlign::Bottom, color);
    }

    const auto visible_history = std::views::reverse(solved_equations) | std::views::enumerate |
                                 std::views::take(HISTORY_LIMIT);

    // Previous equations
    for (auto const& [i, equation] : visible_history) {
        const std::string eq_text = equation->display_solved();

        sdl::text::draw_text(renderer, resources.font, eq_text, CANVAS_X / 2,
                             WINDOW_HEIGHT - 180 - (static_cast<int>(i) * 64), 36,
                             sdl::text::HAlign::Center, sdl::text::VAlign::Middle,
                             sdl::color::LIGHT_GRAY);
    }

    SDL_RenderPresent(renderer);
}
