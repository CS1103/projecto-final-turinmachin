#include "game/game.h"
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_oldnames.h>
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_surface.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <cstddef>
#include <cstdlib>
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
#include "game/sdl/draw.h"
#include "game/sdl/text.h"
#include "game/sdl/texture.h"

namespace game {

    Game::Game(SDL_Renderer* const renderer,
               std::unique_ptr<common::IDigitAgent> agent,
               std::mt19937 rng)
        : agent(std::move(agent)),
          rng(rng),
          renderer(renderer),
          canvas_texture(SDL_CreateTexture(renderer,
                                           SDL_PIXELFORMAT_RGBA32,
                                           SDL_TEXTUREACCESS_TARGET,
                                           CANVAS_WIDTH,
                                           CANVAS_HEIGHT)) {
        if (canvas_texture == nullptr) {
            throw std::runtime_error("Could not create canvas texture: " +
                                     std::string(SDL_GetError()));
        }

        SDL_SetTextureScaleMode(canvas_texture, SDL_SCALEMODE_LINEAR);
        clear_canvas();
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
            case SDL_EVENT_QUIT:
                quit = true;
                break;
            case SDL_EVENT_KEY_DOWN: {
                const SDL_Keycode key = event.key.key;

                if (state == State::Playing && key == SDLK_R) {
                    clear_canvas();
                    current_guess = std::nullopt;
                }
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                mouse_down = true;
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP: {
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
        const std::vector<std::uint8_t> pixel_data =
            sdl::extract_pixel_data(renderer, canvas_texture, CANVAS_WIDTH, CANVAS_HEIGHT, 4);

        const std::vector<std::uint8_t> grayscale_data = make_grayscale(pixel_data, CANVAS_WIDTH);

        const SDL_Rect src_rect = {0, 0, CANVAS_WIDTH, CANVAS_HEIGHT};

        const std::vector<std::uint8_t> downscaled =
            downscale_to_8x8(grayscale_data, src_rect, CANVAS_WIDTH);

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
        using namespace math;

        static const std::array<std::unique_ptr<IEquationFactory>, 3> EQUATION_FACTORIES{
            std::make_unique<AddEquationFactory>(rng),
            std::make_unique<SubtractEquationFactory>(rng),
            std::make_unique<DivideEquationFactory>(rng)};

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

        float mouse_x = 0;
        float mouse_y = 0;
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

            sdl::draw_line(renderer, x1, y1, x2, y2, LINE_THICKNESS, {0, 0, 0, SDL_ALPHA_OPAQUE});

            SDL_SetRenderTarget(renderer, nullptr);
        }

        last_mouse_x = mouse_x;
        last_mouse_y = mouse_y;
    }

    void Game::clear_canvas() const {
        SDL_SetRenderTarget(renderer, canvas_texture);

        SDL_SetRenderDrawColor(renderer, sdl::WHITE.r, sdl::WHITE.g, sdl::WHITE.b, sdl::WHITE.a);
        SDL_RenderClear(renderer);

        SDL_SetRenderTarget(renderer, nullptr);
    }

    void Game::render(const ResourceManager& resources) const {
        // Clear window
        SDL_SetRenderDrawColor(renderer, sdl::GRAY.r, sdl::GRAY.g, sdl::GRAY.b, sdl::GRAY.a);
        SDL_RenderClear(renderer);

        // Game title
        sdl::draw_text(renderer, resources.font, GAME_TITLE, CANVAS_X / 2, 20, 60,
                       sdl::HAlign::Center, sdl::VAlign::Top, sdl::WHITE);

        // Draw texture
        const SDL_FRect canvas_rect = {CANVAS_X, CANVAS_Y, CANVAS_WIDTH, CANVAS_HEIGHT};
        SDL_RenderTexture(renderer, canvas_texture, nullptr, &canvas_rect);

        const bool solved = state == State::WinTransition;

        // Current equation
        if (current_equation) {
            const SDL_Color eq_color = solved ? sdl::GREEN : sdl::WHITE;
            const std::string eq_text =
                solved ? (*current_equation)->display_solved() : (*current_equation)->display();

            sdl::draw_text(renderer, resources.font, eq_text, CANVAS_X / 2, WINDOW_HEIGHT - 86, 48,
                           sdl::HAlign::Center, sdl::VAlign::Bottom, eq_color);
        }

        // Guess hint
        if (!solved && current_guess) {
            const SDL_Color color = solved ? sdl::GREEN : sdl::RED;
            const std::string guess_text = std::format("{}?", *current_guess);

            sdl::draw_text(renderer, resources.font, guess_text, (CANVAS_X / 2), WINDOW_HEIGHT - 24,
                           42, sdl::HAlign::Center, sdl::VAlign::Bottom, color);
        }

        const auto visible_history = std::views::reverse(solved_equations) | std::views::enumerate |
                                     std::views::take(HISTORY_LIMIT);

        // Previous equations
        for (auto const& [i, equation] : visible_history) {
            const std::string eq_text = equation->display_solved();

            sdl::draw_text(renderer, resources.font, eq_text, CANVAS_X / 2,
                           WINDOW_HEIGHT - 180 - (static_cast<int>(i) * 64), 36,
                           sdl::HAlign::Center, sdl::VAlign::Middle, sdl::LIGHT_GRAY);
        }

        SDL_RenderPresent(renderer);
    }

}  // namespace game
