#ifndef INCLUDE_STATE_H
#define INCLUDE_STATE_H

#include <SDL2/SDL_events.h>
#include <SDL2/SDL_render.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>
#include "common/agent.h"
#include "game/math/interfaces.h"
#include "game/render.h"
#include "utec/nn/neural_network.h"

class Game {
    enum class State : std::uint8_t {
        Playing,
        WinTransition,
    };

    bool quit = false;

    std::unique_ptr<agent::IAgent<std::vector<double>, int>> agent;
    std::mt19937 rng;

    SDL_Renderer* renderer;
    SDL_Texture* canvas_texture;

    utec::neural_network::NeuralNetwork<double> net;

    std::optional<std::unique_ptr<math::IEquation>> current_equation = std::nullopt;
    std::vector<std::unique_ptr<math::IEquation>> solved_equations;
    bool mouse_down = false;
    int last_mouse_x = 0;
    int last_mouse_y = 0;
    std::optional<int> current_guess = std::nullopt;

    State state = State::WinTransition;
    double wait_delay = 0.0;

    void transition();

    void process_drawing();

public:
    explicit Game(SDL_Renderer* renderer,
                  std::unique_ptr<agent::IAgent<std::vector<double>, int>> agent,
                  std::mt19937 rng);

    ~Game();

    [[nodiscard]] auto should_quit() const -> bool;

    void handle_event(const SDL_Event& event);

    void update(double delta);

    void clear_canvas() const;

    void render(const render::ResourceManager& resources) const;
};

#endif
