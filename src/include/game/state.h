#ifndef INCLUDE_STATE_H
#define INCLUDE_STATE_H

#include <SDL2/SDL_events.h>
#include <SDL2/SDL_render.h>
#include <cstdint>
#include <set>
#include "utec/nn/neural_network.h"

struct State {
    enum class Pixel : std::uint8_t {
        White,
        Black
    };

    bool quit = false;

    SDL_Renderer* renderer;
    SDL_Texture* canvas_texture;

    utec::neural_network::NeuralNetwork<double> net;

    std::set<SDL_Keycode> keyboard;
    bool mouse_down = false;
    int last_mouse_x = 0;
    int last_mouse_y = 0;

    explicit State(SDL_Renderer* renderer);

    void handle_event(const SDL_Event& event);

    void update(double delta);

    void clear_canvas() const;

    void render() const;

    ~State();
};

#endif
