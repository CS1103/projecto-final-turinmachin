#include "game/sdl/draw.h"
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_render.h>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

namespace game::sdl {

    void draw_line(SDL_Renderer* const renderer,
                   const float x1,
                   const float y1,
                   const float x2,
                   const float y2,
                   const float thickness,
                   const SDL_FColor color) {
        float dx = x2 - x1;
        float dy = y2 - y1;

        const float length = std::sqrt((dx * dx) + (dy * dy));
        dx /= length;
        dy /= length;

        const float px = -dy * (thickness / 2.0F);
        const float py = dx * (thickness / 2.0F);

        const std::vector<SDL_Vertex> vertices = {
            {.position = {x1 + px, y1 + py}, .color = color, .tex_coord = {0, 0}},
            {.position = {x1 - px, y1 - py}, .color = color, .tex_coord = {0, 0}},
            {.position = {x2 - px, y2 - py}, .color = color, .tex_coord = {0, 0}},
            {.position = {x2 + px, y2 + py}, .color = color, .tex_coord = {0, 0}},
        };

        const std::array<int, 6> indices = {0, 1, 2, 0, 2, 3};
        SDL_RenderGeometry(renderer, nullptr, vertices.data(), 4, indices.data(), 6);

        draw_filled_circle(renderer, x1, y1, thickness / 2.0F, color);
        draw_filled_circle(renderer, x2, y2, thickness / 2.0F, color);
    }

    void draw_filled_circle(SDL_Renderer* const renderer,
                            const float cx,
                            const float cy,
                            const float radius,
                            const SDL_FColor color,
                            const int segments) {
        if (radius <= 0 || segments < 3) {
            return;
        }

        std::vector<SDL_Vertex> vertices(segments + 2);
        std::vector<int> indices;
        indices.reserve((segments * 3) - 2);

        vertices[0].position = {.x = cx, .y = cy};
        vertices[0].color = color;

        const float angleStep = 2.0F * std::numbers::pi_v<float> / static_cast<float>(segments);

        for (int i = 0; i <= segments; ++i) {
            const float angle = static_cast<float>(i) * angleStep;
            const float x = cx + (std::cos(angle) * radius);
            const float y = cy + (std::sin(angle) * radius);

            vertices[i + 1].position = {.x = x, .y = y};
            vertices[i + 1].color = color;

            if (i > 0) {
                indices.push_back(0);
                indices.push_back(i);
                indices.push_back(i + 1);
            }
        }

        SDL_RenderGeometry(renderer, nullptr, vertices.data(), static_cast<int>(vertices.size()),
                           indices.data(), static_cast<int>(indices.size()));
    }

}  // namespace game::sdl
