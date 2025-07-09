#include "game/sdl/text.h"
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_surface.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <string>

namespace game::sdl {

    void draw_text(SDL_Renderer* const renderer,
                   TTF_Font* const font,
                   const std::string& text,
                   const float x,
                   const float y,
                   const float font_size,
                   const HAlign h_align,
                   const VAlign v_align,
                   const SDL_Color color) {
        SDL_Surface* const surface = TTF_RenderText_Blended(font, text.c_str(), text.size(), color);
        SDL_Texture* const message = SDL_CreateTextureFromSurface(renderer, surface);

        const float width =
            (font_size * static_cast<float>(surface->w)) / static_cast<float>(surface->h);
        const float height = font_size;

        float aligned_x = 0;

        switch (h_align) {
            case HAlign::Left:
                aligned_x = x;
                break;
            case HAlign::Center:
                aligned_x = x - (width / 2);
                break;
            case HAlign::Right:
                aligned_x = x - width;
                break;
        }

        float aligned_y = 0;

        switch (v_align) {
            case VAlign::Top:
                aligned_y = y;
                break;
            case VAlign::Middle:
                aligned_y = y - (height / 2);
                break;
            case VAlign::Bottom:
                aligned_y = y - height;
                break;
        }

        const SDL_FRect rect = {aligned_x, aligned_y, width, height};
        SDL_RenderTexture(renderer, message, nullptr, &rect);

        SDL_DestroySurface(surface);
        SDL_DestroyTexture(message);
    }

}  // namespace game::sdl
