#include "game/sdl/text.h"
#include <SDL_pixels.h>
#include <SDL_render.h>
#include <SDL_ttf.h>
#include <string>

namespace sdl::text {

    void draw_text(SDL_Renderer* const renderer,
                   TTF_Font* const font,
                   const std::string& text,
                   const int x,
                   const int y,
                   const int font_size,
                   const HAlign h_align,
                   const VAlign v_align,
                   const SDL_Color color) {
        SDL_Surface* const surface = TTF_RenderText_Blended(font, text.c_str(), color);
        SDL_Texture* const message = SDL_CreateTextureFromSurface(renderer, surface);

        const int width = (font_size * surface->w) / surface->h;
        const int height = font_size;

        int aligned_x = 0;

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

        int aligned_y = 0;

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

        const SDL_Rect rect = {aligned_x, aligned_y, width, height};
        SDL_RenderCopy(renderer, message, nullptr, &rect);

        SDL_FreeSurface(surface);
        SDL_DestroyTexture(message);
    }

}  // namespace sdl::text
