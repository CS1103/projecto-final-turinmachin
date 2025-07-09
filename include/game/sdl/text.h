#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_render.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <cstdint>
#include <string>

namespace game::sdl {

    enum class VAlign : std::uint8_t {
        Top,
        Middle,
        Bottom
    };

    enum class HAlign : std::uint8_t {
        Left,
        Center,
        Right
    };

    void draw_text(SDL_Renderer* renderer,
                   TTF_Font* font,
                   const std::string& text,
                   float x,
                   float y,
                   float font_size,
                   HAlign h_align,
                   VAlign v_align,
                   SDL_Color color);

}  // namespace game::sdl
