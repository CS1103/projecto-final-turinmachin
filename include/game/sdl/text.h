#include <SDL_pixels.h>
#include <SDL_render.h>
#include <SDL_ttf.h>
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
                   int x,
                   int y,
                   int font_size,
                   HAlign h_align,
                   VAlign v_align,
                   SDL_Color color);

}  // namespace game::sdl
