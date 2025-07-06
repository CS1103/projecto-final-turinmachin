#ifndef INCLUDE_GAME_CONSTANTS_H
#define INCLUDE_GAME_CONSTANTS_H

constexpr int WINDOW_WIDTH = 960;
constexpr int WINDOW_HEIGHT = 540;
constexpr int FPS = 60;
constexpr double DELTA = 1.0 / FPS;
constexpr long double MAX_TIME_ACCUMULATOR = 4 * DELTA;

constexpr int CANVAS_WIDTH = WINDOW_HEIGHT;
constexpr int CANVAS_HEIGHT = WINDOW_HEIGHT;

constexpr int CANVAS_X = WINDOW_WIDTH - CANVAS_WIDTH;
constexpr int CANVAS_Y = 0;

#endif
