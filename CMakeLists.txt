cmake_minimum_required(VERSION 3.29)
project(brain-ager)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Debug
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Build type" FORCE)
endif()

# Compiler warnings
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED CONFIG REQUIRED)

include(FindPkgConfig)
pkg_search_module(SDL2_gfx REQUIRED SDL2_gfx)

file(GLOB_RECURSE trainer_source CONFIGURE_DEPENDS src/trainer/*.cpp)
file(GLOB_RECURSE trainer_kan_source CONFIGURE_DEPENDS src/trainer_kan/*.cpp)
file(GLOB_RECURSE game_source CONFIGURE_DEPENDS src/game/*.cpp)

add_executable(trainer ${trainer_source})
add_executable(trainer_kan ${trainer_kan_source})
target_include_directories(trainer PUBLIC include)
target_include_directories(trainer_kan PUBLIC include)

add_executable(brain_ager ${game_source} ${font_source})
target_include_directories(brain_ager PUBLIC include)
target_link_libraries(brain_ager PRIVATE SDL2::SDL2)
target_link_libraries(brain_ager PRIVATE SDL2_ttf::SDL2_ttf)
target_link_libraries(brain_ager PRIVATE SDL2_gfx)

include(CTest)
add_subdirectory(tests)
