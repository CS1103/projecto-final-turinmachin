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

option(USE_VENDORED_DEPS
       "Use vendored dependencies instead of system-installed ones" ON)

if(USE_VENDORED_DEPS)
  message(STATUS "Using vendored dependencies")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

  set(SDLTTF_VENDORED ON)

  add_subdirectory(vendored/SDL EXCLUDE_FROM_ALL)
  add_subdirectory(vendored/SDL_ttf EXCLUDE_FROM_ALL)
else()
  message(STATUS "Using system-installed dependencies")

  find_package(SDL3 REQUIRED CONFIG COMPONENTS SDL3-shared)
  find_package(SDL3_ttf REQUIRED CONFIG)
endif()

file(GLOB_RECURSE trainer_source CONFIGURE_DEPENDS src/common/*.cpp
     src/trainer/*.cpp)
file(GLOB_RECURSE game_source CONFIGURE_DEPENDS src/common/*.cpp src/game/*.cpp)

set(SHARE_INSTALL_DIR "share/${PROJECT_NAME}")

configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/config.h)

include_directories(${CMAKE_BINARY_DIR})
include_directories(deps)
include_directories(include)

add_executable(brain_ager_trainer ${trainer_source})

add_executable(brain_ager ${game_source})
target_link_libraries(brain_ager PRIVATE SDL3::SDL3 SDL3_ttf::SDL3_ttf)

install(TARGETS brain_ager DESTINATION bin)
install(TARGETS brain_ager_trainer DESTINATION bin)
install(DIRECTORY share/ DESTINATION ${SHARE_INSTALL_DIR})

include(CTest)
add_subdirectory(tests)
