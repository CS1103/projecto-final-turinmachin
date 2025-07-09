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

find_package(SDL3 REQUIRED CONFIG REQUIRED COMPONENTS SDL3-shared)
find_package(SDL3_ttf REQUIRED CONFIG REQUIRED)

file(GLOB_RECURSE trainer_source CONFIGURE_DEPENDS src/common/*.cpp
     src/trainer/*.cpp)
file(GLOB_RECURSE game_source CONFIGURE_DEPENDS src/common/*.cpp src/game/*.cpp)

set(SHARE_INSTALL_DIR "share/${PROJECT_NAME}")

configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/config.h)

include_directories(${CMAKE_BINARY_DIR})

add_executable(brain_ager_trainer ${trainer_source})
target_include_directories(brain_ager_trainer PUBLIC include)

add_executable(brain_ager ${game_source})
target_include_directories(brain_ager PUBLIC include)
target_include_directories(brain_ager PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(brain_ager PRIVATE SDL3::SDL3 SDL3_ttf::SDL3_ttf)

install(TARGETS brain_ager DESTINATION bin)
install(TARGETS brain_ager_trainer DESTINATION bin)
install(DIRECTORY share/ DESTINATION ${SHARE_INSTALL_DIR})

include(CTest)
add_subdirectory(tests)
