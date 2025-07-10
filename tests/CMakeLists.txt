file(GLOB_RECURSE test_files CONFIGURE_DEPENDS *.cpp)

add_executable(brain_ager_tests ${test_files})
target_include_directories(brain_ager_tests
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(brain_ager_tests
                           PRIVATE ${PROJECT_SOURCE_DIR}/deps/catch)

add_test(NAME all_tests COMMAND brain_ager_tests)
