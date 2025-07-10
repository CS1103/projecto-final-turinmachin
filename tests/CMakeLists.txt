file(GLOB_RECURSE test_files CONFIGURE_DEPENDS *.cpp)

add_executable(brain_ager_tests ${test_files})
target_include_directories(brain_ager_tests PRIVATE catch)

add_test(NAME all_tests COMMAND brain_ager_tests)
