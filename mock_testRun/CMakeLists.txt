cmake_minimum_required(VERSION 3.10)
project(mock_testRun)

set(CMAKE_CXX_STANDARD 11)

add_executable(testRun mock_testRun.cpp)
# Post-build step: Copy testRun (with correct extension) to bin/ directory for all platforms
add_custom_command(TARGET testRun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:testRun>" "${CMAKE_CURRENT_SOURCE_DIR}/bin/$<TARGET_FILE_NAME:testRun>"
)