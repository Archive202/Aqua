cmake_minimum_required(3.14)
project(Aqua)

set(CMAKE_INCLUDE_CURRENT_DIR on)

file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/Aqua/*.h)
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(Aqua STATIC ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(Aqua PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)