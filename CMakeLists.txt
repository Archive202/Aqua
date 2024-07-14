cmake_minimum_required(VERSION 3.14)

project(Aqua)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(Aqua INTERFACE)
target_include_directories(Aqua INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
