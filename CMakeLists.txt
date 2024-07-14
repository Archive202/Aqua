cmake_minimum_required(VERSION 3.14)

project(Aqua)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

add_library(Tensor INTERFACE)
target_include_directories(Tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/Tensor)