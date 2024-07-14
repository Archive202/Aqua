cmake_minimum_required(VERSION 3.14)
project(Aqua)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加 include 目录到包含路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加库
add_library(Aqua src/Tensor.cpp)

# 指定头文件目录
target_include_directories(Aqua PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
