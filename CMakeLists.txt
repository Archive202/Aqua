cmake_minimum_required(VERSION 3.14)
project(Aqua)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加库
add_library(Aqua STATIC src/Aqua.cpp)

# 指定头文件目录
target_include_directories(Aqua PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 安装目标
install(TARGETS Aqua DESTINATION lib)
install(FILES Aqua.hpp DESTINATION include)
