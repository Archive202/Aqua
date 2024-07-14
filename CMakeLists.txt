cmake_minimum_required(VERSION 3.14)
project(Aqua)

# 添加头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 创建静态库或动态库
add_library(Aqua STATIC src/Tensor.cpp)

# 安装目标
install(TARGETS Aqua DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
