cmake_minimum_required(VERSION 3.14)

# 设置项目名称和版本
project(Aqua)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加库
add_library(Aqua INTERFACE)

# 链接头文件目录
target_include_directories(Aqua INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# install(TARGETS Aqua EXPORT AquaTargets)
# install(EXPORT AquaTargets FILE AquaConfig.cmake DESTINATION lib/cmake/Aqua)
# install(DIRECTORY include/ DESTINATION include)