cmake_minimum_required(VERSION 3.16)
project(MultiModuleProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(metadata)
add_subdirectory(deblurring)
add_subdirectory(stitching)