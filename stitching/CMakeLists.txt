cmake_minimum_required(VERSION 3.16)
project(stitching_module)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GDAL REQUIRED gdal)

file(GLOB_RECURSE STITCHING_SRC CONFIGURE_DEPENDS "src/*.cpp")
add_executable(stitcher ${STITCHING_SRC})

target_include_directories(stitcher PRIVATE
    include
    ${GDAL_INCLUDE_DIRS}
)

target_link_libraries(stitcher PRIVATE
    ${GDAL_LIBRARIES}
)
