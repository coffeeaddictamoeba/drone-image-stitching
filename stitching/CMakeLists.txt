cmake_minimum_required(VERSION 3.16)
project(stitching_module)

set(CMAKE_CXX_STANDARD 20)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GDAL REQUIRED gdal)
pkg_check_modules(OpenCV REQUIRED opencv4)

file(GLOB_RECURSE STITCHING_SRC CONFIGURE_DEPENDS "src/*.cpp")
add_executable(stitching_exec ${STITCHING_SRC})

target_include_directories(stitching_exec PRIVATE
    include
    ${GDAL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(stitching_exec 
    PUBLIC
        libmetadata::libmetadata
    PRIVATE
        ${GDAL_LIBRARIES}
        ${OpenCV_LIBRARIES}
)
