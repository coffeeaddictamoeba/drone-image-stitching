cmake_minimum_required(VERSION 3.16)
project(deblurring_module)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE DEBLURRING_SRC "src/*.cpp")

add_executable(deblurring_exec ${DEBLURRING_SRC})

target_include_directories(deblurring_exec PRIVATE
    include
)

find_package(OpenCV REQUIRED)

target_include_directories(deblurring_exec PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(deblurring_exec 
    PUBLIC
        libmetadata::libmetadata
    PRIVATE
        ${OpenCV_LIBS}
)