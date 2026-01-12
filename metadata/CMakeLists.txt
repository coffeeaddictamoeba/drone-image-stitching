add_library(metadata
    src/metadata.cpp
)

target_include_directories(metadata
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# find exiftool executable
find_program(EXIFTOOL_EXECUTABLE exiftool)

if (NOT EXIFTOOL_EXECUTABLE)
    message(FATAL_ERROR "ExifTool was not found on this system.")
endif()

message(STATUS "ExifTool found: ${EXIFTOOL_EXECUTABLE}")
target_compile_definitions(metadata PRIVATE EXIFTOOL_PATH="${EXIFTOOL_EXECUTABLE}")

# platform-specific stuff (e.g., popen on Windows)
if (WIN32)
    target_compile_definitions(metadata PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_library(metadata::metadata ALIAS metadata)
