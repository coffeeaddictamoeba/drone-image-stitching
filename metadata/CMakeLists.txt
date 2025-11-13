add_library(metadata
    src/metadata.cpp
)

target_include_directories(metadata
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# platform-specific stuff (e.g., popen on Windows)
if (WIN32)
    target_compile_definitions(metadata PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_library(metadata::metadata ALIAS metadata)
