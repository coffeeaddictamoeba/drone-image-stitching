add_library(libmetadata
    src/metadata.cpp
)

target_include_directories(libmetadata
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# platform-specific stuff (e.g., popen on Windows)
if (WIN32)
    target_compile_definitions(libmetadata PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_library(libmetadata::libmetadata ALIAS libmetadata)
