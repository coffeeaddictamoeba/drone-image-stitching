CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

SRC_DIR := src
INCLUDE_DIR := include
OBJ_DIR := obj
BIN_DIR := ../bin

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TARGET_NAME := img_deblur

# Detect OS
ifeq ($(OS),Windows_NT)
    SHELL := cmd.exe
    MKDIR_P := powershell -Command "New-Item -ItemType Directory -Force '$@'"
    DEL := powershell -Command "Remove-Item -Force -Recurse '$@'"
    TARGET := $(BIN_DIR)/$(TARGET_NAME).exe
    USE_PKGCONFIG := false
else
    SHELL := /bin/bash
    MKDIR_P := mkdir -p $@
    DEL := rm -rf $@
    TARGET := $(BIN_DIR)/$(TARGET_NAME)
    USE_PKGCONFIG := true
endif

CPPFLAGS := -I$(INCLUDE_DIR)

# OpenCV
ifeq ($(USE_PKGCONFIG),true)
    # Linux: Use pkg-config
    CPPFLAGS += $(shell pkg-config --cflags opencv4)
    LDLIBS := $(shell pkg-config --libs opencv4)
    ifeq ($(LDLIBS),)
        $(error "Error: pkg-config could not find opencv4. Please ensure OpenCV 4 is installed and pkg-config is configured correctly.")
    endif
else
    # Windows: Attempt to find OpenCV automatically
    # Priority: OPENCV_DIR env var -> Registry -> Common paths -> Manual fallback

    # Try to get OPENCV_DIR from environment variable first
    OPENCV_ROOT := $(strip $(shell powershell -Command "echo $$env:OPENCV_DIR"))
    ifeq ($(OPENCV_ROOT),)
        # Try to find from registry (typical for official installers)
        OPENCV_ROOT := $(strip $(shell powershell -Command "Get-ItemProperty 'HKLM:\SOFTWARE\OpenCV' | Select-Object -ExpandProperty InstallLocation -ErrorAction SilentlyContinue"))
    endif
    ifeq ($(OPENCV_ROOT),)
        # Check common installation paths
        ifeq ($(wildcard C:/opencv/build),C:/opencv/build)
            OPENCV_ROOT := C:/opencv
        else ifeq ($(wildcard C:/Program Files/OpenCV/build),C:/Program Files/OpenCV/build)
            OPENCV_ROOT := C:/Program Files/OpenCV
        endif
    endif

    # If OPENCV_ROOT is still not found
    ifeq ($(OPENCV_ROOT),)
        OPENCV_ROOT := C:/opencv # Fallback to a common default, user might need to adjust
        $(warning "WARNING: OpenCV root directory not found automatically on Windows.")
        $(warning "         Please set the OPENCV_DIR environment variable to your OpenCV installation root (e.g., C:/opencv).")
        $(warning "         Or manually adjust OPENCV_ROOT in the Makefile if the default is incorrect.")
    endif

    OPENCV_INCLUDE := $(OPENCV_ROOT)/build/include
    OPENCV_LIB_DIR := $(OPCV_ROOT)/build/x64/vc15/lib # Common path for VS 2017/2019/2022
    ifeq ($(wildcard $(OPENCV_LIB_DIR)),) # Check if the default x64/vc15/lib exists
        OPENCV_LIB_DIR := $(OPENCV_ROOT)/build/x64/vc14/lib # For VS 2015
        ifeq ($(wildcard $(OPENCV_LIB_DIR)),)
            OPENCV_LIB_DIR := $(OPENCV_ROOT)/build/lib # Fallback for some non-standard layouts
        endif
    endif

    CPPFLAGS += -I$(OPENCV_INCLUDE)

    # Automatically detect OpenCV version and construct library names
    # This assumes library names like opencv_core410.lib, opencv_imgproc410.lib, etc.
    OPENCV_LIBS_FOUND := $(shell powershell -Command "Get-ChildItem -Path '$(OPENCV_LIB_DIR)' -Filter 'opencv_core*.lib' | Select-Object -ExpandProperty BaseName -ErrorAction SilentlyContinue")
    OPENCV_VERSION_PART := $(patsubst opencv_core%,%,$(OPENCV_LIBS_FOUND))
    ifeq ($(OPENCV_VERSION_PART),)
        $(error "Error: Could not determine OpenCV version or find core library in $(OPENCV_LIB_DIR).")
        $(error "       Please verify OPENCV_ROOT and OPENCV_LIB_DIR in the Makefile, or ensure OpenCV libraries are present.")
    endif

    # Construct LDLIBS dynamically based on detected version
    LDLIBS := -L$(OPENCV_LIB_DIR) \
              -lopencv_core$(OPENCV_VERSION_PART) \
              -lopencv_imgproc$(OPENCV_VERSION_PART) \
              -lopencv_highgui$(OPENCV_VERSION_PART) \
              -lopencv_imgcodecs$(OPENCV_VERSION_PART)
endif

.PHONY: all clean obj_dir

all: $(BIN_DIR) $(TARGET)

$(BIN_DIR):
	@$(MKDIR_P)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@$(MKDIR_P) $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	@$(DEL) $(OBJ_DIR)
	@$(DEL) $(TARGET)