# IMPORTANT!!
# This is experimental stitching version with Orfeo ToolBox (OTB):
# Link: https://www.orfeo-toolbox.org

# Before running ./mosaic_odm set the OTB environment (source ~/otb/otbenv.profile in my case)

# It would be good to find a better option in future

# I didn't check how it works for Windows
# Windows-based make has some requirements:
# - Run in PowerShell or Git Bash environments
# - Manually modify paths to the specified packages and libraries (OTB, OpenCV, GDAL, etc)


CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

SRC_DIR := src
INCLUDE_DIR := include
OBJ_DIR := obj
BIN_DIR := ../bin

INCOMING_DIR := ../incoming
BATCHES_DIR := ../batches
STITCHED_DIR := ../stitched

# Detect OS
# IMPORTANT: Set your own OTB_DIR path (default is home/otb)!
ifeq ($(OS),Windows_NT)
	EXE := .exe
    SLASH := \\
    SHELL := powershell.exe
    MKDIR := powershell -Command "New-Item -ItemType Directory -Force -Path"
    MKDIR_BIN := powershell -Command "New-Item -ItemType Directory -Force -Path"
    DEL := powershell -Command "Remove-Item -Force -Path"
    RMDIR := powershell -Command "Remove-Item -Force -Recurse -Path"
    OTB_DIR := $(USERPROFILE)\otb
    NULL := $null
else
	EXE :=
	SLASH := /
	MKDIR := mkdir -p $(OBJ_DIR)
	MKDIR_BIN := mkdir -p $(BIN_DIR)
	DEL := rm -f
	RMDIR := rm -rf
	OTB_DIR := $(HOME)/otb
	NULL := /dev/null
	SHELL := /bin/bash
endif

# OTB paths
OTB_INCLUDE_PATH := $(OTB_DIR)$(SLASH)include
OTB_LIBRARY_PATH := $(OTB_DIR)$(SLASH)lib
OTB_PKGCONFIG_PATH := $(OTB_LIBRARY_PATH)$(SLASH)pkgconfig

# Executable and sources
TARGET := $(BIN_DIR)/mosaic_odm$(EXE)
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

CPPFLAGS := -I$(INCLUDE_DIR) -I$(OTB_INCLUDE_PATH)
LDFLAGS := -L$(OTB_LIBRARY_PATH)
ifeq ($(OS),Windows_NT)
	LDFLAGS += -Wl,--enable-auto-import
endif

# OTB OpenCV libs (adapt names if needed)
ifeq ($(OS),Windows_NT)
	OTB_OPENCV_LIBS := \
	    $(OTB_LIBRARY_PATH)\\opencv_core406.lib \
	    $(OTB_LIBRARY_PATH)\\opencv_ml406.lib
else
	OTB_OPENCV_LIBS := \
	    $(OTB_LIBRARY_PATH)/libopencv_core.so.406 \
	    $(OTB_LIBRARY_PATH)/libopencv_ml.so.406
endif

ifeq ($(OS),Windows_NT)
	LDLIBS := \
		$(OTB_LIBRARY_PATH)\gdal_i.lib \
		$(OTB_LIBRARY_PATH)\proj_i.lib \
		$(OTB_LIBRARY_PATH)\openjp2.lib \
		$(OTB_OPENCV_LIBS)
else
	LDLIBS := $(shell PKG_CONFIG_PATH=$(OTB_PKGCONFIG_PATH) pkg-config --libs gdal proj libopenjp2) \
	          $(OTB_OPENCV_LIBS)
endif

.PHONY: all clean restore obj_dir

all: obj_dir $(TARGET)

obj_dir:
	$(MKDIR)
	$(MKDIR_BIN)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

ifeq ($(OS),Windows_NT)
restore:
	@powershell -Command "$$count = 0; \
		Get-ChildItem -Recurse -Include *.jpg,*.jpeg -Path $(BATCHES_DIR) | ForEach-Object { \
			if (Test-Path \$$_.FullName) { \
				Move-Item -Force -LiteralPath \$$_.FullName -Destination $(INCOMING_DIR); \
				$$count++; \
			} \
		}; \
		Write-Host \"$$count images restored.\""
else

restore:
	@count=0; \
	for f in $(BATCHES_DIR)/*/images/*.[jJ][pP][gG] $(BATCHES_DIR)/*/images/*.[jJ][pP][eE][gG]; do \
		[ -f "$$f" ] || continue; \
		mv -n "$$f" $(INCOMING_DIR)/ && count=$$((count+1)); \
	done; \
	echo "$$count images restored."
endif

clean: restore
	-$(DEL) $(TARGET) 2>$(NULL) || true
	-$(RMDIR) $(OBJ_DIR) 2>$(NULL) || true
	-$(RMDIR) $(STITCHED_DIR) 2>$(NULL) || true
	-$(RMDIR) $(BATCHES_DIR) 2>$(NULL) || true
