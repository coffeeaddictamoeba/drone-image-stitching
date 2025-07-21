# IMPORTANT!!
# This is experimental stitching version with Orfeo ToolBox (OTB):
# Link: https://www.orfeo-toolbox.org

# Before running ./mosaic_odm set the OTB environment (source ~/otb/otbenv.profile in my case)

# It would be good to find a better option in future

CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

SRC_DIR := src
INCLUDE_DIR := include
OBJ_DIR := obj

OTB_DIR := $(HOME)/otb
OTB_INCLUDE_PATH := $(OTB_DIR)/include
OTB_LIBRARY_PATH := $(OTB_DIR)/lib
OTB_PKGCONFIG_PATH := $(OTB_LIBRARY_PATH)/pkgconfig

TARGET := ../bin/mosaic_odm
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

INCOMING_DIR := incoming
BATCHES_DIR := batches
STITCHED_DIR := stitched

CPPFLAGS := -I$(INCLUDE_DIR) -I$(OTB_INCLUDE_PATH)

LDFLAGS := -L$(OTB_LIBRARY_PATH) -Wl,-rpath,$(OTB_LIBRARY_PATH)

# OTB's own opencv (OTB 9.1.0) dependencies
OTB_OPENCV_LIBS := \
    $(OTB_LIBRARY_PATH)/libopencv_core.so.406 \
    $(OTB_LIBRARY_PATH)/libopencv_ml.so.406

LDLIBS := $(shell PKG_CONFIG_PATH=$(OTB_PKGCONFIG_PATH) pkg-config --libs gdal proj libopenjp2) \
          $(OTB_OPENCV_LIBS)

.PHONY: all clean restore obj_dir

all: obj_dir $(TARGET)

obj_dir:
	@mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

restore:
	@count=0; \
	for f in $(BATCHES_DIR)/*/images/*.[jJ][pP][gG] $(BATCHES_DIR)/*/images/*.[jJ][pP][eE][gG]; do \
		[ -f "$$f" ] || continue; \
		mv -n "$$f" $(INCOMING_DIR)/ && count=$$((count+1)); \
	done; \
	echo "$$count images restored."

clean: restore
	rm -f $(TARGET)
	rm -rf $(OBJ_DIR) $(STITCHED_DIR) $(BATCHES_DIR)