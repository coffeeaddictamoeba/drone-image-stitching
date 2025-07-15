# IMPORTANT!!
# This is experimental stitching version with Orfeo ToolBox (OTB):
# Link: https://www.orfeo-toolbox.org

# Before running ./mosaic_odm set the OTB environment (source ~/otb/otbenv.profile in my case)

# It would be good to find a better option in future

CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

OTB_DIR := $(HOME)/otb
INCLUDE_PATH := $(OTB_DIR)/include
LIBRARY_PATH := $(OTB_DIR)/lib
PKGCONFIG_PATH := $(LIBRARY_PATH)/pkgconfig

CPPFLAGS := -I$(INCLUDE_PATH)
LDFLAGS := -L$(LIBRARY_PATH) \
           -Wl,-rpath,$(LIBRARY_PATH)

# OTB CLI goes with its own libs preinstalled, however, this may differ on different systems
OPENCV_LIBS := \
    $(LIBRARY_PATH)/libopencv_core.so.406 \
	$(LIBRARY_PATH)/libopencv_ml.so.406

# Link to OTB's GDAL, PROJ, OpenJPEG via .pc files
LDLIBS := $(shell PKG_CONFIG_PATH=$(PKGCONFIG_PATH) pkg-config --libs gdal proj libopenjp2) \
          $(OPENCV_LIBS)

TARGET := mosaic_odm
SRC := main.cpp
INCOMING_DIR := incoming
BATCHES_DIR := batches
STITCHED_DIR := stitched

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

restore:
	@count=0; \
	for f in $(BATCHES_DIR)/*/images/*.[jJ][pP][gG]; do \
		[ -f "$$f" ] || continue; \
		mv -n "$$f" $(INCOMING_DIR)/ && count=$$((count+1)); \
	done; \
	echo "$$count images restored."

clean: restore
	rm -f $(TARGET)
	rm -rf $(STITCHED_DIR) $(BATCHES_DIR)

.PHONY: all clean restore