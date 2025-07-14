CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
PKGCONFIG := pkg-config
LIBS := $(shell $(PKGCONFIG) --cflags --libs opencv4 gdal)

TARGET := mosaic_odm
SRC := main.cpp
INCOMING_DIR := incoming
BATCHES_DIR := batches
STITCHED_DIR := stitched

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

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

.PHONY: all clean
