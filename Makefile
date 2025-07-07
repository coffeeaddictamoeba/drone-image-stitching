CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -O2 -DDEBUG -Iexternal/ctre -fopenmp -lfftw3f
PKGCONFIG := pkg-config
LIBS := $(shell $(PKGCONFIG) --cflags --libs opencv4)

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

TILES_DIR := tiles
RES_IMG_DIR := results
DEBLUR_DIR := deblurred

MOSAIC_BIN := $(BIN_DIR)/mosaic
MOSAIC_SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/ExifToolPipe.cpp $(SRC_DIR)/FeatureMatcher.cpp $(SRC_DIR)/MosaicBuilder.cpp $(SRC_DIR)/MosaicTileManager.cpp
MOSAIC_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MOSAIC_SRCS))

DEBLUR_BIN := $(BIN_DIR)/deblur
DEBLUR_SRCS := $(SRC_DIR)/deblur.cpp
DEBLUR_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(DEBLUR_SRCS))

all: $(MOSAIC_BIN) $(DEBLUR_BIN)

$(MOSAIC_BIN): $(MOSAIC_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(DEBLUR_BIN): $(DEBLUR_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(TILES_DIR) $(RES_IMG_DIR) $(DEBLUR_DIR)

.PHONY: all clean