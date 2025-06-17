CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
PKGCONFIG := pkg-config
LIBS := $(shell $(PKGCONFIG) --cflags --libs opencv4 gdal)

SRC_DIR := src
OBJ_DIR := obj
BIN := mosaic

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN)

.PHONY: all clean