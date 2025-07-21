CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

SRC_DIR := src
INCLUDE_DIR := include
OBJ_DIR := obj
BIN_DIR := ../bin

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TARGET := $(BIN_DIR)/img_deblur

CPPFLAGS := -I$(INCLUDE_DIR) `pkg-config --cflags opencv4`
LDLIBS := `pkg-config --libs opencv4`

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)