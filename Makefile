CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
PKGCONFIG := pkg-config
LIBS := $(shell $(PKGCONFIG) --cflags --libs opencv4 gdal)

TARGET := mosaic_odm
SRC := main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
