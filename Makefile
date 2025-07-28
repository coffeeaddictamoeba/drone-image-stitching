# Root Makefile for Linux/macOS & Windows

# Detect OS
ifeq ($(OS),Windows_NT)
  MKDIR_P     = if not exist $(BIN_DIR) mkdir $(BIN_DIR)
  RM_RF       = rmdir /S /Q
  SEP         = \\
  EXE_EXT     = .exe
else
  MKDIR_P     = mkdir -p $(BIN_DIR)
  RM_RF       = rm -rf
  SEP         = /
  EXE_EXT     =
endif

BIN_DIR    := bin
STITCH_DIR := stitching
DEBLUR_DIR := deblurring

.PHONY: all clean stitching deblurring init

all: init $(BIN_DIR)$(SEP)mosaic_odm$(EXE_EXT) $(BIN_DIR)$(SEP)img_deblur$(EXE_EXT)

# Create bin directory
init:
	$(MKDIR_P)

# Stitching binary
$(BIN_DIR)$(SEP)mosaic_odm$(EXE_EXT):
	$(MAKE) -C $(STITCH_DIR)

# Deblurring binary
$(BIN_DIR)$(SEP)img_deblur$(EXE_EXT):
	$(MAKE) -C $(DEBLUR_DIR)

clean:
	$(MAKE) -C $(STITCH_DIR) clean
	$(MAKE) -C $(DEBLUR_DIR) clean
	$(RM_RF) $(BIN_DIR)