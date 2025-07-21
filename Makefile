.PHONY: all clean stitching deblurring init

all: init bin/mosaic_odm bin/img_deblur

BIN_DIR := bin
init:
	@mkdir -p $(BIN_DIR)

# STITCHING (OTB-based, 9.1.0)
STITCHING_DIR := stitching
bin/mosaic_odm:
	$(MAKE) -C $(STITCHING_DIR)

# DEBLURRING (OpenCV-based)
DEBLURRING_DIR := deblurring
bin/img_deblur:
	$(MAKE) -C $(DEBLURRING_DIR)

clean:
	$(MAKE) -C $(STITCHING_DIR) clean
	$(MAKE) -C $(DEBLURRING_DIR) clean
	rm -f bin/*