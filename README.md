# Drone Image Stitching & Deblurring

This is an application for fast, lightweight and secure drone image stitching.

### Dependencies

---

- [OpenDroneMap (ODM)](https://github.com/OpenDroneMap/ODM)
- [Orfeo ToolBox (OTB)](https://github.com/orfeotoolbox/OTB)
- [ExifTool](https://exiftool.org/)
- [GDAL](https://github.com/OSGeo/gdal)
- [OpenCV](https://opencv.org/)
- FFTW3 for OpenCV
- OpenJP2 and PROJ for GDAL

**NOTE:** This project uses ODM by Docker and OTB CLI (https://www.orfeo-toolbox.org/CookBook/CliInterface.html)

### How to Build

---

As this project uses `cmake`, you can build it as follows:

```
mkdir build && cd build
cmake .. <args> (e.g, if OTB CLI's OpenCV conflicts with your system one, specify -DOpenCV_DIR=/your/path/to/cmake/OpenCV)
make
```

In case of errors double check that all libraries specified in dependencies are installed.

All binaries (`stitching_exec` and `deblurring_exec`) will be inside `build`. To run:

```
./stitching/stitching_exec
./deblurring/deblurring_exec --generate-test
```

**IMPORTANT:** Before running `stitching` code, `source `OTB CLI environment:`source /path/to/otb/cli/otbenv.profile `and run Docker ODM. If you need to build the project once again and see an error, connected with `opencv `or `gdal`, probably you use `otbcli` environment. The easiest way to fix this is to open another shell or reload the current one, build, and then source the OTB environment.

### Use Cases

---

- `./stitching/stitching_exec` - stitches images from default directory `incoming` using Docker ODM. Options it has:
  - `--incoming <dir>` - allows you to specify the directory with your images
  - `--batch-size <value>` - allows you to control batch-size. Default is 10. If you plan to use **deblurred images**, set batch size to **25+ images**.
  - `--timeout <value>` - if batch size wasn't reached till timeout (seconds), the program will try to process a batch with remaining amount of images. Default is 30 s.
  - `--wait-batch-size` - forces the program to wait till batch size
  - `--retry-amount <value>` - allows you to set amount of retries for failed batch processing. Default is 3.
  - `--no-retry` - processes all batches once
  - `--no-bigtiff` - discards images of size more than 4 GB
  - `--save-prev` - saves previous mosaic while updating current one

Other options (`rgb-threshold`, `alpha-threshold`, `blocksize`, etc) are rarely used, as their main purpose is to meet some specific hardware and software needs. Description of these options can be found in `Documentation.docx`.

- `./deblurring/deblurring_exec` - deblurs a single image using OpenCV and ExifTool. Options it has:

  - `--source-dir <dir>` - allows you to specify source images directory for deblurring instead of single image
  - `--target-dir <dir>` - allows you to specify target images directory where your deblurred image(s) will be moved
  - `--blur <img>` - blurs an image instead of deblurring it. Works with `--source-dir` as well
  - `--overwrite-metadata` - adds random synthetic metadata to an image. **Do not use on real images**
  - `--generate-test` - fast image generation feature which creates initial and blurred images
  - `--blur-threshold <value>` - allows you to control image acceptance threshold (as images which were evaluated as "not blurred" are skipped). Default is 100.
  - `--force` - ignores `blur-threshold` and deblurs all the incoming images
  - `--snr <value>` - allows you to control deblurring/artifacts strength. Default value is 1500.
  - `--denoise` - experimental; may add blur to an image. Helpful in case of images with large single-color areas
  - `--sensor-width <value>` and `--sensor-height <value>` - options which may be replaced soon; find description in `Documentation.docx`

  #### Additional
- `./move-images.sh <source-dir> <dest-dir>` - script for moving images. May be helpful in case of `stitching`, as all images from `incoming dir` are placed to `batches/batch_*/images`. Options:

  - `-n | --new-name <name>` - allows to set new name for every image. The name of moved image will be changed to `new_name_index.format`
  - `-c | --copy` - copies images instead of moving them
  - `-d | --delay <value>` - changes delay (seconds) between moving images. Default is 1 s.
  - `-l | --limit <value>` - allows to limit amount of images to move. Default is no limit.
  - `-x | --exiftool` - allows to inspect image metadata while moving it. You can modify EXIF tag list inside script if needed.

### Best Parameters for Run

---

- **Batch Size**:
  - Sharp Images: >= 10
  - Deblurred Images >= 25
- **SNR**: 500-2000
- **Blur Length**: <= 100 px
- **Drone GPS Altitude**: 100 m <= x <= 350 m

Both tests and results can be found in `Documentation.docx`.
