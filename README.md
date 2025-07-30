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

**IMPORTANT:** Before running, `source` OTB CLI environment: `source /path/to/otb/cli/otbenv.profile` and run Docker ODM.
