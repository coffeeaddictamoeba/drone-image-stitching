# BUILD: from project directory (drone-image-stitching):
    # Prerequisites: sudo systemctl start docker docker.socket + sudo usermod -aG docker $USER
    # docker build -t drone-image-stitching .

# IMAGES DIRECTORY MOUNTING:
    # Directory "images" is serving as all images processing container
    # For clear function distribution, "images" will have such subdirectories:
        # incoming - directory with all initial (raw) images, sent right from the drone
        # blurred - directory created mostly for testing for operations on clearly blurred images
        # deblurred - directory where all deblurred (or sharp raw) images moved
        # processing - directory where all image batch data is stored
        # stitched - directory with final orthophoto stitched 

# RUNNING APPLICATION:
    # Start docker-in-docker (as stitching uses Docker ODM) in separate terminal (so another target container can find it):
        # docker run --privileged -d --name dind -e DOCKER_TLS_CERTDIR="" -p 2375:2375 docker:dind dockerd -H tcp://0.0.0.0:2375 -H unix:///var/run/docker.sock
        # (or, if created dind once) docker start dind

    # Check it is running correctly on 2375:
        # docker logs dind | grep API

    # Run with Docker socket and host images
        # docker run -it --rm --link dind:docker -v /absolute/path/to/images:/images -e DOCKER_HOST=tcp://docker:2375 drone-image-stitching:latest
    
    # Run actual code (still in development):
    # Deblurring (works fine right now): 

        # Status: TESTED, WORKING

        # Prerequisites: mount image directory (incoming in this case) by -v /absolute/path/to/images:/images
        # cd build/deblurring
        # (Blur) ./deblurring_exec --blur /images/blurred/waypoint_xx_12345678.jpg --overwrite-metadata
        # (Deblur) ./deblurring_exec /images/blurred/waypoint_xx_12345678.jpg
        # (Directory-wise blur) ./deblurring_exec --blur --source-dir /images/incoming/ --target-dir /images/blurred --overwrite-metadata
        # (Directory-wise deblur) ./deblurring_exec --source-dir /images/incoming/ --target-dir /images/deblurred

    # Stitching (needs to be fixed due to dind + multiple file renaming & moving): 
        
        # Status: REQUIRES FIX

        # Prerequisites: dind (docker-in-docker to run ODM), mounted image incoming directory
        # cd build/stitching
        # (Optional: verify OTB works) otbrun.sh otbcli_Mosaic -help

        # (Does not work for now)./stitching_exec --incoming /incoming --batch-size 15 --wait-batch-size --no-retry --no-bigtiff --save-prev

# IN CASE SOURCE CODE CHANGED:
    # Update the code inside container:
        # docker run -v $PWD:/prototype -it drone-image-stitching:latest
        # cd /prototype/build
        # cmake ..
        # cmake --build . --parallel

        # NOTE: If your build directory has different CMakeCache.txt, wither rename your source build dir to something like "build-source" and inside container just mkdir build
        # or create new build dir in container (like "build-cont") and cmake inside this new build dir

        # exit (as we need to mount image incoming directory)

# EXITING
    # From drone-image-stitching: exit (run from container itself)
    # From dind: docker stop dind (from host)
    # From host: remove dind when not needed: docker rm -f dind

# Base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies, Docker CLI, and exiftool
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgdal-dev \
    gdal-bin \
    libopencv-dev \
    wget \
    curl \
    python3 \
    python3-pip \
    docker.io \
    exiftool \
    && rm -rf /var/lib/apt/lists/*

# Install Orfeo Toolbox 9.1.0
RUN wget https://www.orfeo-toolbox.org/packages/OTB-9.1.0-Linux.tar.gz \
    && mkdir /opt/otb/ \
    && tar -xzf OTB-9.1.0-Linux.tar.gz -C /opt/otb/ \
    && rm OTB-9.1.0-Linux.tar.gz

# Probably unnecessary
# ENV OTB_HOME=/opt/otb
# ENV PATH=$OTB_HOME/bin:$PATH
# ENV LD_LIBRARY_PATH=$OTB_HOME/lib:$LD_LIBRARY_PATH

# Set working directory for your project
# Possibly change name later
WORKDIR /prototype 

# Copy the entire project (make sure Docker build context includes CMakeLists.txt)
COPY . /prototype

# Wrapper for OTB
COPY stitching/otbrun.sh /usr/local/bin/otbrun.sh
RUN chmod +x /usr/local/bin/otbrun.sh

# Create build directory and build your project
RUN rm -rf build && mkdir -p build && cd build && cmake .. && cmake --build . --parallel

# Expose ports if needed
EXPOSE 8000

# Default command
CMD ["/bin/bash"]
