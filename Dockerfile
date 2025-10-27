
# HOW TO BUILD THIS PROJECT WITH DOCKER
# ----------------------------------------------------------------------------------------------------
# Prerequisites:
    # sudo systemctl start docker docker.socket (if not enabled)
    # sudo usermod -aG docker $USER (if not set)
# ----------------------------------------------------------------------------------------------------
# It is highly recommended to use "dockersetup.sh" script for automatic environment setup. However,
# if you wish, you can set it up manually following the instruction below:

# BUILD: 
    # Go to the project root directory (drone-image-stitching)
    # Run "docker build --build-arg USER_ID=$(id -u) \
    #             --build-arg GROUP_ID=$(id -g) \
    #             --build-arg PROJECT_MOUNT_POINT=/someproject \
    #             -t drone-image-stitching ."

# RUNNING CONTAINERS:
    # Start docker-in-docker or DinD (for access to Docker ODM) from host machine terminal:
        # docker run -d --privileged --name dind \
        #            -e DOCKER_TLS_CERTDIR="" \
        #            -p 2375:2375 \
        #            -v /var/lib/docker \
        #            -v /absolute/path/to/your/project/root:/someproject \
        #            docker:dind dockerd \
        #            -H tcp://0.0.0.0:2375 \
        #            -H unix:///var/run/docker.sock

        # Notice, that --build-arg PROJECT_MOUNT_POINT=/someproject should match 
        # this line's mount point: -v /absolute/path/to/your/project/root:/someproject

        # Check it is running correctly on 2375: "docker logs dind | grep API"
    
        # Install OpenDroneMap to DinD:
            # docker exec dind docker pull opendronemap/odm
            # (Verify) docker exec dind docker images
        
    # Run app container with Docker socket and host images
        # docker run -it --rm \
        # --link dind:docker" \
        # -v /absolute/path/to/your/project/root:/someproject \
        # -e DOCKER_HOST=tcp://docker:2375 \
        # --user $(id -u):$(id -g) \
        # drone-image-stitching

# RUNNING CODE (still in development):
    # Deblurring (works fine right now): 
        # Status: TESTED, WORKING

        # Prerequisites: mount image directory (incoming in this case) by -v /absolute/path/to/images:/images

        # cd build/deblurring
        # (Blur) ./deblurring_exec --blur /images/blurred/waypoint_xx_12345678.jpg --overwrite-metadata
        # (Deblur) ./deblurring_exec /images/blurred/waypoint_xx_12345678.jpg
        # (Directory-wise blur) ./deblurring_exec --blur --source-dir /images/incoming/ --target-dir /images/blurred --overwrite-metadata
        # (Directory-wise deblur) ./deblurring_exec --source-dir /images/incoming/ --target-dir /images/deblurred

    # Stitching: 
        # Status: REQUIRES TESTS

        # Prerequisites: dind (docker-in-docker to run ODM), mounted image incoming directory

        # IMAGES DIRECTORY MOUNTING (created automatically when running stitching module):
        # Directory "images" is serving as all images processing container
        # For clear function distribution, "images" will have such subdirectories:
            # incoming - directory with all initial (raw) images, sent right from the drone
            # blurred - directory created mostly for testing for operations on clearly blurred images
            # deblurred - directory where all deblurred (or sharp raw) images moved
            # processing - directory where all image batch data is stored
            # stitched - directory with final orthophoto stitched
        # Pass "/absolute/container/path/to/images" directory as argument for "--data" option in stitching

        # cd build/stitching
        # (Optional: verify OTB works) otbrun.sh otbcli_Mosaic -help
        # (Optional: verify setup for stitching) ./stitching_exec --is-my-setup-ok
        # ./stitching_exec --data /prototype/images --batch-size 15 --wait-batch-size --no-retry --no-bigtiff --save-prev

        # In case of using synthetic data clean synthetic metadata:
            # exiftool -FlightPitchDegree= -FlightRollDegree= -FlightYawDegree= -XMP-drone-dji:FlightXSpeed= \
            # -XMP-drone-dji:FlightYSpeed= -XMP-drone-dji:FlightZSpeed= /prototype/images/incoming/*.jpg

# IN CASE SOURCE CODE CHANGED:
    # Update the code inside container:
        # docker run -v $PWD:/prototype -it drone-image-stitching:latest
        # cd /prototype/build
        # cmake ..
        # cmake --build . --parallel

        # NOTE: If your build directory has different CMakeCache.txt, wither rename your source build dir to something 
        # like "build-source" and inside container just mkdir build or create new build dir in container 
        # (like "build-cont") and cmake inside this new build dir
        
# EXITING
    # From drone-image-stitching: exit (from container itself)
    # From dind: docker stop dind (from host; can be restarted later)
    # From host: remove dind when not needed: docker rm -f dind
# ----------------------------------------------------------------------------------------------------

# Base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Mount point
ARG PROJECT_MOUNT_POINT=/prototype
ENV PROJECT_MOUNT_POINT=${PROJECT_MOUNT_POINT}

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
    lsb-release \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Add user with default parameters
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} somegroup && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash someuser

# Install Orfeo Toolbox 9.1.0
WORKDIR /tmp/otb-install
COPY stitching/otbinstall.sh /tmp/otb-install/otbinstall.sh
RUN chmod +x /tmp/otb-install/otbinstall.sh && /tmp/otb-install/otbinstall.sh --install

RUN bash -lc 'source /opt/otb/otbenv.profile >/dev/null 2>&1 || true && test -f /opt/otb/tools/install_done.txt || true'

# Set working directory for your project
WORKDIR ${PROJECT_MOUNT_POINT}

# Copy the entire project
COPY . ${PROJECT_MOUNT_POINT}

# For ODM
ENV MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p /tmp/matplotlib && chmod 777 /tmp/matplotlib

# Switch to user
USER someuser

# Create build directory and build your project (unnecessary for development, uncomment on prod)
#RUN rm -rf build && mkdir -p build && cd build && cmake .. && cmake --build . --parallel

# Expose ports if needed
EXPOSE 8000

# Default command
CMD ["/bin/bash"]
