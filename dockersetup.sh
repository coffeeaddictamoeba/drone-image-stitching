#!/usr/bin/env bash
set -e

# Script that automates environment setup for Docker container

# Options
# ./dockersetup.sh              - default setup (NO rebuild)
# ./dockersetup.sh --force      - default setup with ALL containers rebuilt
# ./dockersetup.sh --force=dind - default setup with DIND container rebuilt
# ./dockersetup.sh --force=app  - default setup with APP [drone-image-stitching] container rebuilt

# TODO: add --mode option: --mode=dev for development and --mode=prod for production

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIND_NAME="dind"
APP_IMAGE="drone-image-stitching"
ODM_IMAGE="opendronemap/odm"
USER_ID=$(id -u)
GROUP_ID=$(id -g)
DOCKER_PORT=2375
DOCKER_HOST_ADDR="tcp://docker:${DOCKER_PORT}"
HOST_DOCKER_HOST="tcp://localhost:${DOCKER_PORT}"
PROJECT_MOUNT_POINT="/prototype"

FORCE_MODE=""
if [[ "$1" == "--force" ]]; then
    FORCE_MODE="all"
elif [[ "$1" == "--force=dind" ]]; then
    FORCE_MODE="dind"
elif [[ "$1" == "--force=app" ]]; then
    FORCE_MODE="app"
fi

function info() { echo -e "\033[1;34m[INFO]\033[0m $*"; }
function warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
function error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

if [[ "$FORCE_MODE" == "all" || "$FORCE_MODE" == "dind" ]]; then
    if docker ps -a --format '{{.Names}}' | grep -q "^${DIND_NAME}$"; then
        info "Force removing existing DinD container..."
        docker rm -f "${DIND_NAME}" >/dev/null || true
    fi
fi

if [[ "$FORCE_MODE" == "all" || "$FORCE_MODE" == "app" ]]; then
    info "Force rebuilding app image..."
    docker build --no-cache \
        --build-arg USER_ID=${USER_ID} \
        --build-arg GROUP_ID=${GROUP_ID} \
        --build-arg PROJECT_MOUNT_POINT=${PROJECT_MOUNT_POINT} \
        -t "${APP_IMAGE}" .
fi

if ! docker ps -a --format '{{.Names}}' | grep -q "^${DIND_NAME}$"; then
    info "Creating DinD container (tcp://0.0.0.0:${DOCKER_PORT})..."
    docker run -d --privileged --name "${DIND_NAME}" \
        -e DOCKER_TLS_CERTDIR="" \
        -p ${DOCKER_PORT}:${DOCKER_PORT} \
        -v /var/lib/docker \
        -v "${PROJECT_DIR}:${PROJECT_MOUNT_POINT}" \
        docker:dind dockerd \
        -H tcp://0.0.0.0:${DOCKER_PORT} \
        -H unix:///var/run/docker.sock >/dev/null
    info "Waiting for Docker inside DinD to be ready..."
    until docker exec "${DIND_NAME}" docker info >/dev/null 2>&1; do
        sleep 2
    done
else
    if ! docker ps -q -f name="^${DIND_NAME}$" >/dev/null; then
        info "Starting stopped DinD container..."
        docker start "${DIND_NAME}" >/dev/null
        sleep 5
    else
        info "DinD container already running."
    fi
fi

info "Ensuring matching user (${USER_ID}:${GROUP_ID}) inside DinD..."
docker exec "${DIND_NAME}" sh <<'EOF'
addgroup --gid '"${GROUP_ID}"' somegroup 2>/dev/null || true
adduser --disabled-password --uid '"${USER_ID}"' --gid '"${GROUP_ID}"' --gecos "" someuser 2>/dev/null || true
EOF

if docker exec "${DIND_NAME}" docker images --format '{{.Repository}}' | grep -q "^${ODM_IMAGE}$"; then
    info "ODM image '${ODM_IMAGE}' already present in DinD."
else
    info "Pulling ODM image '${ODM_IMAGE}' inside DinD..."
    docker exec "${DIND_NAME}" docker pull "${ODM_IMAGE}"
fi

info "Launching app container with DOCKER_HOST=${DOCKER_HOST_ADDR}..."
docker run -it --rm \
  --link "${DIND_NAME}:docker" \
  -v "${PROJECT_DIR}:${PROJECT_MOUNT_POINT}" \
  -e DOCKER_HOST="${DOCKER_HOST_ADDR}" \
  -e HOST_PROJECT_ROOT="${PROJECT_DIR}" \
  --user "${USER_ID}:${GROUP_ID}" \
  "${APP_IMAGE}"
