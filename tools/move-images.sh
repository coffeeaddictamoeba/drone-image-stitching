#!/bin/bash

# To run:
# ./move-images.sh my/src/images/dir/ my/dst/images/dir/ <- simply moves images from SRC to DST with delay of 1 s, see options below

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

SRC=""
DST=""
NEW_NAME=""
ACTION="Moving"

COPY=false
USE_EXIFTOOL=false

IDX=0
DELAY=1
LIM=-1

help() {
    echo -e "./move-images.sh [args] <source-dir> <dest-dir>"
    echo -e "Options:"
    echo -e " -n | --new-name <name>           - Changes image name to \"name\" and adds an index"
    echo -e " -d | --delay <delay>             - Changes delay to <delay>"
    echo -e " -l | --limit <image-amount>      - Limits moved images amount to <limit>"
    echo -e " -c | --copy                      - Copies images instead of moving them"
    echo -e " -x | --exiftool                  - Inspects metadata of every moved image." # requires exiftool installed
    exit 0
}

# extend the tag list if needed
use_exiftool() {
    local image="$1"
    exiftool -FileName -FileSize -FileAccessDateTime -FilePermissions -FileTypeExtension \
    -CameraModelName -Software -ExposureTime -DateTimeOriginal \
    -FocalLength -GPSLatitude -GPSLongitude -GPSAltitude -GPSImgDirection -ImageWidth -ImageHeight \
    -GPSSpeed $image
    echo -e "------------------------------------------------------------------------"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--new-name)    NEW_NAME="$2"; shift 2;;
        -d|--delay)       DELAY="$2"; shift 2;;
        -l|--limit)       LIM="$2"; shift 2;;
        -c|--copy)        COPY=true; ACTION="Copying"; shift;;
        -x|--exiftool)    USE_EXIFTOOL=true; shift;;
        -*|--*)           echo -e "Unknown option: $1"; help;;
        *)                break;;
    esac
done

if $USE_EXIFTOOL; then
    if ! command -v exiftool >/dev/null 2>&1; then
        echo -e "${RED}Error: exiftool is not installed.${NC}"
        exit 1
    fi
fi

SRC="$1"
DST="$2"

if [[ -z "$SRC" || -z "$DST" ]]; then
    echo -e "${RED}Error: Source and destination directories must be specified.${NC}"
    help
fi

mkdir -p $DST

echo "$ACTION images from '$SRC' to '$DST' with $DELAY s delay."
echo "------------------------------------------------------------------------"

mapfile -t FILES < <(
    find "$SRC" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) |
    sort
)

for FILE in "${FILES[@]}"; do
    IDX=$((IDX + 1))
    if [[ "$LIM" -le 0 ]] || [[ "$IDX" -le "$LIM" ]]; then
        BASENAME=$(basename "$FILE")

        if [[ -n "$NEW_NAME" ]]; then
            EXT="${BASENAME##*.}"
            BASENAME="${NEW_NAME}_${IDX}.${EXT}"
        fi

        if $USE_EXIFTOOL; then
            use_exiftool "$FILE"
        fi

        echo "$ACTION: $BASENAME"
        if $COPY; then
            cp "$FILE" "$DST/$BASENAME"
        else
            mv "$FILE" "$DST/$BASENAME"
        fi
        sleep "$DELAY"
    fi
done

echo -e "${GREEN}Completed! ${ACTION} of ${IDX} images is successful.${NC}" 
