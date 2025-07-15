#!/bin/bash

SRC_DIR=$1

DEST_DIR="./incoming"
DELAY=2

mkdir -p "$DEST_DIR"

echo "Moving images from '$SRC_DIR' to '$DEST_DIR' with $DELAY sec delay."

mapfile -t FILES < <(
    find "$SRC_DIR" -maxdepth 1 -type f -name "waypoint_*_*.jpg" |
    sort -t_ -k2,2n
)

for FILE in "${FILES[@]}"; do
    BASENAME=$(basename "$FILE")
    echo "Moving: $BASENAME"
    mv "$FILE" "$DEST_DIR/"
    sleep $DELAY
done
