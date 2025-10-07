#!/bin/bash

# Wrapper for Orfeo Toolbox CLI that ensures OTB environment is available without conflicting with system libraries.
if ! command -v otbcli_Mosaic >/dev/null 2>&1; then
    if [ -f /opt/otb/otbenv.profile ]; then
        source /opt/otb/otbenv.profile >/dev/null 2>&1 # change path to otbenv.profile if needed
    else
        echo "[ERROR] OTB environment profile not found at /opt/otb/otbenv.profile" >&2
        exit 1
    fi
fi

exec "$@"
