#!/bin/bash

# This is a wrapper for Orfeo Toolbox CLI as it requires specific dependency setup
source /opt/otb/otbenv.profile >/dev/null 2>&1
exec "$@"
