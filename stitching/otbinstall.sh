#!/bin/bash

# Script to install/uninstall Orfeo Toolbox 9.1.0 and generate a wrapper
# Usage: ./otbinstall.sh --install
#        ./otbinstall.sh --uninstall

set -e

OTB_VERSION="9.1.0"
OTB_TAR="OTB-${OTB_VERSION}-Linux.tar.gz"
OTB_URL="https://www.orfeo-toolbox.org/packages/${OTB_TAR}"
INSTALL_DIR="/opt/otb"
WRAPPER_PATH="/usr/local/bin/otbrun.sh"

install_otb() {
    # Install OTB
    echo "Creating installation directory at $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"

    echo "Downloading OTB $OTB_VERSION..."
    wget -q --show-progress "$OTB_URL" -O "/tmp/$OTB_TAR"

    echo "Extracting OTB to $INSTALL_DIR..."
    tar -xzf "/tmp/$OTB_TAR" -C "$INSTALL_DIR" --strip-components=1
    rm "/tmp/$OTB_TAR"

    # Install OTB wrapper
    echo "Creating wrapper script at $WRAPPER_PATH"
    cat << EOF > "$WRAPPER_PATH"
#!/bin/bash

# Wrapper for Orfeo Toolbox CLI that ensures OTB environment is available without conflicting with system libraries.
OTB_INSTALL_DIR="$INSTALL_DIR"

if ! command -v otbcli_Mosaic >/dev/null 2>&1; then
    if [ -f "\$OTB_INSTALL_DIR/otbenv.profile" ]; then
        source "\$OTB_INSTALL_DIR/otbenv.profile" >/dev/null 2>&1
    else
        echo "[ERROR] OTB environment profile not found at \$OTB_INSTALL_DIR/otbenv.profile" >&2
        exit 1
    fi
fi

exec "\$@"
EOF

    chmod +x "$WRAPPER_PATH"

    echo "Installation complete!"
    echo "Use OTB via the wrapper: otbrun.sh <otbcli_command> [args]"
}

uninstall_otb() {
    # Uninstall OTB
    if [ ! -f "$INSTALL_DIR/otbenv.profile" ]; then
        echo "[ERROR] No OTB environment profile found in $INSTALL_DIR. Aborting uninstallation."
        return 1
    fi

    read -p "Are you sure you want to remove OTB at $INSTALL_DIR? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted uninstallation."
        return 1
    fi

    echo "Removing OTB installation at $INSTALL_DIR..."
    rm -rf "$INSTALL_DIR"

    if [ -f "$WRAPPER_PATH" ]; then
        echo "Removing wrapper at $WRAPPER_PATH..."
        rm -f "$WRAPPER_PATH"
    fi

    echo "Uninstallation complete!"
}

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 --install|--uninstall"
    exit 1
fi

case "$1" in
    --install)
        install_otb
        ;;
    --uninstall)
        uninstall_otb
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 --install|--uninstall"
        exit 1
        ;;
esac
