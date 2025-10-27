#!/bin/bash

# Script to install/uninstall Orfeo Toolbox 9.1.0 and generate a wrapper
# Usage: ./otbinstall.sh --install
#        ./otbinstall.sh --uninstall

set -euo pipefail

OTB_VERSION="9.1.0"
OTB_TAR="OTB-${OTB_VERSION}-Linux.tar.gz"
OTB_URL="https://www.orfeo-toolbox.org/packages/${OTB_TAR}"
INSTALL_DIR="/opt/otb"
WRAPPER_BIN="/usr/local/bin/otbwrap"
LINK_DIR="/usr/local/bin"

install_otb() {
  echo "Creating installation directory parent at $(dirname "$INSTALL_DIR")"
  mkdir -p "$(dirname "$INSTALL_DIR")"

  echo "Downloading OTB $OTB_VERSION..."
  wget -q --show-progress "$OTB_URL" -O "/tmp/$OTB_TAR"

  echo "Extracting OTB (staging)..."
  tmpdir="$(mktemp -d)"
  tar -xzf "/tmp/$OTB_TAR" -C "$tmpdir"
  rm -f "/tmp/$OTB_TAR"

  if cand="$(find "$tmpdir" -maxdepth 2 -type f -name 'otbenv.profile' -printf '%h\n' -quit)"; then
    src="$cand"
  elif cand="$(find "$tmpdir" -maxdepth 1 -mindepth 1 -type d -name 'OTB-*' -print -quit)"; then
    src="$cand"
  else
    src="$tmpdir"
  fi

  echo "Placing OTB into $INSTALL_DIR..."
  rm -rf "$INSTALL_DIR"
  mkdir -p "$INSTALL_DIR"
  shopt -s dotglob nullglob
  mv "$src"/* "$INSTALL_DIR"/
  shopt -u dotglob nullglob
  rm -rf "$tmpdir"

  # ------------------------------------------------------------------
  # Build wrapper binary
  echo "Building wrapper at $WRAPPER_BIN"
  SRC_DIR="/usr/local/src"
  mkdir -p "$SRC_DIR"
  cat > "$SRC_DIR/otbwrap.c" <<'EOF_C'
#define _GNU_SOURCE
#include <errno.h>
#include <libgen.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void die(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  vfprintf(stderr, fmt, ap); va_end(ap);
  fputc('\n', stderr);
  _exit(127);
}

int main(int argc, char **argv) {
  const char *install = "/opt/otb";
  const char *sh = "/bin/bash";

  char *invoked = basename(argv[0]);

  // Case 1: invoked via symlink, e.g. "otbcli_Mosaic"
  if (strncmp(invoked, "otbcli_", 7) == 0 ||
      strcmp(invoked, "otbApplicationLauncherCommandLine") == 0) {

    const char *cmd = "source /opt/otb/otbenv.profile >/dev/null 2>&1; exec \"$0\" \"$@\"";

    int new_argc = 5 + (argc - 1);
    char **nargv = calloc(new_argc + 1, sizeof(char*));
    if (!nargv) die("alloc failed");

    nargv[0] = (char*)sh;
    nargv[1] = "-lc";
    nargv[2] = (char*)cmd;
    nargv[3] = "otbwrap";  // becomes $0 inside -c

    char realbin[4096];
    snprintf(realbin, sizeof(realbin), "%s/bin/%s", install, invoked);
    nargv[4] = realbin;

    for (int i = 1; i < argc; ++i) nargv[4 + i] = argv[i];
    nargv[new_argc] = NULL;

    execv(sh, nargv);
    die("execv(%s) failed: %s", sh, strerror(errno));
  }

  // Case 2: direct usage: otbrun <otbcli_command> [args...]
  if (argc < 2) {
    die("Usage: %s <otbcli_command> [args...]", invoked);
  }

  const char *cmd = "source /opt/otb/otbenv.profile >/dev/null 2>&1; exec \"$@\"";

  int new_argc = 4 + (argc - 1);
  char **nargv = calloc(new_argc + 1, sizeof(char*));
  if (!nargv) die("alloc failed");

  nargv[0] = (char*)sh;
  nargv[1] = "-lc";
  nargv[2] = (char*)cmd;
  nargv[3] = "otbwrap"; // becomes $0; not used in this path

  for (int i = 1; i < argc; ++i) nargv[3 + i] = argv[i];
  nargv[new_argc] = NULL;

  execv(sh, nargv);
  die("execv(%s) failed: %s", sh, strerror(errno));
}
EOF_C

  if command -v gcc >/dev/null 2>&1; then
    gcc -O2 -s -o "$WRAPPER_BIN" "$SRC_DIR/otbwrap.c"
  elif command -v clang >/dev/null 2>&1; then
    clang -O2 -s -o "$WRAPPER_BIN" "$SRC_DIR/otbwrap.c"
  else
    echo "[ERROR] No C compiler (gcc/clang) found to build multicall wrapper." >&2
    echo "Install gcc or clang and re-run." >&2
    exit 1
  fi
  chmod 0755 "$WRAPPER_BIN"
  # ------------------------------------------------------------------

  echo "Linking OTB CLI commands..."
  if [ -d "$INSTALL_DIR/bin" ]; then
    while IFS= read -r exe; do
      name="$(basename "$exe")"
      case "$name" in
        otbcli_*|otbApplicationLauncherCommandLine)
          ln -sf "$WRAPPER_BIN" "$LINK_DIR/$name"
          ;;
      esac
    done < <(find "$INSTALL_DIR/bin" -maxdepth 1 -type f -perm -u+x | sort)
  else
    echo "[WARN] $INSTALL_DIR/bin not found — no commands linked."
  fi

  echo "Installation complete!"
}

uninstall_otb() {
  if [ ! -d "$INSTALL_DIR" ]; then
    echo "[WARN] $INSTALL_DIR not found; proceeding with cleanup."
  else
    read -r -p "Are you sure you want to remove OTB at $INSTALL_DIR? [y/N] " confirm
    if [[ ! "${confirm:-}" =~ ^[Yy]$ ]]; then
      echo "Aborted uninstallation."
      return 1
    fi
  fi

  echo "Removing OTB installation at $INSTALL_DIR..."
  rm -rf "$INSTALL_DIR"

  echo "Removing wrapper $WRAPPER_BIN (if exists)..."
  rm -f "$WRAPPER_BIN"

  echo "Removing OTB CLI links pointing to the wrapper..."
  while IFS= read -r link; do
    [ -L "$link" ] && [ "$(readlink -f "$link")" = "$WRAPPER_BIN" ] && rm -f -- "$link" || true
  done < <(find "$LINK_DIR" -maxdepth 1 -type l -name 'otbcli_*' -o -type l -name 'otbApplicationLauncherCommandLine' 2>/dev/null)

  echo "Uninstallation complete!"
}

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 --install|--uninstall"
  exit 1
fi

case "$1" in
  --install)   install_otb ;;
  --uninstall) uninstall_otb ;;
  *) echo "Unknown option: $1"; echo "Usage: $0 --install|--uninstall"; exit 1 ;;
esac
