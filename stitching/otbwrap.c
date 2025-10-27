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