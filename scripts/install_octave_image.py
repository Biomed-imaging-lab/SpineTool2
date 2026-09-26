"""Build Octave's image package with Apple Clang for the local conda prefix.

Run after creating external/octave-env. The conda package contains NUL-padded
paths and flags for its build compiler; replace them before building the package.
"""
import os
from pathlib import Path
import subprocess


CONFIG_KEYS = """
ALL_CFLAGS ALL_CXXFLAGS ALL_FFLAGS ALL_LDFLAGS BLAS_LIBS CFLAGS CPICFLAG
CPPFLAGS CXXFLAGS CXXPICFLAG DL_LDFLAGS FFLAGS FPICFLAG INCFLAGS INCLUDEDIR
LAPACK_LIBS LDFLAGS LD_STATIC_FLAG LIBDIR LIBOCTAVE LIBOCTINTERP LIBOCTMEX
OCTAVE_LINK_OPTS OCTAVE_LIBS OCTAVE_LINK_DEPS OCTINCLUDEDIR OCTLIBDIR
OCT_LINK_DEPS OCT_LINK_OPTS RDYNAMIC_FLAG SPECIAL_MATH_LIB XTRA_CFLAGS
XTRA_CXXFLAGS
""".split()


def main():
    project_root = Path(__file__).resolve().parents[1]
    prefix = project_root / "external" / "octave-env"
    env = os.environ.copy()
    env["OCTAVE_HOME"] = env["OCTAVE_EXEC_HOME"] = str(prefix)
    for key in CONFIG_KEYS:
        raw = subprocess.check_output(
            [str(prefix / "bin" / "mkoctfile"), "-p", key], env=env
        )
        value = raw.replace(b"\0", b"").decode().strip()
        if "FLAGS" in key:
            openmp = "-lomp" if "LDFLAGS" in key else "-Xpreprocessor -fopenmp"
            value = value.replace("-fopenmp", openmp)
        env[key] = value

    for key in ("CXXFLAGS", "ALL_CXXFLAGS"):
        env[key] = env.get(key, "") + " -std=c++17"
    libdir = str(prefix / "lib")
    env["LDFLAGS"] = "-L" + libdir + " -Wl,-rpath," + libdir
    env["ALL_LDFLAGS"] = env.get("ALL_LDFLAGS", "") + " -Wl,-rpath," + libdir
    env["DYLD_FALLBACK_LIBRARY_PATH"] = libdir
    env.update(CC="/usr/bin/clang", CXX="/usr/bin/clang++", CXXLD="/usr/bin/clang++")
    result = subprocess.run(
        [
            str(project_root / "scripts" / "octave-cli"),
            "--no-gui", "--quiet", "--eval",
            'pkg install -global -forge image; pkg load image; disp("IMAGE_PACKAGE_OK");',
        ],
        env=env,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
