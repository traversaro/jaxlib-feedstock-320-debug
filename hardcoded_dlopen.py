#!/usr/bin/env python3

import ctypes
import os
import sys
import traceback


def build_paths_from_conda_prefix() -> list[str]:
    """Construct target .so paths from CONDA_PREFIX and current Python version.

    Default order:
      1) jaxlib/libjax_common.so
      2) jax_plugins/xla_cuda12/xla_cuda_plugin.so
    """
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise RuntimeError("CONDA_PREFIX is not set; run inside the conda/pixi env or export it.")

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = os.path.join(prefix, "lib", pyver, "site-packages")

    

    libjax_common = os.path.join(site, "jaxlib", "libjax_common.so")
    # The failing DSO seen in your log:
    xla_cuda_plugin = os.path.join(site, "jax_plugins", "xla_cuda12", "xla_cuda_plugin.so")
    # System-level gRPC from the same env
    libgrpc = os.path.join(prefix, "lib", "libgrpc.so")

    # Optional extras you can uncomment while bisecting
    # utils_so = os.path.join(site, "jaxlib", "utils.so")
    # jax_so = os.path.join(site, "jaxlib", "_jax.so")

    return [
        libjax_common,
        #libgrpc,
        xla_cuda_plugin,
    ]


def main() -> int:
    try:
        paths = build_paths_from_conda_prefix()
    except Exception as e:
        print(f"Path resolution error: {e}")
        return 0

    mode = getattr(os, "RTLD_NOW", 2) | getattr(os, "RTLD_GLOBAL", 0x100)

    for p in paths:
        print(f"dlopen -> {p}")
        try:
            ctypes.CDLL(p, mode=mode)
            print("  success")
        except OSError as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
