"""Thin standalone wrapper around scripts/ladder_sim.py.

Designed to be the entry point for a PyInstaller --onefile binary. The
existing ``ladder_sim.py`` already contains all the simulation logic; we
just need to:

  1. Make sure ladder_sim is importable when frozen (PyInstaller bundles
     scripts/ as part of the binary, but the working dir at runtime is
     wherever the user launched the binary, so we manually add the
     embedded scripts/ folder to sys.path).
  2. Default --web-dir to the bundled models folder when frozen so the
     binary works on a host with no source checkout.

Everything else (CLI args, ELO logic, JSON output) is reused verbatim
from ladder_sim.main().
"""

from __future__ import annotations

import os
import sys


def _bundle_paths() -> tuple[str, str]:
    """Return (scripts_dir, models_dir) for both frozen and source modes."""
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller --add-data "<src>:scripts" lands files in _MEIPASS/scripts.
        # We bundle ladder_sim.py there so it's importable.
        meipass = sys._MEIPASS
        return os.path.join(meipass, "scripts"), os.path.join(meipass, "models")
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    return here, os.path.join(repo, "web")


def main() -> None:
    scripts_dir, models_dir = _bundle_paths()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    # Inject a default --web-dir when the user didn't provide one. Doing it
    # before the import so ladder_sim's argparse picks it up unchanged.
    if "--web-dir" not in sys.argv:
        sys.argv.extend(["--web-dir", models_dir])

    import ladder_sim  # type: ignore  # resolved via sys.path patch above
    ladder_sim.main()


if __name__ == "__main__":
    main()
