"""Source-checkout wrapper for the saved Body Pose -> Kimodo converter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gem.conversion.cli import main

if __name__ == "__main__":
    main()
