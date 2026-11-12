"""Sweep one refinement module at a time on the seven controlled corruptions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark import CASES, error, inject
from bvh import BVH
from cli import process
from refine import MODULES, config, detect

ROOT = Path(__file__).resolve().parent
LEVELS = (0, 25, 50, 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True, help="new output CSV file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output file already exists")
    inventory = json.loads((args.data / "inventory.json").read_text(encoding="utf-8"))
    samples = {row["name"]: row["sha256"] for row in inventory["clips"]}
    settings = config(strength=50)
    rows = []
    with TemporaryDirectory(prefix="bvh-strength-") as tmp:
        for kind, name, module in CASES:
            source = args.data / f"{name}.bvh"
            if hashlib.sha256(source.read_bytes()).hexdigest() != samples[name]:
                raise ValueError(f"Sample SHA-256 mismatch: {source.name}")
            bvh = BVH.read(source)
            clean = bvh.decode()
            reference = detect(bvh, *bvh.fk(*clean), settings)
            root, local, idx, mode, _ = inject(bvh, clean, reference, kind)
            folder = Path(tmp) / kind
            folder.mkdir()
            damaged_path = folder / f"{kind}.bvh"
            bvh.write(damaged_path, root, local)
            damaged = BVH.read(damaged_path)
            baseline = error(bvh, clean, damaged.decode(), idx, mode)
            for strength in LEVELS:
                stage = folder / str(strength)
                stage.mkdir()
                active = {key: {"enabled": key == module, "strength": strength} for key in MODULES}
                case_config = settings | active | {"floor_m": reference.floor}
                process(damaged_path, stage, case_config)
                refined = BVH.read(stage / f"{kind}.refined.bvh")
                if strength == 0 and refined.raw != damaged.raw:
                    raise RuntimeError(f"Zero-strength bytes changed for {kind}")
                rms = error(bvh, clean, refined.decode(), idx, mode)
                rows.append(dict(case=kind, clean_clip=name, module=module, global_strength=50,
                                 module_strength=strength, rms_unit=mode[0].split("_")[-1],
                                 before_rms=baseline, after_rms=rms,
                                 rms_reduction_pct=100 * (1 - rms / baseline)))
            print(kind, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
