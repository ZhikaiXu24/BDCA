"""Refine G1 Moves human BVH files with CPU-only motion processing."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bvh import BVH
from quality import summarize
from refine import config, refine
from visual import render


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=fields)
        out.writeheader()
        out.writerows(rows)


def flatten(record):
    return {f"{k}_{sub}" if isinstance(v, dict) else k: value
            for k, v in record.items()
            for sub, value in (v.items() if isinstance(v, dict) else [("", v)])}


def process(source, target, settings, video=False, video_start=0.0, video_seconds=None):
    bvh = BVH.read(source)
    before = bvh.decode()
    root, local, reference = refine(bvh, settings)
    output = target / f"{source.stem}.refined.bvh"
    bvh.write(output, root, local)
    written = BVH.read(output)
    after = written.decode()
    if written.header != bvh.header or not np.allclose(root, after[0], atol=1e-8, rtol=0) or not np.allclose(
            local, after[1], atol=3e-7, rtol=0):
        raise RuntimeError(f"Output BVH failed the hierarchy or pose round-trip check for {source.name}")
    stats = summarize(bvh, before, after, reference)
    stats["file"] = source.name
    if video:
        render(target / f"{source.stem}.mp4", bvh, before, after, reference, video_start, video_seconds)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="a BVH file or directory containing BVH files")
    parser.add_argument("output", type=Path, help="a new directory for BVH outputs and metrics")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--strength", type=float, help="override global strength (0–100)")
    parser.add_argument("--video", action="store_true", help="also export side-by-side MP4")
    parser.add_argument("--video-start", type=float, default=0)
    parser.add_argument("--video-seconds", type=float)
    args = parser.parse_args()
    sources = sorted(args.input.glob("*.bvh")) if args.input.is_dir() else [args.input]
    if not sources or any(not p.is_file() or p.suffix.lower() != ".bvh" for p in sources):
        parser.error("provide an existing BVH file or a directory containing .bvh files")
    try:
        settings = config(args.config, args.strength)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent, prefix=".bvh-refine-") as temp:
            stage = Path(temp) / "output"
            stage.mkdir()
            results = []
            for path in sources:
                results.append(process(path, stage, settings, args.video, args.video_start, args.video_seconds))
                print(path.name, flush=True)
            save_json(stage / "metrics.json", {"config": settings, "clips": results})
            save_csv(stage / "metrics.csv", [flatten(row) for row in results])
            if args.output.exists():
                raise FileExistsError(args.output)
            stage.rename(args.output)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
