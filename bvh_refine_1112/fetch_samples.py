"""Fetch only the pinned raw BVH files and the upstream attribution manifest."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, path, sha256):
    if path.is_file() and digest(path) == sha256:
        return
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        subprocess.run(["curl", "-fLsS", "--retry", "2", "--max-time", "90",
                        url, "-o", str(tmp)], check=True)
        if digest(tmp) != sha256:
            raise ValueError(f"SHA-256 mismatch for {url}")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "samples.json").read_text(encoding="utf-8"))
    base = f"{plan['dataset']}/resolve/{plan['revision']}"
    manifest = args.data / "upstream_manifest.json"
    fetch(f"{base}/manifest.json", manifest, plan.pop("manifest_sha256"))
    metadata = json.loads(manifest.read_text(encoding="utf-8"))["clips"]
    checksums = plan.pop("sha256")

    def download(clip):
        category, name, reason = clip
        url = f"{base}/{category}/{name}/capture/{name}.bvh"
        path = args.data / f"{name}.bvh"
        entry = metadata[name]
        fetch(url, path, checksums[name])
        print(name, flush=True)
        return dict(name=name, category=category, reason=reason, url=url,
                    performer=entry.get("performer"), source=entry.get("source"),
                    bytes=path.stat().st_size, sha256=checksums[name])

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, plan["clips"]))
    plan["clips"] = records
    (args.data / "inventory.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
