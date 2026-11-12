"""Run the pinned real-motion and controlled-corruption benchmark."""
import argparse
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import statistics
from tempfile import TemporaryDirectory

import numpy as np

from bvh import BVH, LEGS, matrices, transpose, vectors
from cli import flatten, process, save_csv, save_json
from quality import measure
from refine import MODULES, config, detect, refine, runs, solve_leg

ROOT = Path(__file__).resolve().parent
SWEEP = ("J_Dance0_StepTouch", "B_SpinKarate", "J_ShortDance16_JazzWalk")
PREVIEWS = {"J_ShortDance16_JazzWalk": (0, None), "B_SpinKarate": (4, 12),
            "J_Dance17_Shuffle": (5, 12)}
CASES = (
    ("elbow_jitter", "J_Dance0_StepTouch", "smooth"),
    ("leg_jitter", "B_SpinKarate", "smooth"),
    ("root_jitter", "J_Dance2_Salsa", "smooth"),
    ("stance_slide", "B_HandsUp", "foot"),
    ("ground_sag", "B_HandsUp", "foot"),
    ("torso_spike", "B_BowKarate", "torso"),
    ("speed_spike", "J_ShortDance16_JazzWalk", "speed"),
)


def inject(bvh, clean, contact, kind):
    root, local = (array.copy() for array in clean)
    frames = len(root)
    center = frames // 2
    width = min(frames // 3, round(2.0 / bvh.dt))
    axis = np.array([1.0, 0.0, 0.0])
    if kind in ("stance_slide", "ground_sag"):
        side = 0
        k = 2 * round(0.35 / bvh.dt) + 1
        intervals = [(a, b) for a, b in runs(contact.mask[:, side, 1]) if b - a >= k]
        a, b = max(intervals, key=lambda run: run[1] - run[0])
        width = k
        center = (a + b) // 2
    if kind in ("torso_spike", "speed_spike"):
        width = 5 if kind == "torso_spike" else 1
        if kind == "torso_spike":
            center = int(np.flatnonzero(contact.mask.any(axis=(1, 2)))[len(root)//4])
    first = center - width // 2
    idx = np.arange(first, first + width)
    window = np.hanning(width)
    if kind in ("elbow_jitter", "leg_jitter"):
        joint = bvh.names.index("LeftForeArm" if kind == "elbow_jitter" else "RightLeg")
        signal = np.deg2rad(2.6) * window * np.sin(np.arange(width) * bvh.dt * 2 * np.pi * 21)
        local[idx, joint] = local[idx, joint] @ matrices(signal[:, None] * axis)
        return root, local, idx, ("angle_deg", joint), "2.6° envelope, 21 Hz rotation jitter"
    if kind == "root_jitter":
        root[idx, 2] += 0.010 * window * np.sin(np.arange(width) * bvh.dt * 2 * np.pi * 21)
        return root, local, idx, ("root_mm", 0), "10 mm envelope, 21 Hz root translation jitter"
    if kind in ("torso_spike", "speed_spike"):
        joint = bvh.names.index("Spine1" if kind == "torso_spike" else "RightForeArm")
        degrees = 42 if kind == "torso_spike" else 95
        local[idx, joint] = local[idx, joint] @ matrices(np.deg2rad(degrees) * window[:, None] * axis)
        return root, local, idx, ("angle_deg", joint), f"{degrees}° peak, {width} frames"
    positions, world = bvh.fk(*clean)
    for side in ((0,) if kind == "stance_slide" else (0, 1)):
        foot = bvh.names.index(LEGS[side][2])
        active = np.zeros(frames, dtype=bool)
        active[idx] = True
        target = positions[idx, foot].copy()
        if kind == "stance_slide":
            target[:, 0] += 0.044 * window
        else:
            target[:, 1] -= 0.022 * window
        solve_leg(bvh, local, positions, world, side, active, target, world[idx, foot],
                  np.deg2rad(28))
    axis_index = 0 if kind == "stance_slide" else 1
    return root, local, idx, ("foot_mm", axis_index), (
        "44 mm target lateral drift on the left stance foot" if kind == "stance_slide"
        else "22 mm target downward offset on both feet")


def error(bvh, clean, candidate, idx, mode):
    label, joint_or_axis = mode
    if label == "angle_deg":
        a = clean[1][idx, joint_or_axis]
        b = candidate[1][idx, joint_or_axis]
        diff = np.rad2deg(np.linalg.norm(vectors(transpose(a) @ b), axis=-1))
    elif label == "root_mm":
        diff = 1000 * np.linalg.norm(candidate[0][idx] - clean[0][idx], axis=-1)
    else:
        feet = [bvh.names.index(leg[3]) for leg in (LEGS[:1] if joint_or_axis == 0 else LEGS)]
        a, b = (bvh.fk(*motion)[0][idx][:, feet, joint_or_axis] for motion in (clean, candidate))
        diff = 1000 * (a - b)
    return float(np.sqrt(np.mean(diff**2)))


def controlled(data, stage, settings, videos):
    cases = []
    for kind, sample, module in CASES:
        bvh = BVH.read(data / f"{sample}.bvh")
        clean = bvh.decode()
        reference = detect(bvh, *bvh.fk(*clean), settings)
        root, local, idx, mode, injection = inject(bvh, clean, reference, kind)
        source = stage / f"{kind}.bvh"
        bvh.write(source, root, local)
        damaged = BVH.read(source)
        overrides = {key: dict(settings[key], enabled=(key == module)) for key in MODULES}
        case_config = settings | overrides | {"floor_m": reference.floor}
        if module == "speed":
            case_config["speed"]["strength"] = 100
        output = process(source, stage, case_config, videos and kind == "stance_slide",
                         max(0, idx[0] * bvh.dt - 2), 5)
        after = BVH.read(stage / f"{kind}.refined.bvh").decode()
        before = damaged.decode()
        cases.append({"case": kind, "clean": sample, "injection": injection, "module": module,
                      "first_frame_zero_indexed": int(idx[0]), "last_frame_zero_indexed": int(idx[-1]),
                      "error_unit": mode[0].split("_")[-1],
                      "error_rms_before": error(bvh, clean, before, idx, mode),
                      "error_rms_after": error(bvh, clean, after, idx, mode),
                      "fixed_clean_quality_before": measure(bvh, before, reference),
                      "fixed_clean_quality_after": measure(bvh, after, reference),
                      "input_reference_metrics": output, "config": case_config})
        print(f"synthetic {kind}", flush=True)
    return cases


def pct(before, after):
    return None if before is None or after is None or before == 0 else 100 * (after / before - 1)


def fmt(value, digits=1):
    return "—" if value is None else f"{value:.{digits}f}"


def delta(row, key):
    return pct(row["before"][key], row["after"][key])


def make_report(report):
    real = report["real"]
    synthetic = report["synthetic"]
    skates = [delta(row, "support_skate_m_s") for row in real]
    valid_skates = [x for x in skates if x is not None]
    median_skate = statistics.median(valid_skates) if valid_skates else None
    median_pose = statistics.median(row["fidelity"]["body_world_mean_mm"] for row in real)
    env = ", ".join(f"{key} {value}" for key, value in report["environment"].items())
    lines = [
        "# Benchmark results", "",
        f"{len(real)} G1 Moves clips, {sum(row['frames'] for row in real):,} frames, "
        f"and {len(synthetic)} controlled corruptions. Global strength {report['config']['strength']}.",
        "", f"Environment: {env}.", "",
        f"Mean foot skating decreased in {sum(x < 0 for x in valid_skates)}/{len(real)} clips. "
        f"The median per-clip change was {fmt(median_skate)}%, and the median per-clip "
        f"mean world-space joint displacement was {fmt(median_pose, 2)} mm.", "",
        "## Real motion", "",
        "| Clip | Frames | Foot skating, mm/s before → after | Mean penetration, mm before → after | Jerk P95 change | Mean joint displacement, mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in real:
        b, a = row["before"], row["after"]
        speed = [fmt(v * 1000) if v is not None else "—"
                 for v in (b["support_skate_m_s"], a["support_skate_m_s"])]
        lines.append(f"| {row['name']} | {row['frames']} | {' → '.join(speed)} | "
                     f"{fmt(b['penetration_mean_mm'], 2)} → {fmt(a['penetration_mean_mm'], 2)} | "
                     f"{fmt(delta(row, 'body_jerk_p95_m_s3'))}% | "
                     f"{fmt(row['fidelity']['body_world_mean_mm'], 2)} |")
    lines += [
        "", "## Controlled corruptions", "",
        "RMS is measured against the clean reference within the injected interval. Only the "
        "module under test is enabled. The speed module uses an individual strength of 100.", "",
        "| Corruption | Clean clip | RMS before → after | Change |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in synthetic:
        b, a = row["error_rms_before"], row["error_rms_after"]
        lines.append(f"| {row['case']} | {row['clean']} | {fmt(b, 2)} → {fmt(a, 2)} "
                     f"{row['error_unit']} | {fmt(pct(b, a))}% |")
    lines += [
        "", "## Strength sweep", "",
        "| Clip | Strength | Foot skating change | Jerk P95 change | Mean joint displacement, mm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["sweep"]:
        lines.append(f"| {row['file']} | {row['strength']} | {fmt(delta(row, 'support_skate_m_s'))}% | "
                     f"{fmt(delta(row, 'body_jerk_p95_m_s3'))}% | {fmt(row['fidelity']['body_world_mean_mm'], 2)} |")
    lines += [
        "", "## Interpretation", "",
        "- The input defines the contact intervals and ground estimate for both measurements. "
        "Foot skating uses horizontal heel/toe speed over consecutive support frames.",
        "- Penetration uses virtual foot points and an estimated plane. Deep penetration and large "
        "torso spikes can exceed the bounded corrections.",
        "- Jerk P95 describes high-end motion variation and can also reflect intentional fast motion.",
        "- These are human-motion kinematics. Robot retargeting, simulation and hardware performance "
        "remain untested.",
        f"- G1 Moves by Experiential Technologies and the credited performers, CC BY 4.0. "
        f"[Dataset revision]({report['dataset']}/tree/{report['revision']})",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    settings = config(args.config)
    inventory = json.loads((args.data / "inventory.json").read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix=".benchmark-") as temp:
        stage = Path(temp) / "result"
        real_dir, synth_dir, sweep_dir = (stage / name for name in ("real", "synthetic", "sweep"))
        for directory in (real_dir, synth_dir, sweep_dir):
            directory.mkdir(parents=True)
        results, sweep = [], []
        for sample in inventory["clips"]:
            name = sample["name"]
            source = args.data / f"{name}.bvh"
            if hashlib.sha256(source.read_bytes()).hexdigest() != sample["sha256"]:
                raise ValueError(f"Sample SHA-256 mismatch: {name}")
            preview = PREVIEWS.get(name) if not args.skip_video else None
            result = process(source, real_dir, settings, preview is not None, *(preview or (0, None)))
            results.append(result | sample)
            print(f"real {name}", flush=True)
            if name in SWEEP:
                for strength in (0, 50, 100):
                    target = sweep_dir / str(strength)
                    target.mkdir(exist_ok=True)
                    row = process(source, target, settings | {"strength": strength})
                    if strength == 0 and (target / f"{name}.refined.bvh").read_bytes() != source.read_bytes():
                        raise RuntimeError(f"Zero strength changed bytes: {name}")
                    sweep.append(row | {"strength": strength})
        synthetic = controlled(args.data, synth_dir, settings, not args.skip_video)
        report = {k: v for k, v in inventory.items() if k != "clips"} | {
            "config": settings, "environment": {"python": platform.python_version(),
                **{p: version(p) for p in ("numpy", "scipy", "Pillow")}},
            "real": results, "sweep": sweep, "synthetic": synthetic,
        }
        save_json(stage / "metrics.json", report)
        save_csv(stage / "real_metrics.csv", [flatten(row) for row in results])
        save_csv(stage / "sweep_metrics.csv", [flatten(row) for row in sweep])
        columns = ("case", "clean", "module", "error_unit", "error_rms_before", "error_rms_after",
                   "fixed_clean_quality_before", "fixed_clean_quality_after")
        save_csv(stage / "synthetic_metrics.csv", [flatten({k: row[k] for k in columns}) for row in synthetic])
        (stage / "REPORT.md").write_text(make_report(report), encoding="utf-8")
        if args.output.exists():
            raise FileExistsError(args.output)
        stage.rename(args.output)


if __name__ == "__main__":
    main()
