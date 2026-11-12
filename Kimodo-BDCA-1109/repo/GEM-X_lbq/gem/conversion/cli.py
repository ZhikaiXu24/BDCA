"""Convert saved Body Pose results without loading the neural model again."""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--target", choices=("soma", "g1"), default="soma")
    parser.add_argument("--output", type=Path, help="Single input only; otherwise save beside each .pt")
    kimodo_root = root.parent
    if not (kimodo_root / "kimodo/constraints.py").is_file():
        kimodo_root = root / "third_party/kimodo"
    parser.add_argument("--kimodo-root", type=Path, default=kimodo_root)
    parser.add_argument("--soma-assets", type=Path, default=root / "assets/soma")
    parser.add_argument("--frame-index", type=int, default=0, help="Target Kimodo keyframe index")
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--yaw-degrees", type=float, default=0, help="Additional world Y-axis rotation")
    parser.add_argument("--root-height", type=float, help="Hips height in meters; default: lowest target mesh point at Y=0")
    parser.add_argument("--validate", action="store_true", help="Reload with official Kimodo APIs and check FK")
    parser.add_argument("--render", action="store_true", help="Also render source/target meshes; implies --validate")
    parser.add_argument("--retarget-python", type=Path,
                        default=Path(sys.prefix).parent / "lbq_soma-retarget/bin/python")
    parser.add_argument("--retargeter-root", type=Path, default=root / "third_party/soma-retargeter")
    parser.add_argument("--device", default="cuda", help="Warp device for G1 IK; respects CUDA_VISIBLE_DEVICES")
    parser.add_argument("--hold-frames", type=int, default=30, help="Numerical static IK settling, not observed motion")
    parser.add_argument("--ik-iterations", type=int, default=48)
    parser.add_argument("--limb-rotation-scale", type=float, default=.1,
                        help="Scale official arm rotation objectives; prioritize pose positions")
    parser.add_argument("--reuse-retarget", action="store_true",
                        help="G1: use existing g1_retarget.npz; source/settings must match")
    args = parser.parse_args(argv)
    if args.output is not None and len(args.input) != 1:
        parser.error("--output requires exactly one input")
    if args.frame_index < 0:
        parser.error("--frame-index must be nonnegative")
    if args.target == "g1" and args.hold_frames < 2:
        parser.error("--hold-frames must be at least 2 to check static convergence")

    # Use the source checkout, avoiding Kimodo's conflicting generation dependency pins.
    sys.path.insert(0, str(args.kimodo_root.resolve()))
    from kimodo.constraints import save_constraints_lst
    from .kimodo_soma import KimodoSomaConverter, read_body_pose

    converter = KimodoSomaConverter(args.soma_assets, args.yaw_degrees)
    if args.target == "g1":
        from .kimodo_g1 import convert_g1

        convert_g1(args, converter)
        return
    for path in args.input:
        record = read_body_pose(path, args.instance_index)
        constraint = converter.convert(record, args.frame_index, args.root_height)
        output = args.output or path.with_name("kimodo_soma_fullbody.json")
        save_constraints_lst(str(output), [constraint])
        if args.validate or args.render:
            from .validation import validate_conversion

            report = validate_conversion(
                record, output, converter, constraint, args.soma_assets,
                root_height=args.root_height, render=args.render,
            )
            report["input"] = str(path.resolve())
            report["output"] = str(output.resolve())
            report_path = output.with_name(output.stem + "_validation.json")
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Validated: {output}; report: {report_path}")
            if not report["passed"]:
                failed = [name for name, passed in report["checks"].items() if not passed]
                raise RuntimeError(f"Conversion validation failed: {failed}; see {report_path}")


if __name__ == "__main__":
    main()
