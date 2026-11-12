"""Draw single-image Body Pose skeletons from saved full-image pixel coordinates."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch

from gem.motion.frontend.joint_sets import SOMA29_SOURCE_INDICES, SOMA77_JOINT_NAMES


def load_skeleton(soma_assets):
    """Read the public rig hierarchy without constructing a SOMA geometry runtime."""
    path = Path(soma_assets) / "SOMA_neutral.npz"
    # This versioned reference rig includes virtual Root, unlike saved public77 joints.
    prefix = "t_pose_world__v0.2.0__"
    with np.load(path, allow_pickle=False) as assets:
        names = assets[prefix + "joint_names"].tolist()
        parents = assets[prefix + "parent_ids"].tolist()
    if names != ["Root", *SOMA77_JOINT_NAMES] or len(parents) != len(names):
        raise ValueError("SOMA reference rig does not match the saved public77 joint order")
    edges = []
    for child in range(1, len(names)):
        parent = parents[child]
        if not 0 <= parent < child:
            raise ValueError(f"Invalid SOMA parent for {names[child]}")
        if parent != 0:
            edges.append((parent - 1, child - 1))
    return edges


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _color(index):
    name = SOMA77_JOINT_NAMES[index]
    # BGR: anatomical left is green, right is orange, and center is cyan.
    return (80, 210, 60) if name.startswith("Left") else (
        (0, 140, 255) if name.startswith("Right") else (255, 210, 0)
    )


def draw_skeleton(image, instances, edges, *, layout="body29", radius=3, thickness=2):
    """Draw valid in-raster joints; validity is not an occlusion/confidence score."""
    if layout not in ("body29", "full77"):
        raise ValueError("Skeleton layout must be body29 or full77")
    if radius < 1 or thickness < 1:
        raise ValueError("Point radius and line thickness must be positive")
    selected = list(SOMA29_SOURCE_INDICES) if layout == "body29" else list(range(77))
    selected_edges = [(a, b) for a, b in edges if a in selected and b in selected]
    height, width = image.shape[:2]
    canvas, summaries = image.copy(), []
    for row in instances:
        pixels = _array(row["keypoints_2d"])
        projection_valid = _array(row["projection_valid"]).astype(bool)
        if pixels.shape != (77, 2) or projection_valid.shape != (77,):
            raise ValueError("Body Pose requires keypoints_2d[77,2] and projection_valid[77]")
        finite = np.isfinite(pixels).all(axis=-1)
        in_image = finite & (pixels >= 0).all(-1) & (pixels < [width, height]).all(-1)
        drawable = projection_valid & in_image
        points = np.zeros((77, 2), dtype=int)
        points[drawable] = np.rint(pixels[drawable]).astype(int)
        # A subpixel coordinate at the border can round one pixel outside the raster.
        points = np.clip(points, [0, 0], [width - 1, height - 1])
        drawn_edges = [(a, b) for a, b in selected_edges if drawable[a] and drawable[b]]
        for parent, child in drawn_edges:
            cv2.line(canvas, tuple(points[parent]), tuple(points[child]), _color(child),
                     thickness, cv2.LINE_AA)
        for index in selected:
            if drawable[index]:
                cv2.circle(canvas, tuple(points[index]), radius, (30, 30, 30), -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(points[index]), max(1, radius - 1), _color(index),
                           -1, cv2.LINE_AA)
        summaries.append({
            "track_id": row["track_id"],
            "total_joints": 77,
            "projection_valid": int(projection_valid.sum()),
            "nonfinite": int((~finite).sum()),
            "valid_outside_image": int((projection_valid & finite & ~in_image).sum()),
            "selected_joints": len(selected),
            "drawn_joints": int(drawable[selected].sum()),
            "selected_edges": len(selected_edges),
            "drawn_edges": len(drawn_edges),
        })
    return canvas, summaries


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_skeleton(input_path, output_dir, *, soma_assets, source=None, layout="body29",
                    radius=3, thickness=2):
    """Export a skeleton overlay plus provenance; existing output files are refused."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    output, report = output_dir / "skeleton.png", output_dir / "skeleton.json"
    for path in (output, report):
        if path.exists():
            raise FileExistsError(f"Output already exists; choose a new --output-dir: {path}")
    result = torch.load(input_path, map_location="cpu", weights_only=True)
    if (result.get("format"), result.get("task"), result.get("mode")) != (
        "gem_pose_predictions_v1", "body", "image"
    ):
        raise ValueError("Expected a saved single-image GEM Body Pose result")
    if result.get("coordinates", {}).get("keypoints_2d") != "full_image_pixels":
        raise ValueError("Saved keypoints must use full_image_pixels coordinates")
    if result.get("camera_adapter", {}).get("output_raster", "source") != "source":
        raise ValueError("Expected source-raster predictions")
    if len(result["frames"]) != 1 or result["frames"][0]["frame_id"] != 0:
        raise ValueError("Expected exactly one saved image frame with frame_id=0")
    frame = result["frames"][0]
    if not frame["instances"]:
        raise ValueError("The saved image contains no predicted body instances")
    source = Path(source or result["source"])
    image = cv2.imread(str(source))
    if image is None:
        raise OSError(f"Cannot read source image {source}")
    if frame["image_size"] != [image.shape[1], image.shape[0]]:
        raise ValueError("Source image dimensions differ from the saved Pose result")
    edges = load_skeleton(soma_assets)
    canvas, summaries = draw_skeleton(image, frame["instances"], edges, layout=layout,
                                      radius=radius, thickness=thickness)
    selected = list(SOMA29_SOURCE_INDICES) if layout == "body29" else list(range(77))
    metadata = {
        "format": "gem_body_skeleton_render_v1",
        "input": str(input_path.resolve()), "input_sha256": _sha256(input_path),
        "source": str(source.resolve()), "source_sha256": _sha256(source),
        "soma_assets": str(Path(soma_assets).resolve()),
        "soma_rig_sha256": _sha256(Path(soma_assets) / "SOMA_neutral.npz"),
        "release": result.get("release"),
        "image_size": frame["image_size"],
        "coordinates": "full_image_pixels",
        "keypoint_origin": "GEM 3D joints projected to the source image; not independent 2D detections",
        "mask_semantics": "projection_valid is geometric validity, not visibility or confidence",
        "layout": layout, "radius": radius, "thickness": thickness,
        "selected_indices": selected,
        "joint_names": list(SOMA77_JOINT_NAMES),
        "edges": [(a, b) for a, b in edges if a in selected and b in selected],
        "colors": {"left": "green", "right": "orange", "center": "cyan"},
        "instances": summaries,
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("torch", "numpy", "opencv-python-headless", "py-soma-x")
        }},
        "renderer_sha256": _sha256(__file__),
        "output": str(output.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Cannot write image {output}")
    metadata["output_sha256"] = _sha256(output)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing body.pt; inference is not rerun")
    parser.add_argument("--output-dir", required=True, help="Directory without prior skeleton outputs")
    parser.add_argument("--soma-assets", required=True, help="Directory containing SOMA_neutral.npz")
    parser.add_argument("--source", help="Override the saved source image path; dimensions must match")
    parser.add_argument("--layout", choices=("body29", "full77"), default="body29")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--thickness", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        result = render_skeleton(args.input, args.output_dir, soma_assets=args.soma_assets,
                                 source=args.source, layout=args.layout, radius=args.radius,
                                 thickness=args.thickness)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Skeleton rendering failed: {exc}\n")
    print(f"Saved skeleton to {result['output']}")


if __name__ == "__main__":
    main()
