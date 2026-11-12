"""Integration checks against actual SOMA FK and Kimodo's JSON loading path."""

import itertools
import subprocess
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kimodo.constraints import load_constraints_lst
from kimodo.skeleton import SOMASkeleton30

from gem.common.geometry.rotations import matrix_to_axis_angle
from gem.common.soma import SomaFixedGs1BodyAdapter, SomaFixedGs1ShapeSpec, SomaRuntimeConfig

BODY_BONE_ENDS = (
    "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
    "LeftShin", "LeftFoot", "LeftToeBase", "RightShin", "RightFoot", "RightToeBase",
)


def _bone_angles(source, target, skeleton):
    indices = torch.tensor([skeleton.bone_index[name] for name in BODY_BONE_ENDS])
    parents = skeleton.joint_parents[indices]
    a = (source[indices] - source[parents]).double()
    b = (target[indices] - target[parents]).double()
    return torch.rad2deg(torch.atan2(
        torch.linalg.vector_norm(torch.linalg.cross(a, b), dim=-1), (a * b).sum(-1)
    ))


@torch.inference_mode()
def validate_conversion(record, json_path, converter, expected, soma_assets, *, root_height=None, render=False):
    """Check serialization, reference frames, geometry and SOMA77 -> SOMA30 loading."""
    skeleton = converter.skeleton
    loaded = load_constraints_lst(str(json_path), skeleton)[0]
    source_parameters = {key: value[None] for key, value in record["parameters"].items()}
    adapter = SomaFixedGs1BodyAdapter(
        SomaRuntimeConfig(
            data_root=soma_assets, lod="mid", device="cpu", apply_correctives=False,
            procedural_reference="legacy_pose021",
        ),
        shape_spec=SomaFixedGs1ShapeSpec(require_positive_scales=False),
    )
    native = adapter(
        global_orient=matrix_to_axis_angle(source_parameters["global_orient"].squeeze(1)),
        body_pose=matrix_to_axis_angle(source_parameters["body_pose"]),
        identity_coeffs=source_parameters["identity_coeffs"],
        scale_params=source_parameters["scale_params"],
        pose2rot=True, return_joints_only=True, repose_to_bind_pose=True,
    )
    # This reconstruction is independent of the converter's analytic rotation path.
    reconstruction_error = (
        native["joints"][0] + record["camera_translation"] - record["joints_camera"]
    ).abs().max().item()
    C = converter.camera_to_world
    indices = converter.source_order
    native_global = native["transforms"][:, 1:][:, indices, :3, :3]
    expected_global = C @ native_global @ skeleton.global_rot_offsets.mT
    rotation_error = (loaded.global_joints_rots - expected_global).abs().max().item()
    roundtrip_error = (
        loaded.global_joints_positions - expected.global_joints_positions
    ).abs().max().item()
    root = loaded.global_joints_positions[:, skeleton.root_idx]
    source = (C @ (record["joints_camera"] - record["joints_camera"][0]).T).T[indices]
    target = loaded.global_joints_positions[0] - root[0]
    bone_angles = _bone_angles(source, target, skeleton)
    shape_distances = torch.linalg.vector_norm(source - target, dim=-1)
    vertices = converter.skin.skin(loaded.global_joints_rots, loaded.global_joints_positions, rot_is_global=True)
    floor = vertices[..., 1].min().item()

    # Identity articulation is not identity in Kimodo: the reference offsets differ.
    rest = torch.eye(3).repeat(1, 77, 1, 1)
    rest[:, 0] = C.T
    neutral_native = adapter.runtime.pose(rest, pose2rot=False, fk_only=True, apply_correctives=False)
    rest_global = converter.standard_rotations(rest)
    neutral_expected = (
        C @ neutral_native["transforms"][:, 1:][:, indices, :3, :3]
        @ skeleton.global_rot_offsets.mT
    )
    rest_error = (rest_global - neutral_expected).abs().max().item()

    reduced = SOMASkeleton30().float()
    loaded30 = load_constraints_lst(str(json_path), reduced)[0]
    data, frame_indices = defaultdict(list), defaultdict(list)
    loaded30.update_constraints(data, frame_indices)
    conditioning_points = torch.cat(data["global_joints_positions"])
    body_names = [name for name in reduced.bone_order_names if "Hand" not in name or name.endswith("Hand")]
    full_indices = [skeleton.bone_index[name] for name in body_names]
    small_indices = [reduced.bone_index[name] for name in body_names]
    reduced_body_error = (
        loaded30.global_joints_positions[:, small_indices] - loaded.global_joints_positions[:, full_indices]
    ).abs().max().item()

    checks = {
        "source_pose_reconstruction": reconstruction_error < 1e-4,
        "reference_and_global_rotation": rotation_error < 1e-4,
        "identity_articulation_reference": rest_error < 1e-4,
        "json_reload_fk": roundtrip_error < 1e-5,
        "major_limb_bone_directions": bone_angles.max().item() < 0.1,
        "hips_horizontal_origin": root[:, [0, 2]].abs().max().item() < 1e-6,
        "grounding_or_explicit_height": abs(floor) < 1e-5 if root_height is None else abs(root[0, 1].item() - root_height) < 1e-5,
        "soma30_body_fk": reduced_body_error < 1e-5,
        "soma30_conditioning": list(conditioning_points.shape) == [30, 3] and bool(torch.isfinite(conditioning_points).all()),
        "frame_index_preserved": bool(torch.equal(loaded.frame_indices, expected.frame_indices)),
    }
    import kimodo

    kimodo_root = Path(kimodo.__file__).resolve().parents[1]
    heading = loaded.global_root_heading[0]
    report = {
        "passed": all(checks.values()), "checks": checks,
        "source_reconstruction_max_abs_m": reconstruction_error,
        "global_rotation_max_abs_error": rotation_error,
        "identity_articulation_reference_max_abs_error": rest_error,
        "json_roundtrip_fk_max_abs_m": roundtrip_error,
        "soma30_body_fk_max_abs_m": reduced_body_error,
        "major_limb_bone_direction_max_degrees": bone_angles.max().item(),
        "major_limb_bone_direction_degrees": dict(zip(BODY_BONE_ENDS, bone_angles.tolist())),
        "fixed_vs_fitted_root_aligned_mean_joint_distance_m": shape_distances.mean().item(),
        "fixed_vs_fitted_root_aligned_max_joint_distance_m": shape_distances.max().item(),
        "root_positions_m": root.tolist(), "target_mesh_min_y_m": floor,
        "heading_radians": torch.atan2(heading[1], heading[0]).item(),
        "camera_to_world": C.tolist(),
        "policy": {
            "rotation_reference": "GEM SOMA v0.2.0 -> Kimodo standard_t_pose_global_offsets_rots",
            "units": "meters; axis-angle radians",
            "heading": "preserve camera-relative heading, then apply optional world yaw",
            "camera_assumption": "image up is world up; camera roll and pitch are not estimated",
            "root_xz": "zero; discard camera translation",
            "root_y": "lowest fixed-body skin vertex at Y=0" if root_height is None else "explicit user-supplied height",
            "shape": "Kimodo fixed skeleton/skin; GEM identity and scale are not serialized",
        },
        "kimodo_commit": subprocess.check_output(["git", "-C", str(kimodo_root), "rev-parse", "HEAD"], text=True).strip(),
        "versions": {name: version(name) for name in ["torch", "py-soma-x", "transformers", "peft", "hydra-core"]},
        "scope": "Official JSON loading, FK, SOMA30 conditioning and visualization; no diffusion generation tested",
    }
    if render:
        source_vertices = (C @ (record["vertices_camera"] - record["joints_camera"][0]).T).T + root[0]
        image_path = Path(json_path).with_name(Path(json_path).stem + "_validation.png")
        # Both bodies use the same hips position and cameras, exposing proportion changes.
        render_comparison(source_vertices, vertices[0], record["faces"], converter.skin.faces, image_path)
        report["visualization"] = str(image_path.resolve())
    return report


def render_comparison(source, target, source_faces, target_faces, output,
                      labels=("GEM fitted body", "Kimodo JSON / fixed body")):
    """Render official Kimodo skin reconstructed from the reloaded JSON beside GEM's mesh."""
    from PIL import Image, ImageDraw, ImageFont
    from gem.common.rendering import MotionMeshRenderer

    width, height, label = 480, 640, 30
    bodies = [source.numpy(), target.numpy()]
    faces = [source_faces.numpy(), target_faces.numpy()]
    combined = np.concatenate(bodies)
    lower, upper = combined.min(0), combined.max(0)
    center = (lower + upper) / 2
    corners = np.array(list(itertools.product(*zip(lower, upper))))
    focal = height / (2 * np.tan(np.deg2rad(20)))
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    canvas = Image.new("RGB", (2 * width, 2 * (height + label)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    with MotionMeshRenderer(width, height, backend="software") as renderer:
        for row, (view, direction) in enumerate([("front", [0, .12, 1]), ("side", [1, .12, 0])]):
            direction = np.asarray(direction, dtype=float)
            direction /= np.linalg.norm(direction)
            forward = -direction
            right = np.cross(forward, [0, 1, 0])
            right /= np.linalg.norm(right)
            rotation = np.stack([right, np.cross(forward, right), forward])
            points = (corners - center) @ rotation.T
            distance = max(
                (np.abs(points[:, 0]) * focal / (width * .43) - points[:, 2]).max(),
                (np.abs(points[:, 1]) * focal / (height * .43) - points[:, 2]).max(),
            ) + .2
            eye = center + direction * distance
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rotation
            extrinsic[:3, 3] = -rotation @ eye
            for col, (name, vertices, topology) in enumerate(zip(labels, bodies, faces)):
                rgb = renderer.render(vertices, topology, K, extrinsic)
                top = row * (height + label)
                canvas.paste(Image.fromarray(rgb), (col * width, top + label))
                draw.text((col * width + 8, top + 5), f"{name} - {view}", fill="black", font=font)
    canvas.save(output)
