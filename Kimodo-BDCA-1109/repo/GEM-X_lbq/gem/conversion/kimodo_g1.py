"""Saved GEM Body Pose -> SOMA BVH -> official Newton IK -> Kimodo G1 constraint."""

import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from kimodo.constraints import FullBodyConstraintSet, load_constraints_lst, save_constraints_lst
from kimodo.exports.bvh import save_motion_bvh

from .g1_kinematics import G1RestFrameAdapter, robot_mesh
from .kimodo_soma import read_body_pose


def finish_g1(path, source, soma_converter, args, retargeter_commit):
    data = np.load(path.with_name("g1_retarget.npz"))
    adapter = G1RestFrameAdapter(data)
    skeleton = adapter.skeleton
    calibration_global, calibration_positions = adapter.from_body_transforms(data["calibration_body_q"])
    calibration_fk = adapter.fk_error(calibration_global, calibration_positions, data["calibration_body_q"])
    calibration_q = adapter.to_joint_q(calibration_global, calibration_positions)
    calibration_angles = float(np.abs(calibration_q[:, 7:] - data["calibration_q"][:, 7:]).max())
    global_rots, positions = adapter.from_body_transforms(data["body_q"][None])
    vertices, faces = robot_mesh(skeleton, global_rots[0], positions[0])
    translation = -positions[0, 0].clone()
    translation[1] = -vertices[:, 1].min() if args.root_height is None else args.root_height - positions[0, 0, 1]
    positions += translation
    constraint = FullBodyConstraintSet(skeleton, torch.tensor([args.frame_index]), positions, global_rots)
    output = args.output or path.with_name("kimodo_g1_fullbody.json")
    save_constraints_lst(str(output), [constraint])
    loaded = load_constraints_lst(str(output), skeleton)[0]
    condition_data, condition_indices = defaultdict(list), defaultdict(list)
    loaded.update_constraints(condition_data, condition_indices)
    conditioning_positions = torch.cat(condition_data["global_joints_positions"])
    conditioning_indices = torch.cat(condition_indices["global_joints_positions"])
    recovered = adapter.to_joint_q(loaded.global_joints_rots, loaded.global_joints_positions)
    native = data["joint_q"][-1:].copy()
    native[:, :3] += translation.numpy() @ adapter.matrix
    q_error = float(np.abs(recovered[:, 7:] - native[:, 7:]).max())
    native_fk_error = adapter.fk_error(
        loaded.global_joints_rots, loaded.global_joints_positions, data["body_q"][None], translation.numpy()
    )
    source_indices = [data["source_joint_names"].tolist().index(name) for name in soma_converter.skeleton.bone_order_names]
    bvh_error = float(np.abs(data["source_world"][source_indices, :3] - source.global_joints_positions[0].numpy()).max())
    lower, upper = data["joint_limits"]
    limit_error = float(np.maximum(lower - native[0, 7:], native[0, 7:] - upper).clip(min=0).max())
    convergence = float(np.abs(np.diff(data["joint_q"][-5:, 7:], axis=0)).max())
    vertices, faces = robot_mesh(skeleton, loaded.global_joints_rots[0], loaded.global_joints_positions[0])
    root_error = float(np.abs(recovered[:, :3] - native[:, :3]).max())
    quat_error = float(max(0, 1 - np.abs(np.sum(recovered[:, 3:7] * native[:, 3:7], axis=-1)).min()))
    reload_error = float((loaded.global_joints_positions - positions).abs().max())
    height_ok = abs(float(vertices[:, 1].min())) < 1e-5 if args.root_height is None else abs(float(loaded.root_y_pos[0]) - args.root_height) < 1e-5
    checks = {
        "bvh_roundtrip": bvh_error < 5e-5,
        "all_29_hinge_probe_fk": calibration_fk < 1e-5,
        "all_29_hinge_probe_angles": calibration_angles < 1e-5,
        "native_robot_vs_json_fk": native_fk_error < 1e-5,
        "json_vs_native_hinge_angles": q_error < 1e-5,
        "json_vs_native_root_position": root_error < 1e-5,
        "json_vs_native_root_rotation": abs(quat_error) < 1e-5,
        "official_json_reload_fk": reload_error < 1e-5,
        "generation_conditioning": (
            list(conditioning_positions.shape) == [34, 3]
            and bool(torch.isfinite(conditioning_positions).all())
            and bool((conditioning_indices[:, 0] == args.frame_index).all())
        ),
        "joint_limits": limit_error < 1e-5,
        "static_solver_converged": convergence < 1e-4,
        "ground_or_explicit_height": height_ok,
    }
    # Preserve the full stationary solve for inspection; JSON selects only the final pose.
    history = data["joint_q"].copy()
    history[:, :3] += translation.numpy() @ adapter.matrix
    qpos_path = output.parent / "g1_qpos.csv"
    np.savetxt(qpos_path, adapter.qpos(history), delimiter=",")
    report = {
        "passed": all(checks.values()), "checks": checks,
        "passed_scope": "Numerical adaptation, official JSON loading, FK, joint limits and IK convergence; pose similarity is reported separately",
        "input": str(path.resolve()), "output": str(output.resolve()),
        "qpos_csv": str(qpos_path.resolve()),
        "frame_indices": [args.frame_index], "selected_retarget_frame": len(history) - 1,
        "bvh_fk_max_abs_m": bvh_error, "calibration_fk_max_abs_m": calibration_fk,
        "calibration_hinge_angle_max_rad": calibration_angles,
        "native_robot_vs_json_fk_max_abs_m": native_fk_error,
        "json_hinge_angle_max_error_rad": q_error, "json_root_position_max_error_m": root_error,
        "json_root_quaternion_error": quat_error, "json_reload_fk_max_abs_m": reload_error,
        "joint_limit_violation_rad": limit_error, "last_hold_joint_delta_rad": convergence,
        "joints_at_limit": [
            name for name, angle, lo, hi in zip(adapter.joint_names, native[0, 7:], lower, upper)
            if min(abs(angle - lo), abs(angle - hi)) < 1e-3
        ],
        "root_positions_m": loaded.global_joints_positions[:, 0].tolist(),
        "heading_radians": float(torch.atan2(loaded.global_root_heading[0, 1], loaded.global_root_heading[0, 0])),
        "robot_mesh_min_y_m": float(vertices[:, 1].min()),
        "normalization_translation_yup_m": translation.tolist(),
        "scaled_target_position_residual_m": dict(zip(data["target_names"].tolist(), data["position_residual_m"].tolist())),
        "scaled_target_position_max_residual_m": float(data["position_residual_m"].max()),
        "scaled_target_position_rms_residual_m": float(np.sqrt(np.mean(data["position_residual_m"] ** 2))),
        "pose_fidelity_note": "IK is approximate and may meet joint limits. Inspect the target residuals and image; numerical conversion success does not imply exact human-pose reproduction.",
        "human_scale_ratio": float(data["human_scale_ratio"]),
        "retarget_settings": {"hold_frames": args.hold_frames, "ik_iterations": args.ik_iterations, "limb_rotation_scale": args.limb_rotation_scale, "contact_processing": False, "post_processing": False},
        "retargeter_commit": retargeter_commit,
        "policy": "Camera up assumed world up; preserve predicted heading. Root XZ=0, lowest G1 mesh point at ground unless height is specified. Repeated frames are numerical IK settling, not observed motion or contact evidence.",
        "scope": "Kinematic G1 pose constraint; no Kimodo diffusion rollout, dynamics, balance or hardware execution tested",
    }
    if args.render:
        from .validation import render_comparison

        scale = float(data["human_scale_ratio"])
        source_vertices = soma_converter.skin.skin(source.global_joints_rots, source.global_joints_positions, rot_is_global=True)[0] * scale
        source_vertices += translation
        image_path = output.with_name(output.stem + "_validation.png")
        render_comparison(source_vertices, vertices, soma_converter.skin.faces, faces, image_path,
                          labels=("Scaled SOMA pose", "G1 JSON reload"))
        report["visualization"] = str(image_path.resolve())
    report_path = output.with_name(output.stem + "_validation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"G1 validation failed: {[name for name, ok in checks.items() if not ok]}; see {report_path}")
    print(f"G1 validated: {output}; IK max target residual {report['scaled_target_position_max_residual_m']:.4f} m")


@torch.inference_mode()
def convert_g1(args, soma_converter):
    retargeter_root = args.retargeter_root.resolve()
    if not (retargeter_root / ".git").exists():
        raise ValueError("--retargeter-root must point to a SOMA Retargeter Git checkout.")
    retargeter_commit = subprocess.check_output(
        ["git", "-C", str(retargeter_root), "rev-parse", "HEAD"], text=True
    ).strip()
    sources = []
    for path in args.input:
        source = soma_converter.convert(read_body_pose(path, args.instance_index))
        sources.append(source)
        if not args.reuse_retarget:
            local = soma_converter.skeleton.global_rots_to_local_rots(source.global_joints_rots)
            save_motion_bvh(
                path.with_name("g1_source_soma.bvh"), local.repeat(args.hold_frames, 1, 1, 1),
                source.global_joints_positions[:, 0].repeat(args.hold_frames, 1),
                skeleton=soma_converter.skeleton, fps=30, standard_tpose=False,
            )
    if not args.reuse_retarget:
        command = [str(args.retarget_python), str(Path(__file__).with_name("retarget_worker.py")),
                   "--device", args.device, "--ik-iterations", str(args.ik_iterations),
                   "--limb-rotation-scale", str(args.limb_rotation_scale), "--input"]
        command += [str(path.with_name("g1_source_soma.bvh").resolve()) for path in args.input]
        log_path = args.input[0].parent / "g1_retarget.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(retargeter_root)
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise RuntimeError(f"Official retargeter failed; see {log_path}")
    for path, source in zip(args.input, sources):
        finish_g1(path, source, soma_converter, args, retargeter_commit)
