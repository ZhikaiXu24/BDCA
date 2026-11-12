"""Run NVIDIA SOMA Retargeter in its own Python environment (no Torch dependency)."""

import argparse
from pathlib import Path

import newton
import numpy as np
import warp as wp

from soma_retargeter.io import bvh as bvh_io
from soma_retargeter.io import csv as csv_io
from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
from soma_retargeter.pipelines.utils import get_retargeter_config
from soma_retargeter.utils.newton_utils import get_filtered_joint_names, get_name_from_label


def body_fk(model, coordinates):
    state = model.state()
    newton.eval_fk(model, wp.array(coordinates, dtype=wp.float32), model.joint_qd, state)
    return state.body_q.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ik-iterations", type=int, default=48)
    parser.add_argument("--limb-rotation-scale", type=float, default=.1)
    args = parser.parse_args()
    wp.init()
    wp.set_device(args.device)
    skeleton, first = bvh_io.load_bvh(str(args.input[0]))
    buffers = [first] + [bvh_io.load_bvh(str(path))[1] for path in args.input[1:]]
    for buffer in buffers[1:]:
        if buffer.skeleton.joint_names != skeleton.joint_names or not np.array_equal(
            buffer.skeleton.parent_indices, skeleton.parent_indices
        ):
            raise ValueError("Batch inputs must use the same SOMA hierarchy")
        buffer.skeleton = skeleton
    config = get_retargeter_config("soma", "unitree_g1")
    # Static observations have no measured velocities or contacts.
    config.update(
        enable_contact_processing=False, enable_post_processing=False,
        auto_detect_contacts=False, smooth_joint_filter_weight=0.0,
        ik_iterations=args.ik_iterations,
    )
    # A fullbody keyframe constrains positions; human arm twist is not a robot hinge target.
    for name in ["LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"]:
        config["ik_match_table"][name]["r_weight"] *= args.limb_rotation_scale
    pipeline = SomaRetargetingPipeline(skeleton, "soma", "unitree_g1", config)
    # Same Y-up/+Z-forward -> Z-up/+X-forward mapping used by Kimodo's qpos API.
    matrix = wp.mat33(0., 0., 1., 1., 0., 0., 0., 1., 0.)
    offset = wp.transform(wp.vec3(0.), wp.quat_from_matrix(matrix))
    pipeline.add_input_motions(buffers, [offset] * len(buffers), scale_animation=True)
    results = pipeline.execute()

    names = get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
    body_names = [get_name_from_label(name) for name in pipeline.robot_builder.body_label]
    model = pipeline.ik_model
    calibration = np.repeat(model.joint_q.numpy()[None], len(names) + 1, axis=0)
    calibration[:, 7:] = 0
    for j in range(len(names)):
        calibration[j + 1, 7 + j] = .1
    calibration_body = np.stack([body_fk(model, q) for q in calibration])
    limits = np.stack([
        model.joint_limit_lower.numpy()[-len(names):],
        model.joint_limit_upper.numpy()[-len(names):],
    ])
    for env, (path, source, result) in enumerate(zip(args.input, buffers, results)):
        q = np.asarray(result.data)
        body = body_fk(model, q[-1])
        link_indices = [index for index, weight in pipeline.mapped_body_link_pos_data]
        goals = np.asarray(pipeline.input_targets[env][-1])
        residual = np.linalg.norm(body[link_indices, :3] - goals[:, :3], axis=-1)
        csv_io.save_csv(str(path.with_name("g1_retarget.csv")), names, result)
        source_world = source.compute_global_transforms(0)
        np.savez_compressed(
            path.with_name("g1_retarget.npz"),
            joint_q=q, joint_names=names, body_names=body_names, body_q=body,
            calibration_q=calibration, calibration_body_q=calibration_body,
            joint_limits=limits, source_joint_names=skeleton.joint_names,
            source_world=source_world, target_names=pipeline.mapped_joints,
            target_positions=goals[:, :3], position_residual_m=residual,
            human_scale_ratio=pipeline.human_robot_scaler.human_scale_ratio,
        )
        print(f"Retargeted {path}: frames={len(q)}, max_position_residual={residual.max():.6f} m", flush=True)


if __name__ == "__main__":
    main()
