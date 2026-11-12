"""Fixed-reference diagnostics for recorded human motion."""
import numpy as np

from bvh import LEGS, transform, transpose, vectors
from refine import foot_geometry, norm, unit


G1_MAX_KNEE_FLEX_RAD = 2.8798  # Unitree g1_29dof_mode_15.urdf, left and right


def body_joints(bvh):
    return [i for i, name in enumerate(bvh.names) if not name[-1].isdigit() or name == "Spine1"]


def percentile(values, p):
    return float(np.percentile(values, p)) if values.size else None


def measure(bvh, pose, reference):
    root, local = pose
    pos, world = bvh.fk(root, local)
    points, _ = foot_geometry(bvh, pos, world)
    contact = reference.mask[1:] & reference.mask[:-1]
    speed = norm(np.diff(points[..., [0, 2]], axis=0))[..., 0] / bvh.dt
    depth = np.maximum(0, reference.floor - points[..., 1]).max(axis=(1, 2))
    jerk = norm(np.diff(pos[:, body_joints(bvh)], n=3, axis=0))[..., 0] / bvh.dt**3
    tilt = np.rad2deg(np.arccos(np.clip(world[:, bvh.names.index("Spine1"), 1, 1], -1, 1)))
    flex = []
    for leg in LEGS:
        hip, knee, ankle, _ = [bvh.names.index(name) for name in leg]
        a = unit(pos[:, knee] - pos[:, hip])
        b = unit(pos[:, ankle] - pos[:, knee])
        flex.append(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))
    flex = np.stack(flex, axis=1)
    return {
        "support_skate_m_s": float(speed[contact].mean()) if contact.any() else None,
        "support_skate_p95_m_s": percentile(speed[contact], 95),
        "support_travel_m": float(speed[contact].sum() * bvh.dt) if contact.any() else None,
        "penetration_mean_mm": float(depth.mean() * 1000),
        "penetration_p95_mm": float(np.percentile(depth, 95) * 1000),
        "penetration_pct_over_10mm": float(100 * (depth > 0.010).mean()),
        "body_jerk_p95_m_s3": percentile(jerk, 95),
        "torso_tilt_p95_deg": percentile(tilt, 95),
        "human_knee_flex_p99_deg": float(np.rad2deg(np.percentile(flex, 99))),
        "g1_knee_flex_proxy_over_limit_pct": float(100 * (flex > G1_MAX_KNEE_FLEX_RAD).mean()),
    }


def fidelity(bvh, before, after):
    root_a, local_a = before
    root_b, local_b = after
    pos_a = bvh.fk(*before)[0]
    pos_b = bvh.fk(*after)[0]
    body = body_joints(bvh)
    aligned_a = transform(transpose(local_a[:, 0, None]), pos_a[:, body] - root_a[:, None])
    aligned_b = transform(transpose(local_b[:, 0, None]), pos_b[:, body] - root_b[:, None])
    world_error = norm(pos_b[:, body] - pos_a[:, body])[..., 0]
    root_error = norm(root_b - root_a)[..., 0]
    angle = np.rad2deg(norm(vectors(transpose(local_a) @ local_b))[..., 0])
    modified = (angle[:, body].max(axis=1) > 3) | (world_error.max(axis=1) > 0.020)
    return {
        "body_world_mean_mm": float(world_error.mean() * 1000),
        "body_world_p95_mm": float(np.percentile(world_error, 95) * 1000),
        "body_root_aligned_mean_mm": float(norm(aligned_b - aligned_a).mean() * 1000),
        "local_angle_mean_deg": float(angle[:, body].mean()),
        "local_angle_p95_deg": percentile(angle[:, body], 95),
        "root_rms_mm": float(np.sqrt(np.mean(root_error**2)) * 1000),
        "root_max_mm": float(root_error.max() * 1000),
        "modified_frames_pct": float(100 * modified.mean()),
    }


def summarize(bvh, before, after, reference):
    return {
        "frames": len(bvh.table), "fps": 1 / bvh.dt,
        "duration_s": (len(bvh.table) - 1) * bvh.dt,
        "joints": len(bvh.names), "floor_m": reference.floor,
        "support_frames_pct": float(100 * reference.mask.any(axis=(1, 2)).mean()),
        "contact_samples": int(reference.mask.sum()),
        "before": measure(bvh, before, reference),
        "after": measure(bvh, after, reference),
        "fidelity": fidelity(bvh, before, after),
    }
