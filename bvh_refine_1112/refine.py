"""Conservative motion edits in human BVH space."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from bvh import LEGS, matrices, transform, transpose, vectors

DEFAULT = {
    "strength": 50,
    "foot": {"enabled": True, "strength": 80},
    "smooth": {"enabled": True, "strength": 80},
    "torso": {"enabled": True, "strength": 40},
    "speed": {"enabled": False, "strength": 40},
    "contact": {"height_m": 0.035, "horizontal_m_s": 0.48, "vertical_m_s": 0.23,
                "minimum_s": 0.10},
    "floor_m": None,
}
MODULES = ("foot", "smooth", "torso", "speed")


def config(path=None, strength=None):
    supplied = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if not isinstance(supplied, dict) or supplied.keys() - DEFAULT.keys():
        raise ValueError("Unknown configuration fields or invalid configuration object")
    resolved = {}
    for key, default in DEFAULT.items():
        value = supplied.get(key, default)
        if isinstance(default, dict):
            if not isinstance(value, dict) or value.keys() - default.keys():
                raise ValueError(f"Invalid fields in {key}")
            value = default | value
        resolved[key] = value
    if strength is not None:
        resolved["strength"] = strength

    def number(value, label):
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError(f"{label} must be a finite number")
        return value

    for key in ("strength", *MODULES):
        if key == "strength":
            value = resolved[key]
        else:
            if type(resolved[key]["enabled"]) is not bool:
                raise ValueError(f"{key}.enabled must be a boolean")
            value = resolved[key]["strength"]
        if not 0 <= number(value, key) <= 100:
            raise ValueError(f"{key} strength must be between 0 and 100")
    if resolved["floor_m"] is not None:
        number(resolved["floor_m"], "floor_m")
    for key, value in resolved["contact"].items():
        if number(value, key) <= 0:
            raise ValueError(f"contact.{key} must be positive")
    return resolved


def norm(v):
    return np.linalg.norm(v, axis=-1, keepdims=True)


def unit(v):
    return v / np.maximum(norm(v), 1e-10)


def limited(v, cap):
    return v * np.minimum(1, cap / np.maximum(norm(v), 1e-10))


def runs(mask):
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(boundaries[::2], boundaries[1::2])


def foot_geometry(bvh, positions, world):
    tracks, landmarks = [], []
    for names in LEGS:
        _, _, ankle, toe = (bvh.names.index(name) for name in names)
        front = bvh.offsets[toe]
        rear = front * [0, 1, -0.36]
        offsets = np.array([rear, front])
        tracks.append(positions[:, ankle, None] + transform(world[:, ankle, None], offsets))
        landmarks.append(offsets)
    return np.stack(tracks, axis=1), np.array(landmarks)


@dataclass
class Contact:
    floor: float
    mask: np.ndarray
    tracks: np.ndarray
    offsets: np.ndarray


def detect(bvh, positions, world, settings):
    tracks, offsets = foot_geometry(bvh, positions, world)
    smooth = gaussian_filter1d(tracks, max(0.8, 0.032 / bvh.dt), axis=0, mode="nearest")
    lowest = smooth[..., 1].min(axis=(1, 2))
    floor = settings["floor_m"]
    if floor is None:
        # An isolated low outlier cannot set the whole floor.
        floor = float(np.quantile(lowest, 0.10))
    velocity = np.gradient(smooth, bvh.dt, axis=0) if len(smooth) > 1 else np.zeros_like(smooth)
    c = settings["contact"]
    horizontal = norm(velocity[..., [0, 2]])[..., 0]
    mask = ((smooth[..., 1] < floor + c["height_m"]) &
            (horizontal < c["horizontal_m_s"]) &
            (np.abs(velocity[..., 1]) < c["vertical_m_s"]))
    for side, point in np.ndindex(2, 2):
        line = mask[:, side, point]
        for start, end in runs(~line):
            if start > 0 and end < len(line) and (end - start) * bvh.dt <= 0.035:
                line[start:end] = smooth[start:end, side, point, 1] < floor + c["height_m"]
        for start, end in runs(line):
            if (end - start) * bvh.dt < c["minimum_s"]:
                line[start:end] = False
    return Contact(floor, mask, tracks, offsets)


def smooth_motion(root, local, dt, amount):
    if len(root) < 3:
        return
    lag = max(1, round(0.033 / dt))
    step = vectors(transpose(local[:-1]) @ local[1:])
    # Corrections live in each central frame's local tangent space.
    indices = np.arange(1, len(root) - 1)
    lo, hi = np.maximum(0, indices - lag), np.minimum(len(root) - 1, indices + lag)
    center = transpose(local[indices])
    immediate = (step[indices - 1] - step[indices]) * -0.5
    wide = (vectors(center @ local[lo]) + vectors(center @ local[hi])) * 0.5
    reversal = np.sum(step[indices - 1] * step[indices], axis=-1) < 0
    residual = np.where(reversal[..., None], 0.7 * immediate + 0.3 * wide, immediate)
    score = norm(immediate)
    gate = np.clip((score - np.deg2rad(0.10)) / np.deg2rad(0.8), 0, 1)
    gate *= np.where(reversal[..., None], 1.0, 0.16)
    gain = (4 * amount) / (1 + 4 * amount)
    local[indices] = local[indices] @ matrices(gain * gate * limited(residual, np.deg2rad(1 + 7 * amount)))

    left = root[1:-1] - root[:-2]
    right = root[2:] - root[1:-1]
    residual = (right - left) / 2
    gate = np.clip((norm(residual) - 0.0005) / 0.004, 0, 1)
    gate *= np.where(np.sum(left * right, axis=-1, keepdims=True) < 0, 1.0, 0.12)
    root[1:-1] += gain * gate * limited(residual, 0.002 + 0.015 * amount)


def limit_speed(local, dt, amount):
    if len(local) < 3:
        return
    before = local.copy()
    center = transpose(before[1:-1])
    left = vectors(center @ before[:-2])
    right = vectors(center @ before[2:])
    # Both adjacent steps must be unusually large and reverse at the spike.
    cutoff = np.deg2rad(1700 - 850 * amount) * dt
    over = np.clip((np.minimum(norm(left), norm(right)) - cutoff) / cutoff, 0, 1)
    over *= np.sum(left * right, axis=-1, keepdims=True) > 0
    gain = (3 * amount) / (1 + 3 * amount)
    correction = gain * over * limited((left + right) / 2, np.deg2rad(12 + 30 * amount))
    local[1:-1] = before[1:-1] @ matrices(correction)


def torso_motion(bvh, root, local, contact, amount):
    if len(root) < 5:
        return
    _, world = bvh.fk(root, local)
    chest = bvh.names.index("Spine1")
    up = world[:, chest, :, 1]
    radius = max(2, round(0.15 / bvh.dt))
    baseline = unit(median_filter(up, size=(2 * radius + 1, 1), mode="nearest"))
    tangent = between(up, baseline)
    tilt = np.arccos(np.clip(np.minimum(up[:, 1:2], baseline[:, 1:2]), -1, 1))
    gate = np.clip((norm(tangent) - np.deg2rad(9 - 4 * amount)) / np.deg2rad(12), 0, 1)
    gate *= (tilt > np.deg2rad(22)) & contact.mask.any(axis=(1, 2))[:, None]
    correction = gaussian_filter1d(limited(tangent, np.deg2rad(6 + 27 * amount)) * gate,
                                   max(0.7, 0.018 / bvh.dt), axis=0, mode="nearest")
    parent = bvh.parents[chest]
    gain = (4 * amount) / (1 + 4 * amount)
    local[:, chest] = matrices(transform(transpose(world[:, parent]), correction) * gain) @ local[:, chest]


def between(a, b):
    a, b = unit(a), unit(b)
    cross = np.cross(a, b)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    antiparallel = (dot < 0) & (norm(cross) < 1e-8)
    basis = np.eye(3)[np.argmin(np.abs(a), axis=-1)]
    angle = np.arctan2(norm(cross), dot)
    return np.where(antiparallel, unit(np.cross(a, basis)), unit(cross)) * angle


def solve_leg(bvh, local, positions, world, side, active, target, foot_world, max_bend_delta):
    hip, knee, ankle, _ = (bvh.names.index(name) for name in LEGS[side])
    parent = bvh.parents[hip]
    h, k, a = (positions[active, joint] for joint in (hip, knee, ankle))
    hip_world, knee_world = (world[active, joint] for joint in (hip, knee))
    upper, lower = np.linalg.norm(bvh.offsets[[knee, ankle]], axis=1)
    axis = unit(target - h)
    old_distance = norm(a - h)
    # Limit the actual change in knee flexion. Millimetres of radius error can
    # otherwise create several degrees of knee motion on a nearly straight leg.
    flex = np.arccos(np.clip((old_distance**2 - upper**2 - lower**2) / (2 * upper * lower), -1, 1))
    near = np.clip(flex + max_bend_delta, 0, np.pi - 0.01)
    far = np.maximum(0, flex - max_bend_delta)
    min_radius = np.sqrt(upper**2 + lower**2 + 2 * upper * lower * np.cos(near))
    max_radius = np.sqrt(upper**2 + lower**2 + 2 * upper * lower * np.cos(far))
    d = np.clip(norm(target - h), min_radius, max_radius)
    along = (upper**2 - lower**2 + d**2) / (2 * d)
    height = np.sqrt(np.maximum(upper**2 - along**2, 0))
    old_axis = unit(a - h)
    pole = (k - h) - np.sum((k - h) * old_axis, axis=-1, keepdims=True) * old_axis
    fallback = transform(hip_world, np.array([0., 0., 1.]))
    pole = np.where(norm(pole) < 1e-6, fallback, pole)
    pole = unit(pole - np.sum(pole * axis, axis=-1, keepdims=True) * axis)
    good = (norm(pole)[:, 0] > 0.5) & (norm(axis)[:, 0] > 0.5)
    new_knee = h + along * axis + height * pole
    new_ankle = h + d * axis
    hip_turn = between(k - h, new_knee - h)
    new_hip_world = matrices(hip_turn) @ hip_world
    new_knee_world = matrices(between(a - k, new_ankle - new_knee)) @ knee_world
    idx = np.flatnonzero(active)[good]
    local[idx, hip] = transpose(world[idx, parent]) @ new_hip_world[good]
    local[idx, knee] = transpose(new_hip_world[good]) @ new_knee_world[good]
    local[idx, ankle] = transpose(new_knee_world[good]) @ foot_world[good]


def foot_motion(bvh, root, local, reference, amount):
    positions, world = bvh.fk(root, local)
    tracks, _ = foot_geometry(bvh, positions, world)
    for side in range(2):
        ankle = bvh.names.index(LEGS[side][2])
        weights = np.zeros((len(root), 2))
        goal = tracks[:, side].copy()
        lowest = tracks[:, side, :, 1].min(axis=1)
        rest_bias = np.clip(np.quantile(reference.tracks[:, side, :, 1].min(axis=1), 0.15)
                            - reference.floor, 0, 0.025)
        hover_ceiling = reference.floor + rest_bias + 0.022
        for point in range(2):
            for start, end in runs(reference.mask[:, side, point]):
                segment = reference.tracks[start:end, side, point]
                # A moving median avoids locking many seconds of purposeful pivots
                # to one point. The reference never changes during refinement.
                xz = segment[:, [0, 2]]
                spread = np.quantile(xz, 0.9, axis=0) - np.quantile(xz, 0.1, axis=0)
                still = norm(spread)[0] < 0.028
                window = 2 * max(1, round((0.70 if still else 0.24) / bvh.dt)) + 1
                anchor = median_filter(segment, size=(window, 1), mode="reflect")
                excursion = np.ptp(xz, axis=0)
                if (end - start) * bvh.dt <= 1.1 and norm(excursion)[0] < 0.17:
                    anchor = np.broadcast_to(np.median(segment, axis=0), segment.shape)
                ramp = np.minimum(np.arange(1, end - start + 1), np.arange(end - start, 0, -1))
                ramp = np.clip((ramp * bvh.dt) / 0.09, 0, 1)
                weights[start:end, point] = ramp**2 * (3 - 2 * ramp)
                goal[start:end, point, 0] = anchor[:, 0]
                goal[start:end, point, 2] = anchor[:, 2]
                # Correct hovering only when the whole foot is lifted past its
                # own rest bias. A raised heel alone is normal during a toe pivot.
                y = goal[start:end, point, 1]
                above = np.maximum(0, lowest[start:end] - hover_ceiling)
                goal[start:end, point, 1] = np.maximum(y - above, reference.floor - 0.004)
        # Always resist clear inferred penetration. This term has no horizontal component.
        low = tracks[:, side, :, 1] < reference.floor - 0.009
        missed = low & (weights == 0)
        goal[..., 1][missed] = reference.floor - 0.006
        depth = reference.floor - tracks[:, side, :, 1]
        weights[missed] = np.clip((depth[missed] - 0.009) / 0.024, 0, 0.75)

        both = np.minimum(weights[:, 0], weights[:, 1])[:, None]
        w = 9 * amount * weights[..., None]
        # Correct heading with two supported landmarks while retaining the
        # source pitch and roll of the foot.
        turn = between((tracks[:, side, 1] - tracks[:, side, 0]) * [1, 0, 1],
                       (goal[:, 1] - goal[:, 0]) * [1, 0, 1])
        turn = gaussian_filter1d(limited(turn, np.deg2rad(3 + 9 * amount)) * both * amount,
                                 max(0.7, 0.026 / bvh.dt), axis=0, mode="nearest")
        turn[weights.max(axis=1) == 0] = 0
        foot_world = matrices(turn) @ world[:, ankle]
        predicted = positions[:, ankle, None] + transform(foot_world[:, None], reference.offsets[side])
        error = goal - predicted
        delta = limited(np.sum(w * error, axis=1) / (1 + np.sum(w, axis=1)), 0.015 + 0.075 * amount)
        # Spread the onset and offset across neighbouring supported frames only.
        delta = gaussian_filter1d(delta, max(0.6, 0.015 / bvh.dt), axis=0, mode="nearest")
        active = (weights.max(axis=1) > 0) & ((norm(delta)[:, 0] > 1e-5) | (norm(turn)[:, 0] > 1e-5))
        if active.any():
            target = positions[active, ankle] + delta[active]
            solve_leg(bvh, local, positions, world, side, active, target, foot_world[active],
                      np.deg2rad(1 + 5 * amount))


def refine(bvh, settings):
    root, local = bvh.decode()
    contact = detect(bvh, *bvh.fk(root, local), settings)
    strengths = {key: settings["strength"] * settings[key]["strength"] / 10000
                 if settings[key]["enabled"] else 0 for key in MODULES}
    if strengths["smooth"]:
        smooth_motion(root, local, bvh.dt, strengths["smooth"])
    if strengths["speed"]:
        limit_speed(local, bvh.dt, strengths["speed"])
    if strengths["torso"]:
        torso_motion(bvh, root, local, contact, strengths["torso"])
    if strengths["foot"]:
        foot_motion(bvh, root, local, contact, strengths["foot"])
    return root, local, contact
