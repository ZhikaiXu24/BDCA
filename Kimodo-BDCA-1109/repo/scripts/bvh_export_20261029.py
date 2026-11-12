"""Export generated SOMA arrays directly with Kimodo's official CPU BVH writer."""
import importlib
import io
import math
from pathlib import Path
import re

import numpy as np


def check_bvh_dependencies():
    try:
        for name in ('bvhio', 'glm', 'SpatialTransform'):
            importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError('BVH 导出依赖不可用；尚未启动模型推理。请检查 xzk_kimodo 环境。') from error


def motion_arrays(motion):
    """Select only the two numeric fields needed by BVH; preserve NPZ metadata."""
    rotations = np.asarray(motion['local_rot_mats'])
    positions = np.asarray(motion['root_positions'])
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3) or rotations.shape[1] not in (30, 77):
        raise ValueError('BVH 导出需要 SOMA30/77 的 [T,J,3,3] 局部旋转。')
    if rotations.shape[0] < 1 or positions.shape != (rotations.shape[0], 3):
        raise ValueError('BVH 根位置必须为 [T,3]，且与旋转帧数一致。')
    if rotations.dtype.kind not in 'fiu' or positions.dtype.kind not in 'fiu':
        raise ValueError('BVH 旋转和根位置必须是实数数组。')
    if not np.isfinite(rotations).all() or not np.isfinite(positions).all():
        raise ValueError('BVH 输入含非有限数值。')
    return np.asarray(rotations, dtype=np.float32).copy(), np.asarray(positions, dtype=np.float32).copy()


def validate_bvh_text(text, expected_frames, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('BVH 帧率必须为正数。')
    if 'HIERARCHY' not in text or '\nMOTION' not in text:
        raise ValueError('BVH 缺少骨架层级或动作段。')
    header, motion = text.split('\nMOTION', 1)
    frames_match = re.search(r'^Frames:\s*(\d+)\s*$', motion, re.MULTILINE)
    time_match = re.search(r'^Frame Time:\s*([0-9.eE+-]+)\s*$', motion, re.MULTILINE)
    channels = sum(map(int, re.findall(r'\bCHANNELS\s+(\d+)\b', header)))
    if not frames_match or not time_match or not channels:
        raise ValueError('BVH 头部缺少帧数、帧时间或通道。')
    frames, frame_time = int(frames_match[1]), float(time_match[1])
    if frames != expected_frames or not np.isclose(frame_time, 1/fps, rtol=1e-5, atol=1e-7):
        raise ValueError('BVH 帧数或帧率与生成结果不一致。')
    rows = motion[time_match.end():].strip()
    if not rows:
        raise ValueError('BVH 动作数据为空。')
    values = np.loadtxt(io.StringIO(rows), ndmin=2)
    if values.shape != (frames, channels) or not np.isfinite(values).all():
        raise ValueError('BVH 动作行数、通道数或数值无效。')
    return {'frames': frames, 'frame_time': frame_time, 'channels': channels}


def export_soma_bvh(motion, path, fps):
    """Use an independent CPU skeleton and exclusive output; never reread NPZ."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('BVH 帧率必须为正数。')
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'BVH 已存在，未覆盖：{path}')
    rotations, positions = motion_arrays(motion)
    import torch
    from kimodo.exports.bvh import motion_to_bvh
    from kimodo.skeleton.registry import build_skeleton
    skeleton = build_skeleton(rotations.shape[1])
    text = motion_to_bvh(torch.from_numpy(rotations), torch.from_numpy(positions),
                         skeleton=skeleton, fps=fps, standard_tpose=False)
    validate_bvh_text(text, len(rotations), fps)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(text)
    validate_bvh_text(path.read_text(encoding='utf-8'), len(rotations), fps)
    return str(path)



def export_g1_csv(input_path, output_path, fps):
    """Export final qpos CSV, independently checking BVH FK against robot XML."""
    import json
    import torch
    from g1_keyframes_20261030 import RobotXML
    from kimodo.exports.bvh import motion_to_bvh
    from kimodo.skeleton.registry import build_skeleton
    from bvh_check import read_bvh
    output_path = Path(output_path)
    if output_path.exists(): raise FileExistsError(output_path)
    robot = RobotXML()
    q = np.loadtxt(input_path, delimiter=',', ndmin=2)
    local, roots = robot.local_from_q(q)
    skeleton = build_skeleton(34).to('cpu')
    # Build rest offsets from the exact CSV viewer XML, including its endpoint
    # definitions; do not substitute training-skeleton endpoint offsets.
    from types import SimpleNamespace
    names = robot.model_names()
    parent_indices = [(-1 if robot.parents[robot.names.index(n)] < 0 else names.index(robot.names[robot.parents[robot.names.index(n)]])) for n in names]
    zero = np.zeros((1,36)); zero[:,3] = 1
    rest_local, _ = robot.local_from_q(zero)
    rest_points, _ = robot.fk(zero)
    rest_points = rest_points[0,[robot.names.index(n) for n in names]]
    global_rest = rest_local[0].copy()
    neutral = np.zeros((34,3))
    for i,parent in enumerate(parent_indices):
        if parent < 0: continue
        offset = global_rest[parent].T @ (rest_points[i]-rest_points[parent])
        neutral[i] = neutral[parent] + offset
        global_rest[i] = global_rest[parent] @ global_rest[i]
    skeleton = SimpleNamespace(name='g1_xml34',neutral_joints=torch.tensor(neutral,dtype=torch.float32),
                               bone_order_names=names,joint_parents=torch.tensor(parent_indices),root_idx=0)
    text = motion_to_bvh(torch.from_numpy(local).float(), torch.from_numpy(roots).float(), skeleton=skeleton, fps=fps, standard_tpose=True)
    validate_bvh_text(text, len(q), fps)
    decoded = read_bvh(text)
    points, _ = robot.fk(q)
    names = robot.model_names()
    expected = points[:,[robot.names.index(n) for n in names]]
    actual = decoded['positions'][:,[decoded['names'].index(n) for n in names]]
    error = np.linalg.norm(actual-expected, axis=-1)*1000
    if not np.isfinite(error).all() or error.max()>0.1:
        raise ValueError(f'BVH/CSV FK mismatch: {error.max()} mm')
    with output_path.open('x') as f: f.write(text)
    report = dict(source_csv=str(input_path),bvh=str(output_path),frames=len(q),fps=fps,units='centimetres; Y-up',max_fk_error_mm=float(error.max()),mean_fk_error_mm=float(error.mean()))
    with output_path.with_suffix('.bvh_validation.json').open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2),flush=True)
    return str(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-export an existing SOMA NPZ without model inference.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--g1-csv", action="store_true")
    arguments = parser.parse_args()
    if arguments.g1_csv:
        print(export_g1_csv(arguments.input, arguments.output, arguments.fps))
    else:
        with np.load(arguments.input, allow_pickle=False) as motion:
            print(export_soma_bvh(motion, arguments.output, arguments.fps))
