"""CPU-only validation, absolute position metrics, and target/generated overlays."""
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GENERATED = (50, 220, 235)
TARGET = (255, 204, 51)
BG = (15, 23, 38)


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed 必须是 0 到 4294967295 的整数。")
    return seed


def max_component_step_m(joints):
    return float(np.abs(np.diff(np.asarray(joints), axis=0)).max())


def validate_constraints(data, frames, fps):
    if not isinstance(data, list) or not data:
        raise ValueError("constraints 必须是非空 JSON 数组。")
    if frames < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("无效帧数或帧率。")
    indices = []
    for item in data:
        if not isinstance(item, dict) or item.get("type") != "fullbody":
            raise ValueError("这个入口仅接受 SOMA30/77 fullbody 关键帧。")
        ids = item.get("frame_indices", [])
        if not isinstance(ids, list) or not ids:
            raise ValueError("每组约束必须提供 frame_indices。")
        if any(type(i) is not int or not 0 <= i < frames for i in ids):
            raise ValueError(f"frame_indices 必须为 0 到 {frames-1} 的整数。")
        rotations = np.asarray(item.get("local_joints_rot"), dtype=float)
        if rotations.shape not in ((len(ids), 30, 3), (len(ids), 77, 3)) or not np.isfinite(rotations).all():
            raise ValueError("local_joints_rot 应为 [关键帧数, 30或77, 3] 的有限数值数组。")
        for field, shape in (("root_positions", (len(ids), 3)),):
            a = np.asarray(item.get(field), dtype=float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"{field} 应为有限数值数组，形状 {shape}。")
        if "smooth_root_2d" in item:
            a = np.asarray(item["smooth_root_2d"], dtype=float)
            if a.shape != (len(ids), 2) or not np.isfinite(a).all():
                raise ValueError("smooth_root_2d 应为 [关键帧数, 2] 的有限数值数组。")
        indices.extend(ids)
    if len(set(indices)) != len(indices):
        raise ValueError("同一帧存在重复 fullbody 约束，请先消除冲突。")
    return sorted(indices)


def position_errors(generated, target, indices, fps):
    generated, target = np.asarray(generated), np.asarray(target)
    if generated.shape != target.shape or generated.ndim != 3 or generated.shape[-1] != 3:
        raise ValueError("生成姿态与目标姿态必须同为 [关键帧数, 关节数, 3]。")
    if len(indices) != len(target) or not len(target) or not np.isfinite(generated).all() or not np.isfinite(target).all():
        raise ValueError("关键帧数量或数值无效。")
    errors = np.linalg.norm(generated - target, axis=-1) * 1000
    return {
        "metric": "absolute world-coordinate Euclidean joint position error",
        "units": "millimeters", "alignment_applied": False,
        "mean_joint_error_mm": float(errors.mean()),
        "max_joint_error_mm": float(errors.max()),
        "keyframes": [{"frame_index": int(frame), "time_seconds": float(frame/fps),
                       "mean_joint_error_mm": float(e.mean()), "max_joint_error_mm": float(e.max()),
                       "root_error_mm": float(e[0]), "per_joint_error_mm": e.tolist()}
                      for frame, e in zip(indices, errors)],
    }


def align_horizontal_targets(generated, target, indices, free_frames):
    """Place pose-only targets at generated root X/Z, retaining target height and orientation."""
    generated, target = np.asarray(generated), np.asarray(target).copy()
    if generated.shape != target.shape or len(indices) != len(target):
        raise ValueError('Target comparison shapes must agree')
    free_frames = set(free_frames)
    for k,frame in enumerate(indices):
        if int(frame) in free_frames:
            shift = generated[k,0]-target[k,0]
            shift[1] = 0
            target[k] += shift
    return target


def _project(points):
    p = np.asarray(points)
    return np.stack((.8*p[..., 0]-.6*p[..., 2],
                     -.22*p[..., 0]+.93*p[..., 1]-.30*p[..., 2]), axis=-1)


def overlay_frame(generated, target, parents, title, bounds):
    im = Image.new("RGB", (768, 480), BG)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=17)
    except TypeError:
        font = ImageFont.load_default()
    d.text((22, 14), title, fill=(235, 240, 250), font=font)
    d.text((22, 43), "CYAN: generated", fill=GENERATED, font=font)
    d.text((270, 43), "YELLOW: target keyframe", fill=TARGET, font=font)
    lo_x, hi_x, lo_y, hi_y = bounds
    scale = min(650/max(hi_x-lo_x, .5), 340/max(hi_y-lo_y, .5))

    def xy(p):
        projected = _project(p)
        return np.stack((384+(projected[..., 0]-(lo_x+hi_x)/2)*scale,
                         275-(projected[..., 1]-(lo_y+hi_y)/2)*scale), axis=-1)

    for points, color, width in ((target, TARGET, 6), (generated, GENERATED, 3)):
        p = xy(points)
        for i, parent in enumerate(parents):
            if parent >= 0:
                d.line([tuple(p[parent]), tuple(p[i])], fill=color, width=width)
        for x, y in p:
            d.ellipse((x-2, y-2, x+2, y+2), fill=color)
    return im


def render_overlay(stem, points, targets, indices, parents, fps):
    stem = Path(stem)
    targets = np.asarray(targets)
    cloud = _project(np.concatenate((points.reshape(-1, 3), targets.reshape(-1, 3))))
    lo, hi = cloud.min(0), cloud.max(0)
    bounds = (lo[0]-.15, hi[0]+.15, lo[1]-.15, hi[1]+.15)
    paths = {name: Path(str(stem)+suffix) for name, suffix in
             (("gif", "_overlay.gif"), ("mp4", "_overlay.mp4"), ("keyframes", "_keyframes.png"))}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("预览已存在，不能覆盖。")
    gif_frames, duration_ms, contact_frames = [], [], []
    indices = np.asarray(indices, dtype=int)
    gif_indices = sorted(set(range(0, len(points), 2)) | set(indices.tolist()))
    gif_next = {frame: gif_indices[i+1] if i+1 < len(gif_indices) else len(points)
                for i, frame in enumerate(gif_indices)}
    proc = None
    if shutil.which("ffmpeg"):
        proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "768x480", "-r", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(paths["mp4"])], stdin=subprocess.PIPE)
    else:
        print("未找到 ffmpeg：输出 GIF 和关键帧对比图，不输出 MP4。", flush=True)
    try:
        for frame in range(len(points)):
            nearest = int(np.argmin(abs(indices-frame)))
            is_keyframe = frame == indices[nearest]
            title = (f"t={frame/fps:.2f}s | target @{indices[nearest]/fps:.2f}s | " +
                     ("CONSTRAINED FRAME" if is_keyframe else "target shown for reference"))
            im = overlay_frame(points[frame], targets[nearest], parents, title, bounds)
            if proc:
                proc.stdin.write(im.tobytes())
            if frame in gif_next:
                gif_frames.append(im)
                start = round(frame/fps*100)*10
                end = round(gif_next[frame]/fps*100)*10
                duration_ms.append(max(10, end-start))
            if is_keyframe:
                contact_frames.append(im)
    finally:
        if proc:
            proc.stdin.close()
            if proc.wait(timeout=30):
                raise RuntimeError("MP4 编码失败；已有动作文件保留。")
    gif_frames[0].save(paths["gif"], save_all=True, append_images=gif_frames[1:],
                       duration=duration_ms, loop=0, optimize=False)
    sheet = Image.new("RGB", (768, 480*len(contact_frames)), BG)
    for i, image in enumerate(contact_frames):
        sheet.paste(image, (0, 480*i))
    sheet.save(paths["keyframes"])
    with Image.open(paths["gif"]) as saved:
        duration = 0
        for i in range(saved.n_frames):
            saved.seek(i)
            duration += saved.info.get("duration", 0)
        if abs(duration-len(points)/fps*1000) > 20:
            raise RuntimeError("GIF 时长检查失败。")
    if proc and shutil.which("ffprobe"):
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json",
            str(paths["mp4"])], text=True))
        if int(probe["streams"][0]["nb_read_frames"]) != len(points):
            raise RuntimeError("MP4 帧数检查失败。")
    return {k: str(v) for k, v in paths.items() if v.exists()}


def evaluate_and_render(stem):
    stem = Path(stem)
    with np.load(str(stem)+".npz", allow_pickle=False) as a:
        points = a["posed_joints"].copy()
    with np.load(str(stem)+"_targets.npz", allow_pickle=False) as a:
        indices, targets = a["frame_indices"].copy(), a["positions77"].copy()
        target30, generated30 = a["positions30"].copy(), a["generated_positions30"].copy()
    meta = json.loads(Path(str(stem)+".json").read_text())
    fps = float(meta["fps"])
    free_frames = meta.get('horizontal_free_keyframes', [])
    targets = align_horizontal_targets(points[indices],targets,indices,free_frames)
    target30 = align_horizontal_targets(generated30,target30,indices,free_frames)
    report = {"status": "measured", "acceptance_threshold": None,
        "note": "Postprocessed output; no alignment. SOMA30 is model constraint space; SOMA77 includes expanded finger joints.",
        "input_constraints": meta["constraints_path"], "fps": fps,
        "post_processing": meta["post_processing"],
        "soma30": position_errors(generated30, target30, indices, fps),
        "soma77": position_errors(points[indices], targets, indices, fps),
        "joint_names77": meta["joint_names"], "joint_names30": meta["constraint_joint_names30"]}
    if free_frames:
        report['note'] = 'Horizontal-free pose targets: root X/Z aligned at each marked keyframe; target height and rotations unchanged.'
        report['horizontal_free_keyframes'] = free_frames
        for label in ('soma30','soma77'):
            report[label]['alignment_applied'] = True
            report[label]['alignment'] = 'root X/Z translation only at horizontal_free_keyframes; no Y or rotation alignment'
    report_path = Path(str(stem)+"_errors.json")
    with report_path.open("x") as f:
        json.dump(report, f, indent=2)
    csv_path = Path(str(stem)+"_errors.csv")
    with csv_path.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["skeleton", "frame", "time_seconds", "mean_error_mm", "max_error_mm", "root_error_mm"])
        for skeleton in ("soma30", "soma77"):
            for row in report[skeleton]["keyframes"]:
                writer.writerow([skeleton, row["frame_index"], row["time_seconds"], row["mean_joint_error_mm"],
                                 row["max_joint_error_mm"], row["root_error_mm"]])
    previews = render_overlay(stem, points, targets, indices, meta["joint_parents"], fps)
    for row in report["soma30"]["keyframes"]:
        print(f"关键帧 {row['frame_index']} @ {row['time_seconds']:.3f}s: "
              f"平均误差 {row['mean_joint_error_mm']:.3f} mm，最大误差 {row['max_joint_error_mm']:.3f} mm", flush=True)
    print(json.dumps({"report": str(report_path), "csv": str(csv_path), **previews}, indent=2), flush=True)
