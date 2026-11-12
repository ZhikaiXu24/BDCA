"""Headless side-by-side H.264 preview with a shared moving camera."""
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from quality import body_joints


def render(path, bvh, before, after, reference, start=0.0, duration=None):
    if start < 0 or start >= len(bvh.table) * bvh.dt or (duration is not None and duration <= 0):
        raise ValueError("Video start and duration must select a non-empty interval")
    if path.exists():
        raise FileExistsError(path)
    source = bvh.fk(*before)[0]
    refined = bvh.fk(*after)[0]
    fps = 30
    end = min(len(source) * bvh.dt, start + duration if duration is not None else np.inf)
    indices = np.minimum(np.rint(np.arange(start, end, 1 / fps) / bvh.dt).astype(int), len(source) - 1)
    size = (960, 540)
    font = ImageFont.load_default(size=17)
    headline = ImageFont.load_default(size=23)
    small = ImageFont.load_default(size=13)
    edges = [(bvh.parents[j], j) for j in body_joints(bvh) if j != 0]
    rotation = np.array([[0.866, 0, 0.5], [-0.157, 0.951, 0.268]])
    top = max(np.quantile(source[:, body_joints(bvh), 1], 0.997) - reference.floor, 1.4)
    scale = 316 / top
    label = ("Before", "Refined")
    colors = ("#75caff", "#73e2b8")

    with subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", "960x540", "-r", str(fps), "-i", "-", "-an",
         "-c:v", "libx264", "-threads", "2", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE,
    ) as encoder:
        for frame in indices:
            im = Image.new("RGB", size, "#0c1219")
            draw = ImageDraw.Draw(im)
            draw.text((24, 16), "G1 MOVES  /  HUMAN BVH", fill="#b9c9d8", font=font)
            draw.text((936, 16), f"{frame*bvh.dt:06.2f} s", fill="#b9c9d8", font=font, anchor="ra")
            origin = source[frame, 0].copy()
            origin[1] = reference.floor
            for side, values in enumerate((source, refined)):
                x = 20 + side * 468
                draw.rounded_rectangle((x, 48, x + 452, 472), 12, fill="#131f2b", outline="#344454")
                draw.text((x + 16, 61), label[side], fill=colors[side], font=headline)

                def project(points):
                    return points @ rotation.T * [scale, -scale] + [x + 226, 397]

                for endpoints in (
                    *([[n, 0, -0.55], [n, 0, 0.55]] for n in np.linspace(-0.68, 0.68, 7)),
                    *([[-0.68, 0, n], [0.68, 0, n]] for n in np.linspace(-0.55, 0.55, 5)),
                ):
                    coords = project(np.array(endpoints))
                    draw.line(list(map(tuple, coords)), fill="#294052", width=1)
                joint_pixels = project(values[frame] - origin)
                for a, b in edges:
                    draw.line([tuple(joint_pixels[a]), tuple(joint_pixels[b])], fill=colors[side], width=4)
                for j in body_joints(bvh):
                    cx, cy = joint_pixels[j]
                    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill="#eff7ff")
                for leg, name in enumerate(("L", "R")):
                    state = reference.mask[frame, leg]
                    status = "/".join(n for n, on in zip(("heel", "toe"), state) if on) or "swing"
                    draw.text((x + 16 + leg * 150, 442), f"{name}  {status}",
                              fill="#f4c775" if state.any() else "#a0b0c1", font=small)
            progress = 24 + 912 * (frame + 1) / len(source)
            draw.line((24, 492, 936, 492), fill="#344454", width=4)
            draw.line((24, 492, progress, 492), fill="#73e2b8", width=4)
            draw.text((24, 508), "Shared camera and floor  |  contact labels from the original BVH",
                      fill="#a0b0c1", font=small)
            encoder.stdin.write(im.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("FFmpeg failed to encode comparison video")
