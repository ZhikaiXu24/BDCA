"""GEM Body Pose's SOMA-0.2 reference -> Kimodo's standard SOMA77 frame.

GEM parameters are local rotations relative to historical joint orientations.
Kimodo FK expects rotations relative to its fixed, standard T-pose. In world
space the conversion is C @ G_parameter @ R_reference @ R_kimodo_offset.T.
Only C changes the camera basis; shape/scale do not belong in a constraint.
"""

from pathlib import Path

import numpy as np
import torch

from kimodo.constraints import FullBodyConstraintSet
from kimodo.skeleton import SOMASkeleton77
from kimodo.viz.soma_skin import SOMASkin


def read_body_pose(path: str | Path, instance_index: int = 0) -> dict:
    """Read one person from a trusted, locally generated Body Pose result."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["format"], saved["task"], saved["mode"]) != (
        "gem_pose_predictions_v1", "body", "image"
    ):
        raise ValueError("Expected a GEM Body Pose single-image prediction")
    record = saved["frames"][0]["instances"][instance_index]
    record["faces"] = saved["topology"][record["topology"]]
    return record


class KimodoSomaConverter:
    """Reuse the fixed rig, skin and reference across multiple image results."""

    def __init__(self, soma_assets: str | Path, yaw_degrees: float = 0):
        self.skeleton = SOMASkeleton77().float()
        self.skin = SOMASkin(self.skeleton)
        with np.load(Path(soma_assets) / "SOMA_neutral.npz", allow_pickle=False) as data:
            key = "t_pose_world__v0.2.0"
            names = data[key + "__joint_names"].tolist()
            parents = data[key + "__parent_ids"].tolist()
            reference = torch.from_numpy(data[key].copy()).float()
        # The NPZ has a virtual Root; GEM parameters and Kimodo both omit it.
        self.source_order = torch.tensor([names.index(n) - 1 for n in self.skeleton.bone_order_names])
        self.reference = reference[self.source_order + 1]
        for name, parent in self.skeleton.bone_order_names_with_parents:
            expected_parent = parent or "Root"
            if names[parents[names.index(name)]] != expected_parent:
                raise ValueError(f"SOMA hierarchy differs at {name}")

        angle = np.deg2rad(yaw_degrees)
        yaw = torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ], dtype=torch.float32)
        # OpenCV: right/down/forward -> world: right/up/towards camera.
        self.camera_to_world = yaw @ torch.diag(torch.tensor([1., -1., -1.]))

    def standard_rotations(self, source_local: torch.Tensor) -> torch.Tensor:
        """Return global rotations in Kimodo's standard T-pose frame."""
        raw = source_local[:, self.source_order].float()
        parameter_global, _, _ = self.skeleton.fk(raw, raw.new_zeros(len(raw), 3))
        return (
            self.camera_to_world @ parameter_global @ self.reference
            @ self.skeleton.global_rot_offsets.mT
        )

    @torch.inference_mode()
    def convert(self, record: dict, frame_index: int = 0, root_height: float | None = None):
        parameters = record["parameters"]
        source_local = torch.cat((parameters["global_orient"], parameters["body_pose"]))[None]
        standard_global = self.standard_rotations(source_local)
        local = self.skeleton.global_rots_to_local_rots(standard_global)
        root = local.new_zeros(1, 3)
        global_rots, joints, _ = self.skeleton.fk(local, root)

        if root_height is None:
            vertices = self.skin.skin(global_rots, joints, rot_is_global=True)
            root[0, 1] = -vertices[..., 1].min()
        else:
            root[0, 1] = root_height
        joints = joints + root[:, None]
        return FullBodyConstraintSet(
            self.skeleton, torch.tensor([frame_index]), joints, global_rots
        )
