"""Adapt Newton body frames to Kimodo's fixed G1 rest frames.

The stock qpos inverse is used only to establish the zero-pose convention.
Actual poses come from Newton FK, with a constant per-body rest-frame bridge.
This preserves the real hinge axes, including nontrivial shoulder offsets.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton import G1Skeleton34


def rotation_matrices(quaternions):
    shape = quaternions.shape[:-1]
    return Rotation.from_quat(quaternions.reshape(-1, 4)).as_matrix().reshape(*shape, 3, 3)


class G1RestFrameAdapter:
    def __init__(self, calibration):
        self.skeleton = G1Skeleton34().float()
        self.qpos_converter = MujocoQposConverter(self.skeleton)
        self.matrix = self.qpos_converter.mujoco_to_kimodo_matrix.numpy().astype(float)
        self.joint_names = calibration["joint_names"].tolist()
        self.body_names = calibration["body_names"].tolist()
        xml = ET.parse(self.qpos_converter.xml_path)
        body_to_joint = {"pelvis": "pelvis_skel"}
        self.xml_joint_names = []
        for body in xml.findall(".//worldbody//body"):
            for joint in body.findall("joint"):
                if joint.get("name") and joint.get("type") != "free":
                    body_to_joint[body.get("name")] = joint.get("name").replace("_joint", "_skel")
                    self.xml_joint_names.append(joint.get("name"))
        self.body_indices = [i for i, name in enumerate(self.body_names) if name in body_to_joint]
        self.joint_indices = [self.skeleton.bone_index[body_to_joint[self.body_names[i]]] for i in self.body_indices]
        self.slots = {joint: i for i, joint in enumerate(self.joint_indices)}
        self.hinges = [self.skeleton.bone_index[name.replace("_joint", "_skel")] for name in self.joint_names]

        q0 = calibration["calibration_q"][:1].copy()
        q0[:, 3:7] = q0[:, [6, 3, 4, 5]]
        # The motion utility smooths trajectories; use a stationary clip, not one/two frames.
        reference = self.qpos_converter.qpos_to_motion_dict(
            torch.tensor(np.repeat(q0, 30, axis=0)), 30, mujoco_rest_zero=False
        )
        self.reference_local = reference["local_rot_mats"][0]
        native = rotation_matrices(calibration["calibration_body_q"][:, self.body_indices, 3:7])
        self.bridge = np.swapaxes(self.matrix @ native[0], -1, -2) @ reference["global_rot_mats"][0, self.joint_indices].numpy()

        self.rest_local, self.axes = [], []
        for hinge, j in enumerate(self.hinges):
            child = self.slots[j]
            parent = self.slots[int(self.skeleton.joint_parents[j])]
            rest = native[0, parent].T @ native[0, child]
            probe = native[hinge + 1, parent].T @ native[hinge + 1, child]
            axis = Rotation.from_matrix(rest.T @ probe).as_rotvec() / .1
            self.rest_local.append(rest)
            self.axes.append(axis / np.linalg.norm(axis))

    def from_body_transforms(self, body_q):
        """Return canonical global rotations and FK positions from Newton body transforms."""
        body_q = np.asarray(body_q)
        native = rotation_matrices(body_q[:, self.body_indices, 3:7])
        global_rots = torch.empty(len(body_q), self.skeleton.nbjoints, 3, 3)
        global_rots[:, self.joint_indices] = torch.tensor(self.matrix @ native @ self.bridge, dtype=torch.float32)
        for j in range(self.skeleton.nbjoints):
            if j not in self.slots:
                global_rots[:, j] = global_rots[:, self.skeleton.joint_parents[j]] @ self.reference_local[j]
        local = self.skeleton.global_rots_to_local_rots(global_rots)
        root_body = self.body_indices[self.slots[self.skeleton.root_idx]]
        root = torch.tensor(body_q[:, root_body, :3] @ self.matrix.T, dtype=torch.float32)
        global_rots, positions, _ = self.skeleton.fk(local, root)
        return global_rots, positions

    def to_joint_q(self, global_rots, positions):
        """Recover native xyz/xyzw/hinge-radians, independently checking JSON semantics."""
        native = self.matrix.T @ global_rots[:, self.joint_indices].numpy() @ np.swapaxes(self.bridge, -1, -2)
        q = np.empty((len(native), 7 + len(self.hinges)))
        q[:, :3] = positions[:, self.skeleton.root_idx].numpy() @ self.matrix
        q[:, 3:7] = Rotation.from_matrix(native[:, self.slots[self.skeleton.root_idx]]).as_quat()
        for i, j in enumerate(self.hinges):
            parent = self.slots[int(self.skeleton.joint_parents[j])]
            relative = np.swapaxes(native[:, parent], -1, -2) @ native[:, self.slots[j]]
            delta = self.rest_local[i].T @ relative
            q[:, 7 + i] = Rotation.from_matrix(delta).as_rotvec() @ self.axes[i]
        return q

    def qpos(self, native_q):
        """Pack native coordinates into Kimodo's documented MuJoCo CSV column convention."""
        order = [self.joint_names.index(name) for name in self.xml_joint_names]
        return np.concatenate([native_q[:, :3], native_q[:, [6, 3, 4, 5]], native_q[:, 7:][:, order]], axis=-1)

    def fk_error(self, global_rots, positions, native_body, translation=None):
        expected = native_body[:, self.body_indices, :3] @ self.matrix.T
        if translation is not None:
            expected = expected + translation
        return float(np.abs(positions[:, self.joint_indices].numpy() - expected).max())


def robot_mesh(skeleton, global_rots, positions):
    """Use the same meshes and placement as Kimodo's G1 viewer, without a web server."""
    from kimodo.viz.g1_rig import _load_g1_mesh_data

    items = _load_g1_mesh_data(str(Path(skeleton.folder) / "meshes/g1"), skeleton)
    vertices, triangles, offset = [], [], 0
    for item in items:
        j = item["joint_idx"]
        rotation = global_rots[j].numpy() @ item["geom_rot"]
        position = positions[j].numpy() + global_rots[j].numpy() @ item["geom_pos"]
        v = item["vertices"] @ rotation.T + position
        vertices.append(v)
        triangles.append(item["faces"] + offset)
        offset += len(v)
    return torch.tensor(np.concatenate(vertices), dtype=torch.float32), torch.tensor(np.concatenate(triangles))
