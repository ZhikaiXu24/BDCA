"""BVH IO and rigid-skeleton FK for the G1 Moves capture profile."""
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
from scipy.spatial.transform import Rotation


LEGS = tuple(tuple(side + part for part in ("UpLeg", "Leg", "Foot", "ToeBase"))
             for side in ("Left", "Right"))
MOTION = re.compile(rb"(?m)^[ \t]*MOTION[ \t]*\r?\n[ \t]*Frames:[ \t]*(\d+)[ \t]*\r?\n"
                    rb"[ \t]*Frame Time:[ \t]*([\d.eE+-]+)[ \t]*\r?\n")


def matrices(vectors):
    return Rotation.from_rotvec(vectors.reshape(-1, 3)).as_matrix().reshape(vectors.shape[:-1] + (3, 3))


def vectors(mats):
    return Rotation.from_matrix(mats.reshape(-1, 3, 3)).as_rotvec().reshape(mats.shape[:-2] + (3,))


def transpose(mats):
    return mats.swapaxes(-1, -2)


def transform(mats, points):
    return np.einsum("...ij,...j->...i", mats, points)


@dataclass
class BVH:
    raw: bytes
    header: bytes
    names: list
    parents: np.ndarray
    offsets: np.ndarray
    table: np.ndarray
    dt: float

    @classmethod
    def read(cls, path):
        raw = Path(path).read_bytes()
        match = MOTION.search(raw)
        if not match:
            raise ValueError("Missing BVH MOTION, Frames or Frame Time header")
        count, dt = int(match[1]), float(match[2])
        if count < 1 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("BVH frame count and frame time must be positive")
        hierarchy = raw[:match.start()].decode("utf-8-sig")
        tokens = iter(re.findall(r"[{}]|[^{}\s]+", hierarchy))
        names, parents, offsets = [], [], []

        def expect(token):
            actual = next(tokens)
            if actual != token:
                raise ValueError(f"Expected {token!r}, found {actual!r}")

        def node(parent):
            name = next(tokens)
            i = len(names)
            names.append(name)
            parents.append(parent)
            expect("{")
            expect("OFFSET")
            offsets.append([float(next(tokens)) for _ in range(3)])
            expect("CHANNELS")
            channels = (("Xposition", "Yposition", "Zposition") if parent < 0 else ()) + (
                "Yrotation", "Xrotation", "Zrotation")
            expect(str(len(channels)))
            for channel in channels:
                expect(channel)
            token = next(tokens)
            while token != "}":
                if token == "JOINT":
                    node(i)
                elif token == "End":
                    for word in ("Site", "{", "OFFSET"):
                        expect(word)
                    for _ in range(3):
                        float(next(tokens))
                    expect("}")
                else:
                    raise ValueError(f"Unexpected BVH hierarchy token {token!r}")
                token = next(tokens)

        try:
            expect("HIERARCHY")
            expect("ROOT")
            node(-1)
        except StopIteration as error:
            raise ValueError("Truncated BVH hierarchy") from error
        offsets = np.array(offsets)
        required = {"Spine", "Spine1", "Neck", "Head", *(n for leg in LEGS for n in leg)}
        if (next(tokens, None) is not None or len(names) != 51 or len(set(names)) != 51
                or names[0] != "Hips" or not required.issubset(names)):
            raise ValueError("Expected the 51-joint Y-up G1 Moves capture profile")
        for leg in LEGS:
            chain = [0, *(names.index(joint) for joint in leg)]
            if any(parents[b] != a for a, b in zip(chain, chain[1:])):
                raise ValueError(f"Unexpected leg hierarchy for {leg[0]}")
            if not np.all((np.linalg.norm(offsets[chain[2:4]], axis=1) > 0.15)
                          & (np.linalg.norm(offsets[chain[2:4]], axis=1) < 0.75)):
                raise ValueError("Expected offsets in metres with nondegenerate leg segments")
        if not np.isfinite(offsets).all():
            raise ValueError("Nonfinite skeleton offsets")
        body = raw[match.end():]
        lines = [line for line in body.splitlines() if line.strip()]
        cols = 3 + 3 * len(names)
        if len(lines) != count or any(len(line.split()) != cols for line in lines):
            raise ValueError(f"Expected {count} motion rows of {cols} channels")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            try:
                data = np.fromstring(body.decode("ascii"), sep=" ")
            except DeprecationWarning as error:
                raise ValueError("Invalid numeric BVH motion channel") from error
        if data.size != count * cols or not np.isfinite(data).all():
            raise ValueError("Invalid or nonfinite BVH motion channels")
        return cls(raw, raw[:match.end()], names, np.array(parents), offsets,
                   data.reshape(count, cols), dt)

    def decode(self):
        shape = (len(self.table), len(self.names), 3, 3)
        rotations = Rotation.from_euler("YXZ", self.table[:, 3:].reshape(-1, 3),
                                        degrees=True).as_matrix().reshape(shape)
        return self.table[:, :3].copy() + self.offsets[0], rotations

    def fk(self, root, local):
        positions = np.empty(local.shape[:2] + (3,))
        world = np.empty_like(local)
        positions[:, 0], world[:, 0] = root, local[:, 0]
        for joint in range(1, len(self.names)):
            parent = self.parents[joint]
            positions[:, joint] = positions[:, parent] + transform(world[:, parent], self.offsets[joint])
            world[:, joint] = world[:, parent] @ local[:, joint]
        return positions, world

    def write(self, path, root, local):
        path = Path(path)
        old_root, old_local = self.decode()
        changed = np.max(np.abs(local - old_local), axis=(-1, -2)) > 1e-11
        if not changed.any() and np.array_equal(root, old_root):
            with path.open("xb") as stream:
                stream.write(self.raw)
            return
        output = self.table.copy()
        root_changed = np.any(root != old_root, axis=1)
        output[root_changed, :3] = root[root_changed] - self.offsets[0]
        eulers = output[:, 3:].reshape(len(output), -1, 3)
        reference = eulers[changed]
        # YXZ is Tait-Bryan. Select the equivalent branch nearest the source channels.
        if len(reference):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Gimbal lock detected")
                angles = Rotation.from_matrix(local[changed]).as_euler("YXZ", degrees=True)
            alternate = angles * [1, -1, 1] + [180, 180, 180]
            delta = (angles - reference + 180) % 360 - 180
            other = (alternate - reference + 180) % 360 - 180
            use_other = np.sum(other**2, axis=1) < np.sum(delta**2, axis=1)
            delta[use_other] = other[use_other]
            eulers[changed] = reference + delta
        with path.open("xb") as stream:
            stream.write(self.header)
            np.savetxt(stream, output, fmt="%.9f")
