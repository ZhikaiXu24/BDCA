"""Independent standard BVH parser/FK for export acceptance (centimetres -> metres)."""
import re
import numpy as np


def read_bvh(text, units_to_meters=0.01):
    header, motion = text.split('MOTION',1)
    tokens = iter(re.findall(r'\{|\}|[^\s{}]+',header))
    names, parents, offsets, channels = [], [], [], []
    if next(tokens) != 'HIERARCHY' or next(tokens) != 'ROOT':
        raise ValueError('Invalid BVH hierarchy.')

    def joint(parent):
        name = next(tokens)
        if name in names:
            raise ValueError('Duplicate joint name.')
        index = len(names)
        names.append(name); parents.append(parent); offsets.append(None); channels.append([])
        if next(tokens) != '{':
            raise ValueError('Missing joint block.')
        while True:
            token = next(tokens)
            if token == '}':
                break
            if token == 'OFFSET':
                offsets[index] = [float(next(tokens)) for _ in range(3)]
            elif token == 'CHANNELS':
                channels[index] = [next(tokens) for _ in range(int(next(tokens)))]
            elif token == 'JOINT':
                joint(index)
            elif token == 'End':
                if next(tokens) != 'Site' or next(tokens) != '{' or next(tokens) != 'OFFSET':
                    raise ValueError('Invalid End Site.')
                if not np.isfinite([float(next(tokens)) for _ in range(3)]).all():
                    raise ValueError('Nonfinite End Site offset.')
                if next(tokens) != '}': raise ValueError('Invalid End Site close.')
            else:
                raise ValueError(f'Unsupported hierarchy token: {token}')
        if offsets[index] is None:
            raise ValueError('Missing OFFSET.')
    joint(-1)
    if list(tokens): raise ValueError('Trailing hierarchy tokens.')
    lines = [line.strip() for line in motion.splitlines() if line.strip()]
    frames = int(lines[0].split(':')[1]); frame_time = float(lines[1].split(':')[1])
    data = np.array([[float(v) for v in row.split()] for row in lines[2:]])
    if (frames < 1 or data.shape != (frames,sum(map(len,channels)))
            or not np.isfinite(data).all() or not np.isfinite(offsets).all()
            or not np.isfinite(frame_time) or frame_time <= 0
            or not np.isfinite(units_to_meters) or units_to_meters <= 0):
        raise ValueError('Invalid BVH frames or channels.')
    position = np.zeros((frames,len(names),3))
    rotation = np.zeros((frames,len(names),3,3))
    cursor = 0
    for i,parent in enumerate(parents):
        local_t = np.broadcast_to(offsets[i],(frames,3)).astype(float).copy()
        local_r = np.broadcast_to(np.eye(3),(frames,3,3)).copy()
        for channel in channels[i]:
            axis = 'XYZ'.index(channel[0])
            values = data[:,cursor]; cursor += 1
            if channel[1:] == 'position':
                local_t[:,axis] += values
            elif channel[1:] == 'rotation':
                c,s = np.cos(np.deg2rad(values)),np.sin(np.deg2rad(values))
                matrix = np.broadcast_to(np.eye(3),(frames,3,3)).copy()
                a,b = (axis+1)%3,(axis+2)%3
                matrix[:,a,a]=c; matrix[:,b,b]=c
                matrix[:,a,b]=-s; matrix[:,b,a]=s
                local_r = local_r @ matrix
            else:
                raise ValueError(f'Unknown BVH channel: {channel}')
        if parent < 0:
            position[:,i] = local_t
            rotation[:,i] = local_r
        else:
            position[:,i] = position[:,parent] + np.einsum('bij,bj->bi',rotation[:,parent],local_t)
            rotation[:,i] = rotation[:,parent] @ local_r
    return dict(names=names,parents=parents,positions=position*units_to_meters,
                global_rotations=rotation,frames=frames,fps=1/frame_time)
