"""Reusable staged G1 generation, fullbody constraints and verified robot CSV export.
--check-only validates inputs on CPU without loading a text or motion model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import math
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
XML = ROOT / 'kimodo/assets/skeletons/g1skel34/xml/g1.xml'
DEFAULT_PROMPTS = [
    'A person walks forward on the ground for several steps.',
    'A person jumps forward with both feet, then lands on both feet.',
    'A person walks forward on the ground for several more steps.',
    'A person lowers their body and lies face down flat on the ground.',
    'A person pushes up from lying face down and stands upright on both feet.',
]
DEFAULT_DURATIONS = [2., 2., 2., 1.5, 2.5]
PROMPT = ('A person walks forward for a few steps, jumps forward with both feet '
          'off the ground, lands on both feet, takes a few more steps forward, '
          'then lowers their body and lies face down on the ground, '
          'and finally pushes up from the ground and stands upright again.')


def quat_matrix(q):
    import numpy as np
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Zero root quaternion.')
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(q.shape[:-1]+(3,3))


def axis_matrix(axis, angles):
    import numpy as np
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    angles = np.asarray(angles)
    return np.eye(3) + np.sin(angles)[...,None,None]*cross + (1-np.cos(angles))[...,None,None]*(cross@cross)


def matrix_quat(matrices):
    import numpy as np
    matrices = np.asarray(matrices)
    result = []
    for m in matrices.reshape(-1,3,3):
        # Symmetric eigenproblem avoids singular cases at rotations near pi.
        x,y,z = m[0],m[1],m[2]
        K = np.array([[x[0]-y[1]-z[2],x[1]+y[0],x[2]+z[0],z[1]-y[2]],
                      [x[1]+y[0],y[1]-x[0]-z[2],y[2]+z[1],x[2]-z[0]],
                      [x[2]+z[0],y[2]+z[1],z[2]-x[0]-y[1],y[0]-x[1]],
                      [z[1]-y[2],x[2]-z[0],y[0]-x[1],x[0]+y[1]+z[2]]])/3
        _,v = np.linalg.eigh(K)
        q = v[:,-1][[3,0,1,2]]
        if q[0] < 0: q = -q
        result.append(q)
    return np.asarray(result).reshape(matrices.shape[:-2]+(4,))


def matrix_axis(matrices):
    import numpy as np
    q = matrix_quat(matrices)
    norm = np.linalg.norm(q[...,1:],axis=-1)
    angle = 2*np.arctan2(norm,q[...,0])
    scale = np.divide(angle,norm,out=np.full_like(norm,2),where=norm>1e-9)
    return q[...,1:]*scale[...,None]


class RobotXML:
    """Independent FK for this G1 XML, including sphere/capsule collision proxies.
    qpos uses MuJoCo metres, wxyz and radians. Returned points use Kimodo Y-up.
    No physics stepping, GPU, or mesh loading is performed.
    """
    def __init__(self, path=XML):
        import numpy as np
        self.path = Path(path)
        self.root = ET.parse(self.path).getroot()
        self.defaults = {}
        def defaults(node, inherited):
            values = {k: dict(v) for k,v in inherited.items()}
            for kind in ('joint','geom'):
                child = node.find(kind)
                if child is not None:
                    values.setdefault(kind, {}).update(child.attrib)
            if 'class' in node.attrib:
                self.defaults[node.get('class')] = values
            for child in node.findall('default'):
                defaults(child, values)
        for node in self.root.findall('default'):
            defaults(node, {})
        self.base = self.root.find('worldbody').find('body')
        self.joints = self.base.findall('.//joint')
        self.joint_names = [j.get('name') for j in self.joints]
        if len(self.joints) != 29:
            raise ValueError('Expected a 29-DoF G1 XML.')
        self.index = {n:i for i,n in enumerate(self.joint_names)}
        self.limits = np.array([np.fromstring(self.attrs(j,'joint')['range'], sep=' ') for j in self.joints])
        self.parent_elements = {c:p for p in self.root.iter() for c in p}
        self.names = ['pelvis_skel']
        self.parents = [-1]
        def collect(body, parent):
            for j in body.findall('joint'):
                self.names.append(j.get('name').replace('_joint','_skel'))
                self.parents.append(parent)
                parent = len(self.names)-1
            for child in body.findall('body'):
                collect(child,parent)
        collect(self.base,0)
        self.endpoints = [
            ('left_toe_base','left_ankle_roll_skel',[.13,0,-.025]),
            ('right_toe_base','right_ankle_roll_skel',[.13,0,-.025]),
            ('left_hand_roll_skel','left_wrist_yaw_skel',[.08,0,0]),
            ('right_hand_roll_skel','right_wrist_yaw_skel',[.08,0,0]),
            ('head_fixed','waist_pitch_skel',[.01,0,.42]),
        ]
        for name,parent,_ in self.endpoints:
            self.names.append(name)
            self.parents.append(self.names.index(parent))

    def model_names(self):
        children = {n:[] for n in self.names}
        for i,p in enumerate(self.parents):
            if p >= 0 and self.names[i] != 'head_fixed':
                children[self.names[p]].append(self.names[i])
        result = []
        def visit(n):
            result.append(n)
            for c in children[n]: visit(c)
        visit('pelvis_skel')
        return result

    def local_from_q(self,q):
        import numpy as np
        q = np.atleast_2d(q)
        M = np.array([[0,1,0],[0,0,1],[1,0,0.]])
        names = self.model_names()
        local = np.broadcast_to(np.eye(3),(len(q),34,3,3)).copy()
        local[:,0] = M@quat_matrix(q[:,3:7])@M.T
        for i,j in enumerate(self.joints):
            B = quat_matrix(np.fromstring(self.parent_elements[j].get('quat','1 0 0 0'),sep=' '))
            axis = np.fromstring(self.attrs(j,'joint')['axis'],sep=' ')
            # XML body transform is rest rotation followed by the local hinge.
            local[:,names.index(j.get('name').replace('_joint','_skel'))] = M@B@axis_matrix(axis,q[:,7+i])@M.T
        return local, q[:,:3]@M.T

    def q_from_local(self,local,positions):
        import numpy as np
        M = np.array([[0,1,0],[0,0,1],[1,0,0.]])
        names = self.model_names()
        q = np.zeros((len(local),36))
        q[:,:3] = positions@M
        q[:,3:7] = matrix_quat(M.T@local[:,0]@M)
        for i,j in enumerate(self.joints):
            B = quat_matrix(np.fromstring(self.parent_elements[j].get('quat','1 0 0 0'),sep=' '))
            axis = np.fromstring(self.attrs(j,'joint')['axis'],sep=' ')
            axis /= np.linalg.norm(axis)
            # Undo the XML rest rotation in MuJoCo coordinates before projection.
            R = B.T@M.T@local[:,names.index(j.get('name').replace('_joint','_skel'))]@M
            skew = np.stack([R[:,2,1]-R[:,1,2], R[:,0,2]-R[:,2,0], R[:,1,0]-R[:,0,1]],-1)
            numerator = skew@axis
            denominator = np.trace(R,axis1=1,axis2=2)-np.einsum('i,tij,j->t',axis,R,axis)
            if np.any(np.hypot(numerator,denominator)<1e-8):
                raise ValueError('Ambiguous hinge projection; refusing silent angle selection.')
            q[:,7+i] = np.arctan2(numerator,denominator)
        return q

    def attrs(self, node, kind):
        result = dict(self.defaults.get(node.get('class'),{}).get(kind,{}))
        result.update(node.attrib)
        return result

    def fk(self, q):
        import numpy as np
        q = np.atleast_2d(np.asarray(q, float))
        if q.shape[1] != 36 or not np.isfinite(q).all():
            raise ValueError('Expected finite [frames,36] MuJoCo qpos.')
        points = {'pelvis_skel':q[:,:3]}
        transforms = {'pelvis_skel':(quat_matrix(q[:,3:7]),q[:,:3])}
        bottoms = {}
        def walk(body,R,P,first=False):
            if not first:
                P = P + np.einsum('tij,j->ti',R,np.fromstring(body.get('pos','0 0 0'),sep=' '))
                R = R @ quat_matrix(np.fromstring(body.get('quat','1 0 0 0'),sep=' '))
            for j in body.findall('joint'):
                a = self.attrs(j,'joint')
                if not np.allclose(np.fromstring(a.get('pos','0 0 0'),sep=' '),0):
                    raise ValueError('Nonzero joint anchor requires a different FK implementation.')
                name = j.get('name').replace('_joint','_skel')
                points[name] = P
                R = R @ axis_matrix(np.fromstring(a['axis'],sep=' '),q[:,7+self.index[j.get('name')]])
                transforms[name] = (R,P)
            for geom in body.findall('geom'):
                a = self.attrs(geom,'geom')
                if geom.get('class') != 'collision':
                    continue
                kind = a.get('type','sphere')
                radius = float(a['size'].split()[0])
                if kind == 'capsule' and 'fromto' in a:
                    ends = np.fromstring(a['fromto'],sep=' ').reshape(2,3)
                    world = np.einsum('tij,kj->tki',R,ends)+P[:,None]
                    low = world[:,:,2].min(-1)-radius
                elif kind == 'sphere':
                    center = np.fromstring(a.get('pos','0 0 0'),sep=' ')
                    low = (np.einsum('tij,j->ti',R,center)+P)[:,2]-radius
                else:
                    raise ValueError('Unsupported collision proxy: '+kind)
                bottoms[a['name']] = low
            for child in body.findall('body'):
                walk(child,R,P)
        walk(self.base,*transforms['pelvis_skel'],first=True)
        for name,parent,offset in self.endpoints:
            R,P = transforms[parent]
            points[name] = np.einsum('tij,j->ti',R,offset)+P
        xyz = np.stack([points[n] for n in self.names],1)
        return xyz[:,:,[1,2,0]], bottoms

    def project(self, q):
        import numpy as np
        q = np.asarray(q,float).copy()
        if q.ndim != 2 or q.shape[1] != 36 or not np.isfinite(q).all():
            raise ValueError('Invalid CSV qpos.')
        norm = np.linalg.norm(q[:,3:7],axis=1)
        if np.any(norm < 1e-8):
            raise ValueError('Zero root quaternion.')
        q[:,3:7] /= norm[:,None]
        for i in range(1,len(q)):
            if q[i,3:7]@q[i-1,3:7] < 0:
                q[i,3:7] *= -1
        violation = np.maximum(self.limits[:,0]-q[:,7:],q[:,7:]-self.limits[:,1])
        report = {'out_of_limit_values_before':int((violation>1e-6).sum()),
                  'max_clamp_radians':float(max(0.,violation.max()))}
        q[:,7:] = np.clip(q[:,7:],self.limits[:,0],self.limits[:,1])
        return q,report


def preset_constraints(robot):
    """CPU-authored 10 s boundary targets grounded using G1 collision proxies."""
    import numpy as np
    idx = [0,59,90,119,179,210,224,299]
    labels = ['ground start','walk end / takeoff','airborne','landed',
              'second walk end','prone','prone before rising','standing end']
    q = np.zeros((len(idx),36));q[:,3]=1
    q[:,0] = [0,.8,1.15,1.5,2.3,2.4,2.4,2.4]
    for side in ('left','right'):
        def set_angle(rows,name,value):q[rows,7+robot.index[f'{side}_{name}_joint']]=value
        set_angle(slice(None),'hip_pitch',-.1)
        set_angle(slice(None),'knee',.2)
        set_angle(slice(None),'ankle_pitch',-.1)
        for rows,angles in [([1],[-.25,.5,-.25]),([2],[-.45,.9,-.35]),([3],[-.35,.7,-.35])]:
            for name,angle in zip(['hip_pitch','knee','ankle_pitch'],angles):set_angle(rows,name,angle)
        set_angle([2],'shoulder_pitch',-1.)
        set_angle([2],'elbow',.6)
        for name in ('hip_pitch','knee','ankle_pitch'):set_angle([5,6],name,0.)
        set_angle([5,6],'knee',.35)
        set_angle([5,6],'hip_pitch',.15)
        set_angle([5,6],'shoulder_pitch',1.)
        set_angle([5,6],'elbow',0.)
    q[[5,6],3:7] = [2**-.5,0,2**-.5,0]
    _,bottoms = robot.fk(q)
    low = np.stack(list(bottoms.values()),1).min(1)
    q[:,2] += .003-low
    q[2,2] += .30
    q,clamp = robot.project(q)
    if clamp['out_of_limit_values_before']:
        raise ValueError('Authored target joint limits are invalid.')
    points,bottoms = robot.fk(q)
    all_low = np.stack(list(bottoms.values()),1).min(1)
    foot_low = np.stack([v for k,v in bottoms.items() if 'foot' in k],1).min(1)
    torso_low = np.stack([v for k,v in bottoms.items() if 'torso_collision' in k],1).min(1)
    if np.min(all_low)<-1e-6 or np.max(foot_low[[0,1,3,4,7]])>.01 or foot_low[2]<.2:
        raise ValueError('Target ground/takeoff geometry check failed.')
    if np.max(torso_low[[5,6]])>.03:
        raise ValueError(f'Prone torso is not close to the ground: {torso_low[[5,6]]}; lowest proxies: {[(k,float(v[5])) for k,v in bottoms.items() if v[5]<.02]}')
    local,roots = robot.local_from_q(q)
    recovered = robot.q_from_local(local,roots)
    if np.max(np.abs(recovered[:,7:]-q[:,7:]))>1e-6:
        raise ValueError('Target hinge conversion round trip failed.')
    constraints = [{'type':'fullbody','frame_indices':idx,
                    'local_joints_rot':matrix_axis(local).tolist(),
                    'root_positions':roots.tolist(),'smooth_root_2d':roots[:,[0,2]].tolist()}]
    info = {'frames':idx,'labels':labels,'minimum_collision_height_m':all_low.tolist(),
            'foot_collision_height_m':foot_low.tolist(),'torso_collision_height_m':torso_low.tolist(),
            'note':'Collision-proxy floor checks only; no self-collision or dynamics validation.'}
    return constraints,q,points,info


def make_run_dir():
    parent = ROOT / 'outputs' / 'g1_keyframes'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    run = parent / ('g1_' + stamp)
    run.mkdir(exist_ok=False)
    return run


def generator_source():
    source = (SCRIPTS / 'smoke_text_to_motion.py').read_text()
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('Base generator changed: '+old[:80])
        source = source.replace(old,new)
    replace('args = parser.parse_args()', 'parser.add_argument("--constraints", required=True)\nparser.add_argument("--stages", required=True)\nargs = parser.parse_args()')
    replace('assert args.prompt.strip()', 'assert args.prompt.strip()\nstage_job = json.loads(Path(args.stages).read_text())\nstage_prompts = [sanitize_text(t) for t in stage_job["prompts"]]\nstage_frames = stage_job["stage_frames"]\nassert len(stage_prompts) == len(stage_frames) and sum(stage_frames) == round(args.duration*30)\nunique_prompts = list(dict.fromkeys(stage_prompts))')
    first = source.index('comparison_prompt = ')
    last = source.index('# Only the genuine features', first)
    encoder_block = r'''comparison_prompt = "A person stands still."
if comparison_prompt in unique_prompts:
    comparison_prompt = "A person waves their hands."
with torch.inference_mode():
    features, lengths = real_encoder(unique_prompts + [comparison_prompt])
nencoded = len(unique_prompts) + 1
assert len(bidirectional_checks) == nencoded
assert features.shape == (nencoded, 1, 4096) and lengths == [1]*nencoded
assert torch.isfinite(features).all()
assert bool((torch.linalg.vector_norm(features[:,0],dim=-1)>0).all())
feature_difference = float(torch.linalg.vector_norm(features[:-1]-features[-1:],dim=-1).min())
assert feature_difference > 1e-3
text_encoder_seconds = time.monotonic() - encoder_started
encoder_peak_GiB = torch.cuda.max_memory_allocated(0) / 2**30
print("REAL_STAGE_TEXT_FEATURES", {"prompts": unique_prompts, "shape": list(features.shape)}, flush=True)
class CachedRealTextEncoder(nn.Module):
    def __init__(self, prompts, features):
        super().__init__()
        self.indices = {p.strip():i for i,p in enumerate(prompts)}
        self.register_buffer("features", features.detach().clone())
        self.calls = 0
        self.seen = []
    def forward(self, texts):
        indices = [self.indices[text.strip()] for text in texts]
        self.seen.extend(text.strip() for text in texts)
        self.calls += 1
        return self.features[indices], [1]*len(texts)
encoder = CachedRealTextEncoder(unique_prompts, features[:-1])
'''
    source = source[:first]+encoder_block+source[last:]
    replace('load_seconds = time.monotonic() - load_started', 'load_seconds = time.monotonic() - load_started\nassert model.skeleton.nbjoints == 34\nfrom kimodo.constraints import load_constraints_lst\nconstraint_lst = load_constraints_lst(args.constraints, model.skeleton)')
    replace('args.prompt, num_frames=frames, num_samples=1, num_denoising_steps=100,', 'stage_prompts, num_frames=stage_frames, multi_prompt=True, num_transition_frames=5, num_samples=1, num_denoising_steps=100,')
    replace('constraint_lst=[]', 'constraint_lst=constraint_lst')
    replace('inference_seconds = time.monotonic() - started',
            'inference_seconds = time.monotonic() - started\n'
            'from stage_anchor_20261030 import save_effective_constraints\n'
            'effective_constraints = save_effective_constraints(constraint_lst, str(stem)+"_resolved_constraints.json")')
    replace('frames = int(duration * fps)', 'frames = sum(stage_frames)')
    replace('assert encoder.calls == 1', 'assert set(encoder.seen) == set(unique_prompts) and encoder.calls == len(stage_prompts)')
    replace('if "G1" in args.model:', 'if False:  # Robot CSV is projected and checked by the reusable parent entry.')
    replace('real_text_llama3_llm2vec_no_motion_constraints', 'real_text_llama3_llm2vec_staged_g1_keyframes')
    replace('nonzero text features; no motion constraints', 'real features for every stage; G1 fullbody keyframes')
    replace('"status": "passed", "model": args.model', '"status": "structural_checks_passed", "stages": stage_job["prompts"], "stage_frames": stage_frames, "model": args.model')
    replace('"text_feature_lengths": [1]', '"text_feature_lengths": [1]*len(unique_prompts)')
    compile(source,'<g1-staged-generator>','exec')
    return source


def write_job(run, skel, prompts, durations, seed, extra=None):
    source = generator_source()
    manifest = {
        'model': 'Kimodo-G1-RP-v1', 'prompt': ' | '.join(prompts), 'duration': sum(durations),
        'prompts': prompts, 'stage_durations': durations,
        'stage_frames': [round(d*30) for d in durations],
        'fps': 30, 'seed': seed, 'output_stem': str(run / run.name),
        'joint_names': list(skel.bone_order_names),
        'joint_parents': skel.joint_parents.tolist(),
        'original_generator_sha256': hashlib.sha256((SCRIPTS / 'smoke_text_to_motion.py').read_bytes()).hexdigest(),
        'adapted_generator_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'constraints_sha256': hashlib.sha256((run / 'constraints.json').read_bytes()).hexdigest(),
    }
    manifest.update(extra or {})
    save_json(run / 'job.json', manifest)


def prepare_existing(run, constraint_path, prompts, durations, seed):
    import numpy as np
    import torch
    from kimodo.skeleton.registry import build_skeleton
    from kimodo.constraints import load_constraints_lst
    data = json.loads(constraint_path.read_text())
    frames = sum(round(d*30) for d in durations)
    seen = set()
    if not isinstance(data, list) or not data:
        raise ValueError('Constraints must be a nonempty list of G1 fullbody constraints.')
    for item in data:
        if item.get('type') != 'fullbody':
            raise ValueError('This entry currently accepts G1 fullbody keyframes only.')
        ids = item['frame_indices']
        if not ids or any(type(i) is not int or not 0 <= i < frames for i in ids):
            raise ValueError('Keyframe indices must be integers within the generated clip.')
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('Duplicate keyframe indices.')
        seen.update(ids)
        rotations = np.asarray(item['local_joints_rot'], dtype=np.float32)
        positions = np.asarray(item['root_positions'], dtype=np.float32)
        if rotations.shape != (len(ids), 34, 3):
            raise ValueError('G1 requires [keyframes,34,3] rotations; SOMA 30/77-joint constraints cannot be reused directly.')
        if positions.shape != (len(ids), 3) or not np.isfinite(rotations).all() or not np.isfinite(positions).all():
            raise ValueError('Invalid keyframe root positions or nonfinite pose values.')
    from stage_anchor_20261030 import validate_stage_anchors
    validate_stage_anchors(data, [round(d*30) for d in durations])
    torch.set_num_threads(2)
    skel = build_skeleton(34).to('cpu')
    loaded = load_constraints_lst(data, skel)
    poses = {int(frame): c.global_joints_positions[i].detach().cpu().numpy()
             for c in loaded for i, frame in enumerate(c.frame_indices.tolist())}
    idx = sorted(poses)
    targets = np.stack([poses[i] for i in idx])
    robot = RobotXML()
    if list(skel.bone_order_names) != robot.model_names():
        raise ValueError('Training skeleton order differs from expected XML mapping.')
    from kimodo.geometry import axis_angle_to_matrix
    local_by_frame = {}
    root_by_frame = {}
    for item in data:
        matrices = axis_angle_to_matrix(torch.tensor(item['local_joints_rot'])).numpy()
        for k,i in enumerate(item['frame_indices']):
            local_by_frame[i] = matrices[k]
            root_by_frame[i] = item['root_positions'][k]
    target_q = robot.q_from_local(np.stack([local_by_frame[i] for i in idx]),np.asarray([root_by_frame[i] for i in idx]))
    target_q,clamp = robot.project(target_q)
    if clamp['max_clamp_radians'] > 1e-4:
        raise ValueError('Target constraints violate actual robot joint limits.')
    robot_targets,bottoms = robot.fk(target_q)
    minimum_height = np.stack(list(bottoms.values()),1).min(1)
    if minimum_height.min() < -.005:
        raise ValueError('Target collision proxies penetrate the floor.')
    save_json(run / 'constraints.json', data)
    with (run / 'targets.npz').open('xb') as f:
        np.savez(f, positions=targets, frame_indices=idx, robot_positions=robot_targets, qpos=target_q)
    overlay(run / 'target_keyframes.png', robot_targets, None, robot.parents,
            [f'Frame {i} / {i/30:g} s' for i in idx])
    write_job(run, skel, prompts, durations, seed, {'input_constraints': str(constraint_path),
              'horizontal_free_keyframes':[i for c in data if c.get('horizontal_free',False) for i in c['frame_indices']],
              'robot_joint_names':robot.names,'robot_joint_parents':robot.parents,
              'target_min_collision_height_m':minimum_height.tolist(),
              'robot_xml_sha256':hashlib.sha256(XML.read_bytes()).hexdigest()})


def save_json(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def overlay(path, target, actual, parents, labels):
    import numpy as np
    from PIL import Image, ImageDraw
    if path.exists():
        raise FileExistsError(path)
    count = len(target)
    im = Image.new('RGB', (1200, 40 + 660*((count+1)//2)), '#101827')
    d = ImageDraw.Draw(im)
    views = (np.array([[0, 0], [0, 1], [1, 0.]]),
             np.array([[.8, -.2], [0, .94], [-.6, -.28]]))
    d.text((20, 12), 'G1 targets: yellow | generated: cyan | metres, Y-up', fill='white')
    for i in range(count):
        for v, cam in enumerate(views):
            col, row = i % 2, i // 2
            origin = target[i, 0].copy()
            origin[1] = 0
            sets = [target[i]] if actual is None else [target[i], actual[i]]
            projected = [(x - origin) @ cam for x in sets]
            floor = (np.array([[origin[0], 0, origin[2] - 1.2],
                               [origin[0], 0, origin[2] + 1.2]]) - origin) @ cam
            bounds = np.concatenate(projected + [floor])
            lo, hi = bounds.min(0), bounds.max(0)
            mid = (lo + hi) / 2
            scale = min(530 / max(hi[0] - lo[0], .5), 250 / max(hi[1] - lo[1], .5))
            def screen(x):
                x = (x - mid) * scale
                return np.stack([300 + 600*col + x[:, 0], 205 + 660*row + 330*v - x[:, 1]], -1)
            d.text((20 + 600*col, 40 + 660*row + 330*v), labels[i] + (' | side' if v == 0 else ' | perspective'), fill='white')
            d.line([tuple(x) for x in screen(floor)], fill='#607080', width=2)
            for pts, color in zip(projected, ('#ffcc33', '#38d5ed')):
                xy = screen(pts)
                for j, p in enumerate(parents):
                    if p >= 0:
                        d.line([tuple(xy[p]), tuple(xy[j])], fill=color, width=3)
    with path.open('xb') as f:
        im.save(f, format='PNG')



def generate(run):
    from bvh_export_20261029 import check_bvh_dependencies, export_g1_csv
    check_bvh_dependencies()
    import numpy as np
    import torch
    from kimodo.skeleton.registry import build_skeleton
    job = json.loads((run / 'job.json').read_text())
    stem = Path(job['output_stem'])
    for suffix in ('.npz', '.csv', '.json', '_errors.json', '_keyframes.png'):
        if Path(str(stem) + suffix).exists():
            raise FileExistsError(str(stem) + suffix)
    for name, key in [('constraints.json', 'constraints_sha256')]:
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == job[key]
    assert hashlib.sha256(generator_source().encode()).hexdigest() == job['adapted_generator_sha256']
    selected_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not selected_gpu.isdecimal():
        raise ValueError('Set CUDA_VISIBLE_DEVICES to one physical GPU index.')
    env = dict(os.environ, HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    env.setdefault('HF_HUB_CACHE', str(ROOT / 'hf_cache'))
    env['PYTHONPATH'] = str(SCRIPTS) + os.pathsep + str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run(['nvidia-smi', '-i', selected_gpu], check=True)
    command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--run-dir', str(run)]
    # The existing generator checks for >=24 GiB free before loading any model.
    with (run / 'run.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError('Generation failed; see run.log. Existing files retained.')

    with np.load(run / 'targets.npz', allow_pickle=False) as f:
        targets,idx = f['positions'].copy(),f['frame_indices'].copy()
        robot_targets = f['robot_positions'].copy()
    with np.load(str(stem)+'.npz', allow_pickle=False) as f:
        generated = f['posed_joints'].copy()
        local = f['local_rot_mats'].copy()
        roots = f['root_positions'].copy()
    frames = sum(job['stage_frames'])
    assert generated.shape == (frames,34,3)
    robot = RobotXML()
    if hashlib.sha256(XML.read_bytes()).hexdigest() != job['robot_xml_sha256']:
        raise RuntimeError('Robot XML changed since preparation.')
    effective_path = Path(str(stem)+'_resolved_constraints.json')
    effective = json.loads(effective_path.read_text())
    anchor_resolutions = [c['anchor_resolution'] for c in effective if 'anchor_resolution' in c]
    if anchor_resolutions:
        from kimodo.constraints import load_constraints_lst
        from kimodo.geometry import axis_angle_to_matrix
        skel = build_skeleton(34).to('cpu')
        effective_loaded = load_constraints_lst(effective, skel)
        poses = {int(frame): c.global_joints_positions[k].numpy()
                 for c in effective_loaded for k, frame in enumerate(c.frame_indices.tolist())}
        targets = np.stack([poses[int(i)] for i in idx])
        by_frame = {int(frame): (c['local_joints_rot'][k], c['root_positions'][k])
                    for c in effective for k, frame in enumerate(c['frame_indices'])}
        target_local = axis_angle_to_matrix(torch.tensor([by_frame[int(i)][0] for i in idx])).numpy()
        target_roots = np.asarray([by_frame[int(i)][1] for i in idx])
        effective_q = robot.q_from_local(target_local, target_roots)
        robot_targets, _ = robot.fk(effective_q)
        with (run/'effective_targets.npz').open('xb') as handle:
            np.savez(handle, positions=targets, robot_positions=robot_targets,
                     qpos=effective_q, frame_indices=idx)
    job['stage_anchor_resolutions'] = anchor_resolutions
    job['effective_constraints_path'] = str(effective_path)
    raw_q = robot.q_from_local(local,roots)
    csv,export_report = robot.project(raw_q)
    with Path(str(stem)+'.csv').open('x') as f:
        np.savetxt(f,csv,delimiter=',',fmt='%.10g')
    saved = np.loadtxt(str(stem)+'.csv',delimiter=',')
    assert saved.shape == (frames,36) and np.isfinite(saved).all()
    assert np.max(np.abs(csv-saved)) < 1e-7
    assert np.max(np.abs(np.linalg.norm(saved[:,3:7],axis=1)-1)) < 1e-7
    assert np.all(saved[:,7:] >= robot.limits[:,0]-1e-7)
    assert np.all(saved[:,7:] <= robot.limits[:,1]+1e-7)
    robot_points,bottoms = robot.fk(saved)
    bvh_path = export_g1_csv(Path(str(stem)+'.csv'), Path(str(stem)+'.bvh'), 30)
    from keyframe_tools_20261029 import align_horizontal_targets
    free_frames = job.get('horizontal_free_keyframes',[])
    targets = align_horizontal_targets(generated[idx],targets,idx,free_frames)
    robot_targets = align_horizontal_targets(robot_points[idx],robot_targets,idx,free_frames)
    errors = {'status':'measured','alignment':'none','units':'mm',
              'frames':idx.tolist(),'times_seconds':(idx/30).tolist(),'post_processing':False}
    errors['stage_anchor_resolutions'] = anchor_resolutions
    errors['effective_constraints_path'] = str(effective_path)
    if free_frames:
        errors['alignment'] = 'root X/Z translation only at horizontal_free_keyframes; no Y or rotation alignment'
        errors['horizontal_free_keyframes'] = free_frames
    for label,positions,reference in [('raw_model',generated,targets),('exported_robot',robot_points,robot_targets)]:
        dist = np.linalg.norm(positions[idx]-reference,axis=-1)*1000
        errors[label] = {'mean_per_keyframe_mm':dist.mean(-1).tolist(),
                         'max_per_keyframe_mm':dist.max(-1).tolist()}
    reasons = []
    if max(errors['exported_robot']['mean_per_keyframe_mm']) > 50:
        reasons.append('At least one keyframe mean position error exceeds 50 mm.')
    if max(errors['exported_robot']['max_per_keyframe_mm']) > 150:
        reasons.append('At least one keyframe maximum position error exceeds 150 mm.')
    low = np.stack(list(bottoms.values()),1).min(1)
    foot_low = np.stack([v for k,v in bottoms.items() if 'foot' in k],1).min(1)
    if low.min() < -.02:
        reasons.append('A collision proxy penetrates the floor by more than 20 mm.')
    for i in (0,frames-1):
        if not -.02 <= foot_low[i] <= .05:
            reasons.append(f'Ground boundary frame {i}: lowest foot collision surface {foot_low[i]:.3f} m.')
    if export_report['max_clamp_radians'] > .175:
        reasons.append('Joint limit projection changed an angle by more than 10 degrees.')
    export_report.update({'csv_shape':list(saved.shape),'csv_units':'metres, wxyz, radians',
        'joint_order':robot.joint_names,'xml_sha256':job['robot_xml_sha256'],
        'csv_readback_max_abs_error':float(np.max(np.abs(csv-saved))),
        'minimum_collision_height_m':float(low.min()),
        'start_end_foot_height_m':foot_low[[0,-1]].tolist(),
        'maximum_joint_step_rad':float(np.max(np.abs(np.diff(saved[:,7:],axis=0)))),
        'geometry_reference':'Actual G1 XML forward kinematics, no physics stepping.',
        'quality_status':'needs_review' if reasons else 'screening_passed',
        'quality_reasons':reasons,
        'note':'Screening is not semantic or physical-execution validation.'})
    save_json(Path(str(stem)+'_errors.json'),errors)
    save_json(Path(str(stem)+'_export.json'),export_report)
    overlay(Path(str(stem)+'_keyframes.png'),robot_targets,robot_points[idx],robot.parents,
            [f'Robot frame {i} / {i/30:g} s' for i in idx])
    preview = Path(str(stem)+'_robot')
    with preview.with_suffix('.npz').open('xb') as f:
        np.savez_compressed(f,posed_joints=robot_points)
    preview_job = dict(job,joint_names=robot.names,joint_parents=robot.parents,
                       keyframe_targets=robot_targets.tolist(),
                       keyframe_indices=idx.tolist(),
                       keyframe_overlay_space=('G1 XML FK, metres, Y-up; target X/Z follows generated root at each free keyframe; Y unchanged'
                                               if free_frames else 'G1 XML FK, metres, Y-up; no spatial alignment'))
    save_json(preview.with_suffix('.json'),preview_job)
    subprocess.run([sys.executable,str(SCRIPTS/'render_motion.py'),'--input',str(preview)+'.npz'],
                   cwd=ROOT,env=env,check=True)
    print(json.dumps(export_report,indent=2),flush=True)
    print('BVH:',bvh_path)
    print('CSV:',str(stem)+'.csv')
    print('MP4:',str(preview)+'_skeleton.mp4')
    print('GIF:',str(preview)+'_skeleton.gif')
    print('Keyframe overlay:',str(stem)+'_keyframes.png')
    if reasons:
        print('NEEDS_REVIEW: 文件已保存，但动作检查未通过；请查看 _export.json 和 _errors.json。')
    else:
        print('结构与数值筛查通过；仍需观看视频确认动作顺序与质量。')


def run_worker(run):
    job = json.loads((run / 'job.json').read_text())
    source = generator_source()
    assert hashlib.sha256(source.encode()).hexdigest() == job['adapted_generator_sha256']
    sys.path.insert(0, str(SCRIPTS))
    sys.argv = [str(SCRIPTS / 'smoke_text_to_motion.py'),
                '--model', job['model'], '--prompt', job['prompt'],
                '--duration', str(job['duration']), '--seed', str(job['seed']),
                '--constraints', str(run / 'constraints.json'), '--stages', str(run / 'job.json'), '--output', job['output_stem']]
    # Reuse the verified encoder in memory; do not create a Python file per job.
    exec(compile(source, '<g1-keyframe-generator>', 'exec'),
         {'__name__': '__main__', '__file__': str(SCRIPTS / 'smoke_text_to_motion.py')})


def self_test():
    import ast
    import importlib.util
    import numpy as np
    robot = RobotXML(XML)
    rng = np.random.default_rng(823)
    q = np.zeros((64,36))
    q[:,:3] = rng.normal(size=(64,3))
    q[:,3:7] = rng.normal(size=(64,4))
    q[:,3:7] /= np.linalg.norm(q[:,3:7],axis=-1,keepdims=True)
    q[:,7:] = rng.uniform(robot.limits[:,0],robot.limits[:,1],size=(64,29))
    local,roots = robot.local_from_q(q)
    recovered = robot.q_from_local(local,roots)
    angle_error = float(np.abs(q[:,7:]-recovered[:,7:]).max())
    assert angle_error < 1e-7
    assert np.allclose(quat_matrix(q[:,3:7]),quat_matrix(recovered[:,3:7]),atol=1e-7)
    assert np.allclose(q[:,:3],recovered[:,:3],atol=1e-8)
    bad = q.copy();bad[:,7:] = robot.limits[:,1]+.5
    corrected,clamp = robot.project(bad)
    assert clamp['out_of_limit_values_before'] == 64*29
    assert np.all(corrected[:,7:] <= robot.limits[:,1])
    constraints,targets,_,geometry = preset_constraints(robot)
    source = generator_source()
    tree = ast.parse(source,feature_version=(3,10))
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model']
    assert len(calls)==1
    call = calls[0];keys = {k.arg:k.value for k in call.keywords}
    assert isinstance(call.args[0],ast.Name) and call.args[0].id=='stage_prompts'
    assert isinstance(keys['num_frames'],ast.Name) and keys['num_frames'].id=='stage_frames'
    assert isinstance(keys['multi_prompt'],ast.Constant) and keys['multi_prompt'].value is True
    assert sum(round(x*30) for x in DEFAULT_DURATIONS)==300
    report = {'status':'cpu_checks_passed','model_inference_executed':False,
              'hinge_roundtrip_max_radians':angle_error,'geometry':geometry}
    if importlib.util.find_spec('torch') is not None and importlib.util.find_spec('kimodo') is not None:
        import torch
        from torch import nn
        from kimodo.skeleton.registry import build_skeleton
        from kimodo.exports.mujoco import MujocoQposConverter
        from kimodo.constraints import load_constraints_lst
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()
        skel = build_skeleton(34).to('cpu')
        assert robot.model_names()==list(skel.bone_order_names)
        restored = MujocoQposConverter(skel).qpos_to_motion_dict(q.astype(np.float32),30.)
        official_error = float(np.max(np.abs(restored['local_rot_mats'].numpy()-local)))
        # Compatibility is diagnostic: the official inverse uses a different
        # static-rotation order. Validate geometry independently, not agreement.
        report['official_inverse_matrix_difference'] = official_error
        from kimodo.skeleton.kinematics import fk as native_fk
        _,native_points,_ = native_fk(torch.tensor(local,dtype=torch.float64),
                                     torch.tensor(roots,dtype=torch.float64),skel)
        xml_points,_ = robot.fk(q)
        common = ['pelvis_skel']+[n.replace('_joint','_skel') for n in robot.joint_names]
        native_ids = [robot.model_names().index(n) for n in common]
        xml_ids = [robot.names.index(n) for n in common]
        physical_error = float(np.linalg.norm(native_points.numpy()[:,native_ids]-xml_points[:,xml_ids],axis=-1).max())
        assert physical_error < 2e-6, 'Native skeleton and physical XML FK disagree'
        report['native_skeleton_vs_xml_fk_max_error_m'] = physical_error
        loaded = load_constraints_lst(constraints,skel)
        assert loaded[0].global_joints_positions.shape == (8,34,3)
        # Crouch targets constrain pose/height without pinning world X/Z.
        from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
        from kimodo.postprocess import extract_input_motion_from_constraints
        import copy
        free_checks = {}
        for count,preset in [(34,'g1'),(30,'soma')]:
            test_skel = build_skeleton(count).to('cpu')
            path = ROOT / f'custom_keyframes/{preset}_walk_crouch_walk_10s/constraints.json'
            data = json.loads(path.read_text())
            targets_free = load_constraints_lst(data,test_skel)
            rep_free = KimodoMotionRep(test_skel,30)
            obs,mask = rep_free.create_conditions_from_constraints_batched(
                targets_free,torch.tensor([300]),to_normalize=False,device='cpu')
            root_start = rep_free.slice_dict['smooth_root_pos'].start
            assert not mask[0,:, [root_start,root_start+2]].any(), 'Free-horizontal target must not pin X/Z'
            assert mask[0,180,root_start+1] and mask[0,180,rep_free.slice_dict['local_joints_positions']].all()
            shifted = copy.deepcopy(data)
            shifted[0]['root_positions'][0][0] += 50
            shifted[0]['root_positions'][0][2] -= 30
            moved = load_constraints_lst(shifted,test_skel)
            moved_obs,moved_mask = rep_free.create_conditions_from_constraints_batched(
                moved,torch.tensor([300]),to_normalize=False,device='cpu')
            assert torch.equal(mask,moved_mask) and torch.allclose(obs,moved_obs,atol=1e-5)
            cropped_free = targets_free[0].crop_move(150,210)
            assert cropped_free.horizontal_free and cropped_free.frame_indices.tolist()==[30]
            assert cropped_free.get_save_info()['horizontal_free'] is True
            reference = torch.zeros(300,3);reference[:,0]=7.;reference[:,2]=12.
            hips,_ = extract_input_motion_from_constraints(targets_free,test_skel,300,count,
                                                           root_positions_reference=reference)
            assert torch.equal(hips[180,[0,2]],reference[180,[0,2]])
            assert torch.allclose(hips[180,1],targets_free[0].root_y_pos[0])
            if count == 30:
                expanded = load_constraints_lst(data,build_skeleton(77).to('cpu'))
                target_floor = float(expanded[0].global_joints_positions[...,1].min())
                assert target_floor >= 0
            else:
                from kimodo.geometry import axis_angle_to_matrix
                target_local = axis_angle_to_matrix(torch.tensor(data[0]['local_joints_rot'])).numpy()
                target_q = robot.q_from_local(target_local,np.asarray(data[0]['root_positions']))
                _,target_limits = robot.project(target_q)
                _,target_bottoms = robot.fk(target_q)
                target_floor = min(float(v.min()) for v in target_bottoms.values())
                assert target_floor >= 0 and target_limits['max_clamp_radians']<1e-5
            free_checks[preset] = {'status':'pose/height constrained; X/Z free; cropping and postprocess passed',
                                   'minimum_target_height_m':target_floor}
        report['crouch_horizontal_free'] = free_checks
        # Single squat uses the official position-only fullbody conditioning path.
        squat_checks = {}
        for count,kind in [(34,'g1'),(30,'soma')]:
            squat_skel = build_skeleton(count).to('cpu')
            path = ROOT / f'custom_keyframes/{kind}_walk_squat_single_6s/constraints.json'
            data = json.loads(path.read_text())
            target = load_constraints_lst(data,squat_skel)
            assert len(target)==1 and target[0].frame_indices.tolist()==[180]
            assert not target[0].horizontal_free and not target[0].anchor_to_previous_stage
            rep = KimodoMotionRep(squat_skel,30)
            obs,mask = rep.create_conditions_from_constraints_batched(
                target,torch.tensor([300]),to_normalize=False,device='cpu')
            assert torch.where(mask[0].any(-1))[0].tolist()==[180]
            assert mask[0,180,rep.slice_dict['smooth_root_pos']].all()
            assert mask[0,180,rep.slice_dict['local_joints_positions']].all()
            assert not mask[0,:,rep.slice_dict['global_rot_data']].any()
            P = target[0].global_joints_positions[0].numpy()
            names = list(squat_skel.bone_order_names)
            bends=[]
            for side in (['left','right'] if count==34 else ['Left','Right']):
                joints = ([side+'_hip_yaw_skel',side+'_knee_skel',side+'_ankle_pitch_skel']
                          if count==34 else [side+'Leg',side+'Shin',side+'Foot'])
                hip,knee,ankle = [P[names.index(j)] for j in joints]
                a,b=hip-knee,ankle-knee
                bend=180-float(np.rad2deg(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
                assert 70 < bend < 110, 'Squat target is not a visibly bent-knee pose'
                bends.append(bend)
            if count==34:
                from kimodo.geometry import axis_angle_to_matrix
                L = axis_angle_to_matrix(torch.tensor(data[0]['local_joints_rot'])).numpy()
                q_target=robot.q_from_local(L,np.asarray(data[0]['root_positions']))
                _,limit_report=robot.project(q_target)
                _,bottom=robot.fk(q_target)
                floor=min(float(x.min()) for x in bottom.values())
                assert limit_report['max_clamp_radians']<1e-5 and floor>=0
            else:
                full=load_constraints_lst(data,build_skeleton(77).to('cpu'))
                floor=float(full[0].global_joints_positions[...,1].min())
                assert floor>=0
            squat_checks[kind]={'frame':180,'time_seconds':6.,'knee_flexion_degrees':bends,
                                'minimum_target_height_m':floor,'official_fullbody_mask':True,
                                'root_position':data[0]['root_positions'][0]}
        report['single_squat_targets'] = squat_checks
        # Real conditioning path: a single jump target leaves the outer stages empty.
        # This needs no model, checkpoint, normalization stats or GPU.
        from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
        condition_rep = KimodoMotionRep(skel,30)
        single_path = ROOT / 'custom_keyframes/g1_walk_jump_walk_10s/constraints.json'
        single = load_constraints_lst(str(single_path),skel)
        assert [int(i) for c in single for i in c.frame_indices] == [90]
        stage_masks = []
        for begin,end in ((0,60),(60,120),(120,300)):
            cropped_single = [c.crop_move(begin,end) for c in single]
            observed,mask = condition_rep.create_conditions_from_constraints_batched(
                cropped_single,torch.tensor([end-begin]),to_normalize=False,device='cpu')
            assert torch.isfinite(observed).all() and mask.dtype == torch.bool
            constrained_frames = torch.where(mask[0].any(-1))[0].tolist()
            assert constrained_frames == ([30] if begin == 60 else [])
            stage_masks.append(constrained_frames)
        active = [c.crop_move(60,120) for c in single]
        empty = [c.crop_move(0,60) for c in single]
        pure = condition_rep.create_conditions_from_constraints_batched(
            active,torch.tensor([60]),to_normalize=False,device='cpu')
        mixed = condition_rep.create_conditions_from_constraints_batched(
            empty+active,torch.tensor([60]),to_normalize=False,device='cpu')
        assert all(torch.equal(a,b) for a,b in zip(pure,mixed))
        report['single_keyframe_real_conditioning'] = {
            'stage_local_constraint_frames':stage_masks,
            'empty_plus_active_matches_active_only':True}
        cache = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CachedRealTextEncoder')
        namespace = {'nn':nn}
        exec(compile(ast.Module(body=[cache],type_ignores=[]),'<encoder-dispatch-test>','exec'),namespace)
        features = torch.arange(12.).reshape(3,1,4)
        encoder = namespace['CachedRealTextEncoder'](['walk','jump','rise'],features)
        output,lengths = encoder(['rise','walk','jump','walk'])
        assert torch.equal(output,features[[2,0,1,0]]) and lengths==[1]*4
        try:encoder(['unencoded'])
        except KeyError:pass
        else:raise AssertionError('Unexpected prompt accepted.')
        assert not torch.cuda.is_initialized()
        report['official_inverse_rotation_max_error'] = official_error
        report['actual_torch_stage_cache_dispatch'] = 'passed'
        # Exercise official timeline/cropping code with a deterministic test double.
        # No checkpoint, diffusion, Llama, or generated output file is involved.
        from kimodo.model.kimodo_model import Kimodo
        from types import SimpleNamespace
        cropped = []
        class FixtureRep:
            def create_conditions_from_constraints_batched(self, constraints, lengths, **kwargs):
                for group in constraints:
                    for c in (group if isinstance(group,list) else [group]):
                        assert all(0 <= int(i) < int(lengths.max()) for i in c.frame_indices)
                cropped.append([int(i) for c in constraints if not isinstance(c,list) for i in c.frame_indices])
                shape = (len(lengths),int(lengths.max()),1)
                return torch.zeros(shape),torch.zeros(shape)
            def normalize(self,x):return x
            def unnormalize(self,x):return x
            def translate_2d(self,x,offset):return x
            def inverse(self,x,**kwargs):
                B,T=x.shape[:2]
                rots=torch.eye(3).repeat(B,T,34,1,1)
                roots=torch.zeros(B,T,3)
                gr,joints,_=skel.fk(rots.reshape(-1,34,3,3),roots.reshape(-1,3))
                return {'local_rot_mats':rots,'global_rot_mats':gr.reshape(B,T,34,3,3),
                        'posed_joints':joints.reshape(B,T,34,3),'root_positions':roots,
                        'smooth_root_pos':roots,'foot_contacts':torch.zeros(B,T,4,dtype=torch.bool)}
        calls = []
        def fixture_sampling(texts,max_frames,**kwargs):
            calls.append((texts[0],int(max_frames)))
            return torch.zeros(1,int(max_frames),1)
        fixture = SimpleNamespace(device='cpu',skeleton=skel,motion_rep=FixtureRep(),_generate=fixture_sampling)
        result = Kimodo._multiprompt(fixture,DEFAULT_PROMPTS,[60,60,60,45,75],
                    1,constraint_lst=loaded,num_samples=1,post_processing=False)
        assert result['posed_joints'].shape == (1,300,34,3)
        assert [x[0] for x in calls] == DEFAULT_PROMPTS
        assert [x[1] for x in calls] == [60,65,65,50,80]
        assert cropped[0]==[0,59] and cropped[1]==[30,59]
        assert not torch.cuda.is_initialized()
        report['official_stage_timeline_fixture'] = {'frames':300,'segment_frames_including_overlap':[x[1] for x in calls]}
    else:
        report['official_torch_checks'] = 'not available in this Python environment'
    if importlib.util.find_spec('mujoco') is not None:
        import mujoco
        # A geometry-free reference retains all XML body and joint transforms.
        import copy
        ref = copy.deepcopy(robot.root)
        for parent in ref.iter():
            for child in list(parent):
                if child.tag in ('asset','contact','actuator','sensor','keyframe','geom','site'):
                    parent.remove(child)
        compiler=ref.find('compiler')
        if compiler is not None:compiler.attrib.pop('meshdir',None)
        model = mujoco.MjModel.from_xml_string(ET.tostring(ref,encoding='unicode'))
        data = mujoco.MjData(model)
        pts,_ = robot.fk(q)
        maximum = 0.
        for t in range(0,len(q),4):
            data.qpos[:] = q[t];mujoco.mj_forward(model,data)
            for j in robot.joint_names:
                jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,j)
                body=int(model.jnt_bodyid[jid]);point=data.xpos[body][[1,2,0]]
                i=robot.names.index(j.replace('_joint','_skel'))
                maximum=max(maximum,float(np.linalg.norm(point-pts[t,i])))
        assert maximum<1e-7
        report['mujoco_fk_reference_max_error_m']=maximum
    else:
        report['mujoco_fk_reference']='mujoco not installed; independent XML math only'
    print(json.dumps(report,indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='run', choices=['run', 'generate', '_worker'])
    parser.add_argument('--run-dir')
    parser.add_argument('--constraints', type=Path)
    parser.add_argument('--prompt', '--prompts', nargs='+')
    parser.add_argument('--stage-durations', nargs='+', type=float)
    parser.add_argument('--duration', type=float, help='Optional total duration; must equal the sum of stage durations.')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--check-only', action='store_true', help='Prepare and validate constraints without model loading or generation.')
    parser.add_argument('--self-test', action='store_true', help='CPU conversion tests only; no model load, GPU or generation.')
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if args.prompt is None:
        args.prompt = list(DEFAULT_PROMPTS)
    if args.stage_durations is None:
        args.stage_durations = [args.duration or 10.] if len(args.prompt)==1 else list(DEFAULT_DURATIONS)
    if len(args.prompt)!=len(args.stage_durations):
        parser.error('Supply one stage duration per prompt.')
    if any(not math.isfinite(d) or d < 6/30 or abs(d*30-round(d*30))>1e-6 for d in args.stage_durations):
        parser.error('Each stage must contain at least 6 frames and a whole number of frames at 30 FPS.')
    total = sum(round(d*30) for d in args.stage_durations)/30
    if not total <= 10 or (args.duration is not None and (not math.isfinite(args.duration) or abs(args.duration-total)>1e-6)):
        parser.error('Total duration must be at most 10 seconds and match --duration if supplied.')
    args.duration = total
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error('seed must be between 0 and 2**32-1.')
    if any(not p.strip() for p in args.prompt):
        parser.error('prompt cannot be empty.')
    if args.action == 'run':
        if args.run_dir:
            parser.error('Normal generation automatically creates a timestamped directory; omit --run-dir.')
        if args.constraints is None:
            parser.error('--constraints is required.')
        if not args.constraints.is_absolute() or not args.constraints.resolve().is_relative_to(ROOT.resolve()) or not args.constraints.is_file():
            parser.error('--constraints must be an existing absolute path inside the repository.')
    elif args.run_dir is None:
        parser.error('--run-dir is required for generate.')
    if args.action != 'run' and (args.constraints is not None or args.check_only):
        parser.error('--constraints and --check-only apply to the normal run command.')
    return args


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    if args.self_test:
        self_test()
        return
    if args.action == 'run':
        run = make_run_dir()
        print('本次输出目录:', run, flush=True)
        prepare_existing(run, args.constraints, args.prompt, args.stage_durations, args.seed)
        if args.check_only:
            print('仅准备和校验完成，未加载模型、未运行生成。', flush=True)
        else:
            generate(run)
        return
    run = Path(args.run_dir)
    if not run.is_absolute() or not run.resolve().is_relative_to((ROOT / 'outputs').resolve()) or not run.is_dir():
        raise ValueError("run-dir must be an existing directory inside the repository's outputs directory.")
    if args.action == '_worker':
        run_worker(run)
    else:
        generate(run)


if __name__ == '__main__':
    main()
