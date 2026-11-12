"""Validate dynamic pose anchors and record the targets actually used by sampling."""
import json
from pathlib import Path


def validate_stage_anchors(data, stage_frames):
    boundaries = [0]
    for count in stage_frames:
        boundaries.append(boundaries[-1] + count)
    for item in data:
        enabled = item.get('anchor_to_previous_stage', False)
        if type(enabled) is not bool:
            raise ValueError('anchor_to_previous_stage must be a boolean')
        if not enabled:
            continue
        if item.get('horizontal_free', False):
            raise ValueError('A stage anchor cannot also have horizontal_free enabled')
        ids = item['frame_indices']
        stages = [i for i, (lo, hi) in enumerate(zip(boundaries, boundaries[1:]))
                  if ids and all(lo <= frame < hi for frame in ids)]
        if len(stages) != 1 or stages[0] == 0:
            raise ValueError('Put each previous-stage pose group wholly inside one stage after the first')


def save_effective_constraints(constraints, path):
    records = []
    for constraint in constraints:
        if getattr(constraint, 'anchor_to_previous_stage', False):
            raise ValueError('An unresolved stage anchor remains after sampling')
        record = constraint.get_save_info()
        if hasattr(constraint, 'stage_anchor_resolution'):
            record['anchor_resolution'] = constraint.stage_anchor_resolution
        records.append(record)

    def serializable(value):
        if hasattr(value, 'detach'):
            return value.detach().cpu().tolist()
        if hasattr(value, 'tolist'):
            return value.tolist()
        raise TypeError(type(value).__name__)

    records = json.loads(json.dumps(records, default=serializable))
    with Path(path).open('x') as handle:
        json.dump(records, handle, indent=2)
    return records


def resolve_output_stage_anchors(output_constraints, model_constraints):
    """Move original detailed targets without replacing their omitted joint rotations."""
    if len(output_constraints) != len(model_constraints):
        raise ValueError('Output and model constraint groups must correspond')
    for target, model_target in zip(output_constraints, model_constraints):
        if target.frame_indices.tolist() != model_target.frame_indices.tolist():
            raise ValueError('Output and model constraint frame indices differ')
        if not getattr(target, 'anchor_to_previous_stage', False):
            continue
        resolution = getattr(model_target, 'stage_anchor_resolution', None)
        if resolution is None:
            raise ValueError('The model has not resolved this output pose anchor')
        x, z = resolution['reference_root_xz']
        target.resolve_previous_stage_anchor([x, 0., z], resolution['reference_heading_radians'])
        target.stage_anchor_resolution = dict(resolution)
