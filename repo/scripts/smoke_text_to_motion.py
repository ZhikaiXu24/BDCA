"""Real text-to-motion smoke test using verified local Llama and official adapters."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=("Kimodo-SOMA-RP-v1.2", "Kimodo-G1-RP-v1"))
parser.add_argument("--output", required=True)
parser.add_argument("--prompt", default="A person walks forward.")
parser.add_argument("--duration", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
stem = Path(args.output)
assert stem.is_absolute() and stem.resolve().is_relative_to((root / "outputs").resolve())
paths = {suffix: Path(str(stem) + suffix) for suffix in (".npz", ".json", ".csv", ".bvh")}
assert not any(path.exists() for path in paths.values()), "Output exists; select a fresh output stem."
physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if not physical_gpu.isdecimal():
    raise ValueError("Set CUDA_VISIBLE_DEVICES to one physical GPU index.")
os.environ.setdefault("HF_HUB_CACHE", str(root / "hf_cache"))

import numpy as np
import torch
from torch import nn
from kimodo import load_model
from bvh_export_20261029 import check_bvh_dependencies, export_soma_bvh
if "SOMA" in args.model:
    check_bvh_dependencies()
from kimodo.sanitize import sanitize_text
args.prompt = sanitize_text(args.prompt)
assert args.prompt.strip()
from kimodo.exports.motion_io import save_kimodo_npz
torch.set_num_threads(2)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
print(f"Physical GPU {physical_gpu}: {torch.cuda.get_device_name(0)}; free {free_bytes / 2**30:.2f} GiB", flush=True)
if free_bytes < 24 * 2**30:
    raise RuntimeError("The selected GPU needs at least 24 GiB free.")

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging
transformers_logging.disable_progress_bar()
from peft import PeftModel
from kimodo.model.llm2vec.llm2vec import LLM2Vec
from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel

base_path = os.environ.get("KIMODO_LLAMA_DIR")
if not base_path or not Path(base_path).is_dir():
    raise ValueError("Set KIMODO_LLAMA_DIR to the local Meta-Llama-3-8B-Instruct directory.")
adapters = [
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "31474e395ada192e8ed1586db6be79fb3b70c9c0"),
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"),
]
adapter_paths = [snapshot_download(repo, revision=rev, local_files_only=True, allow_patterns=["adapter_config.json", "adapter_model.safetensors", "tokenizer*.json", "special_tokens_map.json", "config.json"]) for repo, rev in adapters]
print("Loading local BF16 Llama with official bidirectional LLM2Vec implementation.", flush=True)
encoder_started = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(adapter_paths[0], local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
llama = LlamaBiModel.from_pretrained(base_path, local_files_only=True, torch_dtype=torch.bfloat16,
                                    device_map="cuda:0", attn_implementation="sdpa")
assert len(llama.layers) == 32 and all(not layer.self_attn.is_causal for layer in llama.layers)
for adapter_path in adapter_paths:
    print("Applying adapter", adapter_path, flush=True)
    llama = PeftModel.from_pretrained(llama, adapter_path, local_files_only=True).merge_and_unload()
llama.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
llama.config.use_cache = False
# Transformers 5.1 builds masks in forward(), so its forward never calls the
# older LLM2Vec _update_causal_mask override. Supply the equivalent bidirectional
# additive mask explicitly; a 4D mask passes through create_causal_mask unchanged.
bidirectional_checks = []
def supply_bidirectional_mask(module, positional, keyword):
    mask = keyword.get("attention_mask")
    if mask is not None and mask.ndim == 2:
        batch, length = mask.shape
        dtype = next(module.parameters()).dtype
        additive = torch.zeros((batch, 1, length, length), dtype=dtype, device=mask.device)
        additive.masked_fill_(~mask[:, None, None, :].bool(), torch.finfo(dtype).min)
        keyword = dict(keyword, attention_mask=additive, use_cache=False)
    return positional, keyword
def verify_attention_mask(module, positional, keyword):
    mask = keyword.get("attention_mask")
    assert mask is not None and mask.ndim == 4
    # These are unpadded single examples. Earlier tokens must attend to later tokens.
    assert mask.shape[-1] > 2 and bool((mask[..., 0, -1] == 0).all())
    bidirectional_checks.append(True)
llama.register_forward_pre_hook(supply_bidirectional_mask, with_kwargs=True)
llama.layers[0].self_attn.register_forward_pre_hook(verify_attention_mask, with_kwargs=True)
real_encoder = LLM2VecEncoder.__new__(LLM2VecEncoder)
real_encoder.llm_dim = 4096
real_encoder._device = "cuda:0"
real_encoder.model = LLM2Vec(llama, tokenizer).eval()
for parameter in real_encoder.model.parameters():
    parameter.requires_grad = False
assert real_encoder.model.prepare_for_tokenization(args.prompt).startswith("<|start_header_id|>user")
comparison_prompt = "A person stands still." if args.prompt.casefold() != "a person stands still." else "A person walks forward."
with torch.inference_mode():
    features, lengths = real_encoder([args.prompt, comparison_prompt])
assert len(bidirectional_checks) == 2
assert features.shape == (2, 1, 4096) and lengths == [1, 1]
assert torch.isfinite(features).all() and torch.linalg.vector_norm(features[0]) > 0
feature_difference = float(torch.linalg.vector_norm(features[0] - features[1]))
assert feature_difference > 1e-3, "Different text produced identical embeddings."
text_encoder_seconds = time.monotonic() - encoder_started
encoder_peak_GiB = torch.cuda.max_memory_allocated(0) / 2**30
print("REAL_TEXT_FEATURES", {"shape": list(features.shape), "lengths": lengths,
      "different_prompt_feature_distance": feature_difference,
      "encoder_seconds": text_encoder_seconds}, flush=True)

class CachedRealTextEncoder(nn.Module):
    """Serve genuine LLM2Vec features for the exact tested prompt, without fake or zero text."""
    def __init__(self, prompt, feature):
        super().__init__()
        self.prompt = prompt
        self.register_buffer("feature", feature.detach().clone())
        self.calls = 0
    def forward(self, texts):
        assert all(text.strip() == self.prompt.strip() for text in texts), texts
        self.calls += 1
        return self.feature.expand(len(texts), -1, -1), [1] * len(texts)

encoder = CachedRealTextEncoder(args.prompt, features[:1])
# Only the genuine features are needed for diffusion. Free Llama GPU storage before motion sampling.
del real_encoder, llama, features
__import__("gc").collect()
torch.cuda.empty_cache()

print(f"Loading {args.model} with genuine cached LLM2Vec text features.", flush=True)
os.environ["LOCAL_CACHE"] = "True"
load_started = time.monotonic()
model = load_model(args.model, device="cuda:0", text_encoder=encoder)
load_seconds = time.monotonic() - load_started
fps = float(model.fps)
duration = args.duration
assert 0 < duration <= 10
frames = int(duration * fps)
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.reset_peak_memory_stats(0)
post_processing = "SOMA" in args.model

def timeout_handler(signum, frame):
    raise TimeoutError("Inference exceeded the 240 second short-test budget.")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(240)
started = time.monotonic()
try:
    with torch.inference_mode():
        output = model(
            args.prompt, num_frames=frames, num_samples=1, num_denoising_steps=100,
            cfg_type="separated", cfg_weight=[2.0, 2.0], constraint_lst=[], post_processing=post_processing,
            return_numpy=True,
        )
    torch.cuda.synchronize()
finally:
    signal.alarm(0)
inference_seconds = time.monotonic() - started
# Keep the batch dimension through official post-processing, then unwrap one sample.
output = {key: np.asarray(value)[0] for key, value in output.items()}
assert encoder.calls == 1
joints = np.asarray(output["posed_joints"])
assert joints.ndim == 3 and joints.shape[0] == frames and joints.shape[-1] == 3
joint_count = int(joints.shape[1])
assert joint_count == (77 if "SOMA" in args.model else 34)
for key in ("global_rot_mats", "local_rot_mats"):
    assert output[key].shape == (frames, joint_count, 3, 3)
for key in ("smooth_root_pos", "root_positions"):
    assert output[key].shape == (frames, 3)
assert output["foot_contacts"].shape == (frames, 6 if "SOMA" in args.model else 4)
assert output["global_root_heading"].shape == (frames, 2)
arrays = {}
for key, value in output.items():
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.number):
        assert np.isfinite(value).all(), f"Non-finite values in {key}"
    arrays[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
max_joint_step_m = float(np.abs(np.diff(joints, axis=0)).max())
assert max_joint_step_m > 1e-7, "Generated joints are constant."
rotations = np.asarray(output["local_rot_mats"])
orthogonality_error = float(np.max(np.abs(rotations @ np.swapaxes(rotations, -1, -2) - np.eye(3))))
determinants = np.linalg.det(rotations)
assert orthogonality_error < 0.02
assert determinants.min() > 0.98 and determinants.max() < 1.02
skeleton = model.output_skeleton
names = list(skeleton.bone_order_names)
parents = skeleton.joint_parents.detach().cpu().numpy().astype(int).tolist()
assert len(names) == len(parents) == joint_count

csv_shape = None
if "G1" in args.model:
    from kimodo.exports.mujoco import MujocoQposConverter
    converter = MujocoQposConverter(model.skeleton)
    batched = {key: np.asarray(value)[None] for key, value in output.items()}
    qpos = converter.dict_to_qpos(batched, "cuda:0")
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    qpos = np.asarray(qpos).reshape(frames, 36)
    assert np.isfinite(qpos).all()
    csv_shape = list(qpos.shape)

stem.parent.mkdir(parents=True, exist_ok=True)
output["fps"] = np.asarray(fps)
output["conditioning"] = np.asarray("real_text_llama3_llm2vec_no_motion_constraints")
output["prompt"] = np.asarray(args.prompt)
with paths[".npz"].open("xb") as handle:
    save_kimodo_npz(handle, output)
if csv_shape is not None:
    with paths[".csv"].open("x") as handle:
        converter.save_csv(qpos, handle)

# Reopen actual saved files, rather than relying only on in-memory output.
with np.load(paths[".npz"], allow_pickle=False) as saved:
    assert saved["posed_joints"].shape == (frames, joint_count, 3)
    for key in arrays:
        assert list(saved[key].shape) == arrays[key]["shape"]
        if np.issubdtype(saved[key].dtype, np.number):
            assert np.isfinite(saved[key]).all()
if csv_shape is not None:
    saved_csv = np.loadtxt(paths[".csv"], delimiter=",")
    assert list(saved_csv.shape) == csv_shape and np.isfinite(saved_csv).all()
# Export directly from generated arrays: string NPZ metadata never enters Torch.
bvh_path = None
if "SOMA" in args.model:
    print("导出 BVH（CPU）并回读检查帧数、帧率和通道。", flush=True)
    bvh_path = export_soma_bvh(output, paths[".bvh"], fps)
report = {
    "status": "passed", "model": args.model,
    "conditioning": "real Llama3 plus MNTP then supervised LLM2Vec; nonzero text features; no motion constraints",
    "prompt": args.prompt, "not_a_text_to_walk_test": False,
    "text_encoder": "official LLM2VecEncoder with locally loaded Llama3", "encoder_calls": encoder.calls,
    "base_model_path": base_path, "adapters": adapters,
    "bidirectional_mask_verified_calls": len(bidirectional_checks),
    "transformers_5_mask_compatibility": "explicit 4D noncausal mask via helper pre-hook; official source unchanged", "text_encoder_seconds": text_encoder_seconds, "text_encoder_peak_allocated_GiB": encoder_peak_GiB,
    "different_prompt_feature_distance": feature_difference, "comparison_prompt": comparison_prompt, "text_feature_lengths": [1],
    "root_displacement_m": float(np.linalg.norm(joints[-1,0] - joints[0,0])),
    "max_joint_displacement_from_start_m": float(np.linalg.norm(joints-joints[:1],axis=-1).max()),
    "physical_gpu": int(physical_gpu), "visible_gpu": "cuda:0", "gpu_name": torch.cuda.get_device_name(0),
    "cuda_available": True, "python": __import__("sys").version.split()[0],
    "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
    "fps": fps, "duration_seconds": duration, "frames": frames, "joints": joint_count,
    "seed": seed, "diffusion_steps": 100, "cfg_type": "separated", "cfg_weight": [2.0, 2.0],
    "post_processing": post_processing,
    "model_load_and_download_seconds": load_seconds, "inference_seconds": inference_seconds,
    "peak_allocated_GiB": torch.cuda.max_memory_allocated(0) / 2**30,
    "peak_reserved_GiB": torch.cuda.max_memory_reserved(0) / 2**30,
    "max_joint_component_step_m": max_joint_step_m,
    "local_rotation_orthogonality_max_error": orthogonality_error,
    "local_rotation_determinant_range": [float(determinants.min()), float(determinants.max())],
    "joint_names": names, "joint_parents": parents, "arrays": arrays,
    "npz_path": str(paths[".npz"]), "bvh_path": bvh_path,
    "csv_path": str(paths[".csv"]) if csv_shape is not None else None, "csv_shape": csv_shape,
    "hf_cache": os.environ["HF_HUB_CACHE"],
    "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "tracked_source_clean": not subprocess.check_output(["git", "-C", str(root), "diff", "--name-only"], text=True).strip(),
}
with paths[".json"].open("x") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps({key: value for key, value in report.items() if key not in ("joint_names", "joint_parents")}, indent=2), flush=True)
print("SAVED_OUTPUTS_REOPENED_AND_VERIFIED", flush=True)
