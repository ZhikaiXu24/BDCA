# GEM-X 与 Kimodo G1

这个项目将单张图片转换为 Unitree G1 的全身关键帧约束，再用 Kimodo 按文本和关键帧生成连续动作。GEM-X 负责估计人体姿态，SOMA Retargeter 将姿态映射到 G1，Kimodo 生成带前后过渡的轨迹。

| 路径 | 内容 |
| --- | --- |
| `repo/kimodo/` | Kimodo 模型、约束加载、动作生成与导出 |
| `repo/GEM-X_lbq/` | GEM-X 推理及照片姿态到 G1 约束的转换 |
| `repo/scripts/` | 三段约束生成与预览 |
| `repo/lbq_constraint/` | 原始照片约束与最终运行使用的辅助约束 |
| `repo/examples/lbq_tree_pose/` | 最终三段版的 CSV、BVH、NPZ、MP4 和解析后的约束 |

## 安装

从项目根目录执行。GEM-X 环境使用 Python 3.12、PyTorch 2.10+ 和 `transformers==5.3.0`。Kimodo 环境使用 Python 3.10、PyTorch 2.7.1 和 `transformers==5.1.0`。两边都读取同一份 `requirements.txt`，文件根据 Python 版本选择依赖。分别激活下列环境后执行对应命令。GPU 运行前请根据驱动安装相应的 PyTorch CUDA 构建，历史 Kimodo 运行使用 CUDA 11.8 构建。

```bash
# GEM-X 环境（Python 3.12）
python -m pip install -r requirements.txt
python -m pip install -e './repo/GEM-X_lbq[g1,kimodo,video]'

# Kimodo 环境（Python 3.10）
python -m pip install -r requirements.txt
export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
python -m pip install -e ./repo
```

Kimodo 生成需要 Linux、C++ 编译器、CUDA 和 `nvidia-smi`，预览还需要 `ffmpeg` 和 `ffprobe`。仅从已有 JSON 生成动作时，可从 Kimodo 环境开始。若要从新照片制作约束，还需要独立的 SOMA Retargeter 环境。请在另一个 Python 3.12 环境中执行以下命令，系统需有 Git LFS。

```bash
RETARGETER="$PWD/repo/GEM-X_lbq/third_party/soma-retargeter"
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/NVIDIA/soma-retargeter.git "$RETARGETER"
GIT_LFS_SKIP_SMUDGE=1 git -C "$RETARGETER" checkout cbdb2b494a2af82137bf9b6f15122d60167281c4
git -C "$RETARGETER" lfs pull --include="soma_retargeter/assets/robotics/unitree/g1/**,soma_retargeter/assets/soma/soma_zero_frame0.bvh"
python -m pip install -e "$RETARGETER"
```

## 模型与权重

下载后使用本地路径或 Hugging Face Hub 缓存。各模型、资源和第三方组件的许可与访问条件请以其发布页面为准。

| 流程 | 模型和资源 | 获取与配置 |
| --- | --- | --- |
| 图片姿态 | GEM-X Body Pose `body_s240000.ckpt`、YOLOX HumanArt `yolox_humanart.onnx` | [GEM-X 权重下载说明](repo/GEM-X_lbq/docs/PRETRAINED_MODELS.md#archive-model-downloads) |
| 人体几何 | 与 `py-soma-x==0.3.3` 匹配的 SOMA 资源 | [GEM-X 的 SOMA 资源配置](repo/GEM-X_lbq/docs/INSTALL.md#soma-code-and-assets) |
| 动作生成 | `nvidia/Kimodo-G1-RP-v1` | [Kimodo-G1-RP-v1](https://huggingface.co/nvidia/Kimodo-G1-RP-v1)，预先放入 Hub 缓存 |
| 文本编码 | `Meta-Llama-3-8B-Instruct` | [Meta Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)，取得访问权限后将本地权重目录设置为 `KIMODO_LLAMA_DIR` |
| LLM2Vec 第 1 个适配器 | `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` | [固定版本](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/tree/31474e395ada192e8ed1586db6be79fb3b70c9c0) |
| LLM2Vec 第 2 个适配器 | `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised` | [固定版本](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/tree/baa8ebf04a1c2500e61288e7dad65e8ae42601a7) |

在联网环境中可用以下命令准备 Kimodo 和适配器缓存。生成脚本离线读取缓存，并按上表顺序加载两个适配器。

```bash
export HF_HUB_CACHE="$PWD/hf_cache"
hf download nvidia/Kimodo-G1-RP-v1
hf download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
  --revision 31474e395ada192e8ed1586db6be79fb3b70c9c0
hf download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
  --revision baa8ebf04a1c2500e61288e7dad65e8ae42601a7
```

## 图片转成 G1 约束

在 GEM-X 环境执行以下命令，替换图片、模型和解释器路径。默认第 0 帧，`--frame-index 180` 将照片约束安排在 30 FPS 时间轴的第 6 秒。

```bash
mkdir -p repo/outputs/photo
gem-infer pose-image --task body --input /path/to/image.jpg \
  --release body --checkpoint /path/to/body_s240000.ckpt \
  --body-detector /path/to/yolox_humanart.onnx \
  --soma-assets /path/to/soma-assets --device cuda --output repo/outputs/photo/body.pt

gem-convert-pose --target g1 --input repo/outputs/photo/body.pt \
  --soma-assets /path/to/soma-assets --retarget-python /path/to/retarget-env/bin/python \
  --output repo/outputs/photo/kimodo_g1_fullbody.json --frame-index 180 --validate
```

## 用约束和文本生成动作

最终样例保留原始照片的树式姿态，并用辅助站立帧把它接到前后的动作。以下命令从项目根目录在 Kimodo 环境执行，生成总长 10 秒、30 FPS 的三段动作。替换 GPU 编号和本地 Llama 路径，选择至少有 24 GiB 可用显存的 GPU。

```bash
export HF_HUB_CACHE="$PWD/hf_cache"
export KIMODO_LLAMA_DIR=/path/to/Meta-Llama-3-8B-Instruct
export CUDA_VISIBLE_DEVICES=0

python repo/scripts/g1_keyframes_20261030.py \
  --constraints "$PWD/repo/lbq_constraint/kimodo_g1_tree_walk_stand_assisted.json" \
  --prompts \
    "A person walks forward for several steps and stops." \
    "A person stands on the right leg in a tree pose with both hands near the shoulders." \
    "A person lowers the left foot and stands upright on both feet." \
  --stage-durations 4 3.5 2.5 --duration 10 --seed 1234
```

加 `--check-only` 可在 CPU 上检查约束。新结果写到 `repo/outputs/g1_keyframes/`。使用新照片的单个 JSON 时，可直接将 `--constraints` 指向该文件，用 `--prompt` 指定一段描述，并设定 `--duration`。如需复现前进、树式、恢复站立三个阶段，使用仓库提供的辅助约束。

[最终样例](repo/examples/lbq_tree_pose/) 为 300 帧。CSV 每帧保存 3 个根位置分量（米）、4 个 WXYZ 四元数分量和 29 个关节角（弧度）。历史预览确认了三个阶段的顺序，第 6 秒的关节位置误差均值约 26 mm。该轨迹尚未经过真实机器人动力学验证。

## 许可与署名

[Kimodo 许可](repo/LICENSE) · [Kimodo 署名](repo/ATTRIBUTIONS.MD) · [GEM-X 许可](repo/GEM-X_lbq/LICENSE) · [GEM-X 署名与文件例外](repo/GEM-X_lbq/ATTRIBUTIONS.md)
