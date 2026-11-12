# 数据来源与方法参考

## 数据

示例和基准使用 **Experiential Technologies** 发布的 **G1 Moves**，数据许可为 CC BY 4.0。固定数据版本为 `895d064b2385725e6aabf540461c219afc3e0916`。下载脚本生成的清单记录各样本的表演者和来源链接。 [数据集](https://huggingface.co/datasets/exptech/g1-moves) · [固定版本清单](https://huggingface.co/datasets/exptech/g1-moves/resolve/895d064b2385725e6aabf540461c219afc3e0916/manifest.json) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

- `J_ShortDance16_JazzWalk.bvh` 是 Jasmine Coro 表演的原始动捕。配套的 `*.refined.bvh` 是本项目的精修结果，MP4 为处理前后对比。 [原始 BVH](https://huggingface.co/datasets/exptech/g1-moves/resolve/895d064b2385725e6aabf540461c219afc3e0916/dance/J_ShortDance16_JazzWalk/capture/J_ShortDance16_JazzWalk.bvh)
- `stance_slide.mp4` 使用 Jasmine Coro 表演的 `B_HandsUp`，对比人工加入脚滑后的动作和精修结果。 [原始 BVH](https://huggingface.co/datasets/exptech/g1-moves/resolve/895d064b2385725e6aabf540461c219afc3e0916/bonus/B_HandsUp/capture/B_HandsUp.bvh)

## 方法

- 足部模块参考 NVIDIA SOMA Retargeter 和 CoRe 的接触检测与足部约束思路，在人体 BVH 上独立实现。 [SOMA Retargeter 接触处理文档](https://github.com/NVIDIA/soma-retargeter/blob/main/documentation/foot-contact.md) · [CoRe 论文](https://ieeexplore.ieee.org/document/11203055/)
- 人体膝屈筛查指标参考了官方 G1 模型的膝关节上限。这个对比用于诊断，实际 G1 关节余量需在重定向后评估。 [Unitree G1 29-DOF mode 15 URDF](https://github.com/unitreerobotics/unitree_ros/blob/725b59b6de39a26c425632120c89f23c49a36f1e/robots/g1_description/g1_29dof_mode_15.urdf)
