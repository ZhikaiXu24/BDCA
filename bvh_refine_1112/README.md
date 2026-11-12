# 面向 Unitree G1 的人体 BVH 动作精修

在动作重定向前，对原始人体动捕做小幅自动修正。程序输入原始 BVH，输出精修后的人体 BVH，保留骨架、帧数、帧率和时长。工具在 CPU 上独立运行。当前支持 G1 Moves 的 51 关节人体采集格式，输入应符合该格式的米制、Y-up 和 YXZ 旋转通道约定。

[测试总览图](results/bvh-benchmark-summary.png) · [中文结果与分项补测](results/summary-zh.md) · [示例对比视频](examples/J_ShortDance16_JazzWalk.mp4)

## 安装和运行

使用 Python 3.10–3.12，在项目目录中创建虚拟环境。

```bash
python -m venv .venv
```

Windows PowerShell 用 `.\.venv\Scripts\Activate.ps1` 激活。Linux 或 macOS 用 `source .venv/bin/activate` 激活。若 PowerShell 禁用了激活脚本，可用 `.\.venv\Scripts\python.exe` 代替下方的 `python`。

```bash
python -m pip install -r requirements.txt
python cli.py examples/J_ShortDance16_JazzWalk.bvh runs/demo
```

第一个参数是输入 BVH，也可以传入包含 BVH 的目录。第二个参数是新的输出目录，程序会在其中生成 `*.refined.bvh`、`metrics.json` 和 `metrics.csv`。结果目录需尚未存在。若要生成对比 MP4，可加 `--video`，并在 PATH 中配置带 `libx264` 的 FFmpeg。

## 如何调节修正强度

当前通过命令行和 JSON 配置操作，没有图形界面，无需修改 Python 源码。

**只改整体强度**时，在运行命令末尾加 `--strength`。例如，把整体强度设为 70。

```bash
python cli.py examples/J_ShortDance16_JazzWalk.bvh runs/demo-70 --strength 70
```

全局范围为 0–100，默认 50。设为 0 时，输出 BVH 与原文件逐字节一致。

**单独调整某一项**时，可以复制 `config.json` 后修改，也可以新建只包含所需参数的 `tuning.json`。例如，将足部强度提高到 90，关闭躯干修正，其余项使用默认值。

```json
{
  "foot": {"strength": 90},
  "torso": {"enabled": false}
}
```

```bash
python cli.py examples/J_ShortDance16_JazzWalk.bvh runs/demo-tuned --config tuning.json
```

| 分项 | 默认强度和开关 | 处理的问题 |
| --- | --- | --- |
| 足部 `foot` | 80，开启 | 支撑脚滑动及部分近地穿地、轻微悬空 |
| 平滑 `smooth` | 80，开启 | 关节和根节点的短时高频抖动 |
| 躯干 `torso` | 40，开启 | 支撑期的短时异常倾斜 |
| 速度 `speed` | 40，关闭 | 孤立的关节角速度尖峰 |

每项的 `strength` 都是 0–100。全局强度和分项强度各除以 100，再相乘，作为该项的修正系数。例如整体 50、足部 80，足部修正系数为 0.4。`enabled: false` 会关闭该项。命令行 `--strength` 优先于 JSON 中的全局 `strength`。

`floor_m` 用于指定地面高度，单位为米。保留 `null` 时程序从输入动作估计地面。`contact` 里的参数分别控制离地高度、水平速度、垂直速度和最短支撑时间。距离用米，速度用米每秒，时间用秒。

足部修正采用先识别支撑、再约束足部位置的思路。四项修正规则均在人体 BVH 上执行。 [方法参考与数据许可](THIRD_PARTY_NOTICES.md)

## 测试与复现

- **真实样本**直接处理 12 段未额外注入异常的 G1 Moves BVH，覆盖舞蹈步伐、转身和空手道等动作。使用默认配置，前后以原动作推断出的同一组支撑区间和地面为参照。
- **人工异常**在原始动作的副本上，由脚本定量加入脚滑、关节或根节点抖动等 7 项异常。每项单独开启对应模块，在异常区间内比较精修前后与原始动作的均方根误差（RMS）。全局强度为 50，速度尖峰测试将速度分项设为 100。
- **强度对照**对 3 段真实动作比较整体强度 0、50 和 100。另对 7 项人工异常逐个测试对应模块的分项强度 0、25、50 和 100。

需要可联网的环境和 `curl`。脚本只下载固定版本的 12 段 BVH 和数据清单，约 30 MB，并校验 SHA-256。已有依赖时，在项目目录执行。

```bash
python fetch_samples.py
python -m unittest -v
python benchmark.py --output runs/benchmark --skip-video
python benchmark_strength.py --output runs/strength_sweep.csv
```

基准会在 `runs/benchmark/` 生成逐样本结果和报告，分项强度补测会写入 `runs/strength_sweep.csv`。移除 `--skip-video` 可同时生成对比视频。下载的 `data/` 和生成的 `runs/` 已设置为 Git 忽略。

默认强度下，12 段真实动作的支撑脚平均滑动速度和较大的运动突变指标（Jerk P95）均下降，逐段降幅的中位数分别为 **13.6%** 和 **6.7%**。按任一身体关节的局部转角变化超过 3° 或位置变化超过 20 mm 作为明显改动的标准，**98.0%** 的帧未超过阈值。人工脚滑 RMS 误差下降 **78.3%**，足部下沉和躯干异常的恢复幅度较小。 [逐段基准报告（英文）](results/benchmark.md) · [真实样本指标 CSV](results/real_metrics.csv) · [分项强度 CSV](results/strength_sweep.csv)

当前评估的是人体动作中与稳定性相关的运动学指标。机器人质心和支撑关系、持续的大幅动作及硬件执行能力需在重定向后另行验证。本项目尚未进行 G1 动力学仿真或真机测试。
