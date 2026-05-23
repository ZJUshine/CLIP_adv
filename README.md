# CLIP_adv

针对 OpenAI [CLIP](https://github.com/openai/CLIP) 的定向 **L∞ PGD** 对抗攻击演示。脚本在 CIFAR-10 上选取一张图片，在 [0, 1] 像素空间内迭代生成扰动 δ，使 ‖δ‖∞ ≤ ε，并把 CLIP 的分类结果定向迫近到指定目标类别。每若干步保存原图、对抗样本、δ 可视化和类别概率分布的四联图到 `result/`。

## 算法

L∞ PGD（targeted），Madry et al. 2018 的标准实现：


## 目录结构

```
CLIP_adv/
├── CLIP/           # OpenAI CLIP 源码，作为 git submodule 引入
├── CLIP_adv.py     # 对抗攻击主脚本
├── result/         # 迭代过程与最终结果可视化
└── README.md
```

## 环境要求

- Python ≥ 3.8
- PyTorch（建议 CUDA 版本，CPU 也可运行但很慢）
- `ftfy`, `regex`, `tqdm`, `matplotlib`, `numpy`, `torchvision`

## 克隆与初始化

```bash
git clone --recursive https://github.com/ZJUshine/CLIP_adv.git
cd CLIP_adv
```

## 安装依赖

```bash
pip install torch torchvision ftfy regex tqdm matplotlib numpy
pip install -e ./CLIP        # 安装子模块中的 CLIP 包
```

## 运行

```bash
python CLIP_adv.py
```

首次运行会自动下载 CIFAR-10 数据集和 CLIP `ViT-B/32` 权重。

### 可调参数（位于 `CLIP_adv.py` 顶部）

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `DATASET` | 数据集名称 | `"CIFAR10"` |
| `EPS` | L∞ 扰动预算（[0,1] 像素空间） | `8/255` |
| `ALPHA` | PGD 单步步长 | `2/255` |
| `STEPS` | PGD 迭代步数 | `30` |
| `RANDOM_START` | 是否在 ε-ball 内随机初始化 δ | `True` |
| `TARGETED` | 定向 / 非定向攻击 | `True` |
| `TARGET_LABEL` | 攻击目标类别索引（0–9） | `1` (automobile) |
| `SAMPLE_INDEX` | test set 中被攻击图片的索引 | `1` |
| `SNAPSHOT_EVERY` | 每多少步保存一次过程可视化 | `5` |

## 输出

- `result/adv_{step}.png` —— 2×2 四联图：
  - **左上**：原图（标注 clean 预测类别）
  - **右上**：当前对抗样本（标注当前预测类别）
  - **左下**：`adversarial − original`，δ 经 `(δ/(2ε)) + 0.5` 重缩放到 [0, 1] 的 RGB 显示，灰色像素 = 0 扰动
  - **右下**：original vs. adversarial 在 10 类上的概率分布对比条形图
- `result/result.png` —— 收敛后的最终对比图（原图 | 对抗样本 | 概率分布）
- 终端打印每个 snapshot 步的 loss、p[真类]、p[目标类]，以及最终的实际 L∞ 扰动幅度（应当 ≤ ε）

## 效果示例

![result](result/result.png)

## 许可

本仓库脚本仅用于学习与研究。子模块 `CLIP/` 的版权与许可遵循 OpenAI/CLIP 仓库的 [LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE)。
