# CLIP_adv

针对 OpenAI [CLIP](https://github.com/openai/CLIP) 的定向 **L∞ PGD** 对抗攻击演示。脚本在 CIFAR-10 上选取一张图片，在 [0, 1] 像素空间内迭代生成扰动 δ，使 ‖δ‖∞ ≤ ε，并把 CLIP 的分类结果定向迫近到指定目标类别。每若干步保存原图、对抗样本、扰动热力图三联可视化到 `result/`。

## 算法

L∞ PGD（targeted），Madry et al. 2018 的标准实现：

```
δ ← Uniform(-ε, ε)                                  # random start
for t = 1 .. T:
    g  ← ∇_δ  CE( f(clip01(x + δ)),  y_target )     # f 内部自带 CLIP 归一化
    δ  ← δ − α · sign(g)                            # targeted: 反梯度
    δ  ← clip_{L∞≤ε}(δ)                             # 投影回 ε-ball
    δ  ← clip01(x + δ) − x                          # 保持像素 ∈ [0,1]
return x + δ
```

关键实现要点：
- ε 预算施加在 **[0, 1] 像素空间**，CLIP 的 mean/std 归一化封装在 forward 内，避免常见的"在归一化空间设 ε"的语义错误。
- 损失使用 `F.cross_entropy(logits, target_label)`（logits + 类别索引），而非旧版本里把 softmax 概率喂给 CE 的写法。

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

由于 `CLIP/` 是子模块，克隆时需要带上 `--recursive`：

```bash
git clone --recursive https://github.com/<your-name>/CLIP_adv.git
cd CLIP_adv
```

如果已经克隆但忘了 `--recursive`，可以补拉子模块：

```bash
git submodule update --init --recursive
```

更新子模块到上游最新提交：

```bash
git submodule update --remote CLIP
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
| `SNAPSHOT_EVERY` | 每多少步保存一次三联可视化 | `5` |

## 输出

- `result/adv_{step}.png` —— [原图 | 当前对抗样本 | `|δ| / ε` 热力图] 三联图
- `result/result.png` —— 最终对比图：原图预测 vs. 对抗样本预测 vs. 类别概率分布柱状图
- 终端打印每个 snapshot 步的 loss、p[真类]、p[目标类]，以及最终的实际 L∞ 扰动幅度（应当 ≤ ε）

## 效果示例

![result](result/result.png)

## 许可

本仓库脚本仅用于学习与研究。子模块 `CLIP/` 的版权与许可遵循 OpenAI/CLIP 仓库的 [LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE)。
