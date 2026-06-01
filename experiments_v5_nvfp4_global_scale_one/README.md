# NVFP4 `global_scale = 1` 实验（v4 的后续）

## TL;DR

把 `input_global_scale`（记作 `g`）硬编码为 **1** 后，**会不会让 v4 研究的那些分布产生更大的量化误差？**

答案：**会，但误差只集中在「最大的那一档 outlier」上，而且其它档基本不变、甚至更好**。

- **非 outlier 与 ±10 档：完全不变**（MSE 比值 = 1.00）。`g=1` 对它们毫无影响。
- **中档 outlier（±60、±110）：`g=1` 反而更好**（MSE 降到 g_optimal 的 0.44× / 0.45×）。
- **大档 outlier（±230、±350、±410）：`g=1` 明显更差**，越靠近 tensor absmax 越糟。
- **最大的 ±410 档：`g=1` 让 MSE 暴涨 78 倍**（6.25 → 485.2）。

根本原因：`g_optimal = 2688 / absmax` 的设计目的就是把**决定 absmax 的那个 group** 精确放到 FP8 的干净落点 **448**（ε ≈ 0）。`g=1` 等于丢掉这份"保护"，让最大的 outlier 从 448 滑到 s·g ≈ 68 的 FP8 中段（stride 8、相对误差 ~6%），重建误差从 ±5（FP4 量化地板）放大到 ±22。

---

## 1. 这个实验和 v4 的关系

v4 扫描 `gs_used = gs_optimal / K`，`K ∈ [1, 4]`。本实验问的是一个不同的点：

> **如果不做校准、直接把 `g` 钉死为 1，会怎样？**

注意：在 v4 的 K 语言里，`g = 1` 对应

```
K = gs_optimal / gs_used = gs_optimal / 1 = gs_optimal ≈ 6.52
```

也就是说 **`g=1` 落在 v4 扫描区间 [1, 4] 之外**，是一个全新的工作点。数据分布与 v4 **完全一致**（同样的 seed、同样的 6 档 outlier），只换 `g`。

---

## 2. 结果（核心表）

`absmax = 412.25`，`gs_optimal = 6.5203`。

| 档位 | MSE @ g_optimal | MSE @ g=1 | 比值 (g=1 / opt) |
|---|---|---|---|
| non-outlier | 0.0600 | 0.0601 | **1.00** |
| ±10  | 33.80 | 33.79 | **1.00** |
| ±60  | 2.35  | 1.04  | **0.44**（更好）|
| ±110 | 16.90 | 7.55  | **0.45**（更好）|
| ±230 | 31.70 | 100.80 | 3.18 |
| ±350 | 12.32 | 100.96 | 8.19 |
| ±410 | 6.25  | **485.25** | **77.66** |

block_scale 退化情况：`g=1` 下 subnormal / zero / clipped 全为 0；`g_optimal` 下有 1.17% 的 block_scale 触顶 448（正是 absmax group 被放到 448 的结果）。**两边都没有 FP8 下溢**——失效机制依旧是「上端粗粒度舍入」，与 v4 的 (F5) 一致。

---

## 3. 三个关键现象

### (现象 1) `g=1` 不伤害小信号

非 outlier 的理想 block_scale 在 g=1 下是 `s·g = (2.5/6)·1 ≈ 0.42`，仍稳稳落在 FP8 精细区（stride ≈ 0.03，相对误差 ~4%），和 g_optimal 下的 2.7 处于同一精度档。所以 MSE 一模一样（0.06）。**「把 g 调小会让普通激活崩掉」是错的**——只要不进入 subnormal（需要 K > 100 量级），FP8 是浮点、各档相对精度近似恒定。

### (现象 2) 改 g = 在 FP8 网格对齐的"抽奖"里换一个号

`g` 是一个对所有 group 共享的乘子，改变 `g` 只是把每个 group 的 `s·g` **整体平移**到 FP8 数轴的另一段。由于 FP8 在 normal 区是分段恒定相对精度，平移后误差量级**不会系统性变大**，只会因为「这一次落在网格点附近还是网格点中间」而上下震荡。

报告图 (b) 把这点画得很清楚：`g=1`（红竖线）落在各档误差剧烈震荡带的**中间位置**，既不是特别好的点也不是特别坏的点。它对 ±60/±110 恰好抽中好号（更好），对 ±230/±350 抽中差号（更差）。

### (现象 3) `g_optimal` 唯一真正"做了功"的地方，就是保护最大 outlier

`g_optimal = 2688 / absmax` 是**专门**让决定 absmax 的那个 group（这里就是 ±410 档）精确命中 FP8 的 448——一个完全可表示的干净点，ε ≈ 0。这是一次"必中"的人为对齐。

`g=1` 把这次必中扔掉：±410 档的理想 block_scale 变成 `410/6 ≈ 68.4`，落在 FP8 的 [64, 72] 之间（stride = 8），相对舍入误差 ~6%。报告图 (d) 直接对比了最大档的重建误差分布：

- `g_optimal`：误差集中在 ±5（这是 FP4 4-bit 本身的量化地板，已经没法更好）
- `g=1`：误差被推到 ±22 附近 → MSE 从 6.25 跳到 485

越靠近 absmax 的 outlier，被 g_optimal 保护得越多，因此被 `g=1` 伤得越狠（±230 → 3×，±350 → 8×，±410 → 78×）。

---

## 4. 直接回答你的问题

> "假设把 global scale 固定为 1，这些 data distribution 会有更大的量化误差吗？"

**整体（tensor 级）MSE 会变大**——因为 tensor MSE 由最大档主导，而最大档恰恰是 `g=1` 损失最惨的（78×）。

但更准确的结论是：

1. **误差不是均匀变大，而是只在「量级最接近 absmax 的 outlier」上爆炸**。这与 v4 的核心结论一脉相承：`g` 只对 outlier 起作用，尤其是理想 block_scale 落在 FP8 粗粒度区的那些。
2. **普通激活、以及远离 absmax 的小 outlier 完全不受影响**，部分中档 outlier 甚至因为偶然的网格对齐而变好。
3. **`g_optimal` 的价值不在于"普遍降误差"，而在于"用一次确定性的网格命中保护掉最危险的那一档 outlier"**。`g=1` 就是放弃这次命中。

实践含义：如果你的 tensor 有一个明显支配 absmax 的 outlier 量级，**绝不能用 `g=1`**——必须用校准得到的 `g_optimal` 把它钉在 448。反过来，如果 absmax 本身不是离群的庞然大物（分布相对均匀），`g=1` 与 `g_optimal` 的差距会小很多。

---

## 5. 文件清单

```
experiments_v5_nvfp4_global_scale_one/
├── README.md      # 本文档
├── run.py         # 自包含实验脚本（复用 v4 的同一分布）
├── results.csv    # g_optimal / g=1 两个 headline 行 + 一条 g 的几何扫描
└── report.png     # 4 面板报告：
                   #   (a) 各档 MSE 柱状对比 g_optimal vs g=1
                   #   (b) 各档 MSE 随固定 g 的 landscape（标注 g=1 与 g_optimal）
                   #   (c) 各档理想 block_scale 在 FP8 网格上的落点对比
                   #   (d) 最大档 ±410 的重建误差分布对比
```

运行：

```bash
python experiments_v5_nvfp4_global_scale_one/run.py
```

---

## 6. 关键代码引用

- `src/compressed_tensors/quantization/utils/helpers.py:309` — `generate_gparam(min, max)` = `2688 / absmax`
- `src/compressed_tensors/quantization/utils/helpers.py:140` — `compute_dynamic_scales_and_zp`（group-wise block_scale）
- `src/compressed_tensors/quantization/lifecycle/forward.py:149` — `fake_quantize` 端到端 quantize+dequantize
