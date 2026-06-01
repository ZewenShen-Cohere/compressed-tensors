# NVFP4 `input_global_scale` 敏感度研究

## TL;DR

NVFP4 的 `input_global_scale`（下文记作 `g`）本身**不直接量化任何数据**。它的唯一作用是决定每个 group 的 block_scale 在 FP8 E4M3 数轴上的"落点"。当 `g` 偏离最优值时，重建误差**完全集中在大 outlier channel 上**，非 outlier channel 几乎不受影响；且误差随 `g` 的变化是**非单调震荡**的，由 FP8 grid 的离散落点决定，而非简单的"偏差越大越差"。

实验在 `[128, 4096]` 的合成 activation 上扫描 `gs_used = gs_optimal / K`，`K ∈ {1.0, 1.2, ..., 4.0}`，并加入 6 档不同量级（±10, ±60, ±110, ±230, ±350, ±410）的 outlier channel，定量验证了上述结论。

---

## 1. 背景

NVFP4 采用**两级缩放**结构：
- **FP4 E2M1** 存储 4-bit 数据值
- **FP8 E4M3** 存储 per-group block_scale（每 group 一个）
- **FP32** 存储 per-tensor `input_global_scale` `g`（整个 tensor 一个）

部署时 `g` 通常基于**校准集**离线得到，部署时不变。运行时实际激活的分布若与校准集偏离，`g` 就不再是当前 batch 的最优值。本实验的目标是回答：

> "如果 `g` 比最优值小 K 倍（即校准低估了 absmax），重建质量受多大影响？影响落在哪些 channel 上？"

---

## 2. 数学回顾：g 在量化流程里到底干了什么

### 2.1 一个 group 的"真实 scale"

记某个 group 内绝对值最大的数为 `m`，则量化器需要的 effective scale 是

```
s = m / 6        （FP4 E2M1 的最大可表示值 = 6）
```

`s` 是一个实数 — 没有任何精度限制，跟 `g` 无关。

### 2.2 NVFP4 把 s 拆成两层

NVFP4 不直接存 `s`，而是把 `s` 拆分为

```
s = s_FP8_stored / g
```

其中 `s_FP8_stored = round_FP8(s · g)` 用 FP8 E4M3 存储。`g` 是一个**所有 group 共享**的 FP32 标量。

### 2.3 dequantize 公式

```
x_hat = q_FP4 × (s_FP8_stored / g)
```

FP4 grid 看到的"标尺"就是 `s_FP8_stored / g`。

### 2.4 关键洞察：g 自己不参与量化

**假想情形**：FP8 是无穷精度，即 `s_FP8_stored = s · g` 精确成立。则

```
s_FP8_stored / g = (s · g) / g = s
```

g 完美抵消，FP4 看到的标尺永远是 `s = m/6`。**g 是多少都不影响 FP4 量化结果**。

**现实情形**：FP8 有限精度，`s_FP8_stored = s · g + ε`，其中 `ε` 是 FP8 舍入误差。则

```
s_FP8_stored / g = s + ε / g
```

FP4 看到的标尺被歪了 `ε / g`。所有 dequantize 出的值都跟着歪。

**结论**：g 对最终重建结果的全部影响，**都通过 FP8 舍入误差 ε 传递**。g 自己只是一个"放置器"，决定 `s · g` 落在 FP8 grid 的哪一段，从而决定 ε 的大小。

### 2.5 为什么 g_optimal = 2688 / absmax(x)

FP8 E4M3 的最大可表示值是 448。我们希望最大那个 group（决定 tensor absmax 的 group）的 `s · g` 恰好命中 448 — 这是个 FP8 上的"干净落点"，ε ≈ 0。

```
m_max · g / 6 = 448  ⟹  g_optimal = 448 × 6 / m_max = 2688 / m_max
```

这同时把其他较小 group 的 `s · g` 自动散布在 FP8 数轴的更小值区，那里 stride 更细，ε 也更小。

---

## 3. 实验设置

| 参数 | 值 |
|---|---|
| Tensor shape | `[128, 4096]` |
| Group size | 16 |
| Base distribution | `N(0, 1)` |
| Outlier tiers | ±10, ±60, ±110, ±230, ±350, ±410 |
| Channels per tier | 3 |
| Sign per row | 随机 ±（保证 absmax 体现在两侧） |
| K sweep | 1.0, 1.2, 1.4, ..., 4.0（步长 0.2，共 16 个值） |
| 量化路径 | `compressed_tensors` 库的 `fake_quantize` + `compute_dynamic_scales_and_zp` |

**g 的设定**：

```python
gs_optimal = generate_gparam(x.min(), x.max())   # = 2688 / absmax(x)
gs_used = gs_optimal / K                         # 在 K∈[1,4] 下扫描
```

代码入口：`run.py`。运行：

```bash
python experiments_v4_nvfp4_input_global_scale/run.py
```

---

## 4. 结果

### 4.1 三张图

| 文件 | 内容 |
|---|---|
| `report.png` | (a) 各 outlier tier 的 MSE vs K；(b) FP8 subnormal/zero 比例 vs K；(c) 非 outlier 重建误差分布；(d) 每个 tier 的理想 block_scale 与 FP8 grid 的关系 |
| `report_per_row_mse.png` | 每个 K 对应 128 行的 per-row MSE 直方图 |
| `report_outlier_error_dist.png` | 每个 outlier tier 的重建误差分布，叠加多个 K |
| `results.csv` | 数值表格 |

### 4.2 主要发现

**(F1) 非 outlier MSE 在所有 K 下完全恒定（~0.060）。**  
这些 channel 的 group_absmax ≈ 2.5，对应的 `s · g` 落在 FP8 精细区（值 < 30，stride 极细）。K 把它们左右平移，仍在精细区内，FP8 舍入误差 ε 始终很小。

**(F2) FP8 subnormal 在 K ∈ [1, 4] 范围内从未触发。**  
要让非 outlier 的 block_scale 进入 subnormal 区（< 2⁻⁶ ≈ 0.0156），需要 `K > 140`。本实验的失效机制不是下溢，而是**FP8 上端的粗粒度舍入**。

**(F3) 不同量级的 outlier 对 K 的敏感度截然不同。**

| Tier | 理想 s·g @ K=1 | 所在 FP8 区域 | MSE 在 K 扫描下的范围 |
|---|---|---|---|
| ±10  | ~10.9 | 精细 | 33.8（完全不变） |
| ±60  | ~65   | 中段 | 1.1 – 6.9 |
| ±110 | ~120  | 中段偏粗 | 11 – 42 |
| ±230 | ~250  | 上端粗粒度 | 1.2 – 84 |
| ±350 | ~380  | 上端粗粒度 | 1.1 – 392 |
| ±410 | ~446  | 紧邻 FP8 max=448 | 6.3 – 199 |

**(F4) 没有一个 K 对所有 tier 同时最优。**  
- K=1 对 ±410 最优（s·g 恰好落到 448）
- K=1.2 对 ±230 最优（MSE 1.24，比 K=1 的 31.7 低 25 倍）
- K=3.4 对 ±350 最优
- ±10 几乎对任何 K 都没意见

**(F5) MSE 随 K 是非单调震荡的。**  
例如 ±350 tier：K=2.8 时 MSE=392（最差），K=3.4 时 MSE=1.08（最好），相邻两个 K 之间差 360 倍。原因是它的理想 block_scale 在 FP8 数轴上不同位置，会随 K 跳过/命中不同的 FP8 grid 点。

### 4.3 失效机制的定量解释

对一个 outlier group，FP8 舍入误差 ε 传递到重建的最大误差约为：

```
|Δx| ≈ FP4_max × |ε / g| = 6 × |ε| / g
```

FP8 E4M3 在不同区域的 stride：
- `[0.001, 0.1]` ：stride 量级 2⁻⁹（约 0.002）
- `[1, 10]`     ：stride ≈ 1
- `[100, 448]`  ：stride 在 8–32 之间，最大达到 32

`ε` 最坏情况 ≈ stride / 2。把它代入上式：

| s·g 所在区 | ε ≈ stride/2 | g 量级 ~5 时 \|Δx\| ≈ 6ε/g |
|---|---|---|
| 1     | 0.06 | 0.07 |
| 30    | 1    | 1.2 |
| 200   | 8    | 9.6 |
| 400   | 16   | 19  |

对一个 ±410 outlier，重建误差 ~19 是合理的；MSE = (19)² ≈ 360，量级与实测 199 接近。

---

## 5. 实践含义

1. **outlier 是 g 唯一作用的对象**。"input_global_scale 只影响 outlier" 这个直觉是正确的 — 更精确地说，只影响**理想 block_scale 落在 FP8 粗粒度区**的那些 channel。

2. **校准误差被 outlier 放大**。校准集低估 absmax 20%（K=1.2）就可能让大 outlier 的 MSE 恶化 10 倍以上。

3. **"用更小的 g 更安全"是错的**。MSE 不随 K 单调变化，缩小 g 也可能让 outlier 的 block_scale 从一个 FP8 好落点滑到坏落点。

4. **不存在单一最优 g**。如果 activation 有多档量级的 outlier，任何 g 都只能让其中一档命中 FP8 干净落点，其它档的 ε 由它们各自所处 FP8 区域的 stride 决定。

5. **真正的失效模式不是 FP8 下溢**。在常见的 1–10× 校准偏差区间，所有 block_scale 仍在 FP8 normal 区，问题是**上端粗粒度舍入**。要触发下溢需要 K > 100 量级。

---

## 6. 文件清单

```
experiments_v4_nvfp4_input_global_scale/
├── README.md                       # 本文档
├── run.py                          # 实验脚本（自包含，单文件）
├── results.csv                     # K × tier × MSE 数值表
├── report.png                      # 主报告（4 面板）
├── report_per_row_mse.png          # 每个 K 的 per-row MSE 分布
└── report_outlier_error_dist.png   # 每个 outlier tier 的误差分布
```

---

## 7. 关键代码引用

- `src/compressed_tensors/quantization/utils/helpers.py:309` — `generate_gparam(min, max)` 实现 `g_optimal = 2688 / absmax`
- `src/compressed_tensors/quantization/utils/helpers.py:140` — `compute_dynamic_scales_and_zp` 实现 group-wise block_scale 计算
- `src/compressed_tensors/quantization/lifecycle/forward.py:149` — `fake_quantize` 端到端 quantize + dequantize
