"""
g=1 harm vs Gaussian support size:  [-1,1] ... [-3000,3000]
===========================================================
Sweep the support S of a homogeneous Gaussian (x ~ N(0, S/3), clipped to
[-S, S]) and measure how much hard-fixing g=1 hurts relative to g_optimal.

Key threshold
-------------
At g=1 the largest group's stored block scale is absmax/6. FP8 E4M3 max = 448,
so once absmax/6 > 448  <=>  S > 2688, g=1 forces big block scales to CLIP.
Below that, g=1 ~ g_optimal (pure multiplicative shift, constant rel. precision).

Run
---
    python experiments_v5_nvfp4_global_scale_one/gaussian_sweep.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.helpers import (
    compute_dynamic_scales_and_zp,
    generate_gparam,
)

SEED = 42
SHAPE = (128, 4096)
GROUP_SIZE = 16
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FP8_MAX = 448.0
FP4_MAX = 6.0
S_CRIT = FP8_MAX * FP4_MAX  # 2688: above this, g=1 clips block scales


def make_nvfp4_args():
    return QuantizationArgs(
        num_bits=4, type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP, symmetric=True,
        group_size=GROUP_SIZE,
        scale_dtype=torch.float8_e4m3fn, zp_dtype=torch.float8_e4m3fn,
    )


def gen_gaussian(support):
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(*SHAPE, generator=g) * (support / 3.0)
    x = x.clamp(-support, support)
    x[0, 0] = support
    x[0, 1] = -support
    return x


def eval_g(x, g_value, args):
    gs = torch.tensor(float(g_value), dtype=torch.float32)
    scale, zp = compute_dynamic_scales_and_zp(x, args, module=None, global_scale=gs)
    x_hat = fake_quantize(x, scale, zp, args, global_scale=gs)
    mse = ((x_hat - x) ** 2).mean().item()
    bs = scale.float().abs()
    frac_clip = (bs >= FP8_MAX).float().mean().item()
    # normalized MSE (relative to signal power) so scales are comparable
    nmse = mse / (x ** 2).mean().item()
    return mse, nmse, frac_clip


def main():
    args = make_nvfp4_args()
    # Requested range is [-1,1] .. [-3000,3000]; a few points past 3000 are
    # added to reveal the climb once clipping kicks in beyond S_crit=2688.
    supports = [1, 3, 10, 30, 100, 300, 1000, 2000, 2688, 3000, 5000, 10000, 30000]

    rows = []
    print(f"S_crit (g=1 starts clipping) = {S_CRIT:g}\n")
    print(f"  {'support':>8}  {'g_opt':>9}  {'NMSE@opt':>10}  {'NMSE@g=1':>10}  "
          f"{'ratio':>7}  {'clip@g=1':>9}")
    for S in supports:
        x = gen_gaussian(S)
        g_opt = generate_gparam(x.min(), x.max()).item()
        mse_o, nmse_o, clip_o = eval_g(x, g_opt, args)
        mse_1, nmse_1, clip_1 = eval_g(x, 1.0, args)
        ratio = mse_1 / mse_o
        rows.append(dict(S=S, g_opt=g_opt, nmse_o=nmse_o, nmse_1=nmse_1,
                         ratio=ratio, clip_1=clip_1, mse_o=mse_o, mse_1=mse_1))
        print(f"  {S:8g}  {g_opt:9.4f}  {nmse_o:10.5f}  {nmse_1:10.5f}  "
              f"{ratio:7.2f}  {clip_1:9.4f}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    fig.suptitle(
        "g=1 vs g_optimal as Gaussian support grows  (x~N(0,S/3) clipped to [-S,S])",
        fontsize=12, fontweight="bold",
    )
    Ss = [r["S"] for r in rows]

    ax = axes[0]
    ax.plot(Ss, [r["ratio"] for r in rows], "o-", color="#d62728", lw=2)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.axvline(S_CRIT, color="#1f77b4", ls="--", lw=2,
               label=f"S_crit = 2688\n(absmax/6 = FP8 max)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gaussian support  S  (range = [-S, S])")
    ax.set_ylabel("MSE(g=1) / MSE(g_optimal)")
    ax.set_title("(a) Relative harm of g=1 vs support")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(Ss, [r["nmse_o"] for r in rows], "s--", color="#2ca02c", lw=2,
            label="NMSE @ g_optimal")
    ax.plot(Ss, [r["nmse_1"] for r in rows], "o-", color="#d62728", lw=2,
            label="NMSE @ g=1")
    ax2 = ax.twinx()
    ax2.plot(Ss, [r["clip_1"] for r in rows], "^:", color="#ff7f0e", lw=1.6,
             label="frac block-scale clipped @ g=1")
    ax2.set_ylabel("frac block scales clipped @ g=1", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax.axvline(S_CRIT, color="#1f77b4", ls="--", lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gaussian support  S")
    ax.set_ylabel("normalized MSE  (MSE / signal power)")
    ax.set_title("(b) Normalized MSE + clipping fraction")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    png = os.path.join(OUT_DIR, "report_gaussian_sweep.png")
    plt.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png}")


if __name__ == "__main__":
    main()
