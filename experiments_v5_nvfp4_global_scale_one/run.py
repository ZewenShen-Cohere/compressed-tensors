"""
NVFP4 global_scale = 1 study
============================
Question
--------
v4 swept gs_used = gs_optimal / K for K in [1, 4]. Here we ask a different
question on the *same* data distributions:

    "If I hard-fix the global scale to g = 1 (instead of the calibrated optimal),
     do these distributions get a LARGER quantization error?"

Note: in v4's K-framing, g = 1 corresponds to K = gs_optimal (≈ 6.5 here), which
is OUTSIDE the swept [1, 4] range. So g = 1 is a genuinely new operating point.

What we measure
---------------
On the identical activation (same seeds, same 6 outlier tiers as v4) we compare,
per outlier tier and for non-outlier channels, the reconstruction MSE under:
  - g = g_optimal              (the calibrated baseline; max group lands on FP8 448)
  - g = 1                      (the hard-fixed value under study)
  - a fine sweep of fixed g    (to place g=1 in the global landscape)

Outputs
-------
- experiments_v5_nvfp4_global_scale_one/report.png
- experiments_v5_nvfp4_global_scale_one/results.csv

Run
---
    python experiments_v5_nvfp4_global_scale_one/run.py
"""

import csv
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


# Identical setup to v4 so the data distribution is exactly the same.
SEED = 42
SHAPE = (128, 4096)
GROUP_SIZE = 16
OUTLIER_MAGNITUDES = [10.0, 60.0, 110.0, 230.0, 350.0, 410.0]
N_PER_TIER = 3
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FP8_E4M3_MIN_NORMAL = 2 ** -6
FP4_MAX = 6.0
FP8_MAX = 448.0


def make_nvfp4_args():
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        group_size=GROUP_SIZE,
        scale_dtype=torch.float8_e4m3fn,
        zp_dtype=torch.float8_e4m3fn,
    )


def gen_activation():
    """Exactly reproduces v4's activation (same seeds and tiers)."""
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(*SHAPE, generator=g)

    total = len(OUTLIER_MAGNITUDES) * N_PER_TIER
    chan_rng = torch.Generator().manual_seed(SEED + 1)
    picked = torch.randperm(SHAPE[1], generator=chan_rng)[:total]

    tier_channels = {}
    for i, mag in enumerate(OUTLIER_MAGNITUDES):
        cols = picked[i * N_PER_TIER : (i + 1) * N_PER_TIER]
        tier_channels[mag] = cols

        sign_rng = torch.Generator().manual_seed(SEED + 100 + i)
        signs = torch.randint(0, 2, (SHAPE[0], N_PER_TIER), generator=sign_rng) * 2 - 1
        noise_rng = torch.Generator().manual_seed(SEED + 200 + i)
        noise = torch.randn(SHAPE[0], N_PER_TIER, generator=noise_rng)
        x[:, cols] = signs.float() * mag + noise

    return x, tier_channels


def run_quant_with_g(x, g_value, args):
    """Quantize/dequantize with an explicit per-tensor global scale value."""
    gs = torch.tensor(g_value, dtype=torch.float32)
    scale, zp = compute_dynamic_scales_and_zp(x, args, module=None, global_scale=gs)
    x_hat = fake_quantize(x, scale, zp, args, global_scale=gs)
    return x_hat, scale


def measure(x, g_value, args, masks):
    x_hat, block_scale = run_quant_with_g(x, g_value, args)
    sq = (x_hat - x) ** 2
    out = dict(
        g=float(g_value),
        mse_all=sq.mean().item(),
        mse_non_outlier=sq[:, masks["non_outlier"]].mean().item(),
    )
    for mag in OUTLIER_MAGNITUDES:
        out[f"mse_tier_{mag:g}"] = sq[:, masks[mag]].mean().item()

    bs_abs = block_scale.float().abs()
    out["frac_zero"] = (bs_abs == 0).float().mean().item()
    out["frac_subnormal"] = (
        ((bs_abs > 0) & (bs_abs < FP8_E4M3_MIN_NORMAL)).float().mean().item()
    )
    out["frac_clipped"] = (bs_abs >= FP8_MAX).float().mean().item()
    return out, (x_hat - x), block_scale


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    args = make_nvfp4_args()

    x, tier_channels = gen_activation()
    all_outlier_cols = torch.cat(list(tier_channels.values()))
    outlier_mask = torch.zeros(SHAPE[1], dtype=torch.bool)
    outlier_mask[all_outlier_cols] = True

    masks = {"non_outlier": ~outlier_mask}
    for mag, cols in tier_channels.items():
        m = torch.zeros(SHAPE[1], dtype=torch.bool)
        m[cols] = True
        masks[mag] = m

    gs_optimal = generate_gparam(x.min(), x.max()).item()
    absmax = x.abs().max().item()
    print(f"Tensor: shape={tuple(x.shape)}  absmax={absmax:.2f}")
    print(f"gs_optimal = {gs_optimal:.4f}   (g=1  ==  K={gs_optimal:.2f} in v4 framing)\n")

    # ── Two headline operating points: g=1 vs g_optimal ───────────────
    res_opt, err_opt, bs_opt = measure(x, gs_optimal, args, masks)
    res_one, err_one, bs_one = measure(x, 1.0, args, masks)

    print("Per-tier MSE comparison:")
    print(f"  {'tier':>14}   {'g_optimal':>12}   {'g=1':>12}   {'ratio g=1/opt':>14}")
    print(f"  {'non-outlier':>14}   {res_opt['mse_non_outlier']:12.4f}   "
          f"{res_one['mse_non_outlier']:12.4f}   "
          f"{res_one['mse_non_outlier']/res_opt['mse_non_outlier']:14.2f}")
    for mag in OUTLIER_MAGNITUDES:
        k = f"mse_tier_{mag:g}"
        ratio = res_one[k] / res_opt[k] if res_opt[k] > 0 else float("inf")
        print(f"  {('±'+format(mag,'g')):>14}   {res_opt[k]:12.4f}   "
              f"{res_one[k]:12.4f}   {ratio:14.2f}")
    print(f"\n  g=1 block-scale degeneracy: "
          f"subnormal={res_one['frac_subnormal']:.4f}  "
          f"zero={res_one['frac_zero']:.4f}  clipped={res_one['frac_clipped']:.4f}")
    print(f"  g_opt block-scale degeneracy: "
          f"subnormal={res_opt['frac_subnormal']:.4f}  "
          f"zero={res_opt['frac_zero']:.4f}  clipped={res_opt['frac_clipped']:.4f}\n")

    # ── Fine sweep of fixed g to place g=1 in the landscape ───────────
    # Geometric sweep spanning well below 1 up to (and past) g_optimal.
    g_sweep = sorted(set(
        [round(v, 4) for v in np.geomspace(0.1, gs_optimal * 2.0, 40)]
        + [1.0, gs_optimal]
    ))
    sweep_rows = []
    for gv in g_sweep:
        row, _, _ = measure(x, gv, args, masks)
        sweep_rows.append(row)

    # ── CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = list(sweep_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # Tag the two headline rows for easy reading.
        for tag, r in [("g_optimal", res_opt), ("g_one", res_one)]:
            rr = dict(r)
            w.writerow(rr)
        w.writerows(sweep_rows)
    print(f"Wrote {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
    tier_str = ", ".join(f"±{m:g}" for m in OUTLIER_MAGNITUDES)
    fig.suptitle(
        f"NVFP4 global_scale = 1  vs  g_optimal  |  shape={SHAPE}  g={GROUP_SIZE}  "
        f"|  outlier tiers: {tier_str} (×{N_PER_TIER})",
        fontsize=13, fontweight="bold",
    )
    cmap = plt.get_cmap("plasma")
    n_tiers = len(OUTLIER_MAGNITUDES)

    # (a) Grouped bar: per-tier MSE, g_optimal vs g=1
    ax = axes[0, 0]
    labels = ["non-out"] + [f"±{m:g}" for m in OUTLIER_MAGNITUDES]
    opt_vals = [res_opt["mse_non_outlier"]] + [res_opt[f"mse_tier_{m:g}"] for m in OUTLIER_MAGNITUDES]
    one_vals = [res_one["mse_non_outlier"]] + [res_one[f"mse_tier_{m:g}"] for m in OUTLIER_MAGNITUDES]
    xpos = np.arange(len(labels))
    w = 0.38
    ax.bar(xpos - w / 2, opt_vals, w, label="g = g_optimal", color="#2ca02c")
    ax.bar(xpos + w / 2, one_vals, w, label="g = 1", color="#d62728")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=30, fontsize=9)
    ax.set_ylabel("MSE (log)")
    ax.set_title("(a) Per-tier MSE:  g_optimal  vs  g=1")
    ax.legend()
    ax.grid(True, which="both", axis="y", alpha=0.3)

    # (b) Per-tier MSE vs fixed g (landscape), with g=1 and g_opt marked
    ax = axes[0, 1]
    gs = [r["g"] for r in sweep_rows]
    for i, mag in enumerate(OUTLIER_MAGNITUDES):
        ys = [r[f"mse_tier_{mag:g}"] for r in sweep_rows]
        ax.plot(gs, ys, "o-", ms=3, linewidth=1.4,
                color=cmap(i / max(1, n_tiers - 1)), label=f"±{mag:g}")
    ax.plot(gs, [r["mse_non_outlier"] for r in sweep_rows], "s--",
            color="#2ca02c", ms=3, linewidth=1.4, label="non-outlier")
    ax.axvline(1.0, color="#d62728", linestyle="-", linewidth=2, alpha=0.8, label="g = 1")
    ax.axvline(gs_optimal, color="#1f77b4", linestyle="--", linewidth=2,
               alpha=0.8, label="g_optimal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fixed global scale g")
    ax.set_ylabel("MSE (log)")
    ax.set_title("(b) Per-tier MSE landscape vs fixed g")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)

    # (c) Where each tier's ideal block scale lands on the FP8 grid: g=1 vs g_opt
    ax = axes[1, 0]
    fp8_grid = torch.arange(0, 256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    fp8_grid = torch.unique(fp8_grid[(fp8_grid > 0) & torch.isfinite(fp8_grid)]).sort()[0].numpy()
    for v in fp8_grid:
        if 0.05 < v < 500:
            ax.axhline(v, color="gray", alpha=0.13, linewidth=0.6, zorder=0)
    ax.axhline(448, color="red", linestyle="--", alpha=0.6, label="FP8 max=448")
    ax.axhline(FP8_E4M3_MIN_NORMAL, color="purple", linestyle=":", alpha=0.7,
               label="FP8 min normal")
    tiers = OUTLIER_MAGNITUDES
    xpos = np.arange(len(tiers))
    ideal_opt = [(m / FP4_MAX) * gs_optimal for m in tiers]
    ideal_one = [(m / FP4_MAX) * 1.0 for m in tiers]
    ax.scatter(xpos - 0.12, ideal_opt, s=80, color="#2ca02c", zorder=5,
               label="ideal s·g  (g_optimal)")
    ax.scatter(xpos + 0.12, ideal_one, s=80, color="#d62728", marker="D", zorder=5,
               label="ideal s·g  (g=1)")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"±{m:g}" for m in tiers])
    ax.set_ylabel("ideal block scale  s·g = (absmax/6)·g")
    ax.set_title("(c) Where each tier lands on FP8 E4M3 grid")
    ax.legend(fontsize=8, loc="center right")

    # (d) Reconstruction error distribution for the largest tier: g=1 vs g_opt
    ax = axes[1, 1]
    cols_big = tier_channels[OUTLIER_MAGNITUDES[-1]]
    e_opt = err_opt[:, cols_big].flatten().numpy()
    e_one = err_one[:, cols_big].flatten().numpy()
    span = max(np.abs(e_opt).max(), np.abs(e_one).max()) * 1.05
    bins = np.linspace(-span, span, 90)
    ax.hist(e_opt, bins=bins, histtype="step", density=True, linewidth=1.8,
            color="#2ca02c", label=f"g_optimal  (MSE={res_opt[f'mse_tier_{OUTLIER_MAGNITUDES[-1]:g}']:.2f})")
    ax.hist(e_one, bins=bins, histtype="step", density=True, linewidth=1.8,
            color="#d62728", label=f"g=1  (MSE={res_one[f'mse_tier_{OUTLIER_MAGNITUDES[-1]:g}']:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("x̂ - x")
    ax.set_ylabel("density")
    ax.set_title(f"(d) Error dist, largest tier ±{OUTLIER_MAGNITUDES[-1]:g}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    png_path = os.path.join(OUT_DIR, "report.png")
    plt.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
