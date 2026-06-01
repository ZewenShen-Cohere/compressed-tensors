"""
NVFP4 input_global_scale Sensitivity Study
==========================================
Sweep gs_used = gs_optimal / K for K in {1, 1.3, 1.5, 2, 4} on an activation
tensor with 3 large-outlier channels, measure reconstruction MSE.

Setup
-----
- x: [128, 4096], x ~ N(0, 1) except 3 random channels replaced with ~N(500, 1)
  with random ±sign per row.
- NVFP4: group_size=16, FP8 E4M3 block scales, FP32 global scale.
- gs_optimal = (FP8_MAX * FP4_MAX) / absmax(x)  (= generate_gparam(min, max))

Outputs
-------
- experiments_v4_nvfp4_input_global_scale/report.png
- experiments_v4_nvfp4_input_global_scale/results.csv

Run
---
    python experiments_v4_nvfp4_input_global_scale/run.py
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


SEED = 42
SHAPE = (128, 4096)
GROUP_SIZE = 16
# Multiple outlier magnitudes: each tier puts N_PER_TIER channels at ±MAG.
OUTLIER_MAGNITUDES = [10.0, 60.0, 110.0, 230.0, 350.0, 410.0]
N_PER_TIER = 3
K_SWEEP = [1.0] + [round(1.0 + 0.2 * i, 2) for i in range(1, 16)]  # 1.0, 1.2, 1.4, ..., 4.0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# FP8 E4M3 smallest normal magnitude = 2^(1 - bias) with bias=7 ⇒ 2^-6.
FP8_E4M3_MIN_NORMAL = 2 ** -6


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
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(*SHAPE, generator=g)

    total = len(OUTLIER_MAGNITUDES) * N_PER_TIER
    chan_rng = torch.Generator().manual_seed(SEED + 1)
    picked = torch.randperm(SHAPE[1], generator=chan_rng)[:total]

    tier_channels = {}  # mag -> tensor of channel indices
    for i, mag in enumerate(OUTLIER_MAGNITUDES):
        cols = picked[i * N_PER_TIER : (i + 1) * N_PER_TIER]
        tier_channels[mag] = cols

        sign_rng = torch.Generator().manual_seed(SEED + 100 + i)
        signs = torch.randint(0, 2, (SHAPE[0], N_PER_TIER), generator=sign_rng) * 2 - 1
        noise_rng = torch.Generator().manual_seed(SEED + 200 + i)
        noise = torch.randn(SHAPE[0], N_PER_TIER, generator=noise_rng)
        x[:, cols] = signs.float() * mag + noise

    return x, tier_channels


def run_quant(x, K, args):
    gs_optimal = generate_gparam(x.min(), x.max())
    gs = gs_optimal / K
    scale, zp = compute_dynamic_scales_and_zp(x, args, module=None, global_scale=gs)
    x_hat = fake_quantize(x, scale, zp, args, global_scale=gs)
    return x_hat, scale, gs, gs_optimal


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    args = make_nvfp4_args()

    x, tier_channels = gen_activation()
    all_outlier_cols = torch.cat(list(tier_channels.values()))
    outlier_mask = torch.zeros(SHAPE[1], dtype=torch.bool)
    outlier_mask[all_outlier_cols] = True
    non_outlier_mask = ~outlier_mask

    tier_masks = {}
    for mag, cols in tier_channels.items():
        m = torch.zeros(SHAPE[1], dtype=torch.bool)
        m[cols] = True
        tier_masks[mag] = m

    print(f"Tensor: shape={tuple(x.shape)}  absmax={x.abs().max().item():.2f}")
    for mag, cols in tier_channels.items():
        print(f"  outlier tier ±{mag:<6g}: channels {sorted(cols.tolist())}")
    gs_optimal_val = (448.0 * 6.0) / x.abs().max().item()
    print(f"gs_optimal ≈ {gs_optimal_val:.4f}\n")

    results = []
    per_run = {}

    for K in K_SWEEP:
        x_hat, block_scale, gs_used, gs_optimal = run_quant(x, K, args)
        err = x_hat - x
        sq = err ** 2
        mse_all = sq.mean().item()
        mse_out = sq[:, outlier_mask].mean().item()
        mse_non = sq[:, non_outlier_mask].mean().item()

        tier_mses = {mag: sq[:, tier_masks[mag]].mean().item() for mag in OUTLIER_MAGNITUDES}

        bs_abs = block_scale.float().abs()
        frac_zero = (bs_abs == 0).float().mean().item()
        frac_subnormal = ((bs_abs > 0) & (bs_abs < FP8_E4M3_MIN_NORMAL)).float().mean().item()

        row = dict(
            K=K, mse_all=mse_all, mse_outlier=mse_out, mse_non_outlier=mse_non,
            frac_subnormal=frac_subnormal, frac_zero=frac_zero,
            gs_used=gs_used.item(), gs_optimal=gs_optimal.item(),
        )
        for mag in OUTLIER_MAGNITUDES:
            row[f"mse_tier_{mag:g}"] = tier_mses[mag]
        results.append(row)
        per_run[K] = dict(
            x_hat=x_hat, err=err, block_scale=block_scale, tier_mses=tier_mses,
        )

        tier_str = "  ".join(f"±{mag:g}={tier_mses[mag]:.2e}" for mag in OUTLIER_MAGNITUDES)
        print(
            f"  K={K:<4.2g}  gs={gs_used.item():7.4f}  all={mse_all:.3e}  "
            f"non_out={mse_non:.3e}  | {tier_str}"
        )

    # ── CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    tier_str = ", ".join(f"±{m:g}" for m in OUTLIER_MAGNITUDES)
    fig.suptitle(
        f"NVFP4 input_global_scale sensitivity  |  shape={SHAPE}  "
        f"g={GROUP_SIZE}  |  outlier tiers: {tier_str}  (×{N_PER_TIER} each)",
        fontsize=12, fontweight="bold",
    )

    Ks = [r["K"] for r in results]
    mse_non = [r["mse_non_outlier"] for r in results]
    fs_sub = [r["frac_subnormal"] for r in results]
    fs_zero = [r["frac_zero"] for r in results]

    # (a) MSE per outlier tier vs K
    ax = axes[0, 0]
    cmap = plt.get_cmap("plasma")
    n_tiers = len(OUTLIER_MAGNITUDES)
    for i, mag in enumerate(OUTLIER_MAGNITUDES):
        tier_mse = [per_run[K]["tier_mses"][mag] for K in K_SWEEP]
        ax.plot(
            Ks, tier_mse, "o-", linewidth=1.8,
            color=cmap(i / max(1, n_tiers - 1)),
            label=f"outlier ±{mag:g}",
        )
    ax.plot(Ks, mse_non, "s--", color="#2ca02c", linewidth=1.8, label="non-outlier")
    ax.set_xlabel("K  (gs_used = gs_optimal / K)")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([f"{k:g}" for k in Ks], rotation=45, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("(a) Per-tier MSE vs K")

    # (b) block-scale subnormal / zero fraction
    ax = axes[0, 1]
    ax.plot(Ks, fs_sub, "o-", color="#ff7f0e", label="subnormal", linewidth=2)
    ax.plot(Ks, fs_zero, "s-", color="#8c564b", label="zero", linewidth=2)
    ax.set_xlabel("K")
    ax.set_xticks(Ks)
    ax.set_xticklabels([f"{k:g}" for k in Ks], rotation=45, fontsize=8)
    ax.set_ylabel("Fraction of block scales")
    ax.set_title("(b) FP8 block-scale degeneracy (subnormal / zero)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Error histogram for a sparse subset of K (non-outlier channels)
    ax = axes[1, 0]
    hist_Ks = [k for k in [1.0, 1.4, 2.0, 3.0, 4.0] if k in K_SWEEP]
    cmap = plt.get_cmap("viridis")
    for i, K in enumerate(hist_Ks):
        err_non = per_run[K]["err"][:, non_outlier_mask].flatten().numpy()
        ax.hist(
            err_non, bins=200, range=(-0.6, 0.6), histtype="step",
            label=f"K={K:g}", linewidth=1.4, density=True,
            color=cmap(i / max(1, len(hist_Ks) - 1)),
        )
    ax.set_xlabel("x̂ - x  (non-outlier channels)")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title("(c) Reconstruction error distribution (non-outlier channels)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) Tier ideal block-scale (FP4_max=6 normalization) across K — where each
    # outlier tier lands on the FP8 E4M3 representable grid.
    ax = axes[1, 1]
    # Build the FP8 E4M3 positive representable grid (finite values).
    fp8_grid = torch.arange(0, 256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    fp8_grid = torch.unique(fp8_grid[(fp8_grid > 0) & torch.isfinite(fp8_grid)]).sort()[0].numpy()

    for i, mag in enumerate(OUTLIER_MAGNITUDES):
        ideal_bs = []
        for r in results:
            ideal = (mag / 6.0) * r["gs_used"]
            ideal_bs.append(ideal)
        ax.plot(
            Ks, ideal_bs, "o-", linewidth=1.6,
            color=cmap(i / max(1, n_tiers - 1)),
            label=f"±{mag:g}",
        )
    for v in fp8_grid:
        if 0.1 < v < 500:
            ax.axhline(v, color="gray", alpha=0.15, linewidth=0.6, zorder=0)
    ax.axhline(448, color="red", linestyle="--", alpha=0.6, label="FP8 max=448")
    ax.set_yscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("Ideal block scale  (group_absmax / 6) × gs_used")
    ax.set_xticks(Ks)
    ax.set_xticklabels([f"{k:g}" for k in Ks], rotation=45, fontsize=8)
    ax.set_title("(d) Each tier's ideal block scale vs FP8 E4M3 grid (gray lines)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(False)

    png_path = os.path.join(OUT_DIR, "report.png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")

    # ── Per-row MSE distribution per K ────────────────────────────────
    n = len(K_SWEEP)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.0 * nrows),
        layout="constrained", sharex=True,
    )
    fig2.suptitle(
        f"Per-row MSE distribution across 128 rows  |  shape={SHAPE}  g={GROUP_SIZE}  "
        f"|  outlier tiers: {tier_str}  (×{N_PER_TIER} each)",
        fontsize=12, fontweight="bold",
    )

    all_row_mse = {K: (per_run[K]["err"] ** 2).mean(dim=1).numpy() for K in K_SWEEP}
    global_max = max(arr.max() for arr in all_row_mse.values())
    bins = np.linspace(0, global_max * 1.02, 40)

    axes2_flat = axes2.flatten()
    for i, K in enumerate(K_SWEEP):
        ax = axes2_flat[i]
        row_mse = all_row_mse[K]
        ax.hist(row_mse, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="white")
        ax.axvline(row_mse.mean(), color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"mean={row_mse.mean():.3f}")
        ax.set_title(f"K = {K:g}    (gs={results[i]['gs_used']:.3f})", fontsize=10)
        ax.set_xlabel("per-row MSE")
        ax.set_ylabel("count (of 128 rows)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    for j in range(n, len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    png2_path = os.path.join(OUT_DIR, "report_per_row_mse.png")
    plt.savefig(png2_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {png2_path}")

    # ── Per-tier reconstruction error distribution across K ───────────
    n_tiers = len(OUTLIER_MAGNITUDES)
    ncols3 = 3
    nrows3 = (n_tiers + ncols3 - 1) // ncols3
    fig3, axes3 = plt.subplots(
        nrows3, ncols3, figsize=(5.5 * ncols3, 3.6 * nrows3),
        layout="constrained",
    )
    fig3.suptitle(
        f"Per-tier reconstruction error distribution across K  |  shape={SHAPE}  "
        f"g={GROUP_SIZE}",
        fontsize=13, fontweight="bold",
    )
    axes3_flat = axes3.flatten()
    hist_Ks = [k for k in [1.0, 1.4, 2.0, 3.0, 4.0] if k in K_SWEEP]
    cmap3 = plt.get_cmap("viridis")

    for ti, mag in enumerate(OUTLIER_MAGNITUDES):
        ax = axes3_flat[ti]
        cols = tier_channels[mag]
        # Determine a reasonable x range from the worst-K case for this tier.
        all_errs = np.concatenate([
            per_run[K]["err"][:, cols].flatten().numpy() for K in hist_Ks
        ])
        lo, hi = np.percentile(all_errs, [0.1, 99.9])
        span = max(abs(lo), abs(hi)) * 1.05
        bins = np.linspace(-span, span, 80)

        for i, K in enumerate(hist_Ks):
            err_tier = per_run[K]["err"][:, cols].flatten().numpy()
            ax.hist(
                err_tier, bins=bins, histtype="step",
                label=f"K={K:g}", linewidth=1.5, density=True,
                color=cmap3(i / max(1, len(hist_Ks) - 1)),
            )
        ax.set_xlabel("x̂ - x")
        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.set_title(f"outlier tier ±{mag:g}  ({N_PER_TIER} channels)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(n_tiers, len(axes3_flat)):
        axes3_flat[j].set_visible(False)

    png3_path = os.path.join(OUT_DIR, "report_outlier_error_dist.png")
    plt.savefig(png3_path, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    print(f"Wrote {png3_path}")


if __name__ == "__main__":
    main()
