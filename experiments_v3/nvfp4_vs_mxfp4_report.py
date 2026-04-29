"""
NVFP4 vs MXFP4 Comparison (group_size=32): Multi-Distribution Analysis
========================================================================
Compares NVFP4 and MXFP4 quantization at the same group_size=32 across
LLM-realistic activation distributions.

Key architectural difference:
  NVFP4: Two-level scaling — FP8 E4M3 block scale × FP32 global scale
         Effective per-element scale = E4M3_scale / global_scale (fine-grained)
  MXFP4: Single-level — E8M0 power-of-2 block scale (no global scale)
         Effective per-element scale = 2^exponent (coarse, only powers of 2)

Both formats use FP4 E2M1 for data values.

Distributions:
  1. Gaussian(0, 1)                     Post-LayerNorm baseline
  2. Gaussian(μ=0.5, σ=2)              Biased high-variance (FFN intermediate)
  3. Laplace(0, b=1/√2)                Heavy-tailed, unit variance
  4. Uniform(-√3, √3)                  Bounded, light-tailed contrast
  5. Gaussian(0,1) + 2% outlier @15×   Realistic LLM outlier channel pattern

Output:
  experiments_v3/report_01_gaussian_std.png  ... (one per distribution)
  experiments_v3/report_comparison.png

Usage:
    python experiments_v3/nvfp4_vs_mxfp4_report.py
"""

import math
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
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

# ── Config ─────────────────────────────────────────────────────────────
SEED = 42
SHAPE = (64, 1024)
GROUP_SIZE = 32
EPS = 1e-10
OUT_DIR = "experiments_v3"

METHOD_COLORS = {"nvfp4": "#2196F3", "mxfp4": "#F44336"}
METHOD_LABELS = {"nvfp4": "NVFP4", "mxfp4": "MXFP4"}

DIST_COLORS = OrderedDict([
    ("gaussian_std",     "#2196F3"),
    ("gaussian_shifted", "#9C27B0"),
    ("laplace",          "#F44336"),
    ("uniform",          "#FF9800"),
    ("gaussian_outlier", "#4CAF50"),
])

FP4_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_BOUNDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


# ── Quantization Helpers ───────────────────────────────────────────────
def _make_nvfp4_args():
    return QuantizationArgs(
        num_bits=4, type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True, group_size=GROUP_SIZE,
        scale_dtype=torch.float8_e4m3fn, zp_dtype=torch.float8_e4m3fn,
    )


def _make_mxfp4_args():
    return QuantizationArgs(
        num_bits=4, type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True, group_size=GROUP_SIZE,
        scale_dtype=torch.uint8, zp_dtype=torch.uint8,
    )


def quantize_nvfp4(x):
    """NVFP4: FP8 E4M3 block scale + FP32 global scale."""
    args = _make_nvfp4_args()
    gs = generate_gparam(x.min(), x.max())
    scale, zp = compute_dynamic_scales_and_zp(x, args, module=None, global_scale=gs)
    x_hat = fake_quantize(x, scale, zp, args, global_scale=gs)
    eff_scale = scale / gs
    rows, cols = x.shape
    ng = cols // GROUP_SIZE
    scaled = x.reshape(rows, ng, GROUP_SIZE) / eff_scale.unsqueeze(-1)
    return x_hat, scaled.reshape(rows, cols)


def quantize_mxfp4(x):
    """MXFP4: E8M0 power-of-2 block scale, no global scale."""
    args = _make_mxfp4_args()
    scale, zp = compute_dynamic_scales_and_zp(x, args, module=None, global_scale=None)
    x_hat = fake_quantize(x, scale, zp, args, global_scale=None)
    rows, cols = x.shape
    ng = cols // GROUP_SIZE
    scaled = x.reshape(rows, ng, GROUP_SIZE) / scale.unsqueeze(-1)
    return x_hat, scaled.reshape(rows, cols)


def fp4_round_vec(s_vals):
    """Vectorized FP4 E2M1 rounding for positive values."""
    out = torch.zeros_like(s_vals)
    for i, s in enumerate(s_vals):
        sv = s.item()
        if sv <= FP4_BOUNDS[0]:
            out[i] = FP4_GRID[0]
        elif sv < FP4_BOUNDS[1]:
            out[i] = FP4_GRID[1]
        elif sv <= FP4_BOUNDS[2]:
            out[i] = FP4_GRID[2]
        elif sv < FP4_BOUNDS[3]:
            out[i] = FP4_GRID[3]
        elif sv <= FP4_BOUNDS[4]:
            out[i] = FP4_GRID[4]
        elif sv < FP4_BOUNDS[5]:
            out[i] = FP4_GRID[5]
        elif sv <= FP4_BOUNDS[6]:
            out[i] = FP4_GRID[6]
        else:
            out[i] = FP4_GRID[7]
    return out


def compute_results(x):
    """Run both NVFP4 and MXFP4 quantization, return per-method results."""
    R = {}
    for method, qfn in [("nvfp4", quantize_nvfp4), ("mxfp4", quantize_mxfp4)]:
        x_hat, scaled = qfn(x)
        err = (x_hat - x).abs()
        rel = err / (x.abs() + EPS)
        is_sub = scaled.abs() < 1.0
        R[method] = dict(x_hat=x_hat, scaled=scaled, err=err, rel=rel, is_sub=is_sub)
    return R


def excess_kurtosis(x):
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    return (((x - mu) ** 4).mean() / var ** 2 - 3).item()


# ── Distribution Generators ───────────────────────────────────────────
def gen_gaussian_std():
    return torch.randn(SHAPE)


def gen_gaussian_shifted():
    return torch.randn(SHAPE) * 2 + 0.5


def gen_laplace():
    return torch.distributions.Laplace(0, 1.0 / math.sqrt(2)).sample(SHAPE)


def gen_uniform():
    return torch.empty(SHAPE).uniform_(-math.sqrt(3), math.sqrt(3))


def gen_gaussian_outlier():
    x = torch.randn(SHAPE)
    n_outlier = max(1, int(SHAPE[1] * 0.02))
    rng = torch.Generator().manual_seed(SEED + 100)
    outlier_cols = torch.randperm(SHAPE[1], generator=rng)[:n_outlier]
    x[:, outlier_cols] *= 15.0
    return x


DISTRIBUTIONS = OrderedDict([
    ("gaussian_std", {
        "label": "$\\mathcal{N}(0,\\, 1)$",
        "short": "N(0,1)",
        "gen": gen_gaussian_std,
        "desc": "Post-LayerNorm baseline",
    }),
    ("gaussian_shifted", {
        "label": "$\\mathcal{N}(0.5,\\, \\sigma\\!=\\!2)$",
        "short": "N(0.5,σ=2)",
        "gen": gen_gaussian_shifted,
        "desc": "Biased high-var (FFN intermediate)",
    }),
    ("laplace", {
        "label": "Laplace$(0,\\, b\\!=\\!1/\\sqrt{2})$",
        "short": "Laplace",
        "gen": gen_laplace,
        "desc": "Heavy-tailed, unit variance",
    }),
    ("uniform", {
        "label": "$\\mathcal{U}(-\\sqrt{3},\\, \\sqrt{3})$",
        "short": "Uniform",
        "gen": gen_uniform,
        "desc": "Bounded, light-tailed contrast",
    }),
    ("gaussian_outlier", {
        "label": "$\\mathcal{N}(0,1)$ + 2% outlier @15$\\times$",
        "short": "N(0,1)+outlier",
        "gen": gen_gaussian_outlier,
        "desc": "Realistic LLM outlier channel pattern",
    }),
])

METHODS = ["nvfp4", "mxfp4"]


# ── Plot: Per-Distribution 5-Panel Report ──────────────────────────────
def plot_report(x, R, dist_info, out_path):
    kurt = excess_kurtosis(x)
    title = (
        f"NVFP4 vs MXFP4 (g={GROUP_SIZE})  |  {dist_info['label']}  |  "
        f"shape={SHAPE}  |  excess kurtosis={kurt:.1f}"
    )

    fig = plt.figure(figsize=(18, 11), layout="constrained")
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.12, wspace=0.15)
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[1, 4:6])
    fig.suptitle(title, fontsize=13, fontweight="bold")

    n = x.numel()

    # Compute metrics for both methods
    metrics = {}
    sub_fracs = {}
    for m in METHODS:
        r = R[m]
        mse = (r["err"] ** 2).mean()
        metrics[m] = dict(
            rmse=mse.sqrt().item(), mae=r["err"].mean().item(),
            rel_mean=r["rel"].mean().item(), rel_p50=r["rel"].median().item(),
        )
        sub_fracs[m] = r["is_sub"].sum().item() / n

    # ── (a) Aggregate Error Metrics ───────────────────────────────────
    ax = ax_a
    bar_w = 0.28
    xs = range(4)
    for i, m in enumerate(METHODS):
        met = metrics[m]
        vals = [met["rmse"], met["mae"], met["rel_mean"], met["rel_p50"]]
        positions = [xi + (i - 0.5) * bar_w for xi in xs]
        bars = ax.bar(
            positions, vals, bar_w, label=METHOD_LABELS[m],
            color=METHOD_COLORS[m], alpha=0.85,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7,
            )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["RMSE", "MAE", "Mean |relE|", "Median |relE|"])
    ax.set_ylabel("Error")
    ax.set_title("(a) Aggregate Error Metrics")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ── (b) Subnormal vs Normal Fraction ──────────────────────────────
    ax = ax_b
    bar_x = range(len(METHODS))
    sfs = [sub_fracs[m] for m in METHODS]
    nfs = [1 - sf for sf in sfs]
    colors_m = [METHOD_COLORS[m] for m in METHODS]

    ax.bar(bar_x, sfs, 0.5, label="Subnormal", color="#EF5350", alpha=0.8)
    ax.bar(bar_x, nfs, 0.5, bottom=sfs, label="Normal", color="#66BB6A", alpha=0.8)
    for i, (sf, nf) in enumerate(zip(sfs, nfs)):
        ax.text(
            i, sf / 2, f"{sf:.1%}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
        )
        ax.text(
            i, sf + nf / 2, f"{nf:.1%}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
        )
    ax.set_xticks(list(bar_x))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=11)
    ax.set_ylabel("Fraction of elements")
    ax.set_title("(b) Subnormal vs Normal Fraction")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=9)

    # ── (c) Relative Error CDF ────────────────────────────────────────
    ax = ax_c
    for m in METHODS:
        vals = R[m]["rel"].flatten().sort()[0]
        cdf = torch.linspace(0, 1, len(vals))
        step = max(1, len(vals) // 2000)
        ax.plot(
            vals[::step].numpy(), cdf[::step].numpy(),
            label=METHOD_LABELS[m], color=METHOD_COLORS[m], linewidth=1.8,
        )
    ax.axvline(0.25, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(
        0.26, 0.5, "25% (FP4 normal bound)", fontsize=8, color="gray",
        rotation=90, va="center",
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Relative error |$\\hat{x}-x$| / |$x$|")
    ax.set_ylabel("CDF")
    ax.set_title("(c) Per-Element Relative Error CDF")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # ── Precompute FP4 error curves ───────────────────────────────────
    s_vals = torch.linspace(0.01, 6.2, 2000)
    fp4_out = fp4_round_vec(s_vals)
    bins = torch.linspace(0, 6.5, 80)

    # ── (d) FP4 Relative Error + Scaled Value Distributions ───────────
    ax = ax_d
    ax2 = ax.twinx()
    fp4_rel_err = (fp4_out - s_vals).abs() / s_vals
    ax.plot(
        s_vals.numpy(), fp4_rel_err.numpy(), "k-", linewidth=1.2, alpha=0.8,
        label="FP4 rel error", zorder=3,
    )
    ax.axvspan(0, 1.0, color="red", alpha=0.08)
    ax.text(
        0.5, 0.92, "subnormal\nzone", ha="center", fontsize=8,
        color="#C62828", fontstyle="italic", transform=ax.get_xaxis_transform(),
    )
    for m in METHODS:
        sc_abs = R[m]["scaled"].abs().flatten()
        hist_vals = torch.histogram(sc_abs, bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        density = hist_vals.hist / hist_vals.hist.sum() / (bins[1] - bins[0])
        ax2.fill_between(
            centers.numpy(), 0, density.numpy(),
            alpha=0.18, color=METHOD_COLORS[m], label=f"{METHOD_LABELS[m]} density",
        )
        ax2.plot(
            centers.numpy(), density.numpy(),
            color=METHOD_COLORS[m], linewidth=1.2, alpha=0.7,
        )
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("|scaled value|")
    ax.set_ylabel("Relative error (black)")
    ax2.set_ylabel("Density (fills)")
    ax.set_title("(d) FP4 Relative Error + Distribution Shift")
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)

    # ── (e) FP4 Absolute Error + Scaled Value Distributions ───────────
    ax = ax_e
    ax2e = ax.twinx()
    fp4_abs_err = (fp4_out - s_vals).abs()
    ax.plot(
        s_vals.numpy(), fp4_abs_err.numpy(), "k-", linewidth=1.2, alpha=0.8,
        label="FP4 abs error", zorder=3,
    )
    ax.axvspan(0, 1.0, color="red", alpha=0.08)
    ax.text(
        0.5, 0.92, "subnormal\nzone", ha="center", fontsize=8,
        color="#C62828", fontstyle="italic", transform=ax.get_xaxis_transform(),
    )
    for m in METHODS:
        sc_abs = R[m]["scaled"].abs().flatten()
        hist_vals = torch.histogram(sc_abs, bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        density = hist_vals.hist / hist_vals.hist.sum() / (bins[1] - bins[0])
        ax2e.fill_between(
            centers.numpy(), 0, density.numpy(),
            alpha=0.18, color=METHOD_COLORS[m], label=f"{METHOD_LABELS[m]} density",
        )
        ax2e.plot(
            centers.numpy(), density.numpy(),
            color=METHOD_COLORS[m], linewidth=1.2, alpha=0.7,
        )
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, fp4_abs_err.max().item() * 1.15)
    ax.set_xlabel("|scaled value|")
    ax.set_ylabel("Absolute error (black)")
    ax2e.set_ylabel("Density (fills)")
    ax.set_title("(e) FP4 Absolute Error + Distribution Shift")
    ax.legend(loc="upper left", fontsize=8)
    ax2e.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved {out_path}]")
    return metrics, sub_fracs


# ── Plot: Cross-Distribution Comparison ────────────────────────────────
def plot_comparison(all_data, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    fig.suptitle(
        f"NVFP4 vs MXFP4 (g={GROUP_SIZE})  |  Cross-Distribution Comparison",
        fontsize=14, fontweight="bold",
    )

    dist_names = list(all_data.keys())
    n_dist = len(dist_names)
    short_labels = [DISTRIBUTIONS[d]["short"] for d in dist_names]

    # ── (a) Input Distribution PDFs ───────────────────────────────────
    ax = axes[0, 0]
    for name in dist_names:
        x_flat = all_data[name]["x"].flatten().numpy()
        lo, hi = np.percentile(x_flat, [0.5, 99.5])
        bins_pdf = np.linspace(lo, hi, 120)
        counts, edges = np.histogram(x_flat, bins=bins_pdf, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(
            centers, counts, linewidth=1.5, color=DIST_COLORS[name],
            label=DISTRIBUTIONS[name]["short"],
        )
        ax.fill_between(centers, 0, counts, alpha=0.1, color=DIST_COLORS[name])
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Density")
    ax.set_title("(a) Input Distribution Shapes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (b) RMSE: NVFP4 vs MXFP4 per distribution ────────────────────
    ax = axes[0, 1]
    bar_w = 0.3
    x_pos = np.arange(n_dist)
    for i, m in enumerate(METHODS):
        rmses = [all_data[d]["metrics"][m]["rmse"] for d in dist_names]
        pos = x_pos + (i - 0.5) * bar_w
        bars = ax.bar(
            pos, rmses, bar_w, label=METHOD_LABELS[m],
            color=METHOD_COLORS[m], alpha=0.85,
        )
        for b, v in zip(bars, rmses):
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45,
            )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("(b) RMSE: NVFP4 vs MXFP4")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ── (c) Subnormal Fraction ────────────────────────────────────────
    ax = axes[1, 0]
    for i, m in enumerate(METHODS):
        subs = [all_data[d]["sub_fracs"][m] for d in dist_names]
        pos = x_pos + (i - 0.5) * bar_w
        bars = ax.bar(
            pos, subs, bar_w, label=METHOD_LABELS[m],
            color=METHOD_COLORS[m], alpha=0.85,
        )
        for b, v in zip(bars, subs):
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{v:.0%}", ha="center", va="bottom", fontsize=7, rotation=45,
            )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("Subnormal fraction")
    ax.set_title("(c) Subnormal Fraction: NVFP4 vs MXFP4")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ── (d) MXFP4 / NVFP4 RMSE Ratio ─────────────────────────────────
    ax = axes[1, 1]
    ratios = [
        all_data[d]["metrics"]["mxfp4"]["rmse"]
        / all_data[d]["metrics"]["nvfp4"]["rmse"]
        for d in dist_names
    ]
    colors_bar = [DIST_COLORS[d] for d in dist_names]
    bars = ax.bar(x_pos, ratios, 0.5, color=colors_bar, alpha=0.85)
    for b, v in zip(bars, ratios):
        sign = "+" if v >= 1.0 else ""
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
            f"{v:.3f}\n({sign}{(v-1)*100:.1f}%)", ha="center", va="bottom",
            fontsize=8, fontweight="bold",
        )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("RMSE(MXFP4) / RMSE(NVFP4)")
    ax.set_title("(d) MXFP4 vs NVFP4 Error Ratio (>1 = MXFP4 worse)")
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved {out_path}]")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_data = OrderedDict()

    for idx, (name, dist_info) in enumerate(DISTRIBUTIONS.items(), 1):
        print(f"\n{'='*70}")
        print(f"  [{idx}/{len(DISTRIBUTIONS)}] {dist_info['short']} — {dist_info['desc']}")
        print(f"{'='*70}")

        torch.manual_seed(SEED)
        x = dist_info["gen"]()

        kurt = excess_kurtosis(x)
        print(
            f"  shape={tuple(x.shape)}  mean={x.mean():.4f}  std={x.std():.4f}"
            f"  kurtosis={kurt:.2f}"
        )
        print(f"  range=[{x.min():.3f}, {x.max():.3f}]")

        R = compute_results(x)
        out_path = f"{OUT_DIR}/report_{idx:02d}_{name}.png"
        metrics, sub_fracs = plot_report(x, R, dist_info, out_path)

        print(
            f"\n  {'Method':<8s}  {'RMSE':>8s}  {'MAE':>8s}"
            f"  {'relE_mean':>10s}  {'relE_p50':>9s}  {'sub_frac':>9s}"
        )
        print(
            f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*9}  {'─'*9}"
        )
        for m in METHODS:
            met = metrics[m]
            sf = sub_fracs[m]
            print(
                f"  {METHOD_LABELS[m]:<8s}  {met['rmse']:8.5f}  {met['mae']:8.5f}"
                f"  {met['rel_mean']:10.5f}  {met['rel_p50']:9.5f}  {sf:9.1%}"
            )

        ratio = metrics["mxfp4"]["rmse"] / metrics["nvfp4"]["rmse"]
        print(f"\n  RMSE(MXFP4)/RMSE(NVFP4) = {ratio:.3f}  ({'+' if ratio>=1 else ''}{(ratio-1)*100:.1f}%)")

        all_data[name] = dict(x=x, R=R, metrics=metrics, sub_fracs=sub_fracs)

    # ── Cross-Distribution Comparison ─────────────────────────────────
    print(f"\n{'='*70}")
    print("  Generating cross-distribution comparison...")
    print(f"{'='*70}")
    plot_comparison(all_data, f"{OUT_DIR}/report_comparison.png")

    # ── Summary Table ─────────────────────────────────────────────────
    print(
        f"\n  {'Distribution':<20s}  {'Kurt':>5s}  {'NVFP4':>8s}"
        f"  {'MXFP4':>8s}  {'MX/NV':>6s}  {'NV_sub':>7s}  {'MX_sub':>7s}"
    )
    print(
        f"  {'─'*20}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*7}"
    )
    for name, d in all_data.items():
        kurt = excess_kurtosis(d["x"])
        nv = d["metrics"]["nvfp4"]["rmse"]
        mx = d["metrics"]["mxfp4"]["rmse"]
        ratio = mx / nv
        nv_sub = d["sub_fracs"]["nvfp4"]
        mx_sub = d["sub_fracs"]["mxfp4"]
        print(
            f"  {DISTRIBUTIONS[name]['short']:<20s}  {kurt:5.1f}  {nv:8.5f}"
            f"  {mx:8.5f}  {ratio:6.3f}  {nv_sub:7.1%}  {mx_sub:7.1%}"
        )

    print(f"\n  All reports saved to {OUT_DIR}/\n")


if __name__ == "__main__":
    main()
