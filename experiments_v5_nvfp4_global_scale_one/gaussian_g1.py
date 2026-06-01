"""
g=1 harm on a *homogeneous* Gaussian with support [-100, 100]
=============================================================
Follow-up to the outlier study: what if the data is just one Gaussian whose
values fill [-100, 100] (no isolated per-channel outlier tiers)? How much does
hard-fixing g=1 hurt compared to the calibrated g_optimal?

We sample x ~ N(0, sigma), clip to [-100, 100] so absmax = 100, and compare
reconstruction MSE under g=1 vs g_optimal. We also sweep g to show the
landscape, and (for context) report the result for a few sigmas.

Run
---
    python experiments_v5_nvfp4_global_scale_one/gaussian_g1.py
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
SUPPORT = 100.0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_nvfp4_args():
    return QuantizationArgs(
        num_bits=4, type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP, symmetric=True,
        group_size=GROUP_SIZE,
        scale_dtype=torch.float8_e4m3fn, zp_dtype=torch.float8_e4m3fn,
    )


def gen_gaussian(sigma):
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(*SHAPE, generator=g) * sigma
    x = x.clamp(-SUPPORT, SUPPORT)
    # Force the bound to be realized so absmax == SUPPORT exactly.
    x[0, 0] = SUPPORT
    x[0, 1] = -SUPPORT
    return x


def mse_at_g(x, g_value, args):
    gs = torch.tensor(float(g_value), dtype=torch.float32)
    scale, zp = compute_dynamic_scales_and_zp(x, args, module=None, global_scale=gs)
    x_hat = fake_quantize(x, scale, zp, args, global_scale=gs)
    return ((x_hat - x) ** 2).mean().item(), (x_hat - x)


def main():
    args = make_nvfp4_args()

    # sigma chosen so the Gaussian genuinely fills [-100, 100]:
    #   sigma=33.3 -> +-100 is ~3 sigma (tails reach the bound).
    # Also report a couple of others for context.
    sigmas = [33.3, 20.0, 50.0]

    print(f"Homogeneous Gaussian, clipped to support [-{SUPPORT:g}, {SUPPORT:g}]\n")
    print(f"  {'sigma':>7}  {'absmax':>7}  {'g_opt':>7}  "
          f"{'MSE@g_opt':>11}  {'MSE@g=1':>11}  {'ratio':>7}")

    landscape = None
    for si, sigma in enumerate(sigmas):
        x = gen_gaussian(sigma)
        absmax = x.abs().max().item()
        g_opt = generate_gparam(x.min(), x.max()).item()
        mse_opt, err_opt = mse_at_g(x, g_opt, args)
        mse_one, err_one = mse_at_g(x, 1.0, args)
        ratio = mse_one / mse_opt
        print(f"  {sigma:7.1f}  {absmax:7.2f}  {g_opt:7.3f}  "
              f"{mse_opt:11.5f}  {mse_one:11.5f}  {ratio:7.2f}")

        if si == 0:  # build the landscape / plot for the main sigma
            g_sweep = sorted(set(
                [round(v, 4) for v in np.geomspace(0.1, g_opt * 2.0, 40)]
                + [1.0, g_opt]
            ))
            mses = [mse_at_g(x, gv, args)[0] for gv in g_sweep]
            landscape = (sigma, absmax, g_opt, g_sweep, mses,
                         mse_opt, mse_one, err_opt, err_one)

    # ── Plot for the main sigma ───────────────────────────────────────
    sigma, absmax, g_opt, g_sweep, mses, mse_opt, mse_one, err_opt, err_one = landscape
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    fig.suptitle(
        f"Homogeneous Gaussian (sigma={sigma:g}, clipped to [-{SUPPORT:g},{SUPPORT:g}], "
        f"absmax={absmax:.1f})  |  g=1 vs g_optimal",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(g_sweep, mses, "o-", ms=4, color="#1f77b4")
    ax.axvline(1.0, color="#d62728", lw=2, label="g = 1")
    ax.axvline(g_opt, color="#2ca02c", ls="--", lw=2, label="g_optimal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fixed global scale g")
    ax.set_ylabel("tensor MSE (log)")
    ax.set_title(f"(a) MSE vs g   |   g=1: {mse_one:.4f}  vs  g_opt: {mse_opt:.4f}  "
                 f"(ratio {mse_one/mse_opt:.2f})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    e_opt = err_opt.flatten().numpy()
    e_one = err_one.flatten().numpy()
    span = max(np.abs(e_opt).max(), np.abs(e_one).max()) * 1.05
    bins = np.linspace(-span, span, 120)
    ax.hist(e_opt, bins=bins, histtype="step", density=True, lw=1.8,
            color="#2ca02c", label=f"g_optimal (MSE={mse_opt:.4f})")
    ax.hist(e_one, bins=bins, histtype="step", density=True, lw=1.8,
            color="#d62728", label=f"g=1 (MSE={mse_one:.4f})")
    ax.set_yscale("log")
    ax.set_xlabel("x_hat - x  (all elements)")
    ax.set_ylabel("density")
    ax.set_title("(b) Reconstruction error distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    png = os.path.join(OUT_DIR, "report_gaussian_g1.png")
    plt.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png}")


if __name__ == "__main__":
    main()
