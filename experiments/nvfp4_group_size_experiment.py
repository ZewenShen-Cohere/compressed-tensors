"""
NVFP4 Group Size Sensitivity Experiment
========================================
Tests the hypothesis: For Gaussian-distributed activations, NVFP4 with
group_size=16 vs 32 vs 64 does not produce significantly different
quantization errors, because group_size only affects values pushed into
the FP4 subnormal region.

Usage:
    python experiments/nvfp4_group_size_experiment.py
"""

import math

import torch
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.helpers import (
    compute_dynamic_scales_and_zp,
    generate_gparam,
)

# ── Configuration ──────────────────────────────────────────────────────
SEED = 42
TENSOR_SHAPE = (64, 1024)  # 65536 elements, divisible by 16/32/64
GROUP_SIZES = [16, 32, 64]
SIGMA = 1.0
EPS = 1e-10  # for relative error when |x| ~ 0


# ── Helpers ────────────────────────────────────────────────────────────
def make_nvfp4_args(group_size: int) -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        group_size=group_size,
        scale_dtype=torch.float8_e4m3fn,
        zp_dtype=torch.float8_e4m3fn,
    )


def nvfp4_fake_quantize(x: torch.Tensor, group_size: int):
    """Full NVFP4 fake-quantize: compute scales, quantize, dequantize."""
    args = make_nvfp4_args(group_size)
    global_scale = generate_gparam(x.min(), x.max())
    scale, zp = compute_dynamic_scales_and_zp(
        x, args, module=None, global_scale=global_scale
    )
    x_hat = fake_quantize(x, scale, zp, args, global_scale=global_scale)
    return x_hat, scale, global_scale, args


def get_scaled_values(x: torch.Tensor, scale: torch.Tensor,
                      global_scale: torch.Tensor, group_size: int):
    """Compute the scaled values that enter FP4 rounding (per-element)."""
    eff_scale = scale / global_scale  # (rows, num_groups)
    rows, cols = x.shape
    num_groups = cols // group_size
    x_grouped = x.reshape(rows, num_groups, group_size)
    # eff_scale shape: (rows, num_groups) -> (rows, num_groups, 1) for broadcast
    scaled = x_grouped / eff_scale.unsqueeze(-1)
    return scaled.reshape(rows, cols)


def fp4_grid_point(scaled_val: float) -> float:
    """Return the FP4 E2M1 grid point for a single scalar."""
    t = torch.tensor([scaled_val])
    return FP4_E2M1_DATA.cast_to_fp4(t).item()


def print_header(title: str):
    w = 72
    print()
    print("=" * w)
    print(f" {title}")
    print("=" * w)


def print_table(headers: list[str], rows: list[list], col_width: int = 12):
    fmt = " | ".join(f"{{:>{col_width}}}" for _ in headers)
    sep = "-+-".join("-" * col_width for _ in headers)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.6f}")
            elif isinstance(v, int):
                cells.append(str(v))
            else:
                cells.append(str(v))
        print(fmt.format(*cells))


# ── Main ───────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    x = torch.randn(TENSOR_SHAPE) * SIGMA

    print_header("NVFP4 Group Size Sensitivity Experiment")
    print(f"  Tensor shape : {TENSOR_SHAPE}")
    print(f"  dtype        : {x.dtype}")
    print(f"  sigma        : {SIGMA}")
    print(f"  seed         : {SEED}")
    print(f"  |x| range    : [{x.abs().min():.6f}, {x.abs().max():.6f}]")

    # Pre-compute results for all group sizes
    results = {}
    for g in GROUP_SIZES:
        x_hat, scale, gs, args = nvfp4_fake_quantize(x, g)
        scaled = get_scaled_values(x, scale, gs, g)
        err = (x_hat - x).abs()
        rel_err = err / (x.abs() + EPS)
        is_sub = scaled.abs() < 1.0
        results[g] = dict(
            x_hat=x_hat, scale=scale, global_scale=gs, args=args,
            scaled=scaled, err=err, rel_err=rel_err, is_sub=is_sub,
        )
    # Print global_scale (same for all group sizes since it's per-tensor)
    print(f"  global_scale : {results[GROUP_SIZES[0]]['global_scale'].item():.4f}")

    # ── Section 1: Overall Error Metrics ───────────────────────────────
    print_header("Section 1: Overall Error Metrics")

    headers = ["group_size", "RMSE", "MAE", "max_err",
               "relE_mean", "relE_p50", "relE_p95", "SNR_dB"]
    rows = []
    for g in GROUP_SIZES:
        r = results[g]
        mse = (r["err"] ** 2).mean()
        rmse = mse.sqrt().item()
        mae = r["err"].mean().item()
        max_err = r["err"].max().item()
        re = r["rel_err"]
        re_mean = re.mean().item()
        re_p50 = re.median().item()
        re_p95 = torch.quantile(re.float(), 0.95).item()
        snr = 10 * torch.log10((x ** 2).mean() / mse).item()
        rows.append([g, rmse, mae, max_err, re_mean, re_p50, re_p95, snr])
    print_table(headers, rows)

    # ── Section 2: Subnormal vs Normal Region Breakdown ────────────────
    print_header("Section 2: Subnormal vs Normal Region Breakdown")

    # Theoretical prediction
    print("\n  Theoretical predictions (Gaussian, expected absmax ~ sigma * sqrt(2*ln(g))):")
    for g in GROUP_SIZES:
        expected_absmax = SIGMA * math.sqrt(2 * math.log(g))
        threshold = expected_absmax / 6.0
        # P(|X| < t) = erf(t / (sigma * sqrt(2)))
        theory_frac = math.erf(threshold / (SIGMA * math.sqrt(2)))
        print(f"    g={g:3d}: E[absmax]={expected_absmax:.3f}  "
              f"threshold={threshold:.4f}  "
              f"theory_sub_frac={theory_frac:.3f}")

    print()
    headers = ["group_size", "sub_count", "sub_frac", "MAE_sub", "MAE_nrm",
               "relE_sub", "relE_nrm"]
    rows = []
    for g in GROUP_SIZES:
        r = results[g]
        is_sub = r["is_sub"]
        is_nrm = ~is_sub
        n_sub = is_sub.sum().item()
        n_total = x.numel()
        sub_frac = n_sub / n_total

        mae_sub = r["err"][is_sub].mean().item() if n_sub > 0 else 0.0
        mae_nrm = r["err"][is_nrm].mean().item() if is_nrm.sum() > 0 else 0.0
        re_sub = r["rel_err"][is_sub].mean().item() if n_sub > 0 else 0.0
        re_nrm = r["rel_err"][is_nrm].mean().item() if is_nrm.sum() > 0 else 0.0

        rows.append([g, n_sub, sub_frac, mae_sub, mae_nrm, re_sub, re_nrm])
    print_table(headers, rows)

    # ── Section 3: Monotonic Subnormal Growth Check ────────────────────
    print_header("Section 3: Monotonic Subnormal Growth Check")

    sub_16 = results[16]["is_sub"]
    sub_32 = results[32]["is_sub"]
    sub_64 = results[64]["is_sub"]

    # Elements subnormal at g=16 but normal at g=32 (should be ~0 modulo E4M3 rounding)
    sub16_nrm32 = (sub_16 & ~sub_32).sum().item()
    sub16_nrm64 = (sub_16 & ~sub_64).sum().item()
    sub32_nrm64 = (sub_32 & ~sub_64).sum().item()

    # Elements that become newly subnormal
    nrm16_sub32 = (~sub_16 & sub_32).sum().item()
    nrm16_sub64 = (~sub_16 & sub_64).sum().item()
    nrm32_sub64 = (~sub_32 & sub_64).sum().item()

    print(f"  Violations of monotonicity (should be ~0, modulo E4M3 scale rounding):")
    print(f"    subnormal@g=16 but normal@g=32 : {sub16_nrm32}")
    print(f"    subnormal@g=16 but normal@g=64 : {sub16_nrm64}")
    print(f"    subnormal@g=32 but normal@g=64 : {sub32_nrm64}")
    print()
    print(f"  New subnormal elements (strict superset growth):")
    print(f"    normal@g=16  -> subnormal@g=32 : {nrm16_sub32}")
    print(f"    normal@g=16  -> subnormal@g=64 : {nrm16_sub64}")
    print(f"    normal@g=32  -> subnormal@g=64 : {nrm32_sub64}")

    # ── Section 4: Element-Level Diagnostic ────────────────────────────
    print_header("Section 4: Element-Level Diagnostic")
    print("  Shows how the same element maps to different FP4 grid points")
    print("  across group sizes, even when staying in the normal range.\n")

    # Classify elements
    always_normal = ~sub_16 & ~sub_32 & ~sub_64
    always_sub = sub_16 & sub_32 & sub_64
    transition = ~sub_16 & sub_64  # normal at g=16, subnormal at g=64

    def pick_indices(mask, n):
        """Pick up to n element indices from a boolean mask."""
        flat_idx = mask.flatten().nonzero(as_tuple=False).squeeze(-1)
        if len(flat_idx) == 0:
            return []
        step = max(1, len(flat_idx) // n)
        selected = flat_idx[::step][:n]
        rows_idx = selected // x.shape[1]
        cols_idx = selected % x.shape[1]
        return list(zip(rows_idx.tolist(), cols_idx.tolist()))

    categories = [
        ("ALWAYS NORMAL", always_normal, 4),
        ("ALWAYS SUBNORMAL", always_sub, 3),
        ("TRANSITIONING (normal@16 -> subnormal@64)", transition, 3),
    ]

    grid_flip_count_normal = 0
    grid_flip_total_normal = 0

    for cat_name, mask, n_pick in categories:
        indices = pick_indices(mask, n_pick)
        if not indices:
            print(f"  [{cat_name}] -- no elements found\n")
            continue
        print(f"  [{cat_name}]")
        for ri, ci in indices:
            val = x[ri, ci].item()
            print(f"  [{ri:3d},{ci:4d}]  x = {val:+.6f}")

            grid_points_this = []
            for g in GROUP_SIZES:
                r = results[g]
                sc = r["scaled"][ri, ci].item()
                gp = fp4_grid_point(sc)
                xh = r["x_hat"][ri, ci].item()
                ae = r["err"][ri, ci].item()
                re = r["rel_err"][ri, ci].item()
                label = "SUB" if r["is_sub"][ri, ci] else "NRM"
                print(f"    g={g:2d}: eff_s={r['scale'][ri, ci//g].item() / r['global_scale'].item():.6f}"
                      f"  scaled={sc:+8.4f}  fp4={gp:+4.1f}"
                      f"  x_hat={xh:+.6f}  err={ae:.6f}  rel={re:.4f}  [{label}]")
                grid_points_this.append(gp)

            # Track grid flips for always-normal elements
            if cat_name == "ALWAYS NORMAL":
                grid_flip_total_normal += 1
                if len(set(grid_points_this)) > 1:
                    grid_flip_count_normal += 1
                    print(f"    >> DIFFERENT FP4 grid points across group sizes!")
                else:
                    print(f"    >> Same FP4 grid point across all group sizes.")
            print()

    # ── Section 5: Summary Verdict ─────────────────────────────────────
    print_header("Section 5: Summary Verdict")

    rmse_16 = (results[16]["err"] ** 2).mean().sqrt().item()
    rmse_64 = (results[64]["err"] ** 2).mean().sqrt().item()
    rmse_ratio = rmse_64 / rmse_16

    sub_frac_16 = results[16]["is_sub"].sum().item() / x.numel()
    sub_frac_64 = results[64]["is_sub"].sum().item() / x.numel()

    re_mean_16 = results[16]["rel_err"].mean().item()
    re_mean_64 = results[64]["rel_err"].mean().item()

    print(f"  1. RMSE ratio (g=64 / g=16): {rmse_ratio:.3f}")
    if rmse_ratio < 1.10:
        print(f"     -> Within 10% -- supports the hypothesis that error is insensitive to group_size.")
    elif rmse_ratio < 1.20:
        print(f"     -> {(rmse_ratio-1)*100:.1f}% increase -- modest but real difference.")
    else:
        print(f"     -> {(rmse_ratio-1)*100:.1f}% increase -- significant difference, hypothesis weakened.")

    print(f"\n  2. Subnormal fraction: g=16: {sub_frac_16:.3f}  g=64: {sub_frac_64:.3f}"
          f"  (delta: {(sub_frac_64-sub_frac_16)*100:.1f} percentage points)")
    print(f"     The subnormal set grows monotonically. "
          f"New subnormal elements from g=16 to g=64: {nrm16_sub64}")

    print(f"\n  3. Mean relative error: g=16: {re_mean_16:.4f}  g=64: {re_mean_64:.4f}"
          f"  (increase: {(re_mean_64/re_mean_16 - 1)*100:.1f}%)")

    if grid_flip_total_normal > 0:
        print(f"\n  4. Of {grid_flip_total_normal} always-normal elements examined, "
              f"{grid_flip_count_normal} had DIFFERENT FP4 grid points across group sizes.")
        if grid_flip_count_normal > 0:
            print(f"     -> The claim 'if not subnormal, error is the same regardless of group_size'")
            print(f"        is FALSE at the individual element level.")
        else:
            print(f"     -> In this sample, always-normal elements landed on the same grid point.")
            print(f"        (This can happen for values far from FP4 rounding boundaries.)")

    print(f"\n  5. Overall: The aggregate error increase from g=16 to g=64 is "
          f"{(rmse_ratio-1)*100:.1f}% (RMSE).")
    print(f"     FP4's floating-point grid limits the damage of larger group sizes,")
    print(f"     but the effect is not zero: both the growing subnormal fraction")
    print(f"     and grid-point shifts in the normal range contribute.")
    print()


if __name__ == "__main__":
    main()
