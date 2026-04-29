# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import round_to_quantized_type_dtype
from compressed_tensors.quantization.quant_args import (
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import (
    calculate_qparams,
    generate_mx_scales,
    maybe_convert_from_mx_exp,
    round_to_power_2,
    should_generate_mx_scales,
)


def test_should_generate_mx_scales_mxfp8():
    """Test that should_generate_mx_scales returns True for MXFP8 args."""
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    )
    assert should_generate_mx_scales(args) is True


def test_should_generate_mx_scales_mxfp4():
    """Test that should_generate_mx_scales returns True for MXFP4 args."""
    args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    )
    assert should_generate_mx_scales(args) is True


def test_should_generate_mx_scales_regular_fp8():
    """Test that should_generate_mx_scales returns False for regular FP8."""
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
    )
    assert should_generate_mx_scales(args) is False


def test_should_generate_mx_scales_wrong_group_size():
    """Test that should_generate_mx_scales returns False for non-32 group size."""
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
    )
    assert should_generate_mx_scales(args) is False


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64]
)
def test_mxfp8_scales_e2e(dtype):
    """End-to-end test for MXFP8 scale generation and conversion."""
    mock_weight = torch.normal(mean=0.0002, std=0.0576, size=(2880, 2880))

    x = mock_weight.reshape(*mock_weight.shape[:-1], -1, 32).to(dtype)
    min_vals = torch.amin(x, dim=-1)
    max_vals = torch.amax(x, dim=-1)

    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))
    block_max = torch.max(torch.abs(min_vals), torch.abs(max_vals))

    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    )

    scales = generate_mx_scales(block_max, num_bits=8)
    scales = round_to_quantized_type_dtype(scales, dtype=args.scale_dtype)

    converted_ct = maybe_convert_from_mx_exp(args=args, scale=scales)

    scales_exp = torch.log2(converted_ct)
    block_max_exp = torch.floor(torch.log2(round_to_power_2(block_max))) - 8
    assert torch.equal(scales_exp, block_max_exp)


def test_mxfp8_nearest_scale_rounding_matches_current_helper():
    block_max = torch.tensor([200.0, 449.0, 700.0, 1000.0], dtype=torch.float32)

    scales = generate_mx_scales(block_max, num_bits=8, rounding="nearest")
    expected = 127 + torch.floor(torch.log2(round_to_power_2(block_max))) - 8

    assert torch.equal(scales, expected)


def test_mxfp8_ceil_scale_rounding_prevents_overflow():
    block_max = torch.tensor([448.0, 449.0, 896.0, 897.0], dtype=torch.float32)

    scales = generate_mx_scales(block_max, num_bits=8, rounding="ceil")
    scale_values = 2.0 ** (scales - 127)

    assert torch.all(block_max / scale_values <= FP8_E4M3_DATA.max)
    assert torch.all(block_max / (scale_values / 2) > FP8_E4M3_DATA.max)


def test_mxfp8_mse_scale_rounding_selects_lower_error_candidate():
    lower_wins = torch.tensor([449.0] + [0.002] * 31, dtype=torch.float32)
    upper_wins = torch.full((32,), 700.0, dtype=torch.float32)
    observed = torch.stack((lower_wins, upper_wins))
    block_max = torch.amax(torch.abs(observed), dim=-1)

    scales = generate_mx_scales(
        block_max, num_bits=8, rounding="mse", observed=observed
    )

    assert torch.equal(scales, torch.tensor([127.0, 128.0]))


def test_mxfp8_mse_scale_rounding_requires_observed_values():
    block_max = torch.tensor([700.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="observed must be provided"):
        generate_mx_scales(block_max, num_bits=8, rounding="mse")


def test_mxfp8_scale_rounding_from_quantization_args():
    observed = torch.tensor([[449.0] + [0.002] * 31], dtype=torch.float32)
    min_vals = torch.amin(observed, dim=-1)
    max_vals = torch.amax(observed, dim=-1)
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
        mxfp_scale_rounding="ceil",
    )

    scale, _ = calculate_qparams(
        min_vals=min_vals,
        max_vals=max_vals,
        quantization_args=args,
        observed=observed,
    )

    assert torch.equal(scale, torch.tensor([2.0]))
