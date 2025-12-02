"""
Head construction utilities.

These helpers generate transformer_heads.HeadConfig objects for the three
primary strategies we support: single regression head, per-layer multi-head
regression, and multi-head with learned layer weighting.
"""

from typing import List, Tuple

from transformer_heads.config import HeadConfig


def build_single_head(
    hidden_size: int,
    loss_weight: float = 2e-3,
    name: str = "mean_regression",
) -> List[HeadConfig]:
    return [
        HeadConfig(
            name=name,
            layer_hook=-1,
            in_size=hidden_size,
            output_activation="linear",
            is_causal_lm=False,
            pred_for_sequence=True,
            loss_fct="mse",
            num_outputs=1,
            is_regression=True,
            loss_weight=loss_weight,
        )
    ]


def build_multi_head(
    hidden_size: int,
    layer_range: Tuple[int, int],
    per_layer_loss_weight: float = 2e-4,
    regression_loss_weight: float = 2e-3,
    regression_head_name: str = "mean_regression",
    aggregation: str = "average",
) -> List[HeadConfig]:
    """
    Build per-layer regression heads plus an aggregation head.

    layer_range: (start, end) corresponds to negative layer hooks; e.g.,
                 (17, 25) builds heads on layers -17 .. -24.
    aggregation: "average" or "learned_softmax".
    """
    start, end = layer_range
    assert start < end, "layer_range must be (start, end) with start < end"

    heads: List[HeadConfig] = []
    for i in range(start, end):
        heads.append(
            HeadConfig(
                name=f"regression_head_{i}",
                layer_hook=-i,
                in_size=hidden_size,
                output_activation="linear",
                is_causal_lm=False,
                pred_for_sequence=True,
                loss_fct="mse",
                num_outputs=1,
                is_regression=True,
                loss_weight=per_layer_loss_weight,
            )
        )

    heads.append(
        HeadConfig(
            name=regression_head_name,
            layer_hook=-1,
            in_size=hidden_size,
            output_activation="linear",
            is_causal_lm=False,
            pred_for_sequence=True,
            loss_fct="mse",
            num_outputs=1,
            is_regression=True,
            loss_weight=regression_loss_weight,
            aggregation=aggregation,
        )
    )
    return heads


def build_dynamic_weight_heads(
    hidden_size: int,
    layer_range: Tuple[int, int],
    regression_loss_weight: float = 2e-3,
    regression_head_name: str = "mean_regression",
) -> List[HeadConfig]:
    """
    Build heads for dynamic layer weighting:
    - layer embedding heads emit hidden_size outputs
    - final regression head consumes combined embedding (applied in model forward)
    """
    start, end = layer_range
    assert start < end, "layer_range must be (start, end) with start < end"

    heads: List[HeadConfig] = []
    for i in range(start, end):
        heads.append(
            HeadConfig(
                name=f"layer_{i}_embed",
                layer_hook=-i,
                in_size=hidden_size,
                output_activation="linear",
                is_causal_lm=False,
                pred_for_sequence=True,
                loss_fct=None,
                num_outputs=hidden_size,
                is_regression=False,
                loss_weight=0.0,
            )
        )

    heads.append(
        HeadConfig(
            name=regression_head_name,
            layer_hook=-1,
            in_size=hidden_size,
            output_activation="linear",
            is_causal_lm=False,
            pred_for_sequence=True,
            loss_fct="mse",
            num_outputs=1,
            is_regression=True,
            loss_weight=regression_loss_weight,
        )
    )
    return heads

