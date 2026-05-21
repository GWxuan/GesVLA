"""
Policy configuration and loading for the 2VLM model with reference image support.

"""

import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp
from flax import traverse_util

import openpi.models.model_2vlm as _model
import openpi.policies.ges_policy_2vlm_withref as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import gesconfig_2vlm as _config
import openpi.transforms as transforms


logger = logging.getLogger("openpi.policy_config_2vlm_withref")


def _restore_embedder_from_vlm0(model, vlm0_params_path: str) -> None:
    """Restore Embedder weights from a VLM0 pre-training checkpoint.

    """
    vlm0_params_path = pathlib.Path(vlm0_params_path)
    if not vlm0_params_path.exists():
        logger.warning(
            "VLM0 pre-training checkpoint not found: %s — skipping Embedder restore",
            vlm0_params_path,
        )
        return

    print(f"Restoring Embedder from VLM0 pre-training checkpoint: {vlm0_params_path}")

    vlm0_params = _model.restore_params(vlm0_params_path, dtype=jnp.bfloat16)
    flat_vlm0 = traverse_util.flatten_dict(vlm0_params)

    # Locate the Embedder weight by key substring matching
    vlm0_embedder_key = None
    for k in flat_vlm0:
        k_str = "/".join(str(s) for s in k)
        if "embedder" in k_str and "input_embedding" in k_str:
            vlm0_embedder_key = k
            break

    if vlm0_embedder_key is None:
        all_keys = ["/".join(str(s) for s in k) for k in flat_vlm0]
        logger.error(
            "Embedder weight not found in VLM0 checkpoint. "
            "Total keys: %d. First 20: %s",
            len(all_keys),
            all_keys[:20],
        )
        return

    vlm0_embedder_weight = flat_vlm0[vlm0_embedder_key]
    print(
        f"  VLM0 Embedder key : {'/'.join(str(s) for s in vlm0_embedder_key)}\n"
        f"  VLM0 Embedder shape: {vlm0_embedder_weight.shape}, "
        f"dtype: {vlm0_embedder_weight.dtype}"
    )

    # Extract and flatten the current model state
    graphdef, state = nnx.split(model)
    state_dict = state.to_pure_dict()
    flat = traverse_util.flatten_dict(state_dict)

    embedder_key = None
    for k in flat:
        k_str = "/".join(str(s) for s in k)
        if "embedder" in k_str and "input_embedding" in k_str:
            embedder_key = k
            break

    if embedder_key is None:
        logger.error("Embedder key not found in current model parameters")
        return

    old_weight = flat[embedder_key]
    print(f"  Current Embedder shape: {old_weight.shape}, dtype: {old_weight.dtype}")

    if old_weight.shape != vlm0_embedder_weight.shape:
        logger.error(
            "Embedder shape mismatch: current %s vs VLM0 %s",
            old_weight.shape,
            vlm0_embedder_weight.shape,
        )
        return

    # Replace and merge back into the model (NNX modules are mutable)
    flat[embedder_key] = vlm0_embedder_weight.astype(old_weight.dtype)
    state_dict = traverse_util.unflatten_dict(flat)
    state.replace_by_pure_dict(state_dict)
    merged = nnx.merge(graphdef, state)
    model.__dict__.update(merged.__dict__)

    print("  Embedder weights successfully restored from VLM0 pre-training checkpoint")


def create_policy_2vlm_withref(
    config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    vlm0_pretrain_params: str | None = None,
    norm_stats_dir: str | None = None,
    norm_stats_asset_id: str | None = None,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.GesPolicy2VLMWithRef:
    """Create a 2VLM + reference-image deployment policy from a checkpoint.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))

    # Load the 2VLM action model
    print(f"Loading 2VLM model from: {checkpoint_dir}")
    model = config.model.load(
        _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    )

    # Optionally restore VLM0 Embedder weights from a pre-training checkpoint
    if vlm0_pretrain_params is not None:
        _restore_embedder_from_vlm0(model, vlm0_pretrain_params)
    else:
        logger.info("vlm0_pretrain_params not provided — skipping Embedder restore")

    # Build the data configuration and normalisation statistics
    data_config = config.data.create(config.assets_dirs, config.model)

    if norm_stats is None:
        if norm_stats_dir is not None and norm_stats_asset_id is not None:
            try:
                norm_stats = _checkpoints.load_norm_stats(
                    norm_stats_dir, norm_stats_asset_id
                )
            except Exception as exc:
                logger.warning("Failed to load norm_stats: %s — using empty dict", exc)
                norm_stats = {}
        elif data_config.asset_id is not None:
            logger.warning(
                "norm_stats_dir / norm_stats_asset_id not provided; "
                "norm_stats will be empty"
            )
            norm_stats = {}
        else:
            norm_stats = {}

    # Build the 2VLM action transform pipeline
    input_transforms = [
        *repack_transforms.inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
        *repack_transforms.outputs,
    ]

    policy = _policy.GesPolicy2VLMWithRef(
        model=model,
        input_transforms=input_transforms,
        output_transforms=output_transforms,
        sample_kwargs=sample_kwargs,
        metadata={
            "config_name": config.name,
            "checkpoint": str(checkpoint_dir),
        },
    )

    print("2VLM + reference-image policy created successfully")
    return policy
