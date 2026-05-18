"""GesVLA reasoning pre-training script.

Trains the gesture reasoning module (ReasoningModel) using text-loss on
visual-prompt data.  Typically invoked via the shell wrapper:
    bash train_scripts/train_onetwovla_cocktail.sh
"""

import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
from flax.training import common_utils

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.data_loader as _data_loader
import openpi.training.gesconfig as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from openpi.models.reasoning_model import make_attn_mask

matplotlib.use('Agg')  # Non-interactive backend for server training

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        # wandb.init(id=run_id, resume="must", project=config.project_name)
        # we may not resume from the same wandb run
        # as the loaded step from checkpoint might be earlier than wandb's step
        # and wandb only supports monotonically increasing step
        wandb_name = config.exp_name + '-resumed'
        wandb.init(
            name=wandb_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Load a subset of weights from a checkpoint.

    Only leaves whose key exists in *params_shape* **and** whose shape/dtype
    matches the expected specification are kept.  All other leaves are
    silently skipped so that newly-added parameters stay randomly initialised.
    """
    loaded_params = loader.load(params_shape)

    # Subset check (not strict equality) — keep intersection only.
    flat_exp = traverse_util.flatten_dict(params_shape)
    flat_ld  = traverse_util.flatten_dict(loaded_params)

    filtered = {}
    for k, v_ld in flat_ld.items():
        if k not in flat_exp:
            continue  # Skip unexpected keys from checkpoint.
        v_exp = flat_exp[k]

        # Leaf shape/dtype compatibility check.
        exp_shape = getattr(v_exp, "shape", None)
        exp_dtype = getattr(v_exp, "dtype", None)
        ld_shape  = getattr(v_ld,  "shape", None)
        ld_dtype  = getattr(v_ld,  "dtype", None)

        shape_ok = (exp_shape is None or ld_shape is None or tuple(exp_shape) == tuple(ld_shape))
        dtype_ok = (exp_dtype is None or ld_dtype is None or exp_dtype == ld_dtype)

        if not (shape_ok and dtype_ok):
            continue  # Shape/dtype mismatch — keep random init.

        filtered[k] = v_ld

    # Strip jax.ShapeDtypeStruct placeholders; return real arrays only.
    filtered = {k: v for k, v in filtered.items() if not isinstance(v, jax.ShapeDtypeStruct)}
    return traverse_util.unflatten_dict(filtered)

@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss, info = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), info

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, train_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in-place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(train_info)
    return new_state, info


@at.typecheck
def validate_reasoning(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.FuseObservation, _model.Actions],
) -> dict[str, at.Array]:
    """Validate reasoning: compute text loss and decode predicted tokens."""
    if state.ema_decay is None:
        model = nnx.merge(state.model_def, state.params)
    else:
        model = nnx.merge(state.model_def, state.ema_params)
    model.eval()

    observation, _ = batch

    # Build dummy actions (reasoning-only, no action loss).
    dummy_actions = jnp.zeros((observation.state.shape[0], config.model.action_horizon, config.model.action_dim))
    observation = dataclasses.replace(
        observation,
        diffusion_loss_mask=jnp.zeros(observation.state.shape[0], dtype=jnp.bool_)
    )

    # Compute loss via the model's internal method.
    loss, info = model.compute_loss(rng, observation, dummy_actions, train=False)

    # Obtain predicted tokens for display.
    img_txt_tokens, img_txt_mask, img_txt_ar_mask = model.embed_img_txt(observation)
    img_txt_attn_mask = make_attn_mask(img_txt_mask, img_txt_ar_mask)
    positions = jnp.cumsum(img_txt_mask, axis=1) - 1
    (img_txt_pre_logits,), _ = model.PaliGemma.llm([img_txt_tokens],
                                                    mask=img_txt_attn_mask,
                                                    positions=positions)

    txt_logits = model.PaliGemma.llm(
        img_txt_pre_logits[:, -observation.tokenized_prompt.shape[1]:],
        method="embedder_decode",
    )
    
    # Greedy-decode predicted tokens.
    pred_tokens = jnp.argmax(txt_logits, axis=-1)
    
    num_samples = min(observation.tokenized_prompt.shape[0], 6)
    
    return {
        'val_text_loss': info['text_loss'],
        'val_batch_size': observation.state.shape[0],
        'pred_tokens': pred_tokens[:num_samples],
        'true_tokens': observation.tokenized_prompt[:num_samples],
        'token_mask': observation.tokenized_prompt_mask[:num_samples],
        'ar_mask': observation.token_ar_mask[:num_samples],
    }

def decode_predictions(val_info: dict, tokenizer) -> tuple[list, list]:
    """Decode only the suffix (model-generated) portion of predicted tokens."""
    pred_texts = []
    true_texts = []
    
    pred_tokens = val_info['pred_tokens']
    true_tokens = val_info['true_tokens']
    token_mask = val_info['token_mask']
    ar_mask = val_info['ar_mask']
    
    batch_size = pred_tokens.shape[0]
    
    for i in range(batch_size):
        # Find suffix start (where ar_mask transitions to 1).
        suffix_start = find_suffix_start(ar_mask[i])
        
        if suffix_start >= 0:
            suffix_mask = token_mask[i][suffix_start:]
            valid_pred_suffix = pred_tokens[i][suffix_start:][suffix_mask]
            valid_true_suffix = true_tokens[i][suffix_start:][suffix_mask]
            
            try:
                pred_text = tokenizer.extract_thoughts(valid_pred_suffix)
                true_text = tokenizer.extract_thoughts(valid_true_suffix)
            except Exception as e:
                pred_text = f"Decode error: {str(e)}"
                true_text = f"Decode error: {str(e)}"
        else:
            # Fallback: decode the entire valid sequence.
            valid_pred_tokens = pred_tokens[i][token_mask[i].astype(bool)]
            valid_true_tokens = true_tokens[i][token_mask[i].astype(bool)]
            
            try:
                pred_text = tokenizer.extract_thoughts(valid_pred_tokens)
                true_text = tokenizer.extract_thoughts(valid_true_tokens)
            except Exception as e:
                pred_text = f"Decode error: {str(e)}"
                true_text = f"Decode error: {str(e)}"
        
        pred_texts.append(pred_text)
        true_texts.append(true_text)
    
    return pred_texts, true_texts


def find_suffix_start(ar_mask):
    """Find the first position where ar_mask == 1 (suffix start).

    Returns -1 if no suffix is found.
    """
    for i, mask_val in enumerate(ar_mask):
        if mask_val == 1:
            return i
    return -1


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    if config.use_val_dataset:
        val_rng, _ = jax.random.split(train_rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    logging.info("Checkpoint directory: %s", config.checkpoint_dir)
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader, val_data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=2,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    training_utils.inspect_prompts(batch)

    data_config = config.data.create(config.assets_dirs, config.model)
    tokenizer = None
    for transform in data_config.model_transforms.inputs:
        if hasattr(transform, 'tokenizer'):
            tokenizer = transform.tokenizer
            break
    
    if tokenizer is None:
        logging.warning("Could not find tokenizer in model transforms")
    else:
        logging.info(f"Found tokenizer: {type(tokenizer).__name__}")
        logging.info(f"Tokenizer methods: {[method for method in dir(tokenizer) if not method.startswith('_')]}")

    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)
        try:
            val_batch = next(val_data_iter)
            logging.info(f"Initialized validation data loader: {len(val_data_loader)} samples")
            logging.info(f"Validation batch info:\n{training_utils.array_tree_to_info(val_batch)}")
        except StopIteration:
            logging.warning("Validation data loader is empty")
            val_data_iter = None
    else:
        logging.warning("No validation data loader available")
        val_data_iter = None

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    pval_reasoning = jax.jit(
        functools.partial(validate_reasoning, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    text_loss_steps = []
    text_loss_values = []
    val_text_loss_steps = []
    val_text_loss_values = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)

        if 'text_loss' in info:
            text_loss = info['text_loss']
            # Convert to scalar.
            if hasattr(text_loss, '__len__') and len(text_loss) > 0:
                text_loss_value = float(jnp.mean(text_loss))
            else:
                text_loss_value = float(text_loss)
            text_loss_steps.append(step)
            text_loss_values.append(text_loss_value)
        infos.append(info)

        # Periodic validation on the full validation set.
        if step % config.val_interval == 0 and step > start_step and val_data_loader is not None:
            total_val_loss = 0.0
            total_val_batches = 0
            display_batch = None  # Save one batch for text decoding display.
            
            # Iterate over the full validation set.
            for val_batch in val_data_loader:
                with sharding.set_mesh(mesh):
                    val_info = pval_reasoning(val_rng, train_state, val_batch)
                
                batch_loss = float(jnp.mean(val_info['val_text_loss']))
                total_val_loss += batch_loss
                total_val_batches += 1
                
                # Keep the first batch for display.
                if display_batch is None:
                    display_batch = (val_batch, val_info)
            
            # Average validation loss over the full set.
            val_text_loss = total_val_loss / total_val_batches if total_val_batches > 0 else 0.0
            val_text_loss_steps.append(step)
            val_text_loss_values.append(val_text_loss)
            
            # Decode and display sample predictions.
            if display_batch is not None and tokenizer is not None:
                val_batch, val_info = display_batch
                pred_texts, true_texts = decode_predictions(val_info, tokenizer)
                
                pbar.write(f"🚀 Validation at step {step}: text_loss = {val_text_loss:.4f} (full validation set, {total_val_batches} batches)")
                for i, (pred, true) in enumerate(zip(pred_texts, true_texts)):
                    pbar.write(f"   Sample {i}:")
                    pbar.write(f"     True:  {true}")
                    pbar.write(f"     Pred:  {pred}")
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Train at step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
    
    save_training_curves(text_loss_steps, text_loss_values, val_text_loss_steps, val_text_loss_values, config.exp_name)
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
def save_training_curves(train_steps, train_losses, val_steps, val_losses, exp_name):
    """Save training and validation loss curves as a PNG plot and raw TSV data."""
    if len(train_steps) == 0 and len(val_steps) == 0:
        print("No loss data to plot")
        return
    
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.makedirs(result_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    if len(train_steps) > 0:
        plt.plot(train_steps, train_losses, 'b-', linewidth=1.5, alpha=0.7, label='Train Text Loss')
    if len(val_steps) > 0:
        plt.plot(val_steps, val_losses, 'r-', linewidth=2, marker='o', markersize=4, label='Val Text Loss')
    
    plt.title(f'Text Loss Curves - {exp_name}')
    plt.xlabel('Training Step')
    plt.ylabel('Text Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use log scale if the loss range is very large.
    all_losses = train_losses + val_losses
    if len(all_losses) > 1 and max(all_losses) / min([l for l in all_losses if l > 0]) > 100:
        plt.yscale('log')
    
    filename = f"text_loss_curves_{exp_name.replace('/', '_')}.png"
    filepath = os.path.join(result_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to: {filepath}")
    
    # Also save raw loss data as TSV.
    data_file = os.path.join(result_dir, f"loss_data_{exp_name.replace('/', '_')}.txt")
    with open(data_file, 'w') as f:
        f.write("step\ttype\tloss\n")
        for step, loss in zip(train_steps, train_losses):
            f.write(f"{step}\ttrain\t{loss}\n")
        for step, loss in zip(val_steps, val_losses):
            f.write(f"{step}\tval\t{loss}\n")
    print(f"Loss data saved to: {data_file}")
if __name__ == "__main__":
    main(_config.cli())