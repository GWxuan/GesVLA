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
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
from flax.training import common_utils

import openpi.models.model_2vlm as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
# import openpi.training.gesconfig as _config
import openpi.training.gesconfig_2vlm as _config
import openpi.training.data_loader_2vlm as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms_2vlm as _transforms
import wandb

from openpi.models.pi0_ges_2vlm import make_attn_mask
from openpi.models import gemma as _gemma
import einops

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
    """Loads a subset of weights from checkpoint that are a subset of `params_shape`.
    Validates that every loaded leaf matches expected shape/dtype; new params remain init."""
    loaded_params = loader.load(params_shape)

    # Subset validation + filter to intersection.
    flat_exp = traverse_util.flatten_dict(params_shape)
    flat_ld  = traverse_util.flatten_dict(loaded_params)

    filtered = {}
    for k, v_ld in flat_ld.items():
        if k not in flat_exp:
            continue
        v_exp = flat_exp[k]

        # Shape/dtype compatibility check.
        exp_shape = getattr(v_exp, "shape", None)
        exp_dtype = getattr(v_exp, "dtype", None)
        ld_shape  = getattr(v_ld,  "shape", None)
        ld_dtype  = getattr(v_ld,  "dtype", None)

        shape_ok = (exp_shape is None or ld_shape is None or tuple(exp_shape) == tuple(ld_shape))
        dtype_ok = (exp_dtype is None or ld_dtype is None or exp_dtype == ld_dtype)

        if not (shape_ok and dtype_ok):
            continue

        filtered[k] = v_ld

    # Return only actually-loaded leaves (drop ShapeDtypeStruct).
    filtered = {k: v for k, v in filtered.items() if not isinstance(v, jax.ShapeDtypeStruct)}
    return traverse_util.unflatten_dict(filtered)


def _deep_update(dst: dict, src: dict) -> dict:
    """递归合并字典：src 覆盖 dst"""
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _is_under_llm(flat_key: tuple) -> bool:
    # 只在 PaliGemma.llm 子树里做 expert 后缀映射，避免误伤 hand_pose_mlp_2 等
    # 你的 NNX key tuple 里一般会包含这些字符串段
    return ("PaliGemma" in flat_key) and ("llm" in flat_key)


def _strip_expert2_suffix_in_key(flat_key: tuple) -> tuple:
    """把 key tuple 中字符串段末尾的 '_2' 去掉（只对 llm 子树生效）"""
    if not _is_under_llm(flat_key):
        return flat_key

    new = []
    for part in flat_key:
        if isinstance(part, str) and part.endswith("_2"):
            new.append(part[:-2])  # remove "_2"
        else:
            new.append(part)
    return tuple(new)


def _add_expert2_suffix_in_key(flat_key: tuple) -> tuple:
    """把 key tuple 中字符串段加上 '_2'（用于从 source key 复原到 target key）"""
    if not _is_under_llm(flat_key):
        return flat_key

    # 注意：这里我们只对“那些在 target 里确实是 *_2 的段”做恢复。
    # 为了安全，我们不做“全体加 _2”，而是在构建映射时用 target->source 的逆映射来恢复。
    raise RuntimeError("Do not call directly; use mapping dict built from target->source.")

def _load_hand_pose_mlps_from_ckpt(loader, params_shape_puredict: dict) -> dict:
    """
    从 ckpt 里加载 hand_pose_mlp_1 / hand_pose_mlp_2（如果 ckpt 里存在且 shape/dtype 匹配）。
    """
    flat_exp = traverse_util.flatten_dict(params_shape_puredict)

    def is_hand_pose_mlp(k):
        # k 是 tuple path，比如 (..., 'hand_pose_mlp_2', 'kernel') 或类似
        return any(p in ("hand_pose_mlp_1", "hand_pose_mlp_2") for p in k)

    target_keys = [k for k in flat_exp.keys() if is_hand_pose_mlp(k)]
    flat_shape_for_load = {k: flat_exp[k] for k in target_keys}
    shape_for_load = traverse_util.unflatten_dict(flat_shape_for_load)

    loaded = _load_weights_and_validate(loader, shape_for_load)

    # 可选：打印 missing/loaded（强烈建议先开着）
    flat_loaded = traverse_util.flatten_dict(loaded)
    exp = set(target_keys)
    got = set(flat_loaded.keys())
    missing = sorted(exp - got)
    logging.info("hand_pose expected: %d, loaded: %d", len(exp), len(got))
    if missing:
        logging.info("MISSING hand_pose keys: %s", missing)

    return loaded


def _load_vlm0_into_expert2(
    vlm0_loader,
    params_shape_puredict: dict,
) -> dict:
    """
    从旧 checkpoint（含 VLA）里取 expert0(VLM) 权重，灌到当前模型 expert2(*_2)。
    仅针对 PaliGemma.llm 子树中的 *_2 叶子。
    """
    flat_exp = traverse_util.flatten_dict(params_shape_puredict)

    # 1) 找到所有 target keys：位于 llm 子树，并且某个段以 _2 结尾
    target_keys = []
    for k in flat_exp.keys():
        if not _is_under_llm(k):
            continue
        if any(isinstance(p, str) and p.endswith("_2") for p in k):
            target_keys.append(k)

    # 2) 构建 “用于 loader.load 的 shape”：把这些 target keys 映射成 source keys（去掉 _2）
    #    并记录 source->target 映射，方便把 load 回来的叶子写回 *_2 位置
    source_to_target = {}
    flat_shape_for_load = {}

    for tk in target_keys:
        sk = _strip_expert2_suffix_in_key(tk)
        source_to_target[sk] = tk
        flat_shape_for_load[sk] = flat_exp[tk]

    shape_for_load = traverse_util.unflatten_dict(flat_shape_for_load)

    # 3) 用 loader 加载（会从 ckpt 里取到 source keys 对应的数组）
    loaded = _load_weights_and_validate(vlm0_loader, shape_for_load)

    # 4) 把 loaded 的 source keys 改回 target keys（*_2）
    flat_loaded = traverse_util.flatten_dict(loaded)
    flat_out = {}
    for sk, v in flat_loaded.items():
        if sk in source_to_target:
            flat_out[source_to_target[sk]] = v

    return traverse_util.unflatten_dict(flat_out)


# @at.typecheck
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

    # partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    params_shape_dict = train_state_shape.params.to_pure_dict()

    # 先加载 base（VLM1 + action expert）
    base_partial = _load_weights_and_validate(config.weight_loader, params_shape_dict)

    # 再加载 purevl 的 VLM -> expert2 (VLM0)
    vlm0_partial = {}
    hand_pose_partial = {}

    if not isinstance(config.vlm0_weight_loader, _weight_loaders.NoOpWeightLoader):
        vlm0_partial = _load_vlm0_into_expert2(config.vlm0_weight_loader, params_shape_dict)

        # NEW: 同一个 ckpt 里加载 hand_pose_mlp_1/2
        hand_pose_partial = _load_hand_pose_mlps_from_ckpt(config.vlm0_weight_loader, params_shape_dict)

    # 合并：vlm0_partial 命中 llm *_2；hand_pose_partial 命中 hand_pose_mlp_1/2
    partial_params = _deep_update(base_partial, vlm0_partial)
    partial_params = _deep_update(partial_params, hand_pose_partial)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# @at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    # @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss, info = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), info

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, train_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
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
    """
    验证推理能力：使用正确的损失计算 + 完整的解码信息
    """
    if state.ema_decay is None:
        model = nnx.merge(state.model_def, state.params)
    else:
        model = nnx.merge(state.model_def, state.ema_params)
    model.eval()

    observation, _ = batch

    # Compute text loss on the full validation batch.
    dummy_actions = jnp.zeros((observation.state.shape[0], config.model.action_horizon, config.model.action_dim))
    observation = dataclasses.replace(
        observation,
        diffusion_loss_mask=jnp.zeros(observation.state.shape[0], dtype=jnp.bool_)
    )
    
    # 计算损失 - 使用模型内部方法确保正确性
    loss, info = model.compute_loss(rng, observation, dummy_actions, train=False)
    
    # 为了解码，我们需要获取预测的token
    # 这里只用于显示，不用于损失计算
    vlm1_t, vlm1_m, vlm1_ar, vlm0_t, vlm0_m, vlm0_ar = model.embed_img_txt_dual(observation)
    prefix_tokens = jnp.concatenate([vlm1_t, vlm0_t], axis=1)
    prefix_mask = jnp.concatenate([vlm1_m, vlm0_m], axis=1)
    prefix_ar = jnp.concatenate([vlm1_ar, vlm0_ar], axis=1)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (img_txt_pre_logits, _), _ = model.PaliGemma.llm(
        [vlm1_t, vlm0_t], mask=prefix_attn_mask, positions=prefix_positions)

    txt_logits = model.PaliGemma.llm(
        img_txt_pre_logits[:, -observation.tokenized_prompt.shape[1]:],
        method="embedder_decode", 
    )
    
    # 获取预测的token（贪心解码）
    pred_tokens = jnp.argmax(txt_logits, axis=-1)
    
    # 只返回前4个样本
    num_samples = min(observation.tokenized_prompt.shape[0], 4)
    
    return {
        'val_text_loss': info['text_loss'],  # 使用正确的模型计算损失
        'val_batch_size': observation.state.shape[0],
        'pred_tokens': pred_tokens[:num_samples],
        'true_tokens': observation.tokenized_prompt[:num_samples],
        'token_mask': observation.tokenized_prompt_mask[:num_samples],
        'ar_mask': observation.token_ar_mask[:num_samples]  # 添加AR掩码用于区分前缀后缀
    }

def decode_predictions(val_info: dict, tokenizer) -> tuple[list, list]:
    """
    只解码模型应该生成的后缀部分
    """
    pred_texts = []
    true_texts = []
    
    pred_tokens = val_info['pred_tokens']
    true_tokens = val_info['true_tokens']
    token_mask = val_info['token_mask']
    ar_mask = val_info['ar_mask']  # 使用AR掩码区分前缀和后缀
    
    batch_size = pred_tokens.shape[0]
    
    for i in range(batch_size):
        # 找到后缀开始的位置（AR掩码为1的位置）
        suffix_start = find_suffix_start(ar_mask[i])
        
        if suffix_start >= 0:
            # 只解码后缀部分（模型生成的部分）
            suffix_mask = token_mask[i][suffix_start:]
            valid_pred_suffix = pred_tokens[i][suffix_start:][suffix_mask]
            valid_true_suffix = true_tokens[i][suffix_start:][suffix_mask]
            
            try:
                # 解码后缀部分
                pred_text = tokenizer.extract_thoughts(valid_pred_suffix)
                true_text = tokenizer.extract_thoughts(valid_true_suffix)
            except Exception as e:
                pred_text = f"Decode error: {str(e)}"
                true_text = f"Decode error: {str(e)}"
        else:
            # 如果没有找到后缀开始位置，解码整个序列（备选方案）
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
    """
    使用AR掩码找到后缀开始的位置
    AR掩码为0表示前缀，1表示后缀
    """
    for i, mask_val in enumerate(ar_mask):
        if mask_val == 1:  # 找到第一个后缀token
            return i
    return -1  # 没有找到后缀


def _count_expert2_leaves(pure_dict):
    flat = traverse_util.flatten_dict(pure_dict)
    return sum(
        1 for k in flat
        if any(isinstance(p, str) and p.endswith("_2") for p in k)
    )

def _count_expert2_trainable_leaves(params, trainable_filter):
    trainable = params.filter(trainable_filter).to_pure_dict()
    return _count_expert2_leaves(trainable)

def _count_expert2_all_leaves(params):
    return _count_expert2_leaves(params.to_pure_dict())

def _count_hand_pose_leaves(pure_dict):
    flat = traverse_util.flatten_dict(pure_dict)
    def is_hp(k):
        return any(p in ("hand_pose_mlp_1", "hand_pose_mlp_2") for p in k)
    return sum(1 for k in flat if is_hp(k))


import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os

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
    logging.info("Resuming training: %s", resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")


    # Verify freeze configuration.
    e2_all = _count_expert2_all_leaves(train_state.params)
    e2_trainable = _count_expert2_trainable_leaves(train_state.params, config.trainable_filter)
    logging.info("VLM0 (expert2) leaves: %d total, %d trainable (freeze_vlm0=%s)",
                 e2_all, e2_trainable, getattr(config, "freeze_vlm0", "N/A"))

    hp_all = _count_hand_pose_leaves(train_state.params.to_pure_dict())
    trainable_pd = train_state.params.filter(config.trainable_filter).to_pure_dict()
    hp_trainable = _count_hand_pose_leaves(trainable_pd)
    logging.info("hand_pose MLP leaves: %d total, %d trainable", hp_all, hp_trainable)

    # Log frozen/trainable parameter counts.
    flat_all = traverse_util.flatten_dict(train_state.params.to_pure_dict())
    flat_trainable = traverse_util.flatten_dict(
        train_state.params.filter(config.trainable_filter).to_pure_dict()
    )
    all_keys = set(flat_all.keys())
    trainable_keys = set(flat_trainable.keys())
    frozen_keys = sorted(all_keys - trainable_keys, key=lambda x: ".".join(map(str, x)))

    def _count_params(param_dict: dict) -> int:
        """Count total scalar parameters in a flattened param dict."""
        total = 0
        for v in param_dict.values():
            if hasattr(v, "size"):
                total += int(v.size)
            elif hasattr(v, "shape"):
                total += int(np.prod(v.shape))
        return total

    total_params = _count_params(flat_all)
    trainable_params = _count_params(flat_trainable)
    frozen_params = total_params - trainable_params

    logging.info("Parameters: %d total, %d trainable, %d frozen",
                 len(all_keys), len(trainable_keys), len(frozen_keys))
    logging.info("Parameter COUNT: %.2fM total, %.2fM trainable, %.2fM frozen",
                 total_params / 1e6, trainable_params / 1e6, frozen_params / 1e6)
    logging.debug("Frozen keys: %s", [ ".".join(map(str, k)) for k in frozen_keys ])
    logging.debug("Trainable keys: %s", [ ".".join(map(str, k)) for k in sorted(trainable_keys, key=lambda x: ".".join(map(str, x))) ])

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        logging.info("Restored train state from checkpoint.")

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
        ascii = True,
    )

    infos = []
    action_loss_steps = []
    action_loss_values = []
    val_action_loss_steps = []
    val_action_loss_values = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        if 'action_loss' in info:
            action_loss = info['action_loss']
            # 转换为标量
            if hasattr(action_loss, '__len__') and len(action_loss) > 0:
                action_loss_value = float(jnp.mean(action_loss))
            else:
                action_loss_value = float(action_loss)
            
            action_loss_steps.append(step)
            action_loss_values.append(action_loss_value)
        infos.append(info)
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
    
    save_training_curves(action_loss_steps, action_loss_values, val_action_loss_steps, val_action_loss_values, config.exp_name)
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

    # Plot training loss curve.
    if len(train_steps) > 0:
        plt.plot(train_steps, train_losses, 'b-', linewidth=1.5, alpha=0.7, label='Train Text Loss')

    # Plot validation loss curve.
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

    # Save the figure.
    filename = f"text_loss_curves_{exp_name.replace('/', '_')}.png"
    filepath = os.path.join(result_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Loss curves saved to: {filepath}")

    # Also save raw data as TSV.
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