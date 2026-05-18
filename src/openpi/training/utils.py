import logging
import pprint
from collections.abc import Callable
from typing import Any

import jax
import optax
import sentencepiece
from flax import nnx, struct

import openpi.shared.download as download
from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[Any]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")


def inspect_prompts(
        batch: tuple[_model.FuseObservation | _model.Observation, at.Array],
        ) -> str:
    """Converts a PyTree of prompts into a human-readable string for logging."""
    tokenized_prompt = batch[0].tokenized_prompt
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    
    prompts = []
    for tokenized in tokenized_prompt:
        tokens = tokenizer.decode(tokenized.tolist(),)
        prompts.append(tokens)
    logging.info("prompts:")
    logging.info(pprint.pformat(prompts))
