from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model_2vlm as _model
import openpi.training.gesconfig_2vlm as _config
from openpi.training.gesconfig import ReasoningDataConfig
import openpi.transforms_2vlm as _transforms
import openpi.policies.reasoning_dataset as reasoning_dataset
import openpi.policies.umi_ges_dataset_2vlm_withvisualprompt as umi_dataset_2vlm

from openpi.timer_utils import Timer, timed_function

T_co = TypeVar("T_co", covariant=True)

logger = logging.getLogger(__name__)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    @timed_function("TransformedDataset.__getitem__")
    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig, split: str = "train") -> tuple[Dataset, Dataset | None]:
    """Create datasets for training.

    If `data_config.use_val_dataset` is set, will also return a validation dataset.
    """
    with Timer("create_dataset"):
        repo_id = data_config.repo_id

        if repo_id is None:
            raise ValueError("Repo ID is not set. Cannot create dataset.")

        if isinstance(data_config, ReasoningDataConfig):
            logger.info("Creating ReasoningDataset, split: %s", split)
            dataset = reasoning_dataset.ReasoningDataset(data_config, model_config.action_horizon, split=split)

            if split == "train" and getattr(data_config, 'use_val_dataset', False):
                val_dataset = dataset.get_val_dataset()
                return dataset, val_dataset
            else:
                return dataset, None

        if isinstance(data_config, _config.UMIDataConfig):
            logger.info("Creating UMIGesDataset_2vlm, split: %s", split)
            dataset = umi_dataset_2vlm.UMIGesDataset_2vlm(data_config, model_config.action_horizon)
            return dataset, None

        raise ValueError(f"Unsupported data config type: {type(data_config).__name__} with repo_id={repo_id}")


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False, is_training: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    with Timer("transform_dataset"):
        norm_stats = {}
        if data_config.repo_id != "fake" and not skip_norm_stats:
            if data_config.norm_stats is None:
                raise ValueError(
                    "Normalization stats not found. "
                    "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
                )
            norm_stats = data_config.norm_stats

        model_transforms_inputs = []
        for transform in data_config.model_transforms.inputs:
            if isinstance(transform, _transforms.GeometricAugmentation):
                new_transform = _transforms.GeometricAugmentation(apply_augmentation=is_training)
                model_transforms_inputs.append(new_transform)
            else:
                model_transforms_inputs.append(transform)

        model_transforms = dataclasses.replace(data_config.model_transforms, inputs=model_transforms_inputs)

        return TransformedDataset(
            dataset,
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *model_transforms.inputs,
            ],
        )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    iterate_indefinitely: bool = True,
    split: str = "train"
) -> tuple[DataLoader[tuple[_model.Observation, _model.Actions]],\
            DataLoader[tuple[_model.Observation, _model.Actions]] | None]:
    """Create data loaders for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    logger.info("Assets dirs: %s", config.assets_dirs)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset, val_dataset = create_dataset(data_config, config.model, split=split)
    is_training = (split == "train")

    norm_stats = None
    if isinstance(data_config, ReasoningDataConfig):
        skip_norm_stats = True
    else:
        skip_norm_stats = False

    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_training=is_training)
    logger.info("Dataset size: %d", len(dataset))

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        iterate_indefinitely=iterate_indefinitely,
        seed=config.seed,
    )

    if val_dataset is not None:
        val_dataset = transform_dataset(val_dataset, data_config.create_val_config(), skip_norm_stats=True)
        val_data_loader = TorchDataLoader(
            val_dataset,
            local_batch_size=config.val_batch_size // jax.process_count(),
            sharding=sharding,
            shuffle=False,
            num_batches=num_batches,
            num_workers=num_workers,
            iterate_indefinitely=False, # Don't iterate indefinitely for validation.
            seed=config.seed,
        )
    else:
        val_data_loader = None

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.FuseObservation.from_dict(batch), batch["actions"]

        
        def __len__(self) -> int:
            return len(self._data_loader)

    return DataLoaderImpl(data_config, data_loader), \
        DataLoaderImpl(data_config, val_data_loader) if val_data_loader else None


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        iterate_indefinitely: bool = True,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            iterate_indefinitely: Whether to iterate over the dataset indefinitely.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )
        self._iterate_indefinitely = iterate_indefinitely

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if self._iterate_indefinitely:
                        break  # We've exhausted the dataset. Create a new iterator and start over.
                    else:
                        return # We've exhausted the dataset and we're not iterating indefinitely.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
    
    def __len__(self) -> int:
        if self._iterate_indefinitely:
            raise ValueError("Cannot determine the length of an indefinitely iterating data loader.")
        return len(self._data_loader)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)




def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
