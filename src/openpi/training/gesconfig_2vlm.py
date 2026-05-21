"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model_2vlm as _model
import openpi.models.pi0_ges_2vlm as pi0_ges
import openpi.models.tokenizer as _tokenizer
import openpi.policies.umi_policy_2vlm as umi_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms_2vlm as _transforms

import openpi.shared.nnx_utils as nnx_utils


ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:

    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    root: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False

    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False
    local_files_only: bool = False

    def create_val_config(self) -> 'DataConfig':
        return dataclasses.replace(self)

@dataclasses.dataclass(frozen=True)
class UMIDataConfig(DataConfig):
    getitem_type: str = "default"
    use_val_dataset: bool = False
    val_ratio: float = 0.05
    create_train_val_split: bool = False
    norm_stats_dir: str | None = None
    seed: int = 42
    use_reference_image: bool = True
    is_computing_norm_stats: bool = False
    reasoning_json_path: str | None = None
    use_outdated_reasoning: bool = True
    reference_image_dir: str | None = None  # Reference image directory.
    default_instruction: str | None = None  # Default instruction text.
    debug_dir: str | None = None  # Debug output directory.
    ges_parquet_path: str | None = None  # Gesture parquet dataset path.

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:

            case _model.ModelType.PI0_GES:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.GeometricAugmentation(),
                        _transforms.GesTokenizePrompt(
                            _tokenizer.FusePaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractThoughts(
                            _tokenizer.FusePaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ]
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = tyro.MISSING
    root: str | None = None
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        root = self.root
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            root = root,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)
    
@dataclasses.dataclass(frozen=True)
class LeRobotUMIDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[umi_policy.UMIInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[umi_policy.UMIOutputs()],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
    
    def _get_norm_stats_dir(self, assets_dir: epath.Path, asset_id: str | None, getitem_type: str | None) -> str:
        if asset_id is None or getitem_type is None:
            return None
        return str(assets_dir / asset_id / getitem_type)

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        root = self.root if self.root is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or UMIDataConfig(),
            repo_id=repo_id,
            root = root,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id, self.base_config.getitem_type),
            norm_stats_dir=self._get_norm_stats_dir(epath.Path(self.assets.assets_dir or assets_dirs), asset_id, self.base_config.getitem_type),
        )
    
    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None, getitem_type: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id / getitem_type)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

@dataclasses.dataclass(frozen=True)
class PureVLDataConfig(DataConfig):
    use_val_dataset: bool = True
    val_ratio: float = 0.05
    seed: int = 42
    reasoning_json_path: str | None = None
    use_reference_image: bool = False
    skip_norm_stats: bool = True
    parquet_filename: str = "data_test.parquet"

    create_train_val_split: bool = True
    norm_stats_dir: str | None = None


@dataclasses.dataclass(frozen=True)
class PureVLDataConfigFactory(DataConfigFactory):
    repo_id: str = tyro.MISSING
    reasoning_json_path: str | None = None
    use_reference_image: bool = False
    parquet_filename: str = "data_test.parquet"

    repack_transforms=_transforms.Group()
    @override
    def create(self, assets_dirs, model_config):
        return PureVLDataConfig(
            repo_id=self.repo_id,
            asset_id=self.assets.asset_id or self.repo_id,
            norm_stats=None,
            model_transforms=ModelTransformFactory(default_prompt=None)(model_config),
            reasoning_json_path=self.reasoning_json_path,
            use_reference_image=self.use_reference_image,
            repack_transforms=self.repack_transforms,    
            parquet_filename=self.parquet_filename,
        )
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_ges.Pi0GesConfig)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    # Weight loader for VLM0 (expert2).
    vlm0_weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "data/checkpoints"
    # directory for load checkpoint when eval 
    policy_dir: str | None = None
    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    val_batch_size: int = 12
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 1
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 10
    # How often (in steps) to save checkpoints.
    save_interval: int = 5000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 10_000
    # How often (in steps) to evaluate the model. Only used if use_val_dataset is True.
    val_interval: int = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If true, will use the validation dataset for training.
    use_val_dataset: bool = True
    val_ratio: float = 0.05
    # If true, will create a train/val split from the dataset. It's used only when compute norm stats.
    create_train_val_split: bool = False

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


@dataclasses.dataclass(frozen=True)
class UMITrainConfig(TrainConfig):
    repo_id: str = tyro.MISSING
    root: str = tyro.MISSING

    # how to calculate the state
    getitem_type: str = tyro.MISSING
    # enable reasoning
    # whether to use the reference image
    use_reference_image: bool = True
    # whether to use outdated reasoning
    use_outdated_reasoning: bool = True
    is_computing_norm_stats: bool = False
    data: DataConfigFactory = dataclasses.field(init=False)
    reasoning_json_path: str | None = None
    prompt_from_task: bool = True

    # If true, set the `decay_step` to the number of training steps.
    lr_decay_till_end: bool = True

    reference_image_dir: str | None = None  # Reference image directory.
    default_instruction: str | None = None  # Default instruction text.
    debug_dir: str | None = None  # Debug output directory.

    ges_parquet_path: str | None = None  # Gesture parquet dataset path.
    freeze_vlm0: bool = True

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'data', LeRobotUMIDataConfig(
            repo_id=self.repo_id,
            root=self.root,
            base_config=UMIDataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=self.prompt_from_task,
                getitem_type=self.getitem_type,
                use_val_dataset=self.use_val_dataset,
                val_ratio=self.val_ratio,
                create_train_val_split=self.create_train_val_split,
                seed=self.seed,
                use_reference_image=self.use_reference_image,
                is_computing_norm_stats=self.is_computing_norm_stats,
                reasoning_json_path=self.reasoning_json_path,
                use_outdated_reasoning=self.use_outdated_reasoning,
                reference_image_dir=self.reference_image_dir,
                default_instruction=self.default_instruction,
                debug_dir=self.debug_dir,
                ges_parquet_path=self.ges_parquet_path,

            ),
        ))
        if self.lr_decay_till_end:
            assert isinstance(self.lr_schedule, _optimizer.CosineDecaySchedule), "Only CosineDecaySchedule is supported for lr_decay_till_end"
            object.__setattr__(self, 'lr_schedule', dataclasses.replace(self.lr_schedule, decay_steps=self.num_train_steps))
        if self.freeze_vlm0:
            freeze_vlm0 = nnx_utils.PathRegex(r".*_2(/.*)?$")

            # NNX paths are typically like ".../hand_pose_mlp_1/kernel".
            freeze_hand_pose = nnx_utils.PathRegex(r".*hand_pose_mlp_[12](/.*)?$")

            # Freeze embedder to avoid degrading pre-trained token embeddings during training.
            freeze_embedder = nnx_utils.PathRegex(r".*embedder(/.*)?$")

            freeze = nnx.All(nnx.Param, nnx.Any(freeze_vlm0, freeze_hand_pose, freeze_embedder))
            object.__setattr__(self, "freeze_filter", freeze)

            
# Use `get_config` if you need to get a config by name in your code.
# Override via CLI args (e.g. --root /custom/path) or environment variables in shell scripts.
_CONFIGS = [
    UMITrainConfig(
        name="gesvla_2vlm",
        model=pi0_ges.Pi0GesConfig(action_horizon=50,max_token_len=410),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        vlm0_weight_loader=weight_loaders.CheckpointWeightLoader(
            "data/checkpoints/purevl_pretrain/5999/params"
        ),
        num_train_steps=90000,
        batch_size=4,  # overwritten by .sh
        repo_id="gestureVLA_20260121",
        root="data/datasets/gestureVLA_20260121",
        getitem_type="necessary",
        save_interval=60000,
        use_val_dataset=True,
        use_outdated_reasoning=True,
        fsdp_devices=1,
        reasoning_json_path="data/reasoning/test_reasoning_filtered.json",
        reference_image_dir="data/reasoning/reference_images_0121",
        debug_dir="data/debug_samples",
        ges_parquet_path="data/gesture_data/realgesdata_0121/data_test.parquet",
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]