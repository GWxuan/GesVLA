"""Entry point for data generation."""

import argparse
import os
import sys


def _ensure_project_root() -> None:
    """Ensure the project root is on sys.path for relative imports."""
    if __package__ is None or __package__ == "":
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)


def main(detect: bool = False, collect: bool = False) -> None:
    """Main entry point."""
    _ensure_project_root()

    from data_generator.configs import (
        CameraConfig,
        DetectionConfig,
        HandConfig,
        GenerationConfig,
        InstructionTemplates,
    )
    from data_generator.pipeline import DataGenerationPipeline
    from data_generator.detect_train import run_detect
    from data_generator.collect_train import run_collect

    camera_config = CameraConfig()
    detection_config = DetectionConfig()
    hand_config = HandConfig()
    generation_config = GenerationConfig()
    instruction_templates = InstructionTemplates()

    pipeline = DataGenerationPipeline(
        camera_config=camera_config,
        detection_config=detection_config,
        hand_config=hand_config,
        generation_config=generation_config,
        instruction_templates=instruction_templates,
    )

    pipeline.run()

    base_path = generation_config.output_root
    if detect:
        run_detect(base_path)
    if collect:
        run_collect(base_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand VLA data generation pipeline")
    parser.add_argument(
        "--detect",
        default=True,
        help="Run MediaPipe keypoint detection on generated train data"
    )
    parser.add_argument(
        "--collect",
        default=True,
        help="Collect data into parquet and reasoning files"
    )
    args = parser.parse_args()
    main(detect=args.detect, collect=args.collect)
