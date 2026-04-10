"""
Task1-only SFT entrypoint.
Train a dedicated model for task1 data.
"""

from sft_train_common import run_cli, DEFAULT_MODEL_NAME


if __name__ == "__main__":
    run_cli(
        task_name="task1",
        description="Full fine-tuning for SFT task1 (dedicated model)",
        default_model_name=DEFAULT_MODEL_NAME,
    )
