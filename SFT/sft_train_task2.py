"""
Task2-only SFT entrypoint.
Train a dedicated model for task2 data.
"""

from sft_train_common import run_cli, DEFAULT_MODEL_NAME


if __name__ == "__main__":
    run_cli(
        task_name="task2",
        description="Full fine-tuning for SFT task2 (dedicated model)",
        default_model_name=DEFAULT_MODEL_NAME,
    )
