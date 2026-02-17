"""Submit a fine-tuning training job to Azure ML."""

from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes

from shared.config import settings

from .workspace_setup import setup_workspace


def submit_job() -> str:
    """Submit a QLoRA training job to the Azure ML compute cluster.

    Returns the job name for tracking.
    """
    ml_client = setup_workspace()

    job = command(
        display_name="pyspark-query-generator-qlora",
        description="QLoRA fine-tuning of DeepSeek Coder for PySpark generation",
        experiment_name="pyspark-query-generator",
        compute=settings.azure_compute_name,
        environment="pyspark-finetune-env@latest",
        code="../../",
        command=(
            "python -m approach2_finetune.prepare_data "
            "--input ${{inputs.training_data}} "
            "--output data/prepared && "
            "python -m approach2_finetune.train_lora "
            "--data-dir data/prepared "
            "--output-dir ${{outputs.model_output}} "
            f"--base-model {settings.base_model}"
        ),
        inputs={
            "training_data": Input(
                type=AssetTypes.URI_FILE,
                path=str(settings.training_data_path),
            ),
        },
        outputs={
            "model_output": {"type": "uri_folder"},
        },
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted: {returned_job.name}")
    print(f"Studio URL: {returned_job.studio_url}")

    return returned_job.name


if __name__ == "__main__":
    submit_job()
