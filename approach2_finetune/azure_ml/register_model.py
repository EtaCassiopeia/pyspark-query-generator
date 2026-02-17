"""Register a trained model in the Azure ML model registry."""

from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

from .workspace_setup import setup_workspace


def register_model(
    model_path: str,
    model_name: str = "pyspark-query-generator",
    description: str = "QLoRA fine-tuned DeepSeek Coder for PySpark generation",
) -> None:
    """Register a trained LoRA adapter or merged model in the Azure ML registry."""
    ml_client = setup_workspace()

    model = Model(
        path=model_path,
        type=AssetTypes.CUSTOM_MODEL,
        name=model_name,
        description=description,
    )

    registered = ml_client.models.create_or_update(model)
    print(f"Model registered: {registered.name} (version {registered.version})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register model in Azure ML")
    parser.add_argument("--model-path", required=True, help="Path to model artifacts")
    parser.add_argument("--name", default="pyspark-query-generator")
    args = parser.parse_args()

    register_model(model_path=args.model_path, model_name=args.name)
