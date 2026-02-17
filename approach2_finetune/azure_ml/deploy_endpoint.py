"""Deploy a registered model as a managed online endpoint in Azure ML."""

from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
    CodeConfiguration,
)

from shared.config import settings

from .workspace_setup import setup_workspace


def deploy(
    model_name: str = "pyspark-query-generator",
    model_version: str = "1",
    endpoint_name: str = "pyspark-generator-endpoint",
    instance_type: str = "Standard_NC4as_T4_v3",
    instance_count: int = 1,
) -> None:
    """Deploy the model as a managed online endpoint with vLLM."""
    ml_client = setup_workspace()

    # Create endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="PySpark Query Generator — Fine-tuned model",
        auth_mode="key",
    )

    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint '{endpoint_name}' created")

    # Create deployment
    model = ml_client.models.get(model_name, version=model_version)

    deployment = ManagedOnlineDeployment(
        name="primary",
        endpoint_name=endpoint_name,
        model=model,
        instance_type=instance_type,
        instance_count=instance_count,
        environment="pyspark-finetune-env@latest",
        code_configuration=CodeConfiguration(
            code="../../approach2_finetune",
            scoring_script="api.py",
        ),
        request_settings={
            "request_timeout_ms": 120000,
        },
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Deployment 'primary' created on endpoint '{endpoint_name}'")

    # Route 100% traffic to the deployment
    endpoint.traffic = {"primary": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("Traffic routed to deployment")

    # Get scoring URI
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    print(f"Scoring URI: {endpoint.scoring_uri}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy model as managed endpoint")
    parser.add_argument("--model-name", default="pyspark-query-generator")
    parser.add_argument("--model-version", default="1")
    parser.add_argument("--endpoint-name", default="pyspark-generator-endpoint")
    args = parser.parse_args()

    deploy(
        model_name=args.model_name,
        model_version=args.model_version,
        endpoint_name=args.endpoint_name,
    )
