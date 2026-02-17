"""Create or get an Azure ML workspace."""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential

from shared.config import settings


def setup_workspace() -> MLClient:
    """Create the Azure ML workspace if it doesn't exist and return an MLClient."""
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=settings.azure_subscription_id,
        resource_group_name=settings.azure_resource_group,
    )

    try:
        ws = ml_client.workspaces.get(settings.azure_ml_workspace_name)
        print(f"Workspace '{ws.name}' already exists in {ws.location}")
    except Exception:
        print(f"Creating workspace '{settings.azure_ml_workspace_name}'...")
        ws = Workspace(
            name=settings.azure_ml_workspace_name,
            location=settings.azure_region,
        )
        ml_client.workspaces.begin_create_or_update(ws).result()
        print(f"Workspace '{ws.name}' created in {ws.location}")

    # Return client scoped to workspace
    return MLClient(
        credential=credential,
        subscription_id=settings.azure_subscription_id,
        resource_group_name=settings.azure_resource_group,
        workspace_name=settings.azure_ml_workspace_name,
    )


if __name__ == "__main__":
    setup_workspace()
