"""Create GPU compute cluster in Azure ML for training."""

from azure.ai.ml.entities import AmlCompute

from shared.config import settings

from .workspace_setup import setup_workspace


def create_compute() -> None:
    """Create a GPU compute cluster for fine-tuning jobs."""
    ml_client = setup_workspace()

    try:
        compute = ml_client.compute.get(settings.azure_compute_name)
        print(f"Compute '{compute.name}' already exists (size={compute.size})")
    except Exception:
        print(f"Creating compute cluster '{settings.azure_compute_name}'...")
        compute = AmlCompute(
            name=settings.azure_compute_name,
            size=settings.azure_compute_size,
            min_instances=settings.azure_compute_min_nodes,
            max_instances=settings.azure_compute_max_nodes,
            idle_time_before_scale_down=300,
            tier="low_priority",
        )
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"Compute cluster '{compute.name}' created (size={compute.size})")


if __name__ == "__main__":
    create_compute()
