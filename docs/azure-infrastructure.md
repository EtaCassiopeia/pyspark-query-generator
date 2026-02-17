# Azure Infrastructure Setup

## Overview

All resources deploy within a private Azure VNet. No public endpoints are exposed.

## Resource Architecture

```
Azure Resource Group
├── Virtual Network (10.0.0.0/16)
│   ├── Subnet: compute (10.0.1.0/24) — VMs, AKS
│   ├── Subnet: ml (10.0.2.0/24) — Azure ML compute
│   └── Subnet: endpoints (10.0.3.0/24) — Private endpoints
├── Azure ML Workspace
│   ├── Compute Cluster (GPU, scale-to-zero)
│   ├── Model Registry
│   └── Managed Online Endpoints
├── Azure Container Registry
├── Storage Account (training data, FAISS indexes)
└── Key Vault (secrets, API keys)
```

## Setup Steps

### 1. Resource Group

```bash
az group create \
  --name rg-pyspark-generator \
  --location eastus2
```

### 2. Virtual Network

```bash
az network vnet create \
  --resource-group rg-pyspark-generator \
  --name vnet-pyspark \
  --address-prefix 10.0.0.0/16

# Subnets
for subnet in compute:10.0.1.0/24 ml:10.0.2.0/24 endpoints:10.0.3.0/24; do
  IFS=: read name prefix <<< "$subnet"
  az network vnet subnet create \
    --resource-group rg-pyspark-generator \
    --vnet-name vnet-pyspark \
    --name "snet-${name}" \
    --address-prefix "${prefix}"
done
```

### 3. Azure ML Workspace

```bash
az ml workspace create \
  --name mlw-pyspark-generator \
  --resource-group rg-pyspark-generator \
  --location eastus2
```

Or use the Python SDK:

```bash
python -m approach2_finetune.azure_ml.workspace_setup
```

### 4. GPU Compute Cluster

```bash
python -m approach2_finetune.azure_ml.create_compute
```

This creates a `Standard_NC24ads_A100_v4` cluster with:
- Min nodes: 0 (scale to zero when idle)
- Max nodes: 1
- Low-priority (spot) instances for cost savings
- Auto-shutdown after 5 minutes idle

### 5. Container Registry

```bash
az acr create \
  --resource-group rg-pyspark-generator \
  --name acrpysparkgen \
  --sku Standard
```

### 6. Storage Account

```bash
az storage account create \
  --resource-group rg-pyspark-generator \
  --name stpysparkgen \
  --sku Standard_LRS \
  --kind StorageV2
```

## VM Sizing Reference

| Use Case | VM Size | GPU | RAM | Cost/hr |
|---|---|---|---|---|
| Training | Standard_NC24ads_A100_v4 | 1x A100 80GB | 220 GB | ~$3.67 (spot) |
| Inference (GPU) | Standard_NC4as_T4_v3 | 1x T4 16GB | 28 GB | ~$0.53 |
| RAG API (CPU) | Standard_D4s_v5 | None | 16 GB | ~$0.19 |

## Private Endpoints

Configure private endpoints for Azure ML, Storage, and ACR to ensure all traffic stays within the VNet:

```bash
# Example: private endpoint for storage
az network private-endpoint create \
  --resource-group rg-pyspark-generator \
  --name pe-storage \
  --vnet-name vnet-pyspark \
  --subnet snet-endpoints \
  --private-connection-resource-id <storage-resource-id> \
  --group-id blob \
  --connection-name storage-connection
```

## Network Security Groups

Apply NSGs to restrict traffic between subnets. The ML subnet should only be accessible from the compute subnet and the managed endpoint subnet.
