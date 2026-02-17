# Proposal: Automated PySpark Report Generation via Fine-Tuned LLM

## Executive Summary

We propose building an internal tool that automatically generates PySpark report code from plain English descriptions. The tool uses a fine-tuned open-source large language model (DeepSeek Coder 6.7B) deployed entirely within our private Azure network — no data leaves our infrastructure.

By training a model on our own query patterns, we eliminate the repetitive work of writing boilerplate PySpark code for reports that differ only in fields, tables, or join conditions. Engineers describe what they need in natural language, and the system produces production-ready PySpark code in seconds.

## Problem

The data engineering team spends significant time on repetitive PySpark report requests. These reports share common patterns — aggregations, joins, window functions, filters — but differ in specific tables, columns, and business logic. Each request still requires a senior engineer to write, review, and deliver the code manually.

This creates two problems:
1. **Bottleneck**: Senior engineers are pulled from higher-value work to write routine queries.
2. **Slow turnaround**: Requesters wait days for reports that follow well-established patterns.

## Proposed Solution

Fine-tune an open-source code generation model on our internal PySpark patterns using a parameter-efficient technique called QLoRA. The fine-tuned model learns our specific coding style, table schemas, and query patterns, producing higher-quality output than a generic model.

### How It Works

```
                    ONE-TIME TRAINING
   ┌──────────────────────────────────────────────┐
   │                                              │
   │  Example Queries    ──►  QLoRA Training      │
   │  (description +          on Azure ML         │
   │   PySpark code)          (A100 GPU, ~1 hr)   │
   │                              │               │
   │                              ▼               │
   │                      Fine-tuned Model        │
   │                      (registered in          │
   │                       Azure ML registry)     │
   └──────────────────────────────────────────────┘

                    DAILY USAGE
   ┌──────────────────────────────────────────────┐
   │                                              │
   │  Engineer types:         ──►  API returns:   │
   │  "Get total sales per         PySpark code   │
   │   region for Q4, joined       ready to run   │
   │   with customer segments"                    │
   │                                              │
   └──────────────────────────────────────────────┘
```

### Why Fine-Tuning (vs. Generic LLM or External API)

| Concern | Our Approach |
|---|---|
| **Data privacy** | Model runs entirely within our Azure VNet. No data sent to OpenAI, Google, or any third party. |
| **Quality** | Fine-tuned on our specific tables, schemas, and coding conventions — not generic code. |
| **Cost predictability** | Fixed infrastructure cost. No per-token billing that scales with usage. |
| **Latency** | Dedicated GPU instance, typical response in 2-5 seconds. |
| **Control** | We own the model weights. No vendor lock-in. Can retrain anytime. |

## What Needs to Happen

### Phase 1: Data Collection & Preparation (Week 1)

**Goal**: Assemble a high-quality training dataset of 100+ description-to-PySpark pairs.

- Gather existing PySpark reports from our codebase and Jira tickets
- Write a natural language description for each query
- Categorize by complexity (simple / medium / complex) and operation type
- Validate dataset format and quality
- Split into training (90%) and evaluation (10%) sets

**Who**: 1 data engineer, part-time (~16 hrs)

### Phase 2: Infrastructure Setup (Week 1, parallel with Phase 1)

**Goal**: Provision Azure resources for training and serving.

- Create Azure ML Workspace within existing resource group
- Configure GPU compute cluster (A100, scale-to-zero when idle)
- Set up private endpoints so all traffic stays within VNet
- Configure Container Registry for model serving images
- Set up storage for training data and model artifacts

**Who**: 1 platform/DevOps engineer, part-time (~8 hrs)

**Azure resources required**:

| Resource | Purpose | Specification |
|---|---|---|
| Azure ML Workspace | Training orchestration, model registry, endpoint management | Standard tier |
| GPU Compute Cluster | Model training | Standard_NC24ads_A100_v4 (1x A100 80GB), scale 0-1 nodes, low-priority/spot |
| GPU VM for Inference | Model serving | Standard_NC4as_T4_v3 (1x T4 16GB, 28 GB RAM) |
| Container Registry | Docker images for serving | Standard SKU |
| Storage Account | Training data, model artifacts | Standard LRS, ~50 GB |
| Key Vault | API keys, secrets | Standard |
| Virtual Network | Network isolation | Using existing enterprise VNet |

### Phase 3: Model Training (Week 2)

**Goal**: Fine-tune the model and validate quality.

- Convert training data to instruction format (ChatML)
- Run QLoRA fine-tuning on Azure ML (A100 GPU, ~1 hour)
- Evaluate against held-out test set using automated metrics (code correctness, operation coverage, structural similarity)
- Iterate on training hyperparameters if needed (1-2 additional runs)
- Register best model in Azure ML Model Registry

**Who**: 1 ML engineer, ~20 hrs

### Phase 4: Deployment & API (Week 3)

**Goal**: Deploy the model as an internal API endpoint.

- Deploy model to Azure ML Managed Online Endpoint (T4 GPU)
- Stand up FastAPI wrapper with `/generate` and `/health` endpoints
- Add request logging for monitoring and future retraining
- Load testing to confirm latency and throughput targets
- Write user-facing documentation

**Who**: 1 ML engineer + 1 platform engineer, ~24 hrs combined

### Phase 5: Rollout & Feedback (Week 4)

**Goal**: Pilot with the data engineering team and collect feedback.

- Onboard 3-5 engineers as pilot users
- Collect feedback on code quality and usefulness
- Log generated queries for evaluation and retraining data
- Adjust prompts or retrain if quality gaps are identified

**Who**: 1 ML engineer (part-time), pilot users

## Cost Breakdown

### One-Time Setup Costs

| Item | Estimated Cost |
|---|---|
| Training compute (A100 spot, ~3 runs x 1 hr) | ~$11 |
| Container Registry setup | Included in subscription |
| Storage (initial 50 GB) | < $1 |
| **Total one-time** | **~$12** |

### Ongoing Monthly Costs

| Item | Configuration | Monthly Cost |
|---|---|---|
| Inference VM | Standard_NC4as_T4_v3, 24/7 | $380 |
| Inference VM (business hours only, 12hr/day) | Same, with auto-shutdown | $190 |
| Storage | 50 GB Standard LRS | ~$2 |
| Container Registry | Standard SKU | ~$5 |
| Azure ML Workspace | Standard (no additional if existing) | $0 |
| Retraining (monthly, A100 spot, 1 hr) | On-demand | ~$4 |
| **Total (24/7)** | | **~$391/month** |
| **Total (business hours only)** | | **~$201/month** |

### Cost Comparison vs. Engineer Time

| Metric | Current State | With Tool |
|---|---|---|
| Avg. time per PySpark report request | ~2 hours | ~10 minutes (review + adjust) |
| Requests per month (estimated) | ~30 | ~30 |
| Senior engineer hours spent/month | ~60 hrs | ~5 hrs |
| Senior engineer cost at $80/hr | $4,800/month | $400/month |
| **Net savings/month** | — | **~$4,000** |

Even with 24/7 inference ($391/month), the tool pays for itself by reclaiming ~55 senior engineer hours per month.

## Technology Choices

| Component | Choice | Why |
|---|---|---|
| Base model | DeepSeek Coder 6.7B Instruct | Open-weight, code-specialized, strong benchmark performance, fits on a single T4 GPU |
| Fine-tuning method | QLoRA (4-bit quantized LoRA) | Reduces GPU memory from ~26 GB to ~6 GB, trains in under 1 hour, production-proven |
| Serving engine | vLLM | High-throughput inference, native LoRA support, OpenAI-compatible API |
| API framework | FastAPI | Python native, async, auto-generated docs |
| Training platform | Azure ML | GPU clusters with scale-to-zero, model versioning, managed endpoints |
| Network | Private VNet + private endpoints | All data stays within our infrastructure |

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Generated code has errors | Medium | Low — code is reviewed before use | Human review step; automated syntax validation; evaluation metrics track quality over time |
| Model hallucinates table/column names | Medium | Low | Include schema context in prompts; validate output against known table catalog |
| Insufficient training data | Low | Medium | Start with RAG approach (Approach 1) to generate usable output while collecting more examples; 100 examples is sufficient for QLoRA |
| Azure GPU quota unavailable | Low | Medium | Request quota increase in advance; A100 spot instances have good availability in East US 2 |
| Model quality degrades over time | Low | Low | Monthly retraining on accumulated examples; evaluation pipeline runs automatically |

## Success Criteria

| Metric | Target | How Measured |
|---|---|---|
| Report delivery time reduction | 70% faster | Track time from request to delivered code |
| Operation coverage score | > 0.85 | Automated evaluation on held-out test set |
| User satisfaction | > 4 out of 5 | Survey of pilot engineers after 2 weeks |
| Code accepted without major edits | > 60% of requests | Log whether generated code was used as-is or heavily modified |

## Timeline

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Data + infrastructure | Training dataset (100+ examples), Azure resources provisioned |
| 2 | Model trained | Fine-tuned model registered, evaluation report |
| 3 | API deployed | Internal endpoint live, documentation published |
| 4 | Pilot complete | Feedback collected, decision to scale or iterate |

## Team Requirements

| Role | Commitment | Duration |
|---|---|---|
| Data Engineer | Part-time (16 hrs) | Week 1 |
| Platform / DevOps Engineer | Part-time (8 hrs) | Weeks 1 & 3 |
| ML Engineer | Full-time | Weeks 2-4 |

Total estimated effort: **~70 person-hours over 4 weeks**.

## Recommendation

Approve a 4-week pilot with the scope described above. The infrastructure cost is minimal (~$12 one-time, ~$200-390/month ongoing), the engineering investment is modest (~70 hrs), and the projected time savings of ~55 senior engineer hours/month provides a clear return within the first month of production use.

If the pilot meets the success criteria, we proceed with full team rollout and set up a CI/CD pipeline for automated retraining as new examples accumulate.
