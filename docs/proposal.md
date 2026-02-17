# Business Proposal: PySpark Query Generator

## Problem Statement

The data engineering team receives repetitive requests to generate PySpark reports with small modifications — different fields, tables, joins, or aggregations. Each request requires manual effort from a senior engineer, creating bottlenecks and slowing delivery.

## Proposed Solution

Build an internal tool that generates PySpark code from natural language descriptions. We evaluate two approaches and recommend deploying within the existing Azure infrastructure.

## Approach Comparison

| Criteria | Approach 1: RAG + Local LLM | Approach 2: Fine-tuned Model |
|---|---|---|
| **Setup complexity** | Low — index examples, run Ollama | High — training pipeline, Azure ML |
| **Training time** | None (retrieval only) | ~1-2 hours on A100 |
| **Inference cost** | ~$0.19/hr (CPU) or ~$0.53/hr (GPU) | ~$0.53/hr (T4 GPU) |
| **Quality (few examples)** | Good — leverages base model knowledge | Moderate — limited training data |
| **Quality (many examples)** | Good — bounded by context window | Excellent — specialized behavior |
| **Adding new examples** | Re-index (minutes) | Re-train (hours) |
| **Privacy** | Full — all data stays on-premises | Full — all data stays on-premises |
| **Maintenance** | Low | Medium (model versioning, retraining) |

## Recommendation

**Start with Approach 1 (RAG)** for immediate value, then evaluate Approach 2 after collecting 100+ examples from real usage.

### Phase 1 (Weeks 1-2): RAG MVP
- Deploy RAG pipeline with existing examples
- Collect user feedback and new examples
- Estimated cost: ~$140/month (D4s VM, 24/7)

### Phase 2 (Weeks 3-4): Fine-tuning Evaluation
- Prepare fine-tuning dataset from collected examples
- Train and evaluate QLoRA model
- Estimated cost: ~$15 one-time training + ~$380/month inference (T4 VM, 24/7)

### Phase 3 (Month 2+): Production Deployment
- Deploy winning approach behind API gateway
- Add monitoring, logging, feedback loop
- CI/CD pipeline for model updates

## Cost Estimates (Monthly)

| Resource | RAG (Approach 1) | Fine-tuned (Approach 2) |
|---|---|---|
| Inference VM | $140 (D4s, CPU) | $380 (NC4as T4) |
| Training (one-time) | $0 | ~$15 (A100 spot, 4 hrs) |
| Storage | ~$5 | ~$10 |
| **Total/month** | **~$145** | **~$390** |

*Costs based on Azure East US 2 pricing, reserved instances would reduce by ~40%.*

## Risk Mitigation

- **Model quality**: Evaluation framework with BLEU, operation coverage, and structural similarity metrics
- **Hallucinations**: Retrieved examples anchor generation; validation layer checks syntax
- **Data privacy**: All processing within private VNet, no external API calls
- **Availability**: Docker-based deployment with health checks and auto-restart

## Success Metrics

- Reduction in time-to-delivery for PySpark reports (target: 70% faster)
- User satisfaction score from engineering team (target: >4/5)
- Code quality: operation coverage score on evaluation set (target: >0.85)
