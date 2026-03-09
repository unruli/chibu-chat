# ChibuChat Base Model — Pretraining Summary

## Overview

The ChibuChat base language model completed pretraining on **February 7, 2026** after 21,400 optimization steps over ~14.4 hours of GPU compute on the University of Florida HiPerGator cluster. The model was trained using the [nanochat](https://github.com/TrelisResearch/nanochat) framework on the FineWeb Edu 100B dataset.

---

## Model Architecture

| Parameter | Value |
|---|---|
| Architecture | GPT (decoder-only Transformer) |
| Layers | 20 |
| Model dimension | 1,280 |
| Attention heads | 10 (MHA, head dim 128) |
| KV heads | 10 (no GQA) |
| Context length | 2,048 tokens |
| Vocab size | 65,536 (custom BPE tokenizer) |
| **Total parameters** | **560,988,160 (~561M)** |
| FLOPs per token | 3.49 × 10⁹ |
| Precision | BFloat16 |

## Training Configuration

| Setting | Value |
|---|---|
| Dataset | FineWeb Edu 100B (shuffled) |
| Data shards | 19 train / 1 val (20 total) |
| Total training tokens | 11,219,763,200 (~11.2B) |
| Tokens : Params ratio | 20.0 (Chinchilla-optimal) |
| Total batch size | 524,288 tokens |
| Device batch size | 16 |
| Gradient accumulation steps | 16 |
| Optimizer (embeddings) | AdamW (lr=0.2 / 0.004, wd=0.0) |
| Optimizer (matrices) | Muon (lr=0.02) |
| Gradient clipping | 1.0 |
| LR warmup | 0% of steps |
| LR warmdown | 20% of steps (cosine to 0) |
| torch.compile | Enabled |

## Infrastructure

| Resource | Details |
|---|---|
| Cluster | UF HiPerGator (SLURM) |
| GPUs | 2× NVIDIA B200 (183 GB VRAM each) |
| Driver | NVIDIA 570.148.08, CUDA 12.8 |
| Framework | PyTorch 2.9.0+cu128 |
| Peak GPU memory | 40,649 MiB (~39.7 GB) |
| Tracking | Weights & Biases ([project link](https://wandb.ai/c-okocha-university-of-florida/chibuchatGpt)) |

## Training Timeline

| Date | Event |
|---|---|
| Jan 27 | Tokenizer trained (65K BPE vocab) |
| Jan 27 | Base training started (job 23932063), initial run to step 1000 |
| Feb 6 | Diagnosed dataloader epoch-wrap bug causing hangs at step 1900 |
| Feb 6 22:51 | Final training run started (job 24547785) resuming from step 1000 |
| Feb 7 13:05 | **Training completed** — 21,400 steps in 14h 15m wall-clock |

### Bug Fix: Dataloader Epoch Wrap

Both torch.compile-enabled and disabled runs consistently hung at step ~1900. Root cause: when resuming from a checkpoint mid-dataset (shard 10/19), the dataloader exhausted the remaining shards and entered an infinite empty loop instead of wrapping back to shard 0. Fixed by adding `pq_idx = 0` reset in `nanochat/dataloader.py` after the inner shard iteration loop.

## Training Results

### Loss Curve

| Step | Loss | Val BPB | Notes |
|---|---|---|---|
| 1,000 | 3.44 | 1.0003 | Resume checkpoint |
| 5,000 | 2.93 | 0.9034 | — |
| 8,250 | — | **0.8943** | **Best validation BPB** |
| 10,000 | 2.83 | 0.8956 | — |
| 15,000 | 2.67 | 0.8997 | Slight val increase (epoch overlap) |
| 21,400 | 2.49 | 0.8996 | Final |

- **Final training loss:** 2.49
- **Best validation BPB:** 0.8943 (step 8,250)
- **Final validation BPB:** 0.8996

### Training Throughput

| Metric | Value |
|---|---|
| Tokens/sec | ~223,800 |
| Step time | ~2,340 ms |
| MFU | **79.0%** |
| Total training time | 866.4 minutes (14.4 hours) |

### CORE Metric (Aggregate Benchmark)

| Step | CORE Metric |
|---|---|
| 2,000 | 0.1400 |
| 4,000 | 0.1638 |
| 8,000 | 0.1635 |
| 12,000 | 0.1682 |
| 16,000 | **0.1830** |
| 20,000 | 0.1703 |
| 21,400 | 0.1760 |

### Final Benchmark Scores (Step 21,400)

| Benchmark | Shots | Accuracy | Centered |
|---|---|---|---|
| HellaSwag (0-shot) | 0 | 0.4200 | 0.2267 |
| HellaSwag (10-shot) | 10 | 0.4120 | 0.2160 |
| ARC Easy | 10 | **0.6020** | 0.4693 |
| ARC Challenge | 10 | 0.3280 | 0.1040 |
| PIQA | 10 | **0.6760** | 0.3520 |
| COPA | 0 | **0.6600** | 0.3200 |
| Winograd | 0 | 0.6081 | 0.2161 |
| WinoGrande | 0 | 0.5100 | 0.0200 |
| BoolQ | 10 | 0.5660 | −0.1421 |
| CommonsenseQA | 10 | 0.2260 | 0.0325 |
| OpenBookQA | 0 | 0.3480 | 0.1307 |
| LAMBADA | 0 | 0.3280 | 0.3280 |
| BigBench QA Wikidata | 10 | **0.4640** | 0.4640 |
| BigBench CS Algorithms | 10 | 0.4080 | 0.4080 |
| BigBench Lang ID | 10 | 0.2600 | 0.1859 |
| BigBench Operators | 10 | 0.1048 | 0.1048 |
| BigBench Dyck Languages | 10 | 0.0700 | 0.0700 |
| BigBench Repeat Copy Logic | 10 | 0.0000 | 0.0000 |
| AGI Eval LSAT AR | 3 | 0.2217 | 0.0272 |
| Jeopardy | 10 | 0.0780 | 0.0780 |
| SQuAD | 10 | 0.1400 | 0.1400 |
| CoQA | 0 | 0.1200 | 0.1200 |

### Sample Generations (Step 21,400, T=0)

> **The capital of France is** Paris. The capital of France is Paris. The capital of France is Paris.

> **The chemical symbol of gold is** Au. Gold is a yellow, malleable metal that is soft and malleable. It

> **The opposite of hot is** cold. Cold is the opposite of hot. Cold is the opposite of hot.

> **The planets of the solar system are:** Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. The planets are

> **If 5\*x + 3 = 13, then x is** a prime number. *(incorrect — x=2 is prime, but the model didn't solve it)*

## Checkpoints

22 checkpoints saved every 1,000 steps to `~/.cache/nanochat/base_checkpoints/d20/`:

```
model_001000.pt  →  model_021400.pt   (2.0 GB each)
optim_001000_rank0.pt  →  optim_021400_rank0.pt   (2.5 GB each)
meta_001000.json  →  meta_021400.json
```

**Total checkpoint storage:** ~96 GB

## Next Steps

- [ ] Supervised fine-tuning (SFT) for chat capabilities
- [ ] RLHF / DPO alignment
- [ ] Evaluate on downstream tasks (GSM8K, HumanEval, MMLU)
- [ ] Multimodal integration (audio encoders already implemented)
