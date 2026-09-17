# Cloudera Blueprint: Multi-Tool Agent Harness for 5G RAN Energy Optimization

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-success.svg)](#quickstart)

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Use Case](#use-case)
- [Key Features](#key-features)
- [Quickstart](#quickstart)
- [Architecture / Software Components](#architecture--software-components)
- [Target Audience](#target-audience)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Hardware Requirements](#hardware-requirements)
- [Documentation](#documentation)

## Overview

This blueprint extends the [Intent-Based RAN Energy Efficiency Blueprint](https://github.com/athpra/intent-based-ran-energy-optimization) by adding a multi-tool agent harness to the dual-LLM closed loop. Beyond using a VIAVI AI RSG digital twin, the Planner Agent is augmented with five additional runtime tools — historical KPI store, traffic forecast, alarm/fault feed, inter-cell interference model, and energy pricing — injected as structured context on every decision cycle. The result is a research framework for measuring how each tool contributes to energy savings and QoS preservation in 5G RAN automation, running entirely on Cloudera AI.

## Demo

The `analysis/results.ipynb` notebook runs in demo mode with synthetic data — no VIAVI RSG connection required. Open a CML session, run all cells, and seven publication-ready figures are produced comparing all experimental conditions.

To run a live experiment against VIAVI AI RSG, use the experiment runners as CML Jobs (see [Quickstart](#quickstart)).

## Use Case

Mobile operators need to reduce 5G RAN energy consumption during low-traffic periods without violating per-user QoS guarantees. Cell sleeping is the primary lever, but applying it incorrectly causes throughput degradation. This blueprint evaluates how much additional context — historical patterns, traffic forecasts, fault status, interference risk, and energy pricing — improves an AI agent's sleep/wake decisions compared to a digital-twin-only baseline.

## Key Features

- **Tool-augmented agent harness**: five runtime tools inject structured context into every Planner decision cycle, going beyond digital twin simulation alone
- **Dual-LLM closed loop**: a Planner LLM proposes sleep/wake actions; a Validator LLM approves each action against QoS constraints before it is applied
- **Simulation-backed safety**: every candidate action is tested in the VIAVI AI RSG digital twin before being committed to the network state
- **Ablation-ready**: a single `--tools` flag enables or disables any subset of tools, making per-tool contribution analysis a one-line command
- **Private AI by design**: all LLM inference runs on Cloudera AI Inference Service; no network telemetry leaves your environment
- **Reproducible without hardware**: synthetic data generators and demo mode let the full analysis run without a VIAVI RSG connection

## Quickstart

### On Cloudera Machine Learning

1. **Create a new CML project** from Git:
   ```
   https://github.com/athpra/ran-energy-agentharness
   ```

2. **Set project environment variables** (Project Settings → Environment Variables) before the build runs:

   | Variable | Value |
   |---|---|
   | `RSG_HOST` | `3.211.96.252` |
   | `CDSW_API_URL` | Your Cloudera AI Inference endpoint URL |
   | `CDSW_API_KEY` | Your CML API key |
   | `LLM_MODEL` | Model name as registered in the endpoint |
   | `AUTH_MODE` | `jwt` (CML workbench) or `api_key` (external) |

3. **Build the project** — `cdsw-build.sh` installs `requirements.txt` and the VIAVI ADK automatically.

4. **Generate synthetic data** in a session:
   ```bash
   python generators/generate_all.py
   ```

5. **Run the analysis notebook** (demo mode — no RSG needed):
   ```
   analysis/results.ipynb
   ```

6. **Run experiments against VIAVI AI RSG** as CML Jobs:
   ```bash
   python experiments/baseline_rules.py --rsg-host $RSG_HOST --iterations 20
   python experiments/baseline_openloop.py --rsg-host $RSG_HOST --iterations 20
   python experiments/baseline_digital_twin.py --rsg-host $RSG_HOST --iterations 20
   python experiments/full_harness.py --rsg-host $RSG_HOST --iterations 20

   # Ablation — add one tool at a time
   python experiments/full_harness.py --rsg-host $RSG_HOST --tools historical_kpi
   python experiments/full_harness.py --rsg-host $RSG_HOST --tools historical_kpi traffic_forecast
   ```

### Local

```bash
git clone https://github.com/athpra/ran-energy-agentharness.git
cd ran-energy-agentharness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install http://3.211.96.252:8000/adk   # VIAVI ADK
cp .env.example .env                        # fill in LLM endpoint credentials
python generators/generate_all.py
jupyter notebook analysis/results.ipynb
```

## Architecture / Software Components

The system is a fixed-pipeline tool-augmented agent harness. On each decision cycle:

1. All enabled tools are called in parallel and their outputs are assembled into a structured context block
2. The Planner LLM receives current KPIs + tool context and proposes sleep/wake actions
3. Each proposed action is tested in the VIAVI AI RSG digital twin (Sim 1 — no state change)
4. The Validator LLM reviews simulation results against QoS constraints and approves or rejects each action
5. Approved actions are applied to the accumulated network state and replayed into a second simulation (Sim 2)
6. Results are logged for analysis

**Experimental conditions:**

| Condition | Description |
|---|---|
| Baseline A — rule-based | Deterministic threshold policy, no LLM |
| Baseline B — open-loop LLM | Single LLM, no tools, no simulation test, no validator |
| Baseline C — digital twin only | Dual-LLM + VIAVI RSG (current blueprint architecture) |
| Proposed — full harness | Dual-LLM + VIAVI RSG + all five tools |
| Ablation | Digital twin + one tool added at a time |

**Software components:**

- [Cloudera AI Inference Service](https://www.cloudera.com/products/machine-learning.html) — LLM hosting (Planner + Validator)
- [VIAVI TeraVM AI RAN Scenario Generator (AI RSG)](https://www.viavisolutions.com) — 5G RAN digital twin
- [LangChain](https://github.com/langchain-ai/langchain) — LLM orchestration
- Python 3.10+, pandas, numpy, matplotlib, seaborn

## Target Audience

- Network automation engineers evaluating AI-driven RAN energy optimization
- ML engineers building tool-augmented agentic systems on Cloudera AI
- Telecom researchers studying closed-loop RAN control with LLMs
- Solution architects comparing agentic AI patterns against rule-based and open-loop baselines

## Repository Structure

| Path | Description |
|---|---|
| `generators/` | Synthetic dataset generators — run once to produce `data/synthetic/` |
| `tools/` | Runtime tool API modules (historical KPI, forecast, faults, interference, pricing) |
| `agent/` | Agent harness: context assembler, planner, validator, orchestration |
| `experiments/` | Experiment runners — one script per experimental condition |
| `analysis/` | Results analysis notebook producing publication-ready figures |
| `data/synthetic/` | Generated datasets (gitignored — regenerate with `generators/generate_all.py`) |
| `paper/` | LaTeX source for the research paper |
| `cdsw-build.sh` | CML project build script — installs dependencies and VIAVI ADK |
| `.env.example` | Environment variable template |
| `METADATA.yaml` | Cloudera blueprint catalog metadata |

## Prerequisites

- Access to **Cloudera Machine Learning** (CML) with AI Inference Service enabled
- Access to a **VIAVI TeraVM AI RSG** instance with ADK license TVM6238
  - Contact [IB_ES_blueprint@viavisolutions.com](mailto:IB_ES_blueprint@viavisolutions.com) to request access
  - Verify connectivity: `curl http://<rsg-host>:8000/status`
- A deployed **LLM endpoint** on Cloudera AI Inference Service (tested with `Qwen/Qwen2.5-7B-Instruct` and `nvidia/nemotron-3-super-120b-a12b`)
- Python 3.10 or newer (local setup only)

## Hardware Requirements

| Deployment | Minimum |
|---|---|
| Demo / analysis notebook (no RSG) | 2 vCPU, 8 GB RAM CML session |
| Full experiment run with VIAVI RSG | 4 vCPU, 16 GB RAM CML session |
| LLM inference (Cloudera AI Inference) | GPU node sized for the chosen model |

## Documentation

- [Intent-Based RAN Energy Efficiency Blueprint](https://github.com/athpra/intent-based-ran-energy-optimization) — baseline PoC this work extends
- [Fransiscus et al. (2025) — arXiv:2507.14230](https://arxiv.org/abs/2507.14230) — original blueprint paper
- [Cloudera AI Inference Service documentation](https://docs.cloudera.com)

## Contributors

1. [Athul Prasad](https://www.linkedin.com/in/athul-prasad/) — Applied AI, Cloudera

## Disclaimer

*This blueprint is intended for research and proof-of-concept use only. It is not designed for production deployment. Use in production environments is at the user's own risk.*
