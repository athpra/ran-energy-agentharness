# Beyond Digital Twins: A Multi-Tool Agent Harness for Intent-Based 5G RAN Energy Optimization

Research code for the paper of the same title.

## Repository Structure

```
ran-energy-agent/
│
├── generators/             # Synthetic dataset generators (run once)
│   ├── generate_all.py             # Run all generators in order
│   ├── generate_historical_kpi.py
│   ├── generate_traffic_forecast.py
│   ├── generate_faults.py
│   ├── generate_interference.py
│   └── generate_energy_pricing.py
│
├── tools/                  # Runtime tool API used by the agent
│   ├── historical_kpi.py
│   ├── traffic_prediction.py
│   ├── alarm_fault.py
│   ├── interference.py
│   └── energy_pricing.py
│
├── agent/                  # Agent harness (planner, validator, orchestration)
│
├── experiments/            # Experiment runners — one per condition
│
├── analysis/               # Results analysis notebooks
│
├── data/synthetic/         # Generated data (gitignored — regenerate locally)
│
└── paper/                  # Paper source
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in LLM endpoint and RSG host
```

## Generate synthetic data

```bash
python generators/generate_all.py
```

Options:

```bash
python generators/generate_all.py --n-cells 6 --n-days 14 --seed 42
```

This produces five files under `data/synthetic/`:

| File | Description |
|---|---|
| `historical_kpi.csv` | Per-cell throughput, utilization, sleep state over N days |
| `traffic_forecast.csv` | Rolling-mean N-step-ahead forecasts per cell |
| `faults.csv` | Fault events per cell (Bernoulli process, geometric duration) |
| `neighbor_graph.json` | Adjacency list of neighboring cells |
| `load_shift_matrix.csv` | Fraction of load each sleeping cell shifts to each neighbor |
| `energy_pricing.csv` | Time-of-use electricity price per interval |

## Experimental conditions

| Condition | Tools available to agent |
|---|---|
| Baseline A — rule-based | None (deterministic threshold policy) |
| Baseline B — open-loop LLM | None (single LLM, no simulation test, no validator) |
| Baseline C — digital twin only | VIAVI AI RSG (current PoC architecture) |
| Proposed — full harness | VIAVI AI RSG + all 5 synthetic tools |
| Ablation (per tool) | Digital twin + one tool added at a time |

## Reference

Based on the [VIAVI / NVIDIA Intent-Based RAN Energy Efficiency Blueprint](https://github.com/VIAVI-CTOO/es-blueprint-rsg).
