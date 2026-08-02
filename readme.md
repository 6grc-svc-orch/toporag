# AI-native cloud-edge orchestration for 6G metaverse networks: an LLM-guided multi-agent DRL approach

This repository implements the architecture and experiments described in **AI-native cloud-edge orchestration for 6G metaverse networks: an LLM-guided multi-agent DRL approach**. The [paper](.codex/j2.pdf) is the primary specification for the simulator, cloud planner, edge agents, baselines, and evaluation figures.

The project combines:

- a deterministic cloud–edge simulator for attributed infrastructure graphs, regional observations, service DAGs, replicas, placement, routing, mobility, and QoS measurement;
- a topology-aware trace memory and Qwen2-VL planner that produces asynchronous orchestration guidance;
- semantic-guided MAPPO edge agents and adapted comparison methods evaluated on shared seeded workloads; and
- a measured-results pipeline that recreates Figures 4–11 with confidence intervals and spatial utilization maps.

## Setup

Use Python 3.12 and install the pinned dependencies:

```bash
pip install -r requirements.txt
```

`configs/paper.json` selects the reproducible synthetic trace or the OpenCellID, WSDOT, and Alibaba adapters. The adapter paths are grouped under `simulation.external_data`.

Experiment-defining values live in `configs/paper.json`. Runtime planner templates live in `prompts/`, and prepared workload inputs live under `data/`.

## Run

Run the project files in this order from the repository root:

```bash
python prepare_data.py
python llmPlanner.py
python train_agentMulti.py
python evaluate.py
python figures/generate_figures.py
```

Training writes checkpoints and measured logs to the configured runtime directories. Evaluation writes the long-form records and spatial grids to `figures/data/`. The figure generator validates the complete measured dataset and writes Figures 4–11 as PNG and PDF files to `figures/output/`.

## Citation
If you find our work useful, please consider citing it:

```
@article{Ayepah-Mensah2026,
  author  = {Ayepah-Mensah, Daniel and Ghebreziabiher, Amine Kidane and Boateng, Gordon Owusu and Mizouni, Rabeb and Mourad, Azzam and Otrok, Hadi and Bentahar, Jamal},
  title   = {AI-native cloud-edge orchestration for 6G metaverse networks: an LLM-guided multi-agent DRL approach},
  journal = {Complex & Intelligent Systems},
  year    = {2026},
  volume  = {12},
  number  = {4},
  pages   = {141},
  doi     = {10.1007/s40747-026-02249-9},
  url     = {https://doi.org/10.1007/s40747-026-02249-9},
  issn    = {2198-6053}
}
```
