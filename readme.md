# RAG-Assisted DRL for Microservice Deployment

This repository implements the conference paper **A RAG-Assisted DRL Framework for Microservices Deployment in 6G Vehicular Networks**. It combines a Qwen2 semantic planner and SQLite deployment memory with a GAT-encoded PPO scheduler for joint microservice placement and routing. DQN Scheduler and DRP provide the paper baselines.

The default configuration represents 3,233 edge service providers, 16 edge controllers, one cloud, and 4,328 vehicle traces. Synthetic paper-scale inputs are generated deterministically, and the data configuration also accepts the Shanghai Telecom, Shanghai Taxi, and Alibaba trace files.

## Run

Install `requirements.txt`, then run the files in order:

```bash
python prepare_data.py
python llmPlanner.py
python train_agent.py
python evaluate.py
python figures/generate_figures.py
```

`configs/paper.json` contains the experiment parameters. Measured records and Figures 3–6 are written under `figures/`.
