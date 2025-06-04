# FogEnv: Fog Network Simulation for Microservice Placement

This environment models the placement of **individual microservices** in a fog computing network. It is designed for training and evaluating intelligent agents (e.g., via Reinforcement Learning) that make **step-by-step deployment decisions** for a microservice within a distributed fog infrastructure. The environment is implemented using OpenAI's `gymnasium` interface.

> **Note**: The placement process is viewed **from a single microservice perspective** at each environment step. Each action selects a node for the current microservice only.

---

## Environment Overview

The fog network consists of:

* **Fog nodes**: Each with limited compute (CPU, memory), idle/max power, etc.
* **Links between nodes**: Modeled with bandwidth, energy per bit, delay.
* **Services**: Each service comprises multiple microservices (default `N_MS = 5`), placed **sequentially**, one per environment step.

Each episode simulates the placement of microservices requested by the network.

---

## Observation Space

The agent observes a dictionary containing:

* `node_feats`: `(N_NODES, 6)`
  → Available and total \[CPU, MEM], idle power, max power

* `link_feats`: `(N_LINKS, 4)`
  → Available and total bandwidth, energy per bit, delay

* `service_feats`: `(MAX_SRVS, 3)`
  → Deadline, lifetime, arrival rate for each service

* `ms_feats`: `(N_MS * MAX_SRVS, 3)`
  → Microservice-level features like resource requirements

* `link_ms_feats`: `(N_LINKS_MS * MAX_SRVS, 1)`
  → Data size for microservice-to-microservice communication

* `placement_status`:
  → Matrix tracking which microservices are placed on which nodes

* `next_ms_idx`, `next_service_idx`:
  → Track which microservice of which service is currently being placed

---

## Action Space

```python
action_space = Box(low=0, high=1, shape=(10,), dtype=int)
```

Each action selects one of the 10 fog nodes to place the **current microservice**.

---

## Step Logic

Each call to `env.step(action)` performs:

1. **Service Expiration**: Expires old services based on lifetime
2. **Constraint Check**:

   * C1: Does the selected node have enough CPU and MEM?
   * C2: Is there enough link bandwidth for communication?
3. **Placement**: If constraints pass, place the microservice
4. **Advance State**: If this is the last microservice of a service, record completion time and prepare the next

Returns:

```python
state, reward, done, truncated, info
```

---

## File I/O

* Input service requests are read from `requests.json`
* After reading, the request buffer is cleared to avoid duplication
* File read/write is handled using utility functions in `utils.py`

---

## Visualization

Enable optional visualizations via constructor:

```python
env = FogEnv(network=my_net, vis_placement=True)
```

---

## Assertions and Consistency Checks

The environment includes an assertion mixin (`FogEnvAssertionsMixin`) to verify:

* Node/link resource bounds
* Microservice sequencing
* Placement validity
* Resource reclamation after expiration

These checks ensure the simulation reflects realistic resource-constrained behavior.

---

## Reset & Request Fetching

* `reset()`: Initializes network and clears state
* `fetch_requests()`: Loads new services to process (up to buffer size `REQ_BUFFER_SIZE = 10`)

---

## Intended Use

This environment is designed for:

* Training **RL agents** to deploy microservices adaptively
* Evaluating **placement heuristics** under dynamic request patterns
* Exploring energy-aware and latency-sensitive offloading policies

---

## Microservice-Centric Deployment

> Unlike traditional environments where the whole service is placed at once, this environment advances **one microservice at a time**. This granularity allows:

* Better modeling of **inter-microservice dependencies**
* Accurate bandwidth/resource tracking between partial deployments
* Fine-tuned learning and credit assignment for each micro-decision

---

## Requirements

* `Python 3.8+`
* `gymnasium`
* `numpy`
* `matplotlib`

---

## 📁 Files

| File            | Purpose                                |
| --------------- | -------------------------------------- |
| `env.py`        | Main environment definition            |
| `template.py`   | Network/service structures and loading |
| `utils.py`      | JSON file reading/writing              |
| `requests.json` | Request buffer, cleared after loading  |

> Generated with the assistance of GPT-4o.

