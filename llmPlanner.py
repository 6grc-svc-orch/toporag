"""Conference Qwen2 planner and offline demonstration preparation."""

from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import rag


class PlannerHeads(nn.Module):
    """The placement, utility, and reward heads described in the paper."""

    def __init__(self, hidden_size: int, maximum_candidates: int) -> None:
        super().__init__()
        self.placement = nn.Linear(hidden_size, maximum_candidates)
        self.utility = nn.Linear(hidden_size, 1)
        self.reward = nn.Linear(hidden_size, 1)

    def forward(self, planner_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "placement": self.placement(planner_hidden),
            "utility": self.utility(planner_hidden).squeeze(-1),
            "reward": self.reward(planner_hidden).squeeze(-1),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _policy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "service_nodes",
        "service_adjacency",
        "service_mask",
        "infrastructure_nodes",
        "infrastructure_mask",
        "context",
        "placement_mask",
    )
    return {key: _jsonable(observation[key]) for key in keys}


def _feasible_prompt_view(feasible_nodes: list[list[int]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for assignment, nodes in enumerate(feasible_nodes):
        groups.setdefault(tuple(int(node) for node in nodes), []).append(assignment)
    return [
        {"assignments": assignments, "node_ids": list(nodes)}
        for nodes, assignments in groups.items()
    ]


def load_prompt(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    required = {
        "{{GRAPH_EMBEDDING}}",
        "{{TELEMETRY}}",
        "{{SERVICE_DAG}}",
        "{{FEASIBLE_NODES}}",
        "{{RETRIEVED_TRACES}}",
        "{{QOS_INTENT}}",
    }
    missing = [placeholder for placeholder in required if placeholder not in text]
    if missing:
        raise ValueError(f"planner prompt is missing {missing}")
    return text


def render_prompt(
    template: str, query: dict[str, Any], traces: list[dict[str, Any]]
) -> str:
    required = {
        "embedding",
        "telemetry",
        "intent",
        "dag",
        "assignments",
        "assignment_demands",
        "feasible_nodes",
        "resources",
    }
    missing = required.difference(query)
    if missing:
        raise ValueError(f"planner query is missing {sorted(missing)}")
    replacements = {
        "{{GRAPH_EMBEDDING}}": _compact_json(query["embedding"]),
        "{{TELEMETRY}}": _compact_json(
            {"telemetry": query["telemetry"], "resources": query["resources"]}
        ),
        "{{SERVICE_DAG}}": _compact_json(
            {
                "dag": query["dag"],
                "ordered_assignments": query["assignments"],
                "assignment_demands": query["assignment_demands"],
            }
        ),
        "{{FEASIBLE_NODES}}": _compact_json(
            _feasible_prompt_view(query["feasible_nodes"])
        ),
        "{{RETRIEVED_TRACES}}": _compact_json(
            [
                {
                    "placement": np.asarray(trace["placement"], dtype=np.int64)
                    .reshape(-1)
                    .tolist(),
                    "outcome": trace["outcome"],
                    "score": trace["retrieval"]["score"],
                }
                for trace in traces
            ]
        ),
        "{{QOS_INTENT}}": _compact_json(query["intent"]),
    }
    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def _exact_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    value, end = decoder.raw_decode(text.lstrip())
    if text.lstrip()[end:].strip():
        raise ValueError("planner output contains text outside the JSON object")
    if not isinstance(value, dict):
        raise TypeError("planner output must be a JSON object")
    return value


def validate_plan(value: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    if set(value) != {"placement", "utility", "reward"}:
        raise ValueError("planner output has an unexpected schema")
    placement = value["placement"]
    feasible_nodes = query["feasible_nodes"]
    demands = query["assignment_demands"]
    if len(demands) != len(feasible_nodes):
        raise ValueError("assignment demands and feasible-node lists differ")
    if not isinstance(placement, list) or len(placement) != len(feasible_nodes):
        raise ValueError("placement length does not match ordered assignments")
    node_ids = [int(item) for item in query["resources"]["node_ids"]]
    residual = {
        node_id: np.asarray(resources, dtype=np.float64).copy()
        for node_id, resources in zip(
            node_ids, query["resources"]["residual"], strict=True
        )
    }
    for index, (node_id, allowed) in enumerate(zip(placement, feasible_nodes)):
        if (
            isinstance(node_id, bool)
            or int(node_id) != node_id
            or int(node_id) not in allowed
        ):
            raise ValueError(f"placement {index} is infeasible")
        demand = np.asarray(demands[index], dtype=np.float64)
        if demand.shape != (2,) or np.any(~np.isfinite(demand)) or np.any(demand <= 0):
            raise ValueError(f"assignment demand {index} is invalid")
        if int(node_id) not in residual or np.any(
            residual[int(node_id)] + 1e-9 < demand
        ):
            raise ValueError(f"placement {index} exceeds sequential resources")
        residual[int(node_id)] -= demand
    utility = float(value["utility"])
    reward = float(value["reward"])
    if not math.isfinite(utility) or not math.isfinite(reward):
        raise ValueError("utility and reward must be finite")
    return {
        "placement": [int(item) for item in placement],
        "utility": utility,
        "reward": reward,
    }


def _top_transformer_layers(model: Any, trainable_fraction: float) -> list[int]:
    total = int(model.config.num_hidden_layers)
    count = max(1, math.ceil(total * trainable_fraction))
    return list(range(total - count, total))


def load_qwen_planner(config: dict[str, Any], trainable: bool) -> dict[str, Any]:
    """Load text-only Qwen2-7B in FP16 and the conference LoRA adapter."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    llm = config["llm"]
    if llm["dtype"] != "float16":
        raise ValueError("the paper configuration requires float16")
    adapter_path = Path(llm["adapter"])
    if not trainable and not adapter_path.exists():
        raise FileNotFoundError(f"planner adapter not found: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm["model"])
    model = AutoModelForCausalLM.from_pretrained(
        llm["model"], torch_dtype=torch.float16, device_map="auto"
    )
    if trainable:
        layers = _top_transformer_layers(model, float(llm["trainable_fraction"]))
        lora = LoraConfig(
            r=int(llm["lora_rank"]),
            lora_alpha=int(llm["lora_alpha"]),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            layers_to_transform=layers,
            layers_pattern="layers",
        )
        model = get_peft_model(model, lora)
    else:
        model = PeftModel.from_pretrained(model, adapter_path)

    maximum_candidates = (
        math.ceil(config["network"]["esps"] / config["network"]["controllers"]) + 1
    )
    heads = PlannerHeads(int(model.config.hidden_size), maximum_candidates)
    head_path = adapter_path / "planner_heads.pt"
    if not trainable:
        heads.load_state_dict(
            torch.load(head_path, map_location="cpu", weights_only=True)
        )
    heads.to(next(model.parameters()).device, dtype=torch.float32)
    model.train(trainable)
    heads.train(trainable)
    return {"model": model, "tokenizer": tokenizer, "heads": heads}


def _chat_tokens(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def _sequence_scores(
    generated: Any, prompt_length: int, eos_token_id: int
) -> list[float]:
    scores = torch.zeros(
        generated.sequences.shape[0], device=generated.sequences.device
    )
    counts = torch.zeros_like(scores)
    active = torch.ones_like(scores, dtype=torch.bool)
    for offset, logits in enumerate(generated.scores):
        token_ids = generated.sequences[:, prompt_length + offset]
        token_scores = (
            torch.log_softmax(logits.float(), dim=-1)
            .gather(1, token_ids[:, None])
            .squeeze(1)
        )
        scores += token_scores * active
        counts += active
        active &= token_ids != eos_token_id
    return (scores / counts.clamp_min(1)).detach().cpu().tolist()


def generate_candidates(
    runtime: dict[str, Any],
    prompt: str,
    query: dict[str, Any],
    candidate_count: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    device = next(model.parameters()).device
    input_ids = _chat_tokens(tokenizer, prompt, device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=candidate_count,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    likelihoods = _sequence_scores(
        generated, input_ids.shape[1], int(tokenizer.eos_token_id)
    )
    plans = []
    for sequence, likelihood in zip(generated.sequences, likelihoods):
        text = tokenizer.decode(
            sequence[input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        try:
            plan = validate_plan(_exact_json(text), query)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        plan["sequence_log_likelihood"] = float(likelihood)
        plans.append(plan)
    if not plans:
        raise RuntimeError("Qwen produced no valid feasible plan")
    plans.sort(key=lambda item: item["sequence_log_likelihood"], reverse=True)
    confidences = torch.softmax(
        torch.tensor(
            [item["sequence_log_likelihood"] for item in plans], dtype=torch.float32
        ),
        dim=0,
    ).tolist()
    for plan_value, confidence in zip(plans, confidences, strict=True):
        plan_value["plan_confidence"] = float(confidence)
    return plans


def retrieve_context(
    config: dict[str, Any], query: dict[str, Any]
) -> list[dict[str, Any]]:
    return rag.retrieve_traces(
        config["rag"]["database"],
        {
            "embedding": query["embedding"],
            "dag": query["dag"],
            "placement": query["telemetry"]["placement"],
        },
        int(config["rag"]["top_k"]),
        float(config["rag"]["decay_seconds"]),
        time.time(),
    )


def plan(
    runtime: dict[str, Any], config: dict[str, Any], query: dict[str, Any]
) -> list[dict[str, Any]]:
    traces = retrieve_context(config, query)
    prompt = render_prompt(load_prompt(config["llm"]["prompt"]), query, traces)
    plans = generate_candidates(
        runtime,
        prompt,
        query,
        int(config["llm"]["candidate_plans"]),
        int(config["llm"]["max_new_tokens"]),
    )
    model = runtime["model"]
    input_device = next(model.parameters()).device
    prompt_ids = _chat_tokens(runtime["tokenizer"], prompt, input_device)
    head_device = next(runtime["heads"].parameters()).device
    with torch.no_grad():
        output = model(input_ids=prompt_ids, output_hidden_states=True)
        hidden = output.hidden_states[-1][:, -1].to(head_device, dtype=torch.float32)
        predictions = runtime["heads"](hidden)
    candidate_ids = query["feasible_nodes"][0]
    candidate_indices = query["feasible_indices"][0]
    candidate_logits = predictions["placement"][
        0, torch.as_tensor(candidate_indices, device=head_device)
    ]
    probabilities = torch.softmax(candidate_logits, dim=0).cpu().tolist()
    probability_by_node = dict(zip(candidate_ids, probabilities, strict=True))
    for plan_value in plans:
        plan_value["placement_head_probability"] = float(
            probability_by_node[plan_value["placement"][0]]
        )
        plan_value["predicted_utility"] = float(predictions["utility"].item())
        plan_value["predicted_reward"] = float(predictions["reward"].item())
    return plans


def soft_placement_prior(
    plan_value: dict[str, Any],
    assignment_index: int,
    candidate_node_ids: list[int],
    strength: float,
) -> np.ndarray:
    if strength < 0.0:
        raise ValueError("prior strength must be non-negative")
    if assignment_index < 0 or assignment_index >= len(plan_value["placement"]):
        raise IndexError("assignment index is outside the planner sequence")
    logits = np.zeros(len(candidate_node_ids), dtype=np.float32)
    selected = int(plan_value["placement"][assignment_index])
    if selected in candidate_node_ids:
        confidence = float(plan_value["plan_confidence"])
        if assignment_index == 0:
            confidence = 0.5 * (
                confidence + float(plan_value["placement_head_probability"])
            )
        confidence += 1.0
        logits[candidate_node_ids.index(selected)] = strength * confidence
    return logits


def save_offline_samples(path: str, samples: list[dict[str, Any]]) -> None:
    if not samples:
        raise ValueError("offline samples cannot be empty")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    arrays: dict[str, Any] = {
        "candidate_index": np.asarray(
            [sample["candidate_index"] for sample in samples], dtype=np.int64
        ),
    }
    for key in samples[0]["observation"]:
        arrays[f"observation__{key}"] = np.asarray(
            [sample["observation"][key] for sample in samples]
        )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(target)


def load_offline_samples(path: str) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"candidate_index"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"offline sample archive is missing {sorted(missing)}")
        observation_keys = [
            name.removeprefix("observation__")
            for name in archive.files
            if name.startswith("observation__")
        ]
        if not observation_keys:
            raise ValueError("offline sample archive contains no observations")
        loaded = {name: archive[name] for name in archive.files}
        count = len(loaded["candidate_index"])
        if any(len(value) != count for value in loaded.values()):
            raise ValueError("offline sample arrays have different leading lengths")
        return [
            {
                "candidate_index": int(loaded["candidate_index"][index]),
                "observation": {
                    key: loaded[f"observation__{key}"][index].copy()
                    for key in observation_keys
                },
            }
            for index in range(count)
        ]


def _causal_example(
    tokenizer: Any, prompt: str, target: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_ids = _chat_tokens(tokenizer, prompt, device)
    target_ids = tokenizer(
        target + tokenizer.eos_token, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    input_ids = torch.cat((prompt_ids, target_ids), dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return input_ids, labels, int(prompt_ids.shape[1])


def train_qwen_adapter(
    runtime: dict[str, Any], config: dict[str, Any], samples: list[dict[str, Any]]
) -> None:
    model = runtime["model"]
    heads = runtime["heads"]
    tokenizer = runtime["tokenizer"]
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    parameters.extend(heads.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(config["ppo"]["actor_lr"]))
    device = next(model.parameters()).device
    order = list(range(len(samples)))
    generator = random.Random(config["seed"])
    generator.shuffle(order)
    for index in order:
        sample = samples[index]
        input_ids, labels, prompt_length = _causal_example(
            tokenizer, sample["prompt"], sample["target"], device
        )
        output = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
        head_device = next(heads.parameters()).device
        hidden = output.hidden_states[-1][:, prompt_length - 1].to(
            head_device, dtype=torch.float32
        )
        predictions = heads(hidden)
        placement_target = torch.tensor(
            [sample["planner_candidate_index"]], device=device
        )
        utility_target = torch.tensor(
            [sample["utility"]], dtype=torch.float32, device=device
        )
        reward_target = torch.tensor(
            [sample["reward"]], dtype=torch.float32, device=device
        )
        loss = output.loss
        loss = loss + nn.functional.cross_entropy(
            predictions["placement"], placement_target
        )
        loss = loss + nn.functional.mse_loss(predictions["utility"], utility_target)
        loss = loss + nn.functional.mse_loss(predictions["reward"], reward_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    adapter_path = Path(config["llm"]["adapter"])
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    torch.save(heads.state_dict(), adapter_path / "planner_heads.pt")


def _clear_memory(path: str) -> None:
    rag.initialize_memory(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM traces")
    rag.invalidate_memory(path)


def collect_demonstrations(
    config: dict[str, Any], scenario: dict[str, Any], encode_graph: Any
) -> list[dict[str, Any]]:
    """Measure full plans with one rotating placement label per trace."""
    import simulator

    target_samples = int(config["llm"]["offline_samples"])
    target_traces = int(config["rag"]["initial_traces"])
    if target_samples > target_traces:
        raise ValueError("offline_samples cannot exceed initial_traces")
    samples: list[dict[str, Any]] = []
    trace_count = 0
    episode = 0
    template = load_prompt(config["llm"]["prompt"])
    while trace_count < target_traces:
        state = simulator.reset(
            config,
            scenario,
            config["experiment"]["latency_scales"][0],
            config["seed"] + episode,
        )
        done = bool(state["done"])
        while not done and trace_count < target_traces:
            initial_observation = simulator.observe(state)
            done = bool(state["done"])
            if done:
                break
            request_id = int(state["current_request"])
            trace_embedding = _jsonable(encode_graph(initial_observation))
            trace_query = simulator.planner_query(state, trace_embedding)
            collecting_sample = len(samples) < target_samples
            traces = (
                retrieve_context(config, trace_query)
                if trace_count and collecting_sample
                else []
            )
            prompt = (
                render_prompt(template, trace_query, traces)
                if collecting_sample
                else ""
            )
            clone_offset = trace_count % len(trace_query["assignments"])
            clone_observation = None
            clone_action = None
            planner_action = None
            placement: list[int] = []
            completed = None
            completed_placement = None
            while completed is None and not done:
                observation = simulator.observe(state)
                done = bool(state["done"])
                if done or int(state["current_request"]) != request_id:
                    break
                action = simulator.feasible_action(state)
                if not placement:
                    planner_action = int(action["placement_index"])
                if collecting_sample and len(placement) == clone_offset:
                    clone_observation = _policy_observation(observation)
                    clone_action = int(action["placement_index"])
                placement.append(int(action["node_id"]))
                _, _, done, info = simulator.step(state, action)
                completed = info["completed_request"]
                completed_placement = info["completed_placement"]
            if completed is None or len(placement) != len(trace_query["assignments"]):
                continue
            objective = simulator.request_objective(config, completed)
            full_plan = {
                "placement": placement,
                **objective,
            }
            validate_plan(full_plan, trace_query)
            if collecting_sample:
                if (
                    clone_observation is None
                    or clone_action is None
                    or planner_action is None
                ):
                    raise RuntimeError("demonstration missed its cloning decision")
                samples.append(
                    {
                        "prompt": prompt,
                        "target": _compact_json(full_plan),
                        "embedding": trace_embedding,
                        "candidate_index": clone_action,
                        "planner_candidate_index": planner_action,
                        **objective,
                        "observation": clone_observation,
                    }
                )
            rag.record_trace(
                config["rag"]["database"],
                {
                    "timestamp": time.time() - (target_traces - trace_count),
                    "embedding": trace_embedding,
                    "telemetry": trace_query["telemetry"],
                    "intent": trace_query["intent"],
                    "dag": trace_query["dag"],
                    "placement": np.asarray(completed_placement).tolist(),
                    "outcome": {
                        "latency_ms": float(completed["latency_ms"]),
                        "resource_use": float(completed["resource_utilization"]),
                        "sla_violations": float(completed["sla_violation"]),
                        **objective,
                    },
                },
                int(config["rag"]["max_traces"]),
            )
            trace_count += 1
        episode += 1
    if len(samples) != target_samples:
        raise RuntimeError(
            "simulation ended before enough offline samples were collected"
        )
    return samples


def main() -> None:
    import agent
    import simulator

    config = simulator.load_config("configs/paper.json")
    scenario = simulator.load_scenario(config["data"]["scenario"])
    torch.manual_seed(config["seed"])
    bootstrap_config = json.loads(json.dumps(config))
    bootstrap_config["rag"]["initial_traces"] = int(config["llm"]["offline_samples"])
    _clear_memory(config["rag"]["database"])
    initial_encoder = agent.build_graph_encoder(config, scenario)

    def initial_embedding(observation: dict[str, Any]) -> list[float]:
        return (
            agent.graph_embedding(initial_encoder, observation)
            .squeeze(0)
            .cpu()
            .tolist()
        )

    bootstrap = collect_demonstrations(bootstrap_config, scenario, initial_embedding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cloned_policy = agent.build_ppo(bootstrap[0]["observation"], config["ppo"], device)
    agent.behavior_clone(
        cloned_policy,
        bootstrap,
        epochs=int(config["llm"]["behavior_epochs"]),
        minibatch_size=int(config["ppo"]["minibatch_size"]),
    )
    encoder = cloned_policy["model"].actor_dag
    encoder.eval()
    _clear_memory(config["rag"]["database"])

    def encode_graph(observation: dict[str, Any]) -> list[float]:
        return agent.graph_embedding(encoder, observation).squeeze(0).cpu().tolist()

    samples = collect_demonstrations(config, scenario, encode_graph)
    save_offline_samples(config["llm"]["offline_data"], samples)
    runtime = load_qwen_planner(config, trainable=True)
    train_qwen_adapter(runtime, config, samples)
    torch.save(
        encoder.state_dict(), Path(config["llm"]["adapter"]) / "graph_encoder.pt"
    )
    print(
        f"prepared {len(samples)} demonstrations, {rag.memory_size(config['rag']['database'])} traces, "
        f"and {config['llm']['adapter']}"
    )


if __name__ == "__main__":
    main()
