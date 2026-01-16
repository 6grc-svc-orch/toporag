# LLMPlanner.py
import time

import_start = time.time()
import re
import os
import json
import sys
from typing import List, Dict
from rag import initialize_rag, rag_fn
from utils import read_file, read_json, write_json, seed_it
import random
from datetime import datetime
from rich import print

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedModel,
)
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage

import_end = time.time()
print(f"Imports completed in {import_end - import_start:.2f} seconds")

CONTEXT_CAPACITY = 100
device_map = {"": 0}
now = datetime.now().strftime("%m%d_%H%M%S")

key = sys.argv[2]
model_map = {
    "1.5-32b": "Qwen/Qwen1.5-32B-Chat",
    "7b": "Qwen/Qwen2-7B-Instruct",
    "q2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
    "0.5b": "Qwen/Qwen2-0.5B-Instruct",
    "0.36b": "HuggingFaceTB/SmolLM2-360M-Instruct",
}
model_name_or_path = model_map[key]  # "Qwen/Qwen2-7B-Instruct"

run_name = sys.argv[1]
use_rag = True  # False

DIR="drift"
SYS_INST_PATH = "planner/llm_planner_inst.md"
CONTEXT_LOG_PATH = f"logs/context-{run_name}.json"
WEIGHTS_PATH = f"{DIR}/weights-{run_name}-{key}.json"
PLACEMENT_PATH = f"{DIR}/placement-{run_name}-{key}.json"
RAGGER = "BAAI/bge-small-en-v1.5"  # "Qwen/Qwen2-7B-Instruct"
EVERY = 60  # seconds
GPU_ID = int(os.getenv("CUDA_VISIBLE_DEVICES").split(",")[0] if os.getenv("CUDA_VISIBLE_DEVICES") else "0")


print(f"Using model: {model_name_or_path}")
print(f"Using CONTEXT path: {CONTEXT_LOG_PATH}")

nvmlInit()
seed_it(646)


class LLM:  # at the cloud
    def __init__(self, model_name_or_path="Qwen/Qwen2-7B-Instruct", device_map="auto"):
        """Initialize the model and processor"""
        self.model_name_or_path = model_name_or_path
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, torch_dtype=torch.float16, device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

    def process_input(self, raw_input):
        """Process the input data"""
        text = self.processor.apply_chat_template(raw_input, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

        return inputs.to(self.model.device)

    def generate(self, inputs, max_new_tokens=324, **kwargs) -> str:
        """Generate response"""
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    
    def submit_tuning_job(self, data: List[Dict]):
        """Submit a fine-tuning job with the provided data"""
        # This is a placeholder for the actual fine-tuning logic.
        # In practice, this would involve preparing the data, configuring
        # the training parameters, and submitting the job to a training service.
        print("Fine-tuning job submitted with data of length:", len(data))
        # for now let it sleep for 1 minute to simulate
        time.sleep(60)
        
        return True
        


def extract_weights(json_str):
    # Remove any leading non-JSON content (e.g., the word 'json')
    cleaned_str = re.search(r"({.*})", json_str, re.DOTALL)
    if not cleaned_str:
        print("No JSON object found in the input string.")
        return None
    try:
        data: dict = json.loads(cleaned_str.group(1))
        weights = data.get("weights")
        if not weights or "latency" not in weights or "load_balance" not in weights or "utilization" not in weights:
            print("Missing required keys in weights.")
            return None

        # summmation of weights should be 1
        total = weights["latency"] + weights["load_balance"] + weights["utilization"]
        if abs(total - 1.0) > 0.01:
            print("Weights do not sum up to 1.")
            return None
        return weights
    except json.JSONDecodeError as e:
        print("Invalid JSON format.")
        return None


def extract_suggestions(json_str):
    # Remove any leading non-JSON content (e.g., the word 'json')
    cleaned_str = re.search(r"({.*})", json_str, re.DOTALL)
    if not cleaned_str:
        print("No JSON object found in the input string.")
        return None
    try:
        data: dict = json.loads(cleaned_str.group(1))
        placement_suggestion = data.get("placement_suggestion")
        if not placement_suggestion or not isinstance(placement_suggestion, list):
            print("Missing or invalid 'placement_suggestion' key.")
            return None
        if not all(
            all(key in suggestion for key in ["svc_id", "ms_idx", "closest_agent_idx", "node_idx"])
            for suggestion in placement_suggestion
        ):
            print("Missing required keys in placement suggestions.")
            return None
        
        # all the values need to be integers when converted
        for suggestion in placement_suggestion:
            for key in ["svc_id", "ms_idx", "closest_agent_idx", "node_idx"]:
                try:
                    suggestion[key] = int(suggestion[key])
                except ValueError:
                    print(f"Invalid value for {key} in suggestion: {suggestion[key]}")
                    return None
        return placement_suggestion
    except json.JSONDecodeError as e:
        print("Invalid JSON format.")
        return None


def add_suggestions(suggestions: List[Dict]):
    p_suggestions: List[Dict] = read_json(PLACEMENT_PATH, create=True)
    # p_suggestions has svc_id, ms_idx, closest_agent_idx, node_idx
    for i, n_s in enumerate(suggestions):
        for j, p_s in enumerate(p_suggestions):
            if (
                p_s["svc_id"] == n_s["svc_id"]
                and p_s["ms_idx"] == n_s["ms_idx"]
                and p_s["closest_agent_idx"] == n_s["closest_agent_idx"]
            ):
                p_suggestions[j]["node_idx"] = n_s["node_idx"]
                break
        else:
            # append if no match found
            p_suggestions.append(n_s)
    return p_suggestions


def get_context(use_rag, shuffle=True):  # pass top_k?
    data: List[Dict] = read_json(CONTEXT_LOG_PATH)
    if shuffle:
        random.shuffle(data)

    if use_rag:
        q_len = 3
        context = rag_fn(data, q_len=q_len)  # , top_k=10)  # CONTEXT_CAPACITY - q_len
        return context
    else:
        if len(data) > CONTEXT_CAPACITY:
            context = data[-CONTEXT_CAPACITY:]
        else:
            context = data
        return context


def print_gpu_utilization(gpu_id=GPU_ID):
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    mem = nvmlDeviceGetMemoryInfo(handle).used // 1024**2
    power = nvmlDeviceGetPowerUsage(handle) // 1000
    return mem, power


def main():
    lmpf_l = []  # latency-memoryy-power-instfollow
    start = time.time()
    if key == "1.5-32b":
        device_map = "auto"
    else:
        device_map = {"": 0}

    llm = LLM(model_name_or_path=model_name_or_path, device_map=device_map)
    loaded = time.time()
    print(f"Model loaded in {loaded - start:.2f} seconds")

    sys_instruction = read_file(SYS_INST_PATH)
    if use_rag:
        initialize_rag(RAGGER)

    total = int(1e6)  # large number
    for step in range(total):
        start = time.time()
        context = get_context(use_rag=use_rag)

        messages = [{"role": "system", "content": sys_instruction}, {"role": "user", "content": json.dumps(context)}]
        inputs = llm.process_input(messages)

        # Inference
        gen_kwargs = {"temperature": 0.2, "top_p": 0.9, "do_sample": True}
        final_output = llm.generate(inputs, **gen_kwargs)

        inst_follow = 0
        weights = extract_weights(final_output)
        if weights:
            inst_follow += 1
            write_json(WEIGHTS_PATH, weights, atomic=True)

        suggestions = extract_suggestions(final_output)
        if suggestions:
            inst_follow += 1
            all_suggestions = add_suggestions(suggestions)
            write_json(PLACEMENT_PATH, all_suggestions, atomic=True)

        # print(f"Final Output: {repr(final_output)}")

        end = time.time()

        mem, power = print_gpu_utilization()
        lmpf_entry = {
            "latency": end - start,
            "memory": mem,
            "power": power,
            "inst_follow": inst_follow,
        }
        lmpf_l.append(lmpf_entry)
        print(f"[Step {step}/{total}]: {lmpf_entry}")

    avg = lambda key: (sum(entry[key] for entry in lmpf_l) / len(lmpf_l)) if lmpf_l else 0

    avg_latency = avg("latency")
    avg_memory = avg("memory")
    avg_power = avg("power")
    avg_inst_follow = avg("inst_follow") / 2

    print(f"Average Latency: {avg_latency:.2f} seconds")
    print(f"Average Memory: {avg_memory:.2f} MB")
    print(f"Average Power: {avg_power:.2f} W")
    print(f"Average Instruction Follow: {100*avg_inst_follow:.2f}%")

    lmpf = {
        "summary": "average",
        "model_name_or_path": model_name_or_path,
        "ragger": RAGGER if use_rag else "None",
        "latency": avg_latency,
        "memory": avg_memory,
        "power": avg_power,
        "inst_follow": avg_inst_follow,
    }

    # write lmpf to a json file
    write_json(f"{DIR}/{run_name}-{key}.json", lmpf)
    print(f"Logged to {DIR}/{run_name}-{key}.json")


    # check if average reward is below threshold twice in a row
    below_thresold = read_json(f"{DIR}/{run_name}-{key}-reward_threshold.json", create=True)
    if below_thresold.get("below_threshold", True):
        write_json(f"{DIR}/{run_name}-{key}-reward_threshold.json", {"below_threshold": False})
        
        # TODO: fine tune the model
        llm.submit_tuning_job(data=context) # TODO: use all the prev data not just this context
        print("Submitted fine-tuning job.")
        
        
        
        # and then replace the LLM instance with the fine-tuned model  
        pass

if __name__ == "__main__":
    main()
