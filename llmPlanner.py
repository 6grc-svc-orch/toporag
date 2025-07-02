# LLMPlanner.py


import re
import json
import sys
from typing import List, Dict
from utils import read_file, read_json, write_json
import random
import time
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedModel,
)

CONTEXT_CAPACITY = 100
device_map = {"": 0}
now = datetime.now().strftime("%m%d_%H%M%S")
model_name_or_path = "Qwen/Qwen2-7B-Instruct"  # "HuggingFaceTB/SmolLM2-360M-Instruct"

run_name = sys.argv[1] if len(sys.argv) > 1 else "default_run"
TELEMETRY_PATH = f"logs/context-{run_name}.json"
SYS_INST_PATH = "planner/llm_planner_inst.md"
WEIGHTS_PATH = f"planner/weights_.json"
PLACEMENT_PATH = "planner/placement.json"
EVERY = 10  # seconds

print(f"Using model: {model_name_or_path}")
print(f"Using telemetry path: {TELEMETRY_PATH}")


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


def extract_weights(json_str):
    # Remove any leading non-JSON content (e.g., the word 'json')
    cleaned_str = re.search(r"({.*})", json_str, re.DOTALL)
    if not cleaned_str:
        print("No JSON object found in the input string.")
        return None
    try:
        data: dict = json.loads(cleaned_str.group(1))
        weights = data.get("weights")
        if not weights or "latency" not in weights or "load_balance" not in weights:
            print("Missing required keys in weights.")
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
        return placement_suggestion
    except json.JSONDecodeError as e:
        print("Invalid JSON format.")
        return None
    
def add_suggestions(suggestions: List[Dict]):
    p_suggestions: List[Dict] = read_json(PLACEMENT_PATH)
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

    write_json(PLACEMENT_PATH, p_suggestions)


def main():
    start = time.time()
    llm = LLM(model_name_or_path=model_name_or_path, device_map=device_map)
    loaded = time.time()
    print(f"Model loaded in {loaded - start:.2f} seconds")

    sys_instruction = read_file(SYS_INST_PATH)

    prev_len = 0
    start = time.time()
    while True:
        data: List[Dict] = read_json(TELEMETRY_PATH)

        if len(data) > CONTEXT_CAPACITY:
            data = data[-CONTEXT_CAPACITY:]

        context = [{"role": "system", "content": sys_instruction}, {"role": "user", "content": json.dumps(data)}]
        inputs = llm.process_input(context)

        # Inference
        gen_kwargs = {"temperature": 0.2, "top_p": 0.9, "do_sample": True}
        final_output = llm.generate(inputs, **gen_kwargs)

        weights = extract_weights(final_output)
        if weights:
            write_json(WEIGHTS_PATH, weights)

        suggestions = extract_suggestions(final_output)
        if suggestions:
            add_suggestions(suggestions)

        print(f"Final Output: {repr(final_output)}")

        end = time.time()
        sleep = max(0, EVERY - (end - start))
        print(f"Sleeping for {sleep:.2f} seconds before next iteration.")
        time.sleep(sleep)
        start = end


if __name__ == "__main__":
    main()
