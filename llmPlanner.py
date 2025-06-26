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

CONTEXT_CAPACITY = 5
device_map = {"": 0}
now = datetime.now().strftime("%m%d_%H%M%S")
model_name_or_path = "Qwen/Qwen2-7B-Instruct"  # "HuggingFaceTB/SmolLM2-360M-Instruct"

run_name = sys.argv[1] if len(sys.argv) > 1 else "default_run"
TELEMETRY_PATH = f"logs/context-{run_name}.json"
SYS_INST_PATH = "planner/llm_planner_sys_instruction.md"
WEIGHTS_PATH = f"planner/weights.json"
EVERY = 60 # seconds


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
        data = json.loads(cleaned_str.group(1))
        weights = data.get("weights")
        if not weights or "latency" not in weights or "load_balance" not in weights:
            print("Missing required keys in weights.")
            return None
        return weights
    except json.JSONDecodeError as e:
        print("Invalid JSON format.")
        return None


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
        if prev_len != len(data):
            prev_len = len(data)

            if len(data) > CONTEXT_CAPACITY:
                data = random.sample(data, CONTEXT_CAPACITY)

            context = [{"role": "system", "content": sys_instruction}, {"role": "user", "content": json.dumps(data)}]
            inputs = llm.process_input(context)

            # Inference
            gen_kwargs = {"temperature": 0.2, "top_p": 0.9, "do_sample": True}
            final_output = llm.generate(inputs, max_new_tokens=200, **gen_kwargs)

            weights = extract_weights(final_output)
            if weights:
                write_json(WEIGHTS_PATH, weights)

            print(f"Final Output: {repr(final_output)}")

            end = time.time()
            sleep_time = max(0, EVERY - (end - start))
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
