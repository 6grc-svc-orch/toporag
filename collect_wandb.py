import wandb
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

api = wandb.Api()
run_id = "b9opf3p7"
run = api.run(f"my-ku-org/placemeent/{run_id}")

# Collect in chunks
step_size = 1_000
max_step = 1_896_470
chunks = []

for start in tqdm(range(0, max_step, step_size)):
    end = min(start + step_size, max_step)

    r = run.scan_history(min_step=start, max_step=end)
    df = pd.DataFrame(r)
    chunks.append(df)

# Concatenate all chunks
full_df = pd.concat(chunks, ignore_index=True)

# Save to disk
full_df.to_pickle(f"{run_id}-log.pkl")

# Later load:
# df = pd.read_pickle(f"{run_id}-log.pkl")
# print(df.shape)