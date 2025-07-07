# Project Setup and Usage
This repo is intended to implement the experiments of the article titled, **Collaborative LLM-Edge-SLM Enhanced Multi-Agent DRL for On-Demand Microservice Deployment**.

1. **Environment Setup**  
   Create a new conda environment with Python 3.12. Then, install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

2. **Running the Agent**  
   The main entry point of the project for multi agent training is `train_agentMulti.py`. You can run it with:
   ```
   python train_agentMulti.py
   ```
   For more options regarding logging and parameters, use:
   ```
   python train_agentMulti.py -h
   ```

   Copy the `run_name` from the command line output to use it for the next step.

3. **LLM Planning**
   The LLM planning is done in the `llmPlanner.py` file. It generates `weights` for the reward shaping and global placement `plans`. You can run it with:
   ```
   python llmPlanner.py <run_name>
   ```
   Replace `<run_name>` with the name you used in the previous step.  

5. **Model Training and Saving**  
   Once requests are available, the DRL agent will begin training and save the resulting model in the `models` folder.

> This readme.md will be updated with more details as the project progresses.
> Checkout the [envReadme.md](env/envReadme.md) for how the Environment is built.
