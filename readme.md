# Project Setup and Usage
This repo is intended to implement the experiments of the article titled, **Collaborative LLM-Edge-SLM Enhanced Multi-Agent DRL for On-Demand Microservice Deployment**.

1. **Environment Setup**  
   Create a new conda environment with Python 3.12. Then, install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

2. **Running the Agent**  
   The main entry point of the project is `train_agent.py`. You can run it with:
   ```
   python train_agent.py
   ```
   For more options regarding logging and parameters, use:
   ```
   python train_agent.py -h
   ```

3. **Requests File**  
   The trainer expects requests to be present in the `requests.json` file. If no requests are available, the model will remain idle until requests are provided.

4. **Generating Requests**  
   To generate service requests, use the `producer.py` script:
   ```
   python producer.py
   ```
   This will create a `requests.json` file containing the requests to be processed by the agent.

5. **Model Training and Saving**  
   Once requests are available, the DRL agent will begin training and save the resulting model in the `models` folder.

> This readme.md will be updated with more details as the project progresses.
