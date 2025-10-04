# **🚀 TMRL Project**





###### This repository implements a reinforcement learning (RL) pipeline using Tiny Machine Reinforcement Learning (TMRL).

###### It supports both local (venv) execution and containerized (Docker) environments to ensure reproducibility.

##### 

TMRL-Project/

│

├── src/                  # Core source code

│   ├── train.py          # Training loop

│   ├── evaluate.py       # Evaluation script

│   ├── agent.py          # Custom agent implementation (MyAgent)

│   ├── utils.py          # Helper functions

│   └── \_\_init\_\_.py

│

├── config/               # Configuration files

│   └── config.yaml

│

├── results/              # Training logs, graphs, checkpoints

│

├── requirements.txt      # Dependencies

├── Dockerfile            # Docker build file

├── README.md             # Documentation (this file)

└── setup.sh / setup.bat  # Setup script for Linux/Windows







# **⚙️ Installation**



##### Clone the repo:



git clone https://github.com/jmusadiq7/TMRL-Project..git

cd TMRL-Project



##### Create a virtual environment:





python -m venv venv

\# Linux/Mac

source venv/bin/activate

\# Windows

venv\\Scripts\\activate







##### ***Install requirements:***





pip install -r requirements.txt





#### ***🐳 Installation (Docker)***



###### ***Build the Docker image:***



*docker build -t tmrl-project .*



###### ***Run training inside Docker:***



*docker run --rm -v "%cd%:/app" tmrl-project python src/train.py*



###### ***Run evaluation inside Docker:***



*docker run --rm -v "%cd%:/app" tmrl-project python src/evaluate.py --model results/checkpoint.pth*



###### ***Export the image as an artifact (for shipping/deployment):***



*docker save tmrl-project:latest -o tmrl-project.tar*



###### ***Import elsewhere:***



*docker load -i tmrl-project.tar*





## **🎯 Phase 1: Training the Agent**



###### Run the training script:





python -m src.train





###### **or via Docker:**



docker run --rm -v "%cd%:/app" tmrl-project python src/train.py





This will:



 Initialize the environment



 Train the agent (MyAgent)



 Save logs and graphs to results/



    Print progress:



            Episode 5000, Average Reward (last 100): 164.12, Epsilon: 0.010





## **🧪 Phase 2: Evaluating the Agent**





###### *After training, evaluate the agent:*





python -m src.evaluate --model results/checkpoint.pth



###### **or via Docker:**



docker run --rm -v "%cd%:/app" tmrl-project python src/evaluate.py --model results/checkpoint.pth





This will:



  Load the saved model



  Run evaluation episodes



  Output average performance





## **📊 Results \& Graphs**





###### Training curves: results/training\_plot.png, results/logs.csv

###### 

###### Checkpoints: results/checkpoint.pth



## **🔧 Configurations**





###### Hyperparameters are in config/config.yaml:



*episodes: 10000*

*learning\_rate: 0.001*

*gamma: 0.99*

*epsilon\_start: 1.0*

*epsilon\_min: 0.01*

*epsilon\_decay: 0.995*





## **📦 Phase 3: Deployment**



##### Load and use the trained agent:



from agent import MyAgent

from utils import load\_model



agent = MyAgent()

agent = load\_model(agent, "results/checkpoint.pth")



obs = env.reset()

done = False

while not done:

    action = agent.act(obs)

    obs, reward, done, info = env.step(action)















               ----------------------------------------------------------------------------------


