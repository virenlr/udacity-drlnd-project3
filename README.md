# udacity-drlnd-project3
This is the third project done as part of Udacity's Deep Reinforcement Learning Nanodegree.

## Project Details

This project uses the Tennis environment in Unity's ML toolkit.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The GIF below was taken directly from Udacity's project page.

![Project Demo](https://raw.githubusercontent.com/virenlr/udacity-drlnd-project3/main/Demo.gif "Project Demo")

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

#### Step 1: Clone my repository

You will need the Tennis.ipynb file, as well as the Python files titled `maddpg.py`, `ddpg_agent.py` and `model.py`. Please ensure these are all placed in the same folder.

#### Step 2: Setup the required dependencies

1. Create (and activate) a new environment with Python 3.6.

Linux or Mac:

```
conda create --name drlnd python=3.6
source activate drlnd
```

Windows:

```
conda create --name drlnd python=3.6 
activate drlnd
```

2. Perform a minimal installation of OpenAI Gym

```
pip install gym
```

3. Clone the repository from [Udacity's GitHub page](https://github.com/udacity/deep-reinforcement-learning), and navigate to the python/ folder. Then, install several dependencies.

```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an IPython kernel for the `drlnd` environment.

```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. You may have to install other dependencies with pip, like Unity ML and PyTorch. You will know when you start executing the cells in my notebook.

6. Before running the code in the notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.

#### Step 3: Download the environment

Download the Unity environment corresponding to your operating system with the links provided by Udacity.

Linux: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows 32-bit: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows 64-bit: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the same folder as the other files you downloaded from my repo and unzip (or decompress) it.

## Instructions

With the environment set up, you can simply open and execute the cells of the Jupyter notebook, one by one. Feel free to take a look at the code in `maddpg.py`, `ddpg_agent.py` and `model.py` as well.
