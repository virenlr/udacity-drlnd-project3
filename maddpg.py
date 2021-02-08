import numpy as np
from ddpg_agent import Agent as DDPGAgent
import torch
import random
from collections import namedtuple, deque

device = 'cpu'

BUFFER_SIZE = int(1e5)              # replay buffer size
BATCH_SIZE = 256                    # minibatch size

class MADDPG:
    def __init__(self, state_size, action_size, seed):
        super(MADDPG, self).__init__()
        
        self.maddpg_agents = [DDPGAgent(state_size, action_size, 1*seed),
                              DDPGAgent(state_size, action_size, 2*seed)]
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, time_step, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience / reward for the agents
        self.memory.add(states[0], actions[0], rewards[0], next_states[0], dones[0])
        self.memory.add(states[1], actions[1], rewards[1], next_states[1], dones[1])

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            
            for agent in self.maddpg_agents:
                experiences = self.memory.sample()
                agent.step(experiences)
                    
    def act(self, states):
        """For each agent, return the action to take"""
        
        actions = np.zeros([2, 2]) # 2 agents and action consists of 2 values
        
        actions[0, :] = self.maddpg_agents[0].act(states[0])
        actions[1, :] = self.maddpg_agents[1].act(states[1])
        
        return actions
        
    def reset(self):        
        for agent in self.maddpg_agents:
            agent.reset()
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)      

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)