import gymnasium as gym
import torch
from agent import MyAgent

# Load the trained agent
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = MyAgent(state_dim, action_dim)
agent.load("trained_model.pth")   # <-- model saved during training

episodes = 5
for ep in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Episode {ep+1}: Reward = {total_reward}")
