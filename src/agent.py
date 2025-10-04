import numpy as np
import random
import gymnasium as gym

class MyAgent:
    def __init__(self, state_bins=(6, 12, 6, 12), action_size=2, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_bins = state_bins
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table
        self.q_table = np.zeros(state_bins + (action_size,))

    def discretize(self, obs, env_low, env_high):
        ratios = [(obs[i] - env_low[i]) / (env_high[i] - env_low[i]) for i in range(len(obs))]
        new_obs = [int(round((self.state_bins[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.state_bins[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        target = reward + self.gamma * np.max(self.q_table[next_state]) * (not done)
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
