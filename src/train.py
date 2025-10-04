import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from .agent import MyAgent

def train(episodes=5000):
    env = gym.make("CartPole-v1")
    agent = MyAgent()

    rewards = []
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_high[1] = 5  # velocity clip
    env_high[3] = 5
    env_low[1] = -5
    env_low[3] = -5

    for ep in range(episodes):
        obs, _ = env.reset()
        state = agent.discretize(obs, env_low, env_high)
        total_reward = 0

        done = False
        while not done:
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.discretize(next_obs, env_low, env_high)

            agent.learn(state, action, reward, next_state, terminated or truncated)

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {ep+1}, Average Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()

    # Plot learning curve
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress on CartPole-v1")
    plt.savefig("reward_curve.png")
    plt.show()

if __name__ == "__main__":
    train()
