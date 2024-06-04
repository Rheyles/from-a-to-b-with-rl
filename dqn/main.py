import gymnasium as gym
from environment import Environment
from params import NUM_EPISODES
import agent

env = Environment(gym.make("FrozenLake-v1", render_mode='human', is_slippery=False))
agt = agent.DQNAgent(env.n_observations, env.n_actions)

episode_durations = []
for _ in range(NUM_EPISODES):
    episode_durations.extend(env.run_env(agt))

print(f"Average episode duration: {sum(episode_durations) / len(episode_durations)}")

env.close()
