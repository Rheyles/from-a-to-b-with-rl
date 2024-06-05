import gymnasium as gym
from environment import Environment
from params import NUM_EPISODES, RENDER_FPS
import agent

env = Environment(gym.make("FrozenLake-v1", render_mode='human', is_slippery=False))
env.env.metadata['render_fps'] = RENDER_FPS
agt = agent.DQNAgentBase(1, env.env.action_space.n)

for _ in range(NUM_EPISODES):
    agt.episode_duration.append(env.run_episode(agt))
    agt.episode_rewards.append(agt.rewards[-1])
    agt.episode += 1

print(f"Average episode duration: {sum(agt.episode_duration) / len(agt.episode_duration) }")
input('Press any key to close')
env.env.close()
