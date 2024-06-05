import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from environment import Environment
from params import NUM_EPISODES, RENDER_FPS
import agent, os

import torch.distributed as dist

# dist.init_process_group(backend='gloo')

# Initialize Environment
env = Environment(gym.make("CarRacing-v2",render_mode='human', continuous=False))
env.env.metadata['render_fps'] = RENDER_FPS

# Initialize Agent
agt = agent.CarDQNAgent(1, env.env.action_space.n, dropout_rate=0.1)
# agt.load_model("./models/0605_1015DQNAgentObs")


for _ in range(NUM_EPISODES):
    agt.episode_duration.append(env.run_episode(agt))
    agt.episode_rewards.append(agt.rewards[-1])
    agt.episode += 1

save_model = input("Save model ? [y/N]")
if save_model=="y":
    agt.save_model()
    print("Model saved !")

print(f"Average episode duration: {sum(agt.episode_duration) / len(agt.episode_duration) }")
input('Press any key to close')
env.env.close()
