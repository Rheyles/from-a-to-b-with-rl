import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from environment import Environment
from params import NUM_EPISODES, RENDER_FPS
import agent, os

# Initialize Environment
env = Environment(gym.make("FrozenLake-v1",map_name="8x8", render_mode='human', is_slippery=False))
env.env.metadata['render_fps'] = RENDER_FPS

# Initialize Agent
agt = agent.DQNAgentObs(4, env.env.action_space.n, exploration=True, training=True)
agt.load_model("./models/0605_1015DQNAgentObs")


episode_durations = []
for _ in range(NUM_EPISODES):
    agt.steps_done = sum(episode_durations)
    episode_durations.append(env.run_episode(agt))

save_model = input("Save model ? [y/N]")
if save_model=="y":
    agt.save_model()
    print("Model saved !")

print(f"Average episode duration: {sum(episode_durations) / len(episode_durations)}")
input('Press any key to close')
env.env.close()
