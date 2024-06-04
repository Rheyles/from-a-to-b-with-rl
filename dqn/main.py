import gymnasium as gym
from environment import Environment
from params import NUM_EPISODES, RENDER_FPS
import agent

env = Environment(gym.make("FrozenLake-v1", render_mode='human', is_slippery=False))
env.env.metadata['render_fps'] = RENDER_FPS
agt = agent.DQNAgent(1, env.env.action_space.n)

episode_durations = []
for _ in range(NUM_EPISODES):
    agt.steps_done = sum(episode_durations)
    episode_durations.append(env.run_episode(agt))


print(f"Average episode duration: {sum(episode_durations) / len(episode_durations)}")

user_input = input('Do you want to save the model & params ? Y/n')
if user_input.lower() == "y":
    agt.save_model()

env.env.close()
