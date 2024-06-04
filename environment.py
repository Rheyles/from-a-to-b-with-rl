import gymnasium as gym
import numpy as np
import os

def play_frozen_lake():
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False, render_mode='human')

    state, info = env.reset()
    env.render()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)

        state = new_state

    env.close()

if __name__ == "__main__":
    play_frozen_lake()

