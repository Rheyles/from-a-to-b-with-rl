import gymnasium as gym
from environment import Environment
from params import NUM_EPISODES, RENDER_FPS, FROZEN_MAP_NAME
import frozen_agent as agent
from frozen_maps import maps
from params import *

# Initialize Environment
render = RENDER_MODE if not RECORD_VIDEO else 'rgb_array_list'
env = Environment(gym.make("FrozenLake-v1", desc=maps[FROZEN_MAP_NAME], render_mode=render, is_slippery=False))
env.env.metadata['render_fps'] = RENDER_FPS
env.env.spec.max_episode_steps = 250
env.env._max_episode_steps = 250
print(f'\n~~~~~ FROZEN LAKE USING {DEVICE} ~~~~~ Map is {FROZEN_MAP_NAME} ~~~~~~~~~ ')
print(f'Saving video : {RECORD_VIDEO}, saving models/video every {SAVE_EVERY}')

# Initialize Agent
agt = agent.FrozenDQNAgentBase(env.env.action_space.n)
# agt.load_model("./models/0608_0008_FrozenDQNAgentBase")
print(f'Agent {agt.__class__.__name__}: exploration {agt.exploration}, training {agt.training}')
print(f'Optimizer : {OPTIMIZER} optimizer, {LOSS} loss, {REGULARIZATION} L2 regularization coeff.')
print(f'Network : {agt.policy_net.__class__.__name__}, {NETWORK_REFRESH_STRATEGY} net refresh , {DROPOUT_RATE} dropout rate\n') 


try:
    save_model = True
    for _ in range(NUM_EPISODES):
        env.run_episode(agt)
        agt.end_episode()
        if RECORD_VIDEO: env.recording(agt)

except KeyboardInterrupt:
    print('\n\n\nInterrupted w. Keyboard !')
    save_model = input("Save model ? [y/N]").lower() == 'y'

finally:
    if save_model:
        agt.save_model(add_episode=True)
        print("\n\nModel saved !")

print(f"Average episode duration: {sum(agt.episode_duration) / len(agt.episode_duration) }")
input('Press any key to close')
env.env.close()
