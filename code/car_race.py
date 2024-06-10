import gymnasium as gym
from environment import Environment
from params import *
import car_agent as agent


# Initialize Environment
render_mode = 'rgb_array_list' if RECORD_VIDEO else RENDER_MODE
env = Environment(gym.make("CarRacing-v2", render_mode=render_mode, continuous=False))
env.env.metadata['render_fps'] = RENDER_FPS
print(f'\n~~~~~ CAR RACING USING {DEVICE} ~~~~~')

# Initialize Agent
agt = agent.CarA2CAgent(n_actions=env.env.action_space.n, dropout_rate=0.1)
#agt.load_model("./models/0607_1520_CarDQNAgent")
print(f'Agent details : {LOSS} loss, {NETWORK_REFRESH_STRATEGY} net refresh, {OPTIMIZER} optimizer')
print(f'Agent : exploration {agt.exploration}, training {agt.training}, multiframe {MULTIFRAME}\n')

# If run_episode is called, check the value of the variable MULTIFRAME, has to be set to 1 to work
try:
    save_model = True
    for _ in range(NUM_EPISODES):
        if MULTIFRAME == 1:
            env.run_episode(agt)
        else:
            env.run_episode_memory(agt)

        env.recording(agt)
        agt.end_episode()

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
