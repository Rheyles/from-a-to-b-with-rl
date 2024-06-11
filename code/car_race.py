import gymnasium as gym
from environment import Environment
from params import *
import car_agent as agent


# Initialize Environment
render_mode = 'rgb_array_list' if RECORD_VIDEO else RENDER_MODE
env = Environment(gym.make("CarRacing-v2", render_mode=render_mode, continuous=False))
env.env.metadata['render_fps'] = RENDER_FPS
print(f'\n~~~~~ CAR RACING USING {DEVICE} ~~~~~')
print(f'Saving video : {RECORD_VIDEO}, saving models/video every {SAVE_EVERY}')

# Initialize Agent
agt = agent.CarA2CAgent(env.env.action_space.n, dropout_rate=DROPOUT_RATE, network=NETWORK)
# agt.load_model("./models/0610_0002_CarDQNAgent")
print(f'Agent : exploration {agt.exploration}, training {agt.training}, {MULTIFRAME} multiframe , {IDLENESS} idleness')
print(f'Optimizer : {OPTIMIZER} optimizer, {LOSS} loss, {REGULARIZATION} L2 regularization coeff.')
print(f'Network : {NETWORK}, {NETWORK_REFRESH_STRATEGY} net refresh , {DROPOUT_RATE} dropout rate\n')

# If run_episode is called, check the value of the variable MULTIFRAME, has to be set to 1 to work
try:
    save_model = True
    for ep in range(NUM_EPISODES):
        if MULTIFRAME == 1:
            env.run_episode(agt)
        else:
            env.run_episode_memory(agt)

        if ep % SAVE_EVERY == 0:
            if RECORD_VIDEO: env.recording(agt)
            agt.save_model()

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
