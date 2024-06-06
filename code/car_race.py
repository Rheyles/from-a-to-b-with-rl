import gymnasium as gym
from environment import Environment
from params import NUM_EPISODES, RENDER_FPS, DEVICE
import car_agent as agent


# Initialize Environment
env = Environment(gym.make("CarRacing-v2",render_mode='rgb_array', continuous=False))
env.env.metadata['render_fps'] = RENDER_FPS
print('\n~~~~~CAR RACING USING {DEVICE} ~~~~~')

# Initialize Agent
agt = agent.CarDQNAgent(env.env.action_space.n, dropout_rate=0.1)
# agt.load_model("./models/0605_1015DQNAgentObs")

try:
    for _ in range(NUM_EPISODES):
        env.run_episode(agt)
        agt.end_episode()

except KeyboardInterrupt:
    print('Interrupted w. Keyboard !')
    save_model = input("Save model ? [y/N]")
    if save_model.lower() == "y":
        agt.save_model(add_episode=True)
        print("Model saved !")

print(f"Average episode duration: {sum(agt.episode_duration) / len(agt.episode_duration) }")
input('Press any key to close')
env.env.close()
