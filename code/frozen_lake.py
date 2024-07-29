import gymnasium as gym
from environment import Environment
from params import NUM_EPISODES, RENDER_FPS
import frozen_agent as agent
from params import *

# Initialize Environment
env = Environment(gym.make("FrozenLake-v1" , map_name='4x4',render_mode='rgb_array', is_slippery=False))
env.env.metadata['render_fps'] = RENDER_FPS
print(f'\n~~~~~ FROZEN LAKE USING {DEVICE} ~~~~~')


# Initialize Agent
agt = agent.FrozenDQNAgentBase(env.env.action_space.n, env_map = env.env.get_wrapper_attr('desc'))
# agt.load_model("./models/2024-06-10 193218.600447FrozenDQNAgentObs")
print(f'Agent details : {LOSS} loss, {NETWORK_REFRESH_STRATEGY} net refresh, {OPTIMIZER} optimizer.')
print(f'Agent explores {agt.exploration} | agent trains {agt.training}\n ')
save_model = True

try:
    for _ in range(NUM_EPISODES+1):
        env.run_episode(agt)
        agt.end_episode()

        if _ % 500 == 0:
            import matplotlib.pyplot as plt
            with torch.no_grad():
                fig, ax = plt.subplots()
                all_states = torch.arange(0,16, dtype=torch.float32).unsqueeze(-1)
                all_values = agt.policy_net.forward(all_states)
                ax.imshow(all_values.T, vmin=-0.1, vmax=1)
                
                plt.pause(1)
                plt.close(fig)


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
