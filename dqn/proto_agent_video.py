import gymnasium as gym # type: ignore
from gymnasium.utils.save_video import save_video

filming =''
while filming not in ['y','n']:
    filming = input("Voulez-vous enregistrer une vidéo de l'essai (écran invisible si oui) ? [y/n]")

if filming == 'y':
    render_mode = "rgb_array_list"
else:
    render_mode = "human"

env = gym.make("FrozenLake-v1", render_mode=render_mode, is_slippery=False)
observation, info = env.reset()

step_starting_index = 0
episode_index = 0

for step_index in range(25):
    print(f"Action space : {env.action_space}")
    #action = env.action_space.sample()  # agent policy that uses the observation and info
    action = int(input("Quelle action pour le lutin ?\n0.Left 1.Down 2.Right 3.Up\n"))
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"observation : {observation}")
    print(f"reward : {reward}")
    print(f"terminated : {terminated}")
    print(f"truncated : {truncated}")
    print(f"info : {info}")

    if terminated or truncated:
        if filming == 'y':
            save_video(
                env.render(),
                "videos",
                fps=env.metadata["render_fps"],
                step_starting_index=step_starting_index,
                episode_index=episode_index
            )
            step_starting_index = step_index + 1
            episode_index += 1
        observation, info = env.reset()

env.close()
