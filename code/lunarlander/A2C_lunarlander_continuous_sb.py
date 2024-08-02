from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("LunarLander-v2", n_envs=4, env_kwargs={'continuous':True})

model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log='./a2c_ll_tensorboard/')
model.learn(total_timesteps=300000)
model.save("stable_baselines_lunar_lander")

del model # remove to demonstrate saving and loading

model = A2C.load("stable_baselines_lunar_lander")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")