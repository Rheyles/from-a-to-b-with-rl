from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from rl_zoo3.train import train


# Parallel environments
vec_env = make_vec_env("LunarLander-v2", n_envs=4, env_kwargs={'continuous':True})

model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log='./a2c_ll_tensorboard/')
train()
model.save("stable_baselines_lunar_lander")

del model # remove to demonstrate saving and loading

model = A2C.load("stable_baselines_lunar_lander")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")


## HYperparameters from the RL Zoo

"""
# Tuned
LunarLanderContinuous-v2:
  normalize: true
  n_envs: 4
  n_timesteps: !!float 5e6
  policy: 'MlpPolicy'
  ent_coef: 0.0
  max_grad_norm: 0.5
  n_steps: 8
  gae_lambda: 0.9 # Generalised advantage Estimation : 0 is regular A2C, otherwise we have a geometrical sum of the future A's 
  vf_coef: 0.4 # Value function coefficient : proportion between actor and critic loss when both are computed together to optimize the models
  gamma: 0.99 # Reward discount rate
  use_rms_prop: True # 
  normalize_advantage: False
  learning_rate: lin_7e-4
  use_sde: True
  policy_kwargs: "dict(log_std_init=-2, ortho_init=False)"
"""