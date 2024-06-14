import gymnasium as gym

from stable_baselines3 import PPO
from gymnasium.wrappers.frame_stack import FrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.logger import configure

def run():
    vec_env = make_vec_env('CarRacing-v2', env_kwargs= {'continuous':True})
    new_logger = configure('./PPOmodels/', ["stdout", "csv", "tensorboard"])
    st_env = VecFrameStack(vec_env, n_stack=3)

    model = PPO(policy='CnnPolicy',
                env=st_env,
                learning_rate=0.001,
                n_steps=1024,
                batch_size=32,
                n_epochs=10,
                gamma=0.99)

    model.set_logger(new_logger)
    model.learn(total_timesteps=10000,
                tb_log_name='toto')

    model.save('PPOmodels/stable_baseline.model')


if __name__ == '__main__':
    run()
