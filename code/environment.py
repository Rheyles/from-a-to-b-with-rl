import gymnasium as gym
from itertools import count
import torch
from params import DEVICE, MULTIFRAME, NETWORK_REFRESH_STRATEGY, SAVE_EVERY
from gymnasium.utils.save_video import save_video # type: ignore
from super_agent import DQNAgent

class Environment():
    def __init__(self, env_gym: gym.Env) -> None:
        self.env = env_gym
        self.type = env_gym.unwrapped.spec.id
        self.skip_steps = 50 if 'CarRacing' in self.type else 0 

    def run_episode(self, agent:DQNAgent) -> int:
        """
        Runs a single episode of the environment using the provided agent.
        Store transition in memory and move to the next state.
        Performance optimization and update target.

        CHECK VALUE OF MULTIFRAME global variable. Has to be set to 1 for this method to be called correctly

        Args:
            agent (_type_): Component that makes the decision of what action to take
        """
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        for t in count():
            action = agent.select_action(self.env.action_space, state) # , self.env.get_wrapper_attr('desc')
            observation, reward, terminated, truncated, _ = self.env.step(action.item())
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            reward = torch.tensor([reward], device=DEVICE)
            not_done = torch.tensor([(not terminated) and (not truncated)], device=DEVICE)

            # Store the transition in memory
            reset = agent.update_memory(state, action, next_state, reward, not_done, skip_steps=self.skip_steps) # NOTE : Skip_steps should be 50 for CAR_RACE, 0 for FROZEN_LAKE
            agent.logging()

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            # Update of the target network's weights
            agent.update_agent(strategy=NETWORK_REFRESH_STRATEGY)

            if terminated or truncated or reset: break

        return t

    def run_episode_memory(self, agent:DQNAgent) -> int:
        """
        Runs a single episode of the environment using the provided agent. With a multiframed input
        Store transition in memory and move to the next state.
        Performance optimization and update target.

        Args:
            agent (_type_): Component that makes the decision of what action to take
        """
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        batch = []
        for _ in range(MULTIFRAME): #Creates a batch of mutiple frames
            batch.append(state)


        for t in count():

            state = torch.cat(batch, -1).to(DEVICE) #Transforms batch into right format
            action = agent.select_action(self.env.action_space, state) # , self.env.get_wrapper_attr('desc')
            observation, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device=DEVICE)

            not_done = torch.tensor([(not terminated) and (not truncated)], device=DEVICE)
            batch.append(torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0))
            batch.pop(0)
            next_state = torch.cat(batch,-1)

            # Store the transition in memory
            reset = agent.update_memory(state, action, next_state, reward, not_done) # , self.env.get_wrapper_attr('desc')
            agent.logging()

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            # Update of target net weights (hard or soft)
            agent.update_agent(strategy=NETWORK_REFRESH_STRATEGY)

            if terminated or truncated or reset: break

        return t

    def recording(self, agent):
        # If we don't train, it's nice to have a video !
        save_video(
            frames = self.env.render(),
            video_folder=agent.folder,
            episode_trigger = lambda x: x % SAVE_EVERY == 0,
            fps=30,
            name_prefix='recording',
            step_starting_index=0,
            episode_index=agent.episode,
        )
