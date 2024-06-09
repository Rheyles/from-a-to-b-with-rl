import os
import torch
import torch.nn as nn
from PIL import Image

import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from gymnasium.spaces.utils import flatdim

from params import *
from super_agent import DQNAgent
from network import *
from buffer import Transition
from display import Plotter

networks = {'ConvDQN_3layers_small':ConvDQN_3layers_small, 
            'ConvDQN_3layers_classic':ConvDQN_3layers_classic,
            'ConvDQN_2layers_small':ConvDQN_2layers_small,
            'ConvDQN_2layers_classic':ConvDQN_2layers_classic,
            'ConvA2C':ConvA2C
              }

class CarDQNAgent(DQNAgent):

    def __init__(self, y_dim: int, reward_threshold:float=20, reset_patience:int=250, **kwargs) -> None:
        ChosenNetwork = networks[NETWORK]
        self.policy_net = ChosenNetwork(y_dim, dropout_rate=DROPOUT_RATE).to(DEVICE)
        self.target_net = ChosenNetwork(y_dim, dropout_rate=DROPOUT_RATE).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        super().__init__(**kwargs)

        self.reward_threshold = reward_threshold
        self.max_reward = 0
        self.reset_patience = reset_patience
        self.batch = []

    def end_episode(self) -> None:
        """
        All the actions to proceed when an episode is over

        Args:
            episode_duration (int): length of the episode
        """
        self.episode_rewards.append(sum(self.rewards[-1 * self.episode_duration[-1]:]))
        self.episode_duration.append(0)
        self.episode += 1
        self.scheduler.step(metrics=self.episode_rewards[-1])
        if self.episode_rewards[-1]>=self.reward_threshold + self.max_reward:
            self.max_reward = self.episode_rewards[-1]
            self.save_model()

    def prepro(self, state: torch.Tensor, crop=True) -> torch.Tensor:

        if state is None:
            return None

        state = state[:,:,:,1::3] / 256
        if crop:
            crop_height = int(state.shape[1] * 0.88)
            crop_w = int(state.shape[2] * 0.07)
        state = state[:, :crop_height, crop_w:-crop_w, :]

        return state.moveaxis(-1, 1)


    def select_action(self, act_space : torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """

        Agent selects one of four actions to take either as a prediction of the model or randomly:
        The chances of picking a random action are high in the beginning and decrease with number of iterations

        Args:
            act_space : Action space of environment
            state (gym.ObsType): Observation

        Returns:
            act ActType : Action that the self performs
        """
        state = self.prepro(state)
        sample = random.random()
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-self.steps_done / EPS_DECAY)
        self.time = (datetime.now() - self.creation_time).total_seconds()
        self.steps_done+=1 #Update the number of steps within one episode
        self.episode_duration[-1]+=1 #Update the duration of the current episode

        if self.steps_done % IDLENESS == 0:
            if sample > self.epsilon or not self.exploration:
                with torch.no_grad():
                    # torch.no_grad() used when inference on the model is done
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.

                    result = self.policy_net(state)
                    action = result.max(1).indices.view(1, 1)
            else:
                # If action is selected at random, give a bit of extra weight
                # on hitting the gas
                action = np.random.choice(flatdim(act_space), p=[0.10, 0.2, 0.2, 0.30, 0.2])
                action = torch.tensor([[action]], device=DEVICE, dtype=torch.long)

            self.last_action = action
            return action
        else:
            return self.last_action

    def optimize_model(self) -> list:
        """

        This function runs the optimization of the model:
        it takes a batch from the buffer, creates the non final mask and computes:
        Q(s_t, a) and V(s_{t+1}) to compute the Hubber Loss, performs backprop and then clips gradient
        returns the computed loss

        Args:
            device (_type_, optional): Device to run computations on. Defaults to DEVICE.

        Returns:
            losses (list): Calculated Loss
        """
        if not self.training: # Si on ne s'entraîne pas, on ne s'entraîne pas
            return

        if len(self.memory) < BATCH_SIZE: return 0

        (state_batch, 
         action_batch, 
         next_state_batch, 
         reward_batch, 
         not_done_batch) = self.memory.sample(BATCH_SIZE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        rewards_tensor = torch.tile(reward_batch, (1,5)).to(DEVICE)

        with torch.no_grad():
            future_state_values = rewards_tensor + not_done_batch * \
                self.target_net(next_state_batch) * GAMMA
            best_action_values = future_state_values.max(1).values

        # Compute MSE loss
        loss = self.lossfun(state_action_values, best_action_values.unsqueeze(1))
        self.losses.append(float(loss))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        rwd_ep = self.episode_rewards[-1]
        lr = self.scheduler.optimizer.param_groups[0]['lr']
        act = self.last_action.item()

        print(f" 🏎️  🏎️  || {'t':7s} | {'Step':7s} | {'Episode':14s} | {'Loss':8s} |" \
            + f" {'ε':7s} | {'η':8s} | {'Rwd/ep':7s} | {'Action'}")
        print(f" 🏎️  🏎️  || " \
            + f'{self.time:7.1f} | {self.steps_done:7.0f} | ' \
            + f'{self.episode:7.0f} / {NUM_EPISODES:4.0f} | ' \
            + f'{self.losses[-1]:.2e} | {self.epsilon:7.4f} |'\
            + f' {lr:.2e} | {rwd_ep:7.2f} | {act:7.0f}')

        print("\033[F"*2, end='')

        return self.losses
