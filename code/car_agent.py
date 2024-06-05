import torch
import torch.nn as nn
from PIL import Image

import numpy as np
import random
import matplotlib.pyplot as plt

from params import *
from super_agent import DQNAgent
from network import ConvDQN
from buffer import Transition
from display import Plotter, dqn_diagnostics


class CarDQNAgent(DQNAgent):

    def __init__(self, y_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.policy_net = ConvDQN(y_dim, dropout_rate=kwargs.get('dropout_rate',0.0))
        self.target_net = ConvDQN(y_dim, dropout_rate=kwargs.get('dropout_rate',0.0))

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=LR)

    def prepro(self, state: torch.Tensor) -> torch.Tensor:

        #state = state.numpy()

        crop_height = int(state.shape[1] * 0.88)
        state = state[:, :crop_height, :, :]

        #state = Image.open(state).convert("L")
        #state = np.asarray(state)
        #plt.imshow(state, cmap='gray', vmin=0, vmax=255)
        #plt.show()

        r, g, b = state[:, :, :, 0], state[:, :, :, 1], state[:, :, :, 2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        gray = g // 64

        #plt.imshow(gray.squeeze(0), cmap='gray')
        #plt.show()
        #input('COntinue ?')

        return gray.unsqueeze(0)


    def select_action(self, act_space : torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """

        Agent selects one of four actions to take either as a prediction of the model or randomly:
        The chances of picking a random action are high in the beginning and decrease with number of iterations

        Args:
            act_space : Action space of environment
            state (gym.ObsType): Observation

        Returns:
            act ActType : Action that the agent performs
        """
        state = self.prepro(state)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-self.steps_done / EPS_DECAY)
        self.steps_done+=1 #Update the number of steps within one episode
        if sample > eps_threshold or not self.exploration:
            with torch.no_grad():
                # torch.no_grad() used when inference on the model is done
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                result = self.policy_net(state)
                return result.max(1).indices.view(1, 1)
        else:
            return torch.tensor([[act_space.sample()]], device = DEVICE, dtype=torch.long)


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

        transitions = self.memory.sample(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.

        batch = Transition(*zip(*transitions)) # Needs to pass this from buffer class

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        all_next_states = [s if s is not None else -1
                           for s in batch.next_state ]

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        future_state_values = torch.zeros((BATCH_SIZE,5), dtype=torch.float32)
        rewards_tensor = torch.tile(reward_batch, (5,1)).T

        with torch.no_grad():
            future_state_values[non_final_mask,:] = self.target_net(non_final_next_states)

        future_state_values = (future_state_values * GAMMA) + rewards_tensor
        best_action = future_state_values.argmax(1)
        best_action_values = future_state_values.max(1).values

        # print('\n\n')
        # print(torch.cat((state_batch.unsqueeze(1), action_batch, future_state_values, best_action_values.unsqueeze(1)), dim=1))
        # print('\n')

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, best_action_values.unsqueeze(1))
        self.losses.append(float(loss))

        #Plotting

        if self.steps_done % DISPLAY_EVERY == 0:
            Plotter().plot_data_gradually('Loss', self.losses)
            Plotter().plot_data_gradually('Rewards', self.rewards, cumulative=True)
            Plotter().plot_data_gradually('RewardRate', self.episode_rewards, rolling=30)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if self.show_diagnostics:
            dqn_diagnostics(self, action_batch, best_action,
                            state_batch, reward_batch, all_next_states,
                            state_action_values,future_state_values,
                            best_action_values)
        else:
            eps = EPS_END + (EPS_START - EPS_END) \
                * np.exp(- self.steps_done / EPS_DECAY)

            print(f'Step : {self.steps_done:5.0f} \t' \
                + f'episode {self.episode:4.0f} / {NUM_EPISODES:4.0f} \t'\
                + f'loss = {self.losses[-1]:.3e}, ε = {eps:7.4f}'
                  , end='\r')

        return self.losses

    def update_memory(self, state, action, next_state, reward) -> None:
        state = self.prepro(state)
        next_state = self.prepro(next_state)
        self.memory.push(state, action, next_state, reward)
        self.rewards.append(reward[0].item())
        #print(self.rewards)
        #print(reward)
