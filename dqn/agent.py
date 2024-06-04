import torch
import torch.optim as optim
import torch.nn as nn

from numpy import exp
import random
import typing

import gymnasium as gym
from datetime import datetime
import json
import os

from params import *
from network import DQN
from buffer import ReplayMemory, Transition

class SuperAgent():
    def __init__(self) -> None:
        pass
    def select_action()-> gym.ActType:
        pass
    def optimize_model() -> list:
        pass

class DQNAgent():

    def __init__(self, x_dim:int, y_dim:int) -> None:
        """

        Args:
            x_dim (int): Size of model input
            y_dim (int): Size of model output
        """
        self.policy_net = DQN(x_dim, y_dim)
        self.target_net = DQN(x_dim, y_dim)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.memory = ReplayMemory(10000)
        self.loss = []
        self.rewards = []

    def select_action(self, act_space, state: gym.ObsType) -> gym.ActType:
        """

        Agent selects one of four actions to take either as a prediction of the model or randomly:
        The chances of picking a random action are high in the beginning and decrease with number of iterations

        Args:
            act_space : XXXXXX
            state (gym.ObsType): Observation

        Returns:
            act ActType : Action that the agent performs
        """

        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            exp(-steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # torch.no_grad() used when inference on the model is done
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                result = self.policy_net(state)
                return result.max(0).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=DEVICE, dtype=torch.long)


    def optimize_model(self):
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

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = self.policy_net(state_batch.unsqueeze(-1)).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.

        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

        with torch.no_grad():
            result = self.target_net(non_final_next_states.unsqueeze(-1))
            next_state_values[non_final_mask] = result.max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        print(f"Loss: {float(loss)}")
        print(f'Reward in batch:  {reward_batch.sum()}')

        self.losses.append(float(loss))
        #plot_loss()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return self.losses

    def update_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def soft_update_agent(self):
        """
        Performs a soft update of the agent's networks: i.e. updates the weights of the target net according to the changes
        observed in the policy net. In initial code, updated at every episode
        """
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def save_model(self, path='model/'):
        my_date = datetime.strftime(datetime.now(), "%m%d_%h%")
        os.makedirs(path + my_date, exist_ok=True)
        torch.save(self.policy_net.state_dict(), 'models/' + my_date + '/policy.model')
        torch.save(self.target_net.state_dict(), 'models/' + my_date + '/target.model')

        with open(...) as my_file:
            json.dump(my_dict, my_file)


    def load_model(self, folder):
        self.policy_net.load_state_dict('models/' + folder + '/policy.model')
        self.target_net.load_state_dict('models/' + folder + '/target.model')
        with open(...) as my_file:
            hyper_dict = json.load(my_file)
            print([f"{key} : {val} \n" for key, val in hyper_dict.items])
