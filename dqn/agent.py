import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import random
import typing

import gymnasium as gym
from datetime import datetime
import json
import os

from params import *
from network import LinearDQN, ConvDQN
from buffer import ReplayMemory, Transition
from display import Plotter, dqn_diagnostics, plot_success_rate

class SuperAgent():
    def __init__(self, exploration=True, training=True) -> None:
        self.exploration = exploration
        self.training = training
        self.steps_done = 0
        self.episode = 0
        self.losses = [0]
        self.rewards = [0]
        self.episode_rewards = [0]
        self.episode_duration = [0]

    def select_action()-> torch.Tensor:
        pass
    def optimize_model() -> list:
        pass
    def save_model():
        pass
    def load_model():
        pass


class FrozenDQNAgentBase(SuperAgent):

    def __init__(self, x_dim:int, y_dim:int, show_diagnostics=False, **kwargs) -> None:
        """

        Args:
            x_dim (int): Size of model input
            y_dim (int): Size of model output
            show_diagnostics (bool): Whether you want to show detailed info
            on the DQN network while it works. Defaults to False
        """
        super().__init__(kwargs.get('exploration',True), kwargs.get('training',True))
        self.policy_net = LinearDQN(x_dim, y_dim)
        self.target_net = LinearDQN(x_dim, y_dim)

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEM_SIZE)
        self.show_diagnostics = show_diagnostics


    def select_action(self, act_space : torch.Tensor, state: torch.Tensor, _env_map: np.ndarray) -> torch.Tensor:
        """

        Agent selects one of four actions to take either as a prediction of the model or randomly:
        The chances of picking a random action are high in the beginning and decrease with number of iterations

        Args:
            act_space : Action space of environment
            state (gym.ObsType): Observation

        Returns:
            act ActType : Action that the agent performs
        """
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
                return result.max(0).indices.view(1, 1)
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
        all_next_states = [s.item() if s is not None else -1
                           for s in batch.next_state ]

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
        future_state_values = torch.zeros((BATCH_SIZE,4), dtype=torch.float32)
        rewards_tensor = torch.tile(reward_batch, (4,1)).T

        with torch.no_grad():
            future_state_values[non_final_mask,:] = self.target_net(non_final_next_states.unsqueeze(-1))

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

    def update_memory(self, state, action, next_state, reward, _env_map) -> None:
        self.memory.push(state, action, next_state, reward)
        self.rewards.append(reward[0].item())
        #print(self.rewards)
        #print(reward)

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

    def save_model(self, folder='models/') -> None:
        """
        Save the model (NN and hyperparameters).

        Args:
            folder (str, optional): Where to save the model. Defaults to 'models/'.
        """
        my_date = datetime.strftime(datetime.now(), "%m%d_%H%M")
        folder_name = folder + my_date + type(self).__name__
        os.makedirs(folder_name, exist_ok=True)
        torch.save(self.policy_net.state_dict(), folder_name + '/policy.model')
        torch.save(self.target_net.state_dict(), folder_name + '/target.model')

        with open(folder_name + '/params.json', 'w') as my_file:
            import params as prm
            my_dict = prm.__dict__
            my_dict = {key : val for key, val in my_dict.items()
                       if '__' not in key
                       and key != 'torch'
                       and key != 'DEVICE'}

            json.dump(my_dict, my_file)


    def load_model(self, folder : str) -> None:
        """
        Load a model (ex: agt.load_model("./models/0605_1015DQNAgentObs"))

        Args:
            folder (str): Folder to the model
        """
        self.policy_net.load_state_dict(torch.load(folder + '/policy.model'))
        self.target_net.load_state_dict(torch.load(folder + '/target.model'))
        with open(folder + '/params.json') as my_file:
            hyper_dict = json.load(my_file)
            print(''.join([f"- {key} : {val} \n" for key, val in hyper_dict.items()]))



class CarDQNAgent(FrozenDQNAgentBase):

    def __init__(self, x_dim: int, y_dim: int, show_diagnostics=False, **kwargs) -> None:
        super().__init__(x_dim, y_dim, show_diagnostics, **kwargs)
        self.policy_net = ConvDQN(y_dim, dropout_rate=kwargs.get('dropout_rate',0.0))
        self.target_net = ConvDQN(y_dim, dropout_rate=kwargs.get('dropout_rate',0.0))

    def select_action(self, act_space : torch.Tensor, state: torch.Tensor, _env_map: np.ndarray) -> torch.Tensor:
        """

        Agent selects one of four actions to take either as a prediction of the model or randomly:
        The chances of picking a random action are high in the beginning and decrease with number of iterations

        Args:
            act_space : Action space of environment
            state (gym.ObsType): Observation

        Returns:
            act ActType : Action that the agent performs
        """
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

                state = torch.moveaxis(state,-1,1)

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

        state_batch = torch.moveaxis(state_batch, -1, 1)
        non_final_next_states = torch.moveaxis(non_final_next_states, -1, 1)

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




class FrozenDQNAgentObs(FrozenDQNAgentBase):

    def prepare_observation(self, state: torch.Tensor, env_map: np.ndarray) -> torch.Tensor:
        """
        !!! SPECIFIC TO FROZEN LAKE !!!
        Args:
            state (torch.Tensor): number of the tile where the agent stands
            env_map (np.ndarray): map of the environment

        Returns:
            np.ndarray: Array of the encoded close tiles
        """
        TILE_VALUE_ENCODING = {'F':1,
                               'S':1,
                               'H':-1,
                               'G':2,
                               '':0}

        state = int(state[0].item())
        result = np.zeros((4,1))
        map_x, map_y = env_map.shape

        if state - map_x >=0 :
            tile = env_map[(state-map_x)//map_x][(state-map_x)%map_x]
        else :
            tile = b''
        result[0] = TILE_VALUE_ENCODING[tile.decode('UTF-8')]

        if state + map_x < map_x * map_y :
            tile = env_map[(state+map_x)//map_x][(state+map_x)%map_x]
        else :
            tile = b''
        result[1] = TILE_VALUE_ENCODING[tile.decode('UTF-8')]

        if (state+1)//map_x == state//map_x :
            tile = env_map[(state+1)//map_x][(state+1)%map_x]
        else :
            tile = b''
        result[2] = TILE_VALUE_ENCODING[tile.decode('UTF-8')]

        if (state-1)//map_x == state//map_x :
            tile = env_map[(state-1)//map_x][(state-1)%map_x]
        else :
            tile = b''
        result[3] = TILE_VALUE_ENCODING[tile.decode('UTF-8')]

        return torch.rot90(torch.Tensor(result))


    def select_action(self, act_space : torch.Tensor, state: torch.Tensor, env_map: np.ndarray) -> torch.Tensor:
        """

        Agent selects one of four actions to take either as a prediction of the model or randomly:
        The chances of picking a random action are high in the beginning and decrease with number of iterations

        Args:
            act_space : Action space of environment
            state (gym.ObsType): Observation

        Returns:
            act ActType : Action that the agent performs
        """
        observation = self.prepare_observation(state, env_map)
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

                result = self.policy_net(observation)
                return result.max(1).indices.view(1, 1) # avant c'était max(0)
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

        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

        with torch.no_grad():
            result = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = result.max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # print(f"Loss: {float(loss)}")
        # print(f'Reward in batch:  {reward_batch.sum()}')

        self.losses.append(float(loss))

        #Plotting
        if self.steps_done % DISPLAY_EVERY == 0:
            Plotter().plot_data_gradually('Loss', self.losses)
            # print(type(self.losses))
            plot_success_rate(self.episode_rewards)



        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return self.losses

    def update_memory(self, state, action, next_state, reward, env_map):
        observation = self.prepare_observation(state, env_map)
        if next_state is None:
            next_observation = None
        else:
            next_observation = self.prepare_observation(next_state, env_map)
        self.memory.push(observation, action, next_observation, reward)
        self.rewards.append( self.rewards[-1] + reward[0].item() )

        #Plotting
        if self.steps_done % DISPLAY_EVERY == 0:
            Plotter().plot_data_gradually('Rewards', self.rewards)

        #print(self.rewards)
        #print(reward)


if __name__ == '__main__':
    agt = DQNAgentBase(16, 8)
    # agt.save_model()
    agt.load_model('models/0604_1729')
