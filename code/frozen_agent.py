import torch
import torch.nn as nn

import numpy as np
import random

from params import *
from datetime import datetime
from super_agent import DQNAgent
from network import LinearDQN
from buffer import Transition
from display import Plotter, dqn_diagnostics

class FrozenDQNAgentBase(DQNAgent):

    def __init__(self, y_dim:int, **kwargs) -> None:
        """

        Args:
            x_dim (int): Size of model input
            y_dim (int): Size of model output
            show_diagnostics (bool): Whether you want to show detailed info
            on the DQN network while it works. Defaults to False
        """
        x_dim = 1
        self.policy_net = LinearDQN(x_dim, y_dim).to(DEVICE)
        self.target_net = LinearDQN(x_dim, y_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        super().__init__(**kwargs)

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

        sample = random.random()
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-self.steps_done / EPS_DECAY)
        self.episode_duration[-1]+=1 #Update the duration of the current episode
        self.steps_done+=1 #Update the number of steps within one episode
        self.time = (datetime.now() - self.creation_time).total_seconds()

        if sample > self.epsilon or not self.exploration:
            with torch.no_grad():
                # torch.no_grad() used when inference on the model is done
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                result = self.policy_net(state)
                action = result.max(0).indices.view(1, 1)
        else:
            action = torch.tensor([[act_space.sample()]], device = DEVICE, dtype=torch.long)

        self.last_action = action
        return action

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
        future_state_values = torch.zeros((BATCH_SIZE,4), dtype=torch.float32, device = DEVICE)
        rewards_tensor = torch.tile(reward_batch, (4,1)).T.to(DEVICE)

        with torch.no_grad():
            future_state_values[non_final_mask,:] = self.target_net(non_final_next_states.unsqueeze(-1))

        future_state_values = (future_state_values * GAMMA) + rewards_tensor
        best_action_values = future_state_values.max(1).values

        # Compute Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, best_action_values.unsqueeze(1))
        self.losses.append(float(loss))

         #Plotting
        if self.steps_done % DISPLAY_EVERY == 0:
            Plotter().plot_data_gradually('Loss', self.losses)
            Plotter().plot_data_gradually('Rewards',
                                          self.episode_rewards,
                                          rolling=10,
                                          per_episode=True)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Fancy print
        rwd_ep = self.episode_rewards[-1]
        lr = self.scheduler.optimizer.param_groups[0]['lr']
        act = self.last_action.item()

        print(f" 🎄  🎄  || {'t':7s} | {'Step':7s} | {'Episode':14s} | {'Loss':8s} |" \
            + f" {'ε':7s} | {'η':8s} | {'Rwd/ep':7s} | {'Action'}")
        print(f" 🎄  🎄  || " \
            + f'{self.time:7.1f} | {self.steps_done:7.0f} | ' \
            + f'{self.episode:7.0f} / {NUM_EPISODES:4.0f} | ' \
            + f'{self.losses[-1]:.2e} | {self.epsilon:7.4f} |'\
            + f' {lr:.2e} | {rwd_ep:7.2f} | {act:7.0f}')

        print("\033[F"*2, end='')

        return self.losses

class FrozenDQNAgentObs(FrozenDQNAgentBase):

    def __init__(self, y_dim:int, **kwargs) -> None:
        """

        Args:
            x_dim (int): Size of model input
            y_dim (int): Size of model output
            show_diagnostics (bool): Whether you want to show detailed info
            on the DQN network while it works. Defaults to False
        """
        x_dim = 4
        self.policy_net = LinearDQN(x_dim, y_dim).to(DEVICE)
        self.target_net = LinearDQN(x_dim, y_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        super().__init__(**kwargs)

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
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-self.steps_done / EPS_DECAY)
        self.steps_done+=1 #Update the number of steps within one episode
        self.episode_duration[-1]+=1 #Update the duration of the current episode

        if sample > self.epsilon or not self.exploration:
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
            Plotter().plot_data_gradually('Rewards',
                                          self.episode_rewards,
                                          rolling=30,
                                          per_episode=True)

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

        #print(self.rewards)
        #print(reward)
