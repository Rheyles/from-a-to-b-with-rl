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
    def select_action()-> torch.Tensor:
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

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEM_SIZE)
        self.steps_done = 0
        self.losses = []
        self.rewards = 0

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
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            exp(-self.steps_done / EPS_DECAY)
        self.steps_done+=1 #Update the number of steps within one episode
        if sample > eps_threshold:
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

        future_possible_state_values = torch.zeros((BATCH_SIZE,4), dtype=torch.float32)
        best_action = torch.zeros(BATCH_SIZE, dtype=int)
        best_action_values = torch.zeros(BATCH_SIZE, device=DEVICE)

        with torch.no_grad():
            result = self.target_net(non_final_next_states.unsqueeze(-1))
            rewards_tensor = torch.tile(reward_batch[non_final_mask], (4,1)).T
            future_possible_state_values[non_final_mask,:] = \
                (result * GAMMA) + rewards_tensor

            best_action[non_final_mask] = future_possible_state_values[non_final_mask].argmax(1)
            best_action_values[non_final_mask] = future_possible_state_values[non_final_mask].max(1).values

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, best_action_values.unsqueeze(1))
        self.losses.append(float(loss))
        #plot_loss()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        show_diagnostics = True
        if show_diagnostics:

            with torch.no_grad():
                # pass
                action_dict = {0:'←', 1:'↓', 2:'→', 3:'↑'}
                action_arrows = [action_dict[elem.item()] for elem in action_batch]
                best_act_arrs = [action_dict[elem.item()] for elem in best_action]

                states_str     = 'Current state ' + ' | '.join([f"{elem:5.0f}" for elem in state_batch])
                action_str     = 'Current action' + ' | '.join([f"    {elem}"   for elem in action_arrows])
                reward_str     = 'Reward (s,a)  ' + ' | '.join([f"{elem:5.0f}" for elem in reward_batch])
                next_state_str = 'Future state  ' + ' | '.join([f"{elem:5.0f}"  for elem in all_next_states])
                current_Q_str  = 'Current Q     ' + ' | '.join([f"{elem:+4.2f}" for elem in torch.squeeze(state_action_values)])
                Q_left_str     = 'Estimated Q ← ' + ' | '.join([f"{elem:+4.2f}" for elem in future_possible_state_values[:,0]])
                Q_down_str     = 'Estimated Q ↓ ' + ' | '.join([f"{elem:+4.2f}" for elem in future_possible_state_values[:,1]])
                Q_right_str    = 'Estimated Q → ' + ' | '.join([f"{elem:+4.2f}" for elem in future_possible_state_values[:,2]])
                Q_up_str       = 'Estimated Q ↑ ' + ' | '.join([f"{elem:+4.2f}" for elem in future_possible_state_values[:,3]])
                expected_Q_str = 'Best known Q  ' + ' | '.join([f"{elem:+4.2f}" for elem in best_action_values])
                best_act_str   = 'Best action   ' + ' | '.join([f"    {elem}"   for elem in best_act_arrs])

                print("\033[A"*16)
                print(f'\nStep {self.steps_done:4.0f}, loss {float(loss):5.2e}, total rewards {self.rewards.item():3.0f}')
                print('-'*len(states_str))
                print(states_str)
                print(action_str)
                print(reward_str)
                print(next_state_str)
                print(current_Q_str)
                print(Q_left_str)
                print(Q_down_str)
                print(Q_right_str)
                print(Q_up_str)
                print(expected_Q_str)
                print(best_act_str)
                print('-'*len(states_str))

        return self.losses

    def update_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)
        self.rewards += reward

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

    def save_model(self, folder='saved_model/') -> None:
        """SAVE_MODEL() : saves our Pytorch model and
        associated parameters. Program will create a folder [folder]/MMDD_HHmm/
        and puts the policy.model, target.model and params.json files in there.

        Args:
            folder (str, optional): _description_. Defaults to 'saved_model/'.
        """
        my_date = datetime.strftime(datetime.now(), "%m%d_%H%M")
        os.makedirs(folder + my_date, exist_ok=True)
        torch.save(self.policy_net.state_dict(), folder + my_date + '/policy.model')
        torch.save(self.target_net.state_dict(), folder + my_date + '/target.model')

        with open(folder + my_date + '/params.json', 'w') as my_file:
            import params as prm
            my_dict = prm.__dict__
            my_dict = {key: val for key, val in my_dict.items()
                       if ('__' not in key)
                       and key != 'torch'
                       and key != 'DEVICE'}
            json.dump(my_dict, my_file)

    def load_model(self, folder: str) -> None:
        """LOAD_MODEL() : Loads an existing model
        based on a folder into a model. Will also
        print the hyperparameters of the model when
        it was instantiated.

        Args:
            folder (str): The folder you want to load from. Needs
            to have a policy.model, target.model and params.json file.
        """
        self.policy_net.load_state_dict(torch.load(folder + '/policy.model'))
        self.target_net.load_state_dict(torch.load(folder + '/target.model'))
        with open(folder + '/params.json') as my_file:
            hyper_dict = json.load(my_file)
            print(f'\nHyperparameters for model in {folder}')
            print(''.join([f"- {key:13s} : {val} \n"
                   for key, val in hyper_dict.items()]))

if __name__ == '__main__':
    my_agent = DQNAgent(16, 4)
    my_agent.save_model()
    my_agent.load_model('saved_model/0604_1538')
