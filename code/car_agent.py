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
from super_agent import DQNAgent, SuperAgent
from network import ConvDQN, ConvA2CDiscrete, ConvA2CContinuousActor, ConvA2CContinuousCritic
from buffer import Transition, TransitionPPO, ReplayMemoryPPO
from display import Plotter, dqn_diagnostics


class CarDQNAgent(DQNAgent):

    def __init__(self, y_dim: int, reward_threshold:float=20, reset_patience:int=250, **kwargs) -> None:
        self.policy_net = ConvDQN(y_dim, dropout_rate=kwargs.get('dropout_rate',0.0)).to(DEVICE)
        self.target_net = ConvDQN(y_dim, dropout_rate=kwargs.get('dropout_rate',0.0)).to(DEVICE)
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

    def prepro(self, state: torch.Tensor) -> torch.Tensor:

        if state is None:
            return None

        crop_height = int(state.shape[1] * 0.88)
        crop_w = int(state.shape[2] * 0.07)
        state = state[:, :crop_height, crop_w:-crop_w, :]
        g = state[:, :, :, 1::3]
        gray = (g // 16) / 16
        gray = torch.moveaxis(gray, -1, 1)

        # print(gray.shape)
        # plt.imshow(gray[:,1,:,:], cmap='gray')
        # plt.show()
        # input('Continue ?')
        return gray


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
                action = np.random.choice(flatdim(act_space), p=[0.1, 0.2, 0.2, 0.3, 0.2])
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
                           for s in batch.next_state]

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
        future_state_values = torch.zeros((BATCH_SIZE,5), dtype=torch.float32, device = DEVICE)
        rewards_tensor = torch.tile(reward_batch, (5,1)).T.to(DEVICE)

        with torch.no_grad():
            future_state_values[non_final_mask,:] = self.target_net(non_final_next_states)

        future_state_values = (future_state_values * GAMMA) + rewards_tensor
        best_action_values = future_state_values.max(1).values

        # print('\n\n')
        # print(torch.cat((state_batch.unsqueeze(1), action_batch, future_state_values, best_action_values.unsqueeze(1)), dim=1))
        # print('\n')

        # Compute MSE loss
        loss = self.lossfun(state_action_values, best_action_values.unsqueeze(1))
        self.losses.append(float(loss))

        #Plotting
        if self.steps_done % DISPLAY_EVERY == 0:
            Plotter().plot_data_gradually('Loss', self.losses)
            # Plotter().plot_data_gradually('Rewards', self.rewards, cumulative=True, episode_durations=self.episode_duration)
            Plotter().plot_data_gradually('Reward per Episode',
                                          self.episode_rewards,
                                          per_episode=True)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
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

    def update_memory(self, state, action, next_state, reward) -> bool:

        self.rewards.append(reward[0].item())
        current_episode_rewards = sum(self.rewards[-self.episode_duration[-1]:])
        episode_is_done = current_episode_rewards < -8

        if action == 0:
            reward -= 0.2

        if self.episode_duration[-1] < 50: # On ne met pas en mémoire le zoom de début d'épisode
            return episode_is_done

        state = self.prepro(state)
        next_state = self.prepro(next_state)
        self.memory.push(state, action, next_state, reward)

        return episode_is_done


class CarA2CAgentDiscrete(SuperAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_actions = kwargs.get("n_actions", 5)
        self.net = ConvA2CDiscrete(self.n_actions)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=INI_LR)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (optimizer=self.optimizer,
             mode='max',
             factor=0.5,
             min_lr = 1e-4,
             patience=SCHEDULER_PATIENCE)
        self.reset_patience = 250

    def prepro(self, state: torch.Tensor) -> torch.Tensor:

        if state is None:
            return None

        crop_height = int(state.shape[1] * 0.88)
        crop_w = int(state.shape[2] * 0.07)
        state = state[:, :crop_height, crop_w:-crop_w, :]
        g = state[:, :, :, 1::3]
        gray = (g // 16) / 16
        gray = torch.moveaxis(gray, -1, 1)

        # print(gray.shape)
        # plt.imshow(gray[:,1,:,:], cmap='gray')
        # plt.show()
        # input('Continue ?')
        return gray

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

        if self.steps_done % IDLENESS == 0:
            self.steps_done+=1 #Update the number of steps within one episode
            self.episode_duration[-1]+=1 #Update the duration of the current episode
            if sample > self.epsilon or not self.exploration:
                with torch.no_grad():
                    # torch.no_grad() used when inference on the model is done
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.

                    _ , result = self.net(state)
                    action = result.max(1).indices.view(1, 1)
            else:
                # If action is selected at random, give a bit of extra weight
                # on hitting the gas
                choice = np.random.choice(flatdim(act_space), p=[0.1, 0.2, 0.2, 0.3, 0.2])
                action = torch.tensor([[choice]], device = DEVICE, dtype=torch.long)


            self.last_action = action
            return action
        else:
            self.steps_done+=1 #Update the number of steps within one episode
            self.episode_duration[-1]+=1 #Update the duration of the current episode
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
                           for s in batch.next_state]

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)


        y_val_pred, y_pol_pred = self.net(state_batch,action_batch)

        future_state_values = torch.zeros((BATCH_SIZE,5), dtype=torch.float32, device = DEVICE)
        rewards_tensor = torch.tile(reward_batch, (5,1)).T.to(DEVICE)

        with torch.no_grad():
            _, next_actions = self.net(non_final_next_states)
            next_actions = next_actions.max(1).indices
            future_state_values[non_final_mask,:], _ = self.net(non_final_next_states, next_actions.unsqueeze(-1))
        y_val_true = rewards_tensor + GAMMA * future_state_values

        adv = y_val_true - y_val_pred
        val_loss = 0.5 * torch.square(adv)

        pol_loss = - (adv * torch.log(y_pol_pred+1e-6))

        loss = (val_loss+pol_loss).mean()
        # print(loss.item())
        self.losses.append(float(loss))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self.losses

    def update_memory(self, state, action, next_state, reward) -> None:
        self.rewards.append(reward[0].item())

        if self.episode_duration[-1] < 50: # On ne met pas en mémoire le zoom de début d'épisode
            return None

        state = self.prepro(state)
        next_state = self.prepro(next_state)

        if sum(self.rewards[-1*min(self.reset_patience,self.episode_duration[-1]):]) <= (self.reset_patience-1)*-0.1:
            reward[0]=-100
            self.memory.push(state, action, next_state, reward)
            self.rewards[-1]=-100
            return True

        self.memory.push(state, action, next_state, reward)

        return None

    def update_agent(self, strategy=NETWORK_REFRESH_STRATEGY):
            """Empty function that just allows the environment code to run

            Args:
                strategy (_type_, optional): _description_. Defaults to NETWORK_REFRESH_STRATEGY.
            """
            pass



class CarA2CAgentContinous(SuperAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_actions = kwargs.get("n_actions", 4)
        self.actor_net = ConvA2CContinuousActor(self.n_actions).to(DEVICE)
        self.critic_net = ConvA2CContinuousCritic().to(DEVICE)
        self.optimizer_actor = torch.optim.AdamW(self.actor_net.parameters(), lr=INI_LR)
        self.optimizer_critic = torch.optim.AdamW(self.critic_net.parameters(), lr=INI_LR)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (optimizer=self.optimizer_actor,
             mode='max',
             factor=0.5,
             min_lr = 1e-4,
             patience=SCHEDULER_PATIENCE)
        self.reset_patience = 250
        self.reward_threshold = 50
        self.max_reward = 0

    def prepro(self, state: torch.Tensor) -> torch.Tensor:

        if state is None:
            return None

        crop_height = int(state.shape[1] * 0.88)
        crop_w = int(state.shape[2] * 0.07)
        state = state[:, :crop_height, crop_w:-crop_w, :]
        g = state[:, :, :, 1::3]
        gray = (g  / 255)
        gray = torch.moveaxis(gray, -1, 1)

        # print(gray.shape)
        # plt.imshow(gray[:,1,:,:], cmap='gray')
        # plt.show()
        # input('Continue ?')
        return gray

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
        #sample = random.random()
        #self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            #np.exp(-self.steps_done / EPS_DECAY)
        self.time = (datetime.now() - self.creation_time).total_seconds()

        if self.steps_done % IDLENESS == 0:
            self.steps_done+=1 #Update the number of steps within one episode
            self.episode_duration[-1]+=1 #Update the duration of the current episode
            # if sample > self.epsilon or not self.exploration:
            with torch.no_grad():
                # torch.no_grad() used when inference on the model is done
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                mu, sigma = self.actor_net(state.to(DEVICE))

                print('\n Choosing action \n')
                print(mu)
                print(sigma)
                print('\n\n')

                sigma = torch.clamp(sigma,torch.Tensor([1e-6,1e-6,1e-6,1e-6]).to(DEVICE))
                dist = torch.distributions.Normal(mu, sigma)
                action = dist.sample()
                action = torch.clamp(action, torch.Tensor([0,0,0,0]).to(DEVICE), torch.Tensor([1,1,1,1]).to(DEVICE))

            self.last_action = action
            return action
        else:
            self.steps_done+=1 #Update the number of steps within one episode
            self.episode_duration[-1]+=1 #Update the duration of the current episode
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

        if self.episode_duration[-1] < 50: # On ne s'entraîne pas pendant le zoom de début d'épisode
            return None

        if len(self.memory) < BATCH_SIZE: return 0

        transitions = self.memory.sample(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.

        batch = Transition(*zip(*transitions)) # Needs to pass this from buffer class

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        mu_v, var_v = self.actor_net(state_batch)
        var_v = torch.clamp(var_v,torch.Tensor([1e-6,1e-6,1e-6,1e-6]).to(DEVICE))

        value_v = self.critic_net(state_batch, action_batch)

        loss_value_v = nn.functional.mse_loss(value_v.squeeze(-1), reward_batch)

        adv_v = reward_batch.unsqueeze(dim=-1) - value_v.detach()

        p1 = - (mu_v - action_batch).pow(2) / (2 * var_v)
        p2 = - torch.log(torch.square(2 * var_v * np.pi))


        log_prob_v = adv_v * (p1 + p2)
        loss_policy_v = -log_prob_v.mean()
        entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*np.pi*var_v) + 1)/2).mean()

        loss_v = loss_policy_v + entropy_loss_v + loss_value_v

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_v.backward()

        torch.nn.utils.clip_grad_value_(list(self.actor_net.parameters()) + \
                                        list(self.critic_net.parameters()), 100)

        self.optimizer_actor.step()
        self.optimizer_critic.step()

        rwd_ep = self.episode_rewards[-1]
        lr = self.scheduler.optimizer.param_groups[0]['lr']

        # print(f" 🏎️  🏎️  || {'t':7s} | {'Step':7s} | {'Episode':14s} | {'Loss':8s}  |" \
        #     + f" {'ε':7s}    | {'η':8s} | {'Rwd/ep':7s}")
        # print(f" 🏎️  🏎️  || " \
        #     + f'{self.time:7.1f} | {self.steps_done:7.0f} | ' \
        #     + f'{self.episode:7.0f} / {NUM_EPISODES:4.0f} | ' \
        #     + f'{self.losses[-1]:.2e} |'\
        #     + f' {lr:.2e} | {rwd_ep:7.2f}')

        # print("\033[F"*2, end='')

        return self.losses


    def old_optimize_model(self) -> list:
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
                           for s in batch.next_state]

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        y_val_pred = self.critic_net(state_batch)
        y_pol_mu_pred, y_pol_sigma_pred = self.actor_net(state_batch)
        y_pol_sigma_pred = torch.clamp(y_pol_sigma_pred,torch.Tensor([1e-6,1e-6,1e-6]))

        future_state_values = torch.zeros((BATCH_SIZE,1), dtype=torch.float32, device = DEVICE)
        # rewards_tensor = torch.tile(reward_batch, (5,1)).T.to(DEVICE)

        with torch.no_grad():
            mu, sigma = self.actor_net(non_final_next_states)
            sigma = torch.clamp(sigma,torch.Tensor([1e-6,1e-6,1e-6]))
            dist = torch.distributions.Normal(mu, sigma)
            next_actions = dist.sample()
            next_actions = torch.clamp(next_actions, torch.Tensor([-1,0,0]), torch.Tensor([1,1,1]))
            future_state_values[non_final_mask] = self.critic_net(non_final_next_states)
        y_val_true = reward_batch + GAMMA * future_state_values

        adv = y_val_true - y_val_pred
        val_loss = 0.5 * torch.square(adv)

        dist = torch.distributions.Normal(y_pol_mu_pred, y_pol_sigma_pred)
        log_probs = dist.log_prob(action_batch).sum(axis=1)
        pol_loss = -(log_probs * adv)

        p1 = - (y_pol_mu_pred - action_batch).pow(2) / (2 * y_pol_sigma_pred)
        # print('p1')
        # print(p1)

        p2 = - torch.log(torch.square(2 * y_pol_sigma_pred * np.pi))
        # print('p2')
        # print(p2)

        pol_loss = (p1 + p2)


        # Entropy loss
        # print(log_probs)
        # print(action_batch)
        # print(log_probs)
        # print(y_pol_mu_pred)
        # print(y_pol_sigma_pred)
        # print('\n\n')
        entropy = dist.entropy().sum(axis=1)
        # print("pol_loss")
        # print(pol_loss.mean())
        # print("val_loss")
        # print(val_loss.mean())
        loss = (pol_loss + val_loss - ENTROPY_BETA * entropy).mean()
        # print('loss')
        # print(loss)
        # print(loss.item())
        self.losses.append(float(loss))

        # if self.steps_done > 10:
        #     exit()

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        val_loss.mean().backward(retain_graph=False)
        pol_loss.mean().backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(list(self.actor_net.parameters()) + \
                                        list(self.critic_net.parameters()), 100)

        self.optimizer_critic.step()
        self.optimizer_actor.step()

        rwd_ep = self.episode_rewards[-1]
        lr = self.scheduler.optimizer.param_groups[0]['lr']

        # print(f" 🏎️  🏎️  || {'t':7s} | {'Step':7s} | {'Episode':14s} | {'Loss':8s}  |" \
        #     + f" {'ε':7s}    | {'η':8s} | {'Rwd/ep':7s}")
        # print(f" 🏎️  🏎️  || " \
        #     + f'{self.time:7.1f} | {self.steps_done:7.0f} | ' \
        #     + f'{self.episode:7.0f} / {NUM_EPISODES:4.0f} | ' \
        #     + f'{self.losses[-1]:.2e} |'\
        #     + f' {lr:.2e} | {rwd_ep:7.2f}')

        # print("\033[F"*2, end='')

        return self.losses

    def update_memory(self, state, action, next_state, reward) -> None:
        self.rewards.append(reward[0].item())

        if self.episode_duration[-1] < 50: # On ne met pas en mémoire le zoom de début d'épisode
            return None

        state = self.prepro(state)
        next_state = self.prepro(next_state)

        if sum(self.rewards[-1*min(self.reset_patience,self.episode_duration[-1]):]) <= (self.reset_patience-1)*-0.1:
            reward[0]=-100
            self.memory.push(state, action, next_state, reward)
            self.rewards[-1]=-100
            return True

        self.memory.push(state, action, next_state, reward)

        return None

    def update_agent(self, strategy=NETWORK_REFRESH_STRATEGY):
            """Empty function that just allows the environment code to run

            Args:
                strategy (_type_, optional): _description_. Defaults to NETWORK_REFRESH_STRATEGY.
            """
            pass

    def logging(self):
        """Logs some statistics on the agent running as a function of time
        in a .csv file"""

        if not os.path.exists(self.folder() + 'log.csv'):
            with open(self.folder() + 'log.csv', 'w') as log_file:
                log_file.write('Time,Step,Episode,Loss,Reward,Eta,Action0,Action1,Action2\n')

        lr = self.scheduler.optimizer.param_groups[0]['lr']

        self.log_buffer.append([self.time,
                                     self.steps_done,
                                     self.episode,
                                     self.losses[-1],
                                     self.rewards[-1],
                                     lr,
                                     self.last_action.tolist()[0][0],
                                     self.last_action.tolist()[0][1],
                                     self.last_action.tolist()[0][2]])

        if self.steps_done % LOG_EVERY == 0:
            array_test = np.vstack(self.log_buffer)
            self.log_buffer = []

            with open(self.folder() + 'log.csv', 'a') as myfile:
                np.savetxt(myfile, array_test, delimiter=',',
                           fmt=["%7.2f", "%6d", "%4d",
                                "%5.3e", "%5.3e", "%5.3e",
                                "%5.3f", "%5.3f", "%5.3f"])



class CarPPOAgentContinous(CarA2CAgentContinous):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.memory = ReplayMemoryPPO(MEM_SIZE)

        self.last_mu = None
        self.last_sigma = None

        self.clip_param = 0.2


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
        #sample = random.random()
        #self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            #np.exp(-self.steps_done / EPS_DECAY)
        self.time = (datetime.now() - self.creation_time).total_seconds()

        if self.steps_done % IDLENESS == 0:
            self.steps_done+=1 #Update the number of steps within one episode
            self.episode_duration[-1]+=1 #Update the duration of the current episode
            # if sample > self.epsilon or not self.exploration:
            with torch.no_grad():
                # torch.no_grad() used when inference on the model is done
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                self.last_mu, self.last_sigma = self.actor_net(state.to(DEVICE))

                print("Reward")
                print(str(self.episode_rewards[-1]) + '\n')

                print('\n Choosing action \n')
                print(self.last_mu)
                print(self.last_sigma)
                print('\n\n')

                self.last_sigma = torch.clamp(self.last_sigma,torch.Tensor([1e-6,1e-6,1e-6,1e-6]).to(DEVICE))
                dist = torch.distributions.Normal(self.last_mu, self.last_sigma)
                action = dist.sample()
                action = torch.clamp(action, torch.Tensor([0,0,0,0]).to(DEVICE), torch.Tensor([1,1,1,1]).to(DEVICE))

            self.last_action = action
            return action, self.last_mu, self.last_sigma
        else:
            self.steps_done+=1 #Update the number of steps within one episode
            self.episode_duration[-1]+=1 #Update the duration of the current episode
            return self.last_action, self.last_mu, self.last_sigma


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

        if self.episode_duration[-1] < 50: # On ne s'entraîne pas pendant le zoom de début d'épisode
            return None

        if len(self.memory) < BATCH_SIZE: return 0

        transitions = self.memory.sample(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.

        batch = TransitionPPO(*zip(*transitions)) # Needs to pass this from buffer class

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        mu_batch = torch.cat(batch.mu)
        sigma_batch = torch.cat(batch.sigma)
        reward_batch = torch.cat(batch.reward)

        mu_v, var_v = self.actor_net(state_batch)
        var_v = torch.clamp(var_v,torch.Tensor([1e-6,1e-6,1e-6,1e-6]).to(DEVICE))

        value_v = self.critic_net(state_batch, action_batch)

        loss_value_v = nn.functional.mse_loss(value_v.squeeze(-1), reward_batch)

        adv_v = reward_batch.unsqueeze(dim=-1) - value_v.detach()

        dist_old = torch.distributions.Normal(mu_batch, sigma_batch)
        dist_new = torch.distributions.Normal(mu_v, var_v)

        log_probs_old = dist_old.log_prob(action_batch)
        log_probs_new = dist_new.log_prob(action_batch)

        ratio = log_probs_new / log_probs_old

        surr1 = ratio * adv_v
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_v
        action_loss = -torch.min(surr1, surr2).mean()

        p1 = - (mu_v - action_batch).pow(2) / (2 * var_v)
        p2 = - torch.log(torch.square(2 * var_v * np.pi))


        log_prob_v = adv_v * (p1 + p2)
        loss_policy_v = -log_prob_v.mean()
        entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*np.pi*var_v) + 1)/2).mean()

        # loss_v = loss_policy_v + entropy_loss_v + loss_value_v
        loss_v = action_loss + entropy_loss_v + loss_value_v

        # print("Loss")
        # print(str(loss_v) + '\n')

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_v.backward()

        torch.nn.utils.clip_grad_value_(list(self.actor_net.parameters()) + \
                                        list(self.critic_net.parameters()), 100)

        self.optimizer_actor.step()
        self.optimizer_critic.step()

        rwd_ep = self.episode_rewards[-1]
        lr = self.scheduler.optimizer.param_groups[0]['lr']

        # print(f" 🏎️  🏎️  || {'t':7s} | {'Step':7s} | {'Episode':14s} | {'Loss':8s}  |" \
        #     + f" {'ε':7s}    | {'η':8s} | {'Rwd/ep':7s}")
        # print(f" 🏎️  🏎️  || " \
        #     + f'{self.time:7.1f} | {self.steps_done:7.0f} | ' \
        #     + f'{self.episode:7.0f} / {NUM_EPISODES:4.0f} | ' \
        #     + f'{self.losses[-1]:.2e} |'\
        #     + f' {lr:.2e} | {rwd_ep:7.2f}')

        # print("\033[F"*2, end='')

        return self.losses

    def update_memory(self, state, action, mu, sigma, next_state, reward) -> None:
        self.rewards.append(reward[0].item())

        if self.episode_duration[-1] < 50: # On ne met pas en mémoire le zoom de début d'épisode
            return None

        state = self.prepro(state)
        next_state = self.prepro(next_state)

        if sum(self.rewards[-1*min(self.reset_patience,self.episode_duration[-1]):]) <= (self.reset_patience-1)*-0.1:
            reward[0]=-100
            self.memory.push(state, action, mu, sigma, next_state, reward)
            self.rewards[-1]=-100
            return True

        self.memory.push(state, action, mu, sigma, next_state, reward)

        return None
