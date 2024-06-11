import torch

import numpy as np
import random
from datetime import datetime
from gymnasium.spaces.utils import flatdim

from params import *
from super_agent import DQNAgent, SuperAgent
from network import *
from display import Plotter
from buffer import ReplayMemory, TorchMemory


networks = {'ConvDQN3layersSmall':ConvDQN3layersSmall,
            'ConvDQN3layersClassic':ConvDQN3layersClassic,
            'ConvDQN2layersSmall':ConvDQN2layersSmall,
            'ConvDQN2layersClassic':ConvDQN2layersClassic,
            'ConvDQN2layersBrice':ConvDQN2layersBrice,
            'ConvA2C':ConvA2CBrice
              }

class CarDQNAgent(DQNAgent):
    """The Car DQN Agent class, deriving from the DQN Agent class. We add a
    few specific things in terms of preprocessing.

    ARGS :
        y_dim : it should correspond to the action space (N actions in discrete case)
        reward_threshold : if we exceed our best score by that threshold, we
            save the model
        reset_patience : ????? no longer sure about it
        crop_image [True/False] : whether you want to crop the image before
            injecting it into the network. In that case, the agent has no information
            about its speed / etc. through these controls.
    """

    def __init__(self, y_dim: int,
                 reward_threshold:float=20,
                 reset_patience:int=250,
                 crop_image=True, **kwargs) -> None:
        ChosenNetwork = networks[NETWORK]
        self.policy_net = ChosenNetwork(y_dim, dropout_rate=DROPOUT_RATE).to(DEVICE)
        self.target_net = ChosenNetwork(y_dim, dropout_rate=DROPOUT_RATE).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        super().__init__(**kwargs)

        self.crop_image = crop_image
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
        """Preprocessing for CarDQNAgent. Converts the image to b&w
        using the GREEN channel of each successive image.

        Args:
            state (torch.Tensor): a single (or multiple) observation

        Returns:
            torch.Tensor: the preprocessed frame(s)
        """

        state = state[:,:,:,1::3] / 256
        if self.crop_image:
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
                action = np.random.choice(flatdim(act_space), p=[0.05, 0.2, 0.2, 0.30, 0.25])
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
            best_action_values = future_state_values.max(1).values.unsqueeze(-1)

        # Compute MSE loss
        loss = self.lossfun(state_action_values, best_action_values)
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

class CarA2CAgent(SuperAgent):
    def __init__(self, y_dim: int,
                 reward_threshold:float=20,
                 reset_patience:int=250,
                 crop_image=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.net = networks[NETWORK](y_dim).to(DEVICE) # To check
        self.crop_image = crop_image
        self.reward_threshold = reward_threshold
        self.max_reward = 0
        self.reset_patience = reset_patience
        self.batch = []
        self.adv = [0]
        self.pol_loss = [0]

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=INI_LR)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (optimizer=self.optimizer,
             mode='max',
             factor=SCHEDULER_FACTOR,
             min_lr = MIN_LR,
             patience=SCHEDULER_PATIENCE)

        if MEM_TYPE.lower() == 'legacy':
            self.memory = ReplayMemory(MEM_SIZE)
        elif MEM_TYPE.lower() == 'torch':
            self.memory = TorchMemory(MEM_SIZE)

    def prepro(self, state: torch.Tensor) -> torch.Tensor:
        """Preprocessing for CarDQNAgent. Converts the image to b&w
        using the GREEN channel of each successive image.

        Args:
            state (torch.Tensor): a single (or multiple) observation

        Returns:
            torch.Tensor: the preprocessed frame(s)
        """

        state = state[:,:,:,1::3] / 256
        if self.crop_image:
            crop_height = int(state.shape[1] * 0.88)
            crop_w = int(state.shape[2] * 0.07)
        state = state[:, :crop_height, crop_w:-crop_w, :]

        return state.moveaxis(-1, 1)

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
            act ActType : Action that the agent performs
        """
        state = self.prepro(state)
        self.episode_duration[-1]+=1 #Update the duration of the current episode
        self.steps_done+=1 #Update the number of steps within one episode
        self.time = (datetime.now() - self.creation_time).total_seconds()

        with torch.no_grad():
            # torch.no_grad() used when inference on the model is done
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            _ , probs = self.net(state)
            probs = probs.detach().numpy().T.squeeze(-1)
            action = torch.tensor(np.random.choice(flatdim(act_space), p= probs))
        self.last_action = action
        print(" 🏎️  🏎️  || Idle | Left | Right | Gas | Break")
        print(probs)
        print("\033[F"*2, end='')
        return action

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

        y_val_pred, y_pol_pred = self.net(state_batch)

        with torch.no_grad():
            #next_actions = self.actor_net(next_state_batch*not_done_batch)
            future_state_values, _ = self.net(next_state_batch)#*not_done_batch

            y_val_true = reward_batch.unsqueeze(-1) + GAMMA * future_state_values

        # with torch.no_grad():
        #     _, next_actions = self.net(non_final_next_states.unsqueeze(-1))
        #     next_actions = next_actions.max(1).indices
        #     future_state_values[non_final_mask,:], _ = self.net(non_final_next_states.unsqueeze(-1), next_actions.unsqueeze(-1))
        # y_val_true = reward_batch.unsqueeze(-1) #+ GAMMA * future_state_values

        #Critic Loss - Advantage
        adv = y_val_true - y_val_pred
        val_loss = torch.mean(0.5 * torch.square(adv))

        self.optimizer.zero_grad()
        val_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 25)

        # print(adv.detach())
        # print(y_pol_pred)
        #Actor Loss - LogLikelihood
        pol_loss = torch.mean(-(adv * torch.log(y_pol_pred+1e-6)))
        pol_loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 25)
        self.optimizer.step()

        # self.losses.append(float(loss))
        self.adv.append(float(adv[0][0]))
        self.pol_loss.append(float(pol_loss.item()))

        return self.adv

    def update_agent(self, strategy=NETWORK_REFRESH_STRATEGY):
        """Empty function that just allows the environment code to run

        Args:
            strategy (_type_, optional): _description_. Defaults to NETWORK_REFRESH_STRATEGY.
        """
        pass

    def update_memory(self, state:torch.Tensor,
                      action:torch.Tensor,
                      next_state:torch.Tensor,
                      reward:torch.Tensor,
                      not_done:torch.Tensor,
                      skip_steps=50) -> None:
        """ UPDATE_MEMORY (SUPER_AGENT) defines
        what to do in general when we update the
        agent memory.

        Optional arg: skip_step (default 50), forces the programme to not store
        any states at the beginning of each episode (e.g. for CarRace, when they
        are not useful)
        """

        state = self.prepro(state)
        next_state = self.prepro(next_state)
        reward = reward.unsqueeze(-1)
        not_done = not_done.unsqueeze(-1)
        action = action.unsqueeze(-1)

        self.rewards.append(reward[0].item())
        current_episode_rewards = sum(self.rewards[-self.episode_duration[-1]:])
        episode_is_done = current_episode_rewards < EARLY_STOPPING_SCORE

        if self.episode_duration[-1] < skip_steps: # On ne met pas en mémoire le zoom de début d'épisode
            return episode_is_done

        self.memory.push(state, action, next_state, reward, not_done)

        return episode_is_done
