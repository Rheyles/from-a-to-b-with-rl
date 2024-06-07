import torch
from datetime import datetime
import json
import os
import glob

from params import *
from buffer import ReplayMemory

class SuperAgent():
    def __init__(self, **kwargs) -> None:
        self.exploration = kwargs.get('exploration',True)
        self.training = kwargs.get('training',True)
        self.show_diagnostics = kwargs.get('show_diagnostics',False)
        self.creation_time = datetime.now()
        self.time = (datetime.now() - self.creation_time).total_seconds()
        self.log_every = kwargs.get('log_every', 100)
        self.log_buffer = []
        self.folder = 'models/' \
            + datetime.strftime(datetime.now(), "%m%d_%H%M_") \
            + str(self.__class__.__name__) + '/'


        self.steps_done = 0
        self.episode = 0
        self.losses = [0]
        self.rewards = [0]
        self.episode_rewards = [0]
        self.episode_duration = [0,0]

        self.memory = ReplayMemory(MEM_SIZE)

    def select_action(self)-> torch.Tensor:
        pass
    def optimize_model(self) -> list:
        pass
    def soft_update_agent(self, *args, **kwargs):
        pass
    def save_model(self, **kwargs):
        pass
    def load_model(self):
        pass
    def logging(self):
        pass


class DQNAgent(SuperAgent):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.policy_net = None
        self.target_net = None
        self.epsilon = EPS_START

    def update_memory(self, state, action, next_state, reward) -> None:
        self.memory.push(state, action, next_state, reward)
        self.rewards.append(reward[0].item())
        #print(self.rewards)
        #print(reward)
        return None

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

    def save_model(self, add_episode=True) -> None:
        """
        Save the model (NN and hyperparameters).

        Args:
            folder (str, optional): Where to save the model. Defaults to 'models/'.
        """

        episode_str = ''
        if add_episode:
            episode_str = f'_{self.episode}'

        os.makedirs(self.folder, exist_ok=True)
        torch.save(self.policy_net.state_dict(),
                   f'{self.folder}/policy{episode_str}.model')
        torch.save(self.target_net.state_dict(),
                   f'{self.folder}/target{episode_str}.model')

        with open(self.folder + '/params.json', 'w') as my_file:
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

        policy_file = glob.glob(folder + '/policy_*.model')[-1]
        target_file = glob.glob(folder + '/target_*.model')[-1]
        self.policy_net.load_state_dict(torch.load(policy_file, map_location=DEVICE))
        self.target_net.load_state_dict(torch.load(target_file, map_location=DEVICE))
        with open(folder + '/params.json') as my_file:
            print(f'Loaded model {policy_file} from {folder}')
            hyper_dict = json.load(my_file)
            print(''.join([f"- {key} : {val} \n" for key, val in hyper_dict.items()]))
