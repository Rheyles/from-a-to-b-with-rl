import torch
from datetime import datetime
import json
import os

from params import *
from buffer import ReplayMemory

class SuperAgent():
    def __init__(self, **kwargs) -> None:
        self.exploration = kwargs.get('exploration',True)
        self.training = kwargs.get('training',True)
        self.show_diagnostics = kwargs.get('show_diagnostics',False)

        self.steps_done = 0
        self.episode = 0
        self.losses = [0]
        self.rewards = [0]
        self.episode_rewards = [0]
        self.episode_duration = [0]

        self.memory = ReplayMemory(MEM_SIZE)

    def select_action()-> torch.Tensor:
        pass
    def optimize_model() -> list:
        pass
    def save_model():
        pass
    def load_model():
        pass


class DQNAgent(SuperAgent):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.policy_net = None
        self.target_net = None

    def update_memory(self, state, action, next_state, reward) -> None:
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
