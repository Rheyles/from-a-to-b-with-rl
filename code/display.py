import matplotlib.pyplot as plt
import numpy as np
import torch
import typing
import sys
from params import *


class Plotter():

    plt.ion()
    fig_data_list = []

    @classmethod
    def plot_data_gradually(cls, data_name: str, data: list[float,int] | np.ndarray,
                            show_result: bool=False, cumulative=False, episode_durations:list[int]=[0],
                            rolling: int=0) -> None:
        """
        Plots gradually the data as a function of time. Each data is plotted in
        its own figure
        ex : Plotter().plot_data_gradually(data_name,data)

        Args:
            data_name (str): the name of the data you want to plot
            data (list[float,int]): the data you want to plot
            show_result (bool, optional): true if the data won't change anymore.
                                          Defaults to False.
            cumulative (bool, default=False) : plot the _cumulative sum_ of
                                            a variable instead of the variable instead.
            rolling (int, default=0) : plots a _rolling average_ of a variable
                                            instead of a variable
        """

        # We get the correct Figure object
        title = 'Result' if show_result else 'Training ...'
        if cumulative:
            if len(episode_durations) < 1:
                data = np.cumsum(data)
            else:
                temp_data = np.array([])
                for i in range(len(episode_durations)-1):
                    cum_data = np.cumsum(data[sum(episode_durations[:i+1]):sum(episode_durations[:i+2])])
                    temp_data = np.concatenate((temp_data,cum_data))
                data = temp_data
        if rolling:
            dummy = np.convolve(np.ones_like(data), np.ones(rolling)/rolling, mode='valid')
            data = np.convolve(data, np.ones(rolling)/rolling, mode='valid')/dummy

        if data_name not in cls.fig_data_list:
            cls.fig_data_list.append(data_name)
            fig = plt.figure(len(cls.fig_data_list))
            ax = plt.axes()
            ax.set_title(title)
            ax.plot(data)
            ax.set_xlabel('Step')
            ax.set_ylabel(data_name)
        else:
            ax = plt.figure(cls.fig_data_list.index(data_name) + 1).gca()
            ax._children[0].set_data(np.arange(len(data)), data)
            ax.axis([0, len(data)+1, min(data)-0.0001, max(data)+0.001])

        plt.pause(0.001)  # pause a bit so that plots are updated

        if show_result:
            plt.show()

        return ax


def plot_success_rate(episode_rewards:list, rolling=10):

    roll_rewards = np.convolve(np.array(episode_rewards),
                               np.ones(rolling)/rolling, mode='same')
    ax = Plotter().plot_data_gradually('RollingRewards', data=roll_rewards)


def dqn_diagnostics(agent,
                    action_batch:torch.Tensor,
                    best_action:torch.Tensor,
                    state_batch:torch.Tensor,
                    reward_batch:torch.Tensor,
                    all_next_states:torch.Tensor,
                    state_action_values:torch.Tensor,
                    future_state_values:torch.Tensor,
                    best_action_values:torch.Tensor):
    """
        DQN_diagnostics (plenty of things) : allows you to view
        detailed diagnotics on the DQN model in the terminal.

        ARGS
        ----
        * agent : your agent
        * action_batch, best_action, state_batch, all_next_states,
        state_action_values, future_state_values, best_action_values : things
        that are usually available from optimize, so call this function
        from your "optimize" in the agent.

    """

    with torch.no_grad():
        action_dict = {0:'←', 1:'↓', 2:'→', 3:'↑'}
        action_arrows = [action_dict[elem.item()] for elem in action_batch]
        best_act_arrs = [action_dict[elem.item()] for elem in best_action]

        states_str     = 'Current state ' + ' | '.join([f"{elem:5.0f}" for elem in state_batch])
        action_str     = 'Current action' + ' | '.join([f"    {elem}"   for elem in action_arrows])
        reward_str     = 'Reward (s,a)  ' + ' | '.join([f"{elem:5.0f}" for elem in reward_batch])
        next_state_str = 'Future state  ' + ' | '.join([f"{elem:5.0f}"  for elem in all_next_states])
        current_Q_str  = 'Current Q     ' + ' | '.join([f"{elem:+4.2f}" for elem in torch.squeeze(state_action_values)])
        Q_left_str     = 'Estimated Q ← ' + ' | '.join([f"{elem:+4.2f}" for elem in future_state_values[:,0]])
        Q_down_str     = 'Estimated Q ↓ ' + ' | '.join([f"{elem:+4.2f}" for elem in future_state_values[:,1]])
        Q_right_str    = 'Estimated Q → ' + ' | '.join([f"{elem:+4.2f}" for elem in future_state_values[:,2]])
        Q_up_str       = 'Estimated Q ↑ ' + ' | '.join([f"{elem:+4.2f}" for elem in future_state_values[:,3]])
        expected_Q_str = 'Best known Q  ' + ' | '.join([f"{elem:+4.2f}" for elem in best_action_values])
        best_act_str   = 'Bst futr actn ' + ' | '.join([f"    {elem}"   for elem in best_act_arrs])

        loss = agent.losses[-1]
        step = agent.steps_done
        rewards = np.sum(agent.rewards)
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-agent.steps_done / EPS_DECAY)

        print("\033[A"*16)
        print(f'\nStep {step:4.0f}, \t loss : {float(loss):5.2e}, \t rewards : {rewards:3.0f}, \t epsilon : {eps:.3f}')
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




if __name__ == '__main__':
    # import numpy as np

    # x = np.random.uniform(size=(4,10))
    # y = np.random.uniform(size=(4,10))
    # pretty_print((x,y), names=('fake x', 'real x'))

    data = []
    data_2 = []
    for i in range(50):
        data.append(i**2)
        data_2.append(i+1)
        Plotter().plot_data_gradually('Test', data)
        Plotter().plot_data_gradually('Toto', data_2)
    data.append(i**2)
    Plotter().plot_data_gradually('Test', data, show_result=True)
    Plotter().plot_data_gradually('Toto', data_2, show_result=True)
