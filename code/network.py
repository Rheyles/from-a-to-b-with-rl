import torch.nn as nn
import torch.nn.functional as F
import torch
from params import MULTIFRAME


class LinearDQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(LinearDQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class LinearA2C(nn.Module):

    def __init__(self, n_observations, n_actions, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.actor = nn.Sequential(
            nn.Linear(n_observations, 300), nn.ReLU(inplace = True),
            nn.Linear(300, n_actions),
            nn.Softmax(),
        )

        self.critic = nn.Sequential(
            nn.Linear(in_features = 9249, out_features = 300),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 300, out_features = 1),
            nn.Tanh()
        )

    def forward(self, state, action = None):

        y_pol = self.actor(state)
        if action is None:
            return None, y_pol
        y_val = self.critic(torch.concat((state,action), dim = 1))
        return y_val, y_pol


class ConvDQN(nn.Module):

    def __init__(self, n_actions, dropout_rate=0.0):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(MULTIFRAME, 32, kernel_size=9, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate))
        self.lin1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6272, n_actions, bias=True))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.lin1(out)
        return F.softmax(out)



class ConvA2C(nn.Module):

    def __init__(self, n_actions, dropout_rate=0.0):
        super(ConvA2C, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=9, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rate))

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(p=dropout_rate))


        self.conv_net  = nn.Sequential(self.conv1,
                                       self.conv2,
                                       nn.Flatten())


        self.actor = nn.Sequential(
            nn.Linear(9248, 300), nn.ReLU(inplace = True),
            nn.Linear(300, n_actions),
            nn.Softmax(),
        )

        self.critic = nn.Sequential(
            nn.Linear(in_features = 9249, out_features = 300),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 300, out_features = 1),
            nn.Tanh()
        )




    def forward(self, state, action = None):

        out = self.conv_net(state)
        y_pol = self.actor(out)
        if action is None:
            return None, y_pol
        y_val = self.critic(torch.concat((out,action), dim = 1))
        return y_val, y_pol
