import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.distributions.categorical import Categorical



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    try:
        assert(torch.backends.mps.is_available())
        device = torch.device("mps")
    except:
        device = torch.device('cpu')

class MyPPO(nn.Module):
    """Implementation of a PPO model. The same backbone is used to get actor and critic values.
    Includes a backbone (to change for a cnn application), an actor and a critic"""

    def __init__(self, MULTIFRAME, in_shape, n_actions, hidden_d=100, share_backbone=False):
        # Super constructor
        super(MyPPO, self).__init__()

        # Attributes
        self.in_shape = in_shape
        self.n_actions = n_actions
        self.hidden_d = hidden_d
        self.share_backbone = share_backbone

        # Shared backbone for policy and value functions
        in_dim = np.prod(in_shape)

        self.convnet = nn.Sequential(
            nn.Conv2d(MULTIFRAME, 16, kernel_size=7, stride=3,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0),
            nn.Conv2d(16, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0),
            nn.Flatten()
        )


        # State action function
        self.actor = nn.Sequential(
            nn.Linear(1152, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

        # Value function
        self.critic = nn.Sequential(
            nn.Linear(1152, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.convnet(x)
        action = self.actor(features)
        value = self.critic(features)
        return Categorical(action).sample(), action, value


def prepro(state: torch.Tensor,crop_image :bool) -> torch.Tensor:
        """Preprocessing for CarDQNAgent. Converts the image to b&w
        using the GREEN channel of each successive image.

        Args:
            state (torch.Tensor): a single (or multiple) observation

        Returns:
            torch.Tensor: the preprocessed frame(s)
        """
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        state = state[:,:,:,1::3] / 256
        if crop_image:
            crop_height = int(state.shape[1] * 0.88)
            crop_w = int(state.shape[2] * 0.07)
            state = state[:, :crop_height, crop_w:-crop_w, :]

        return state.moveaxis(-1, 1)


@torch.no_grad()
def run_timestamps(env, model, MULTIFRAME,timestamps=1024, render=False,device="cpu"):
    """Runs the given policy on the given environment for the given amount of timestamps.
     Returns a buffer with state action transitions and rewards."""
    buffer = []
    state = env.reset()[0]

    batch = []
    # Running timestamps and collecting state, actions, rewards and terminations
    for ts in range(timestamps):
        # Taking a step into the environment
        model_input = prepro(state = state, crop_image=False)
        action, action_logits, value = model(model_input)
        new_state, reward, terminated, truncated, info = env.step(action.item())

        # Rendering / storing (s, a, r, t) in the buffer
        if render:
            env.render()
        else:
            buffer.append([model_input, action, action_logits, value, reward, terminated or truncated])

        # Updating current state
        state = new_state

        # Resetting environment if episode terminated or truncated
        if terminated or truncated:
            state = env.reset()[0]
    return buffer




def compute_cumulative_rewards(buffer, gamma):
    """Given a buffer with states, policy action logits, rewards and terminations,
    computes the cumulative rewards for each timestamp and substitutes them into the buffer."""
    curr_rew = 0.

    # Traversing the buffer on the reverse direction
    for i in range(len(buffer) - 1, -1, -1):
        r, t = buffer[i][-2], buffer[i][-1]

        if t:
            curr_rew = 0
        else:
            curr_rew = r + gamma * curr_rew

        buffer[i][-2] = curr_rew

    # Getting the average reward before normalizing (for logging and checkpointing)
    avg_rew = np.mean([buffer[i][-2] for i in range(len(buffer))])

    # Normalizing cumulative rewards
    mean = np.mean([buffer[i][-2] for i in range(len(buffer))])
    std = np.std([buffer[i][-2] for i in range(len(buffer))]) + 1e-6
    for i in range(len(buffer)):
        buffer[i][-2] = (buffer[i][-2] - mean) / std

    return avg_rew


def get_losses(model, batch, epsilon, annealing, device="cpu"):
    """Returns the three loss terms for a given model and a given batch and additional parameters"""
    # Getting old data
    n = len(batch)
    states = torch.cat([batch[i][0] for i in range(n)])
    actions = torch.cat([batch[i][1] for i in range(n)]).view(n, 1)
    logits = torch.cat([batch[i][2] for i in range(n)])
    values = torch.cat([batch[i][3] for i in range(n)])
    cumulative_rewards = torch.tensor([batch[i][-2] for i in range(n)]).view(-1, 1).float().to(device)

    # Computing predictions with the new model
    _, new_logits, new_values = model(states)

    # Loss on the state-action-function / actor (L_CLIP)
    advantages = cumulative_rewards - values
    margin = epsilon * annealing
    # print(f'margin {margin}, epsilon {epsilon}, annealing {annealing}')
    ratios = new_logits.gather(1, actions) / logits.gather(1, actions)

    l_clip = torch.mean(
        torch.min(
            torch.cat(
                (ratios * advantages,
                 torch.clip(ratios, 1 - margin, 1 + margin) * advantages),
                dim=1),
            dim=1
        ).values
    )

    # Loss on the value-function / critic (L_VF)
    l_vf = torch.mean((cumulative_rewards - new_values) ** 2)

    # Bonus for entropy of the actor
    entropy_bonus = torch.mean(torch.sum(-new_logits * (torch.log(new_logits + 1e-5)), dim=1))

    return l_clip, l_vf, entropy_bonus


def training_loop(env, model, max_iterations, n_actors, gamma, epsilon, n_epochs, batch_size, lr,
                  c1, c2, device, MODEL_PATH, MULTIFRAME, env_name=""):
    """Train the model on the given environment using multiple actors acting up to n timestamps."""

    # Training variables
    max_reward = float("-inf")
    optimizer = Adam(model.parameters(), lr=lr, maximize=True)
    scheduler = LinearLR(optimizer, 1, 0, max_iterations * n_epochs)
    anneals = np.linspace(1, 0, max_iterations)

    # Training loop
    for iteration in range(max_iterations):
        buffer = []
        annealing = anneals[iteration]

        # Collecting timestamps for all actors with the current policy
        for actor in range(1, n_actors + 1):
            # print(f'actor {actor}')
            buffer.extend(run_timestamps(env, model, MULTIFRAME = MULTIFRAME ,render=False, device=device))
            # print(len(buffer))

        # Computing cumulative rewards and shuffling the buffer
        avg_rew = compute_cumulative_rewards(buffer, gamma)

        np.random.shuffle(buffer)
        losses = []
        # Running optimization for a few epochs
        for epoch in range(n_epochs):
            for batch_idx in range(len(buffer) // batch_size):
                # Getting batch for this buffer
                start = batch_size * batch_idx
                end = start + batch_size if start + batch_size < len(buffer) else -1
                batch = buffer[start:end]

                # Zero-ing optimizers gradients
                optimizer.zero_grad()

                # Getting the losses
                l_clip, l_vf, entropy_bonus = get_losses(model, batch, epsilon, annealing, device)

                # Computing total loss and back-propagating it
                loss = l_clip - c1 * l_vf + c2 * entropy_bonus
                losses.append(loss)
                loss.backward()
                # Optimizing
                optimizer.step()
            scheduler.step()

        #Logging information to stdout
        curr_loss = losses[-1].item()
        log = f"Iteration {iteration + 1} / {max_iterations}: " \
              f"Average Reward: {avg_rew:.2f}\t" \
              f"Loss: {curr_loss:.3f} " \
              f"(L_CLIP: {l_clip.item():.1f} | L_VF: {l_vf.item():.1f} | L_bonus: {entropy_bonus.item():.1f})"
        if avg_rew > max_reward:
            torch.save(model.state_dict(), MODEL_PATH)
            max_reward = avg_rew
            log += " --> Stored model with highest average reward"
        print(log)

def testing_loop(env, model, MULTIFRAME,n_episodes, device):
    """Runs the learned policy on the environment for n episodes"""
    for _ in range(n_episodes):
        run_timestamps(env, model, MULTIFRAME = MULTIFRAME ,render=True, device=device)


def main():
    # Parsing program arguments
    max_iterations = 100 # Number of iterations of training
    n_actors = 12 # Number of actors for each update
    epsilon = 0.1 # Epsilon parameter, controls the margin decrease
    n_epochs = 10 # Number of training epochs per iteration
    batch_size = 64 # Batch size
    lr = 1e-4 # Learning Rate
    gamma = 0.9 # Discount factor gamma, accounts for the importance of previous rewards when calculating avg reward
    c1 = 0.9 # Weight for the value function in the loss function,
            #controls term ~ mean((cumulative_rewards - new_values) ** 2)
    c2 = 0.01 # Weight for the entropy bonus in the loss function
    n_test_episodes = 10 # Number of episodes to render
    MODEL_PATH = "/home/ralph/code/rhage183/from-a-to-b-with-rl/code/PPOmodels"
    MULTIFRAME = 1


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            assert(torch.backends.mps.is_available())
            device = torch.device("mps")
        except:
            device = torch.device('cpu')


    # Creating environment (discrete action space)
    env_name = "CarRacing-v2"
    env = gym.make(env_name, continuous = False)

    # Creating the model (both actor and critic)
    # model = MyPPO(MULTIFRAME = MULTIFRAME, in_shape=env.observation_space.shape, n_actions=env.action_space.n).to(device)

    # # Training
    # training_loop(env, model, max_iterations, n_actors, gamma, epsilon,
    #               n_epochs, batch_size, lr, c1, c2, device, MULTIFRAME = MULTIFRAME, MODEL_PATH = MODEL_PATH ,env_name = env_name)

    # Loading best model
    model = MyPPO(MULTIFRAME = MULTIFRAME, in_shape=env.observation_space.shape, n_actions=env.action_space.n).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # # Testing
    env = gym.make(env_name, render_mode="human", continuous = False)
    testing_loop(env=env, model=model, MULTIFRAME=MULTIFRAME, n_episodes=n_test_episodes, device=device)
    env.close()

if __name__ == '__main__':
    main()
