import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if GPU is to be used
RENDER_FPS = 150
NUM_EPISODES = 200

# MEM_SIZE is the size of the ReplayMemory buffer
MEM_SIZE = 10000

# BATCH_SIZE is the number of transitions sampled from the replay buffer
BATCH_SIZE = 64

# GAMMA is the discount factor of long-term reward
GAMMA = 0.8

# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1000

# TAU is the update rate of the target network
TAU = 0.005

# LR is the learning rate of the ``AdamW`` optimizer
LR = 1e-4
