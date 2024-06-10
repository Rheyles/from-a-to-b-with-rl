import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        assert(torch.backends.mps.is_available())
        DEVICE = torch.device("mps")
    except:
        DEVICE = torch.device('cpu')

RENDER_MODE = 'human' # or 'rgb_array' or 'state_pixels'
RENDER_FPS = 150
NUM_EPISODES = 5000
DISPLAY_EVERY = 100 # Display / update optimization graphs every XXX steps
LOG_EVERY = 500 # Log info every XXX steps
RECORD_VIDEO = False

# MEM_SIZE is the size of the ReplayMemory buffer
MEM_SIZE = 10000

# BATCH_SIZE is the number of transitions sampled from the replay buffer
BATCH_SIZE = 32

# GAMMA is the discount factor of long-term reward
GAMMA = 0.8

# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 10000

# TAU is the update rate of the target network
# NETWORK REFRESH STRATEGY ('hard', 'soft') tells you if you either
# progressively replace your target net (mixing itself with tau * policy_net)
# or if you do it completely every 1/TAU steps
NETWORK_REFRESH_STRATEGY = 'soft'
TAU = 0.003

# LR is the learning rate of the optimizer
LOSS = 'MSE'   # HUBER, MSE, MAE
OPTIMIZER = 'RMSPROP' # ADAM, RMSPROP, ADAMW
INI_LR = 5e-4
MIN_LR = 5e-4
# IDLENESS is the amount where agent choose the same action
IDLENESS = 3

#MULTIFRAME is the amount of frames you feed into the network
# HAS TO BE SET TO 1 if the method run_episode is called in the main
MULTIFRAME = 1

# Patience of the scheduler that decrease the learning rate in the car race env
SCHEDULER_FACTOR = 0.7
SCHEDULER_PATIENCE = 500
