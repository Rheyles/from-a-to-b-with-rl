import torch


NUM_EPISODES = 5000

# Logging and whatnot -----------------------------------------------------------------

RECORD_VIDEO = False # Supersedes RENDER_MODE !
RENDER_MODE = 'human' # 'human' or 'rgb_array' or 'state_pixels'
RENDER_FPS = 150
LOG_EVERY = 500 # Write Buffered Log info every XXX steps
SAVE_EVERY = 100
DISPLAY_EVERY = 50 # Display / update optimization graphs every XXX steps


# Game parameter -----------------------------------------------------------------
# IDLENESS is the amount of frames during which the agent chooses the same action
FROZEN_MAP_NAME = '4x4'     # NOTE: ONLY FOR FROZEN_LAKE :  lake_4x4, lake_5x5_easy, lake_5x5_hard, lake_6x6_easy, lake_6x6_hard, lake_8x8
EARLY_STOPPING_SCORE = -20  # The score under which we trigger an early stopping of the episode
MULTIFRAME = 1              # NOTE: ONLY FOR CAR_RACE // MULTIFRAME is the amount of frames you feed into the network // NOTE : HAS TO BE SET TO 1 if the method run_episode is called in the main
IDLENESS = 1                # NOTE : ONLY FOR CAR_RACE

# Everything related to the Neural Network --------------------------------------------
# TAU is the update rate of the target network
# NETWORK REFRESH STRATEGY ('hard', 'soft') tells you if you either
# progressively replace your target net (mixing itself with tau * policy_net)
# or if you do it completely every 1/TAU steps
NETWORK = 'LinearDQN' 
           # ******* FROZEN_LAKE *******
           # LinearDQN, LinearA2C
           # ******* CAR RACE **********
           # ConvDQN_2layers_classic (from Manu), ConvDQN_2layers_small (from https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN/)
           # ConvDQN_3layers_classic (what we used in week 1), ConvDQN_3layers_small,
           # ConvA2C
NETWORK_REFRESH_STRATEGY = 'soft'
TAU = 0.02

# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 2500

# MEM_SIZE is the size of the ReplayMemory buffer
MEM_TYPE = 'torch' # 'Torch' or 'Legacy' 
MEM_SIZE = 5000

# BATCH_SIZE is the number of transitions sampled from the replay buffer
BATCH_SIZE = 32

# GAMMA is the discount factor of long-term reward
GAMMA = 0.7

# Everything related to the optimizer -----------------------------------------------
LOSS = 'HUBER'   # HUBER, MSE, MAE
OPTIMIZER = 'ADAMW' # ADAM, RMSPROP, ADAMW
REGULARIZATION = 0 # L2 Regularization coefficient (L1 and ElasticNet not simply implemented ...)
DROPOUT_RATE = 0
INI_LR = 1e-3   # LR is the learning rate of the optimizer
MIN_LR = 1e-3   # How low it can (will) get due to the optimizer
SCHEDULER_FACTOR = 0.7 
SCHEDULER_PATIENCE = 50 # Patience of the scheduler that decrease the learning rate in the car race env



# Leave this alone unless you know what you are doing
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        assert(torch.backends.mps.is_available())
        DEVICE = torch.device("mps")
    except:
        DEVICE = torch.device('cpu')