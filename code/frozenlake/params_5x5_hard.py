NUM_EPISODES = 3001
OBS_AGENT = False # Whether the agent "sees" its environment
MAP_NAME = '5x5_hard' # 4x4, 5x5_easy, 5x5_hard, 8x8
NEG_REWARD_LAKE = False

# Neural Network parameters
GAMMA = 0.8 # GAMMA is the discount factor of long-term reward
NN_SIZE = 32 # The size of the neural network

# DQN-specific parameters
DQN_EPS_START = 1 # EPS_START is the starting value of the exploration rate
DQN_EPS_END = 0.05 # EPS_END is the final value of exploration rate
DQN_EPS_DECAY = 3000 # EPS_DECAY is the typical number of steps to reduce epsilon (characteristic time of the exponential decay)
DQN_MEM_SIZE = 2000 # DQN_MEM_SIZE is the size of the ReplayMemory buffer
DQN_MEM_BATCH_SIZE = 16 # and we sample DQN_MEM_BATCH_SIZE from this buffer every step
DQN_NETWORK_REFRESH_STRATEGY = 'soft' # tells you if you either progressively replace your target net (mixing itself with tau * policy_net) or if you do it completely every 1/TAU steps
DQN_TAU = 0.01 # TAU is the update rate of the target network
DQN_LR = 1e-3 # LR is the initial learning rate of the optimizer
DQN_L2 = 0 # L2 regularization factor of the DQN Network
DQN_SCHEDULER_FACTOR = 0.7 # The scheduler will reduce the learning rate when the model gets better
DQN_SCHEDULER_PATIENCE = 500 # Number of steps (updates) before the scheduler acts
DQN_SCHEDULER_MIN_LR = 3e-5 # The minimum learning rate that the scheduler can set

# Note : in Frozen Lake, the rewards do not need to be normalized
# Note : I chosse an AdamW optimizer
