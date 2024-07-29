NUM_EPISODES = 2000
OBS_AGENT = False # Whether the agent "sees" its environment
MAP_NAME = '4x4' # 4x4, 5x5_easy, 5x5_hard, 8x8
NEG_REWARD_LAKE = True

# Neural Network parameters
GAMMA = 0.9 # GAMMA is the discount factor of long-term reward
NN_SIZE = 16 # The size of the neural network

# DQN-specific parameters
DQN_EPS_START = 1 # EPS_START is the starting value of the exploration rate
DQN_EPS_END = 0.05 # EPS_END is the final value of exploration rate
DQN_EPS_DECAY = 5000 # EPS_DECAY is the typical number of steps to reduce epsilon (characteristic time of the exponential decay)
DQN_MEM_SIZE = 5000 # DQN_MEM_SIZE is the size of the ReplayMemory buffer
DQN_MEM_BATCH_SIZE = 64 # and we sample DQN_MEM_BATCH_SIZE from this buffer every step
DQN_NETWORK_REFRESH_STRATEGY = 'hard' # tells you if you either progressively replace your target net (mixing itself with tau * policy_net) or if you do it completely every 1/TAU steps
DQN_TAU = 0.02 # TAU is the update rate of the target network
DQN_LR = 0.001 # LR is the learning rate of the optimizer
DQN_L2 = 0 # L2 regularization factor of the DQN Network
DQN_GRADIENT_CLIP = 100 # How large the gradients (with respect to network parameters) can be before they are clipped (used to avoid too sudden changes in policy network)
# Note : in Frozen Lake, the rewards do not need to be normalized
# Note : I chosse an AdamW optimizer
