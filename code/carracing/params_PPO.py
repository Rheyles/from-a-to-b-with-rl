# Agent tweaks
MAX_STEPS = 10000000 # Note : if N_IDLE is not 1, we will collect info only on some of these steps
N_IMGS = 3 # Number of successive images that will make up the observation
N_IDLE = 2 # Number of successive frames / steps during which agent will keep the same action (and ignore the next states) but still reap the rewards
N_START_SKIP = 30 # Number of frames to skip at the beginning of the episode / please choose it to be higher than N_IMGS !

# Neural Network parameters
GAMMA = 0.95 # GAMMA is the discount factor of long-term reward
N_FILTERS = 16 # The size of the neural network
DROPOUT_RATE = 0
ACTOR_LR = 5e-4
CRITIC_LR = 1e-3
L2_ALPHA = 0
SCHEDULER_PATIENCE = 200
SCHEDULER_MIN_LR = 1e-5
SCHEDULER_FACTOR = 0.5

# PPO Parameters
BUFFER_SIZE = 5000 # Update networks every xxxx steps
UPDATE_EPOCHS = 7 # Number of epochs to use when learning and updating the policy
ENTROPY_BETA = 0.001
PPO_CLIP_VAL = 0.2