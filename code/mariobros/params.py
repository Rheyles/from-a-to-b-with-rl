# Agent tweaks
MAX_STEPS = 5_000_000 # Note : if N_IDLE is not 1, we will collect info only on some of these steps
N_IMGS = 3 # Number of successive images that will make up the observation
N_IDLE = 3 # Number of successive frames / steps during which agent will keep the same action (and ignore the next states) but still reap the rewards
N_START_SKIP = 3 # Number of frames to skip at the beginning of the episode / please choose it to be higher than N_IMGS !
STAGES=['1-1', '2-1', '3-1', '3-2', '4-1','5-1', '6-1', '7-1'] # Mario levels to be played

# Neural Network parameters
GAMMA = 0.95 # GAMMA is the discount factor of long-term reward
N_FILTERS = 48 # The size of the neural network
DROPOUT_RATE = 0
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
L2_ALPHA = 0
SCHEDULER_PATIENCE = 200
SCHEDULER_MIN_LR = 3e-5
SCHEDULER_FACTOR = 0.5


# PPO Parameters
STEPS_TO_UPDATE = 2000 # Update networks every xxxx steps
UPDATE_EPOCHS = 5 # Number of epochs to use when learning and updating the policy
ENTROPY_BETA = 0.001
PPO_CLIP_VAL = 0.2
MINIBATCH_SIZE = 64 # Size of the mini batches used during optimization step (BUFFER_SIZE / MINIBATCH_SIZE steps will be taken to make one epoch)
