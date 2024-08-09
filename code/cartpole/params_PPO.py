# Agent tweaks
MAX_STEPS = 150000

# Neural Network parameters (common for actor and critic I am lazy)
GAMMA = 0.99 # GAMMA is the discount factor of long-term reward
DROPOUT_RATE = 0
L2_ALPHA = 0
SCHEDULER_PATIENCE = 200
SCHEDULER_MIN_LR = 1e-5
SCHEDULER_FACTOR = 0.5

# Actor thingies
ACTOR_SIZE = 6
ACTOR_LR = 2e-3

# Critic thingies
CRITIC_SIZE = 6
CRITIC_LR = 5e-3

# PPO Parameters
BUFFER_SIZE = 300 # When buffer is full, we update networks
UPDATE_EPOCHS = 7 # Number of epochs to use when learning and updating the policy
PPO_CLIP_VAL = 0.2
ENTROPY_BETA = 1e-4
