import random
import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
from collections import namedtuple, deque
from params import DEVICE

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*random.sample(self.memory, batch_size))) # Needs to pass this from buffer class

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        state_batch = torch.cat(batch.state).to(DEVICE)
        action_batch = torch.cat(batch.action).to(DEVICE)
        next_state_batch = torch.cat(batch.next_state).to(DEVICE)
        reward_batch = torch.cat(batch.reward).to(DEVICE)
        not_done_batch = torch.cat(batch.not_done).to(DEVICE)

        # print(state_batch.shape)
        # print(action_batch.shape)
        # print(reward_batch.shape)
        # print(not_done_batch.shape)

        return state_batch, action_batch, next_state_batch, reward_batch, not_done_batch

    def __len__(self):
        return len(self.memory)




class TorchMemory():
    """ Maybe a more optimized memory based on PytorchRL 
    package (not big I promise). Has retro-compatibility
    with the old (' legacy ' ) memory. 
    NOTE : Thought 
    it would be stored directly on GPU, but in the end
    it is not possible"""

    def __init__(self, capacity):
        self.memory = TensorDictReplayBuffer(
            storage = LazyMemmapStorage(
                max_size=capacity,
                device='cpu'))

    def push(self, *args):
        self.memory.add(TensorDict({
            "state": args[0],
            "action": args[1],
            "next_state": args[2],
            "reward": args[3],
            "not_done": args[4],
        }, batch_size=[]))

    def sample(self, batch_size):
        batch = self.memory.sample(batch_size)
        state_batch = batch.get('state').squeeze(1).type(torch.FloatTensor).to(DEVICE)
        action_batch = batch.get('action').squeeze(1).to(DEVICE)
        next_state_batch = batch.get('next_state').squeeze(1).type(torch.FloatTensor).to(DEVICE)
        reward_batch = batch.get('reward').squeeze(1).to(DEVICE)
        not_done_batch = batch.get('not_done').squeeze(1).to(DEVICE)

        # print(state_batch.shape)
        # print(action_batch.shape)
        # print(reward_batch.shape)
        # print(not_done_batch.shape)

        return state_batch, action_batch, next_state_batch, reward_batch, not_done_batch
    
    def __len__(self):
        return self.memory.__len__()