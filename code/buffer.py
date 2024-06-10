import random
import torch
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from collections import namedtuple, deque
from params import DEVICE, BATCH_SIZE

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayMemory(object):
    '''Legacy memory class. It has the advantage of storing data on the GPU
    (might clog its memory), which might not be possible with the Pytorch
    memory.'''

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Pushes individual step data to the Transition (legacy) memory

        Args (in order !):
            - state (torch.Tensor, should be (dims_of_observation))
            - action (torch.Tensor, should be 1 or 1x1)
            - next_state (torch.Tensor of same dims as state)
            - reward (torch.Tensor of size 1 or 1x1)
            - not_done (torch.Tensor of size 1 or 1x1)

        Returns: None
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size :int):
        """Takes a batch_sample from the Pytorch memory

        Args:
            batch_size (int): size of the batch you are using.
            Should probably be equal to the global param BATCH_SIZE

        Returns:
            - state_batch [batch_size x (dims of observation) torch.Tensor]
            - action_batch [batch_size x 1 torch.Tensor]
            - next_state_batch [same size as state_batch]
            - reward_batch [batch_size x 1 torch.Tensor]
            - not_done_batch [batch_size x 1 torch.Tensor]
        """

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
        if state_batch.ndim == 1: state_batch = state_batch.unsqueeze(-1)
        if next_state_batch.ndim == 1: next_state_batch = next_state_batch.unsqueeze(-1)

        return state_batch, action_batch, next_state_batch, reward_batch, not_done_batch

    def __len__(self):
        return len(self.memory)


class TorchMemory():
    """ Maybe a more optimized memory based on PytorchRL
    package (not big I promise). Has retro-compatibility
    with the old (' legacy ' ) memory. NOTE : Thought
    it would be stored directly on GPU, but in the end
    it is not possible. Has the same functionalities as
    the regular memory"""

    def __init__(self, capacity):
        self.memory = TensorDictReplayBuffer(
            storage = LazyTensorStorage(
                device=DEVICE,
                max_size=capacity),
            batch_size=BATCH_SIZE)

    def push(self, *args):
        """Pushes individual step data to the Pytorch memory

        Args (in order !):
            - state (torch.Tensor, should be (dims_of_observation))
            - action (torch.Tensor, should be 1 or 1x1)
            - next_state (torch.Tensor of same dims as state)
            - reward (torch.Tensor of size 1 or 1x1)
            - not_done (torch.Tensor of size 1 or 1x1)

        Returns: None
        """

        state = args[0].unsqueeze(-1) if args[0].ndim == 1 else args[0]
        next_state = args[2].unsqueeze(-1) if args[2].ndim == 1 else args[2]

        self.memory.add(TensorDict({
            "state": state.to('cpu'),
            "action": args[1].to('cpu'),
            "next_state": next_state.to('cpu'),
            "reward": args[3].to('cpu'),
            "not_done": args[4].to('cpu'),
        }, batch_size=[]))

    def sample(self, batch_size : int):
        """Takes a batch_sample from the Pytorch memory

        Args:
            batch_size (int): size of the batch you are using.
            Should probably be equal to the global param BATCH_SIZE

        Returns:
            - state_batch [batch_size x (dims of observation) torch.Tensor]
            - action_batch [batch_size x 1 torch.Tensor]
            - next_state_batch [same size as state_batch]
            - reward_batch [batch_size x 1 torch.Tensor]
            - not_done_batch [batch_size x 1 torch.Tensor]
        """
        batch = self.memory.sample(batch_size)
        state_batch = batch.get('state').squeeze(1).to(DEVICE)
        action_batch = batch.get('action').squeeze(1).to(DEVICE)
        next_state_batch = batch.get('next_state').squeeze(1).to(DEVICE)
        reward_batch = batch.get('reward').squeeze(1).to(DEVICE)
        not_done_batch = batch.get('not_done').squeeze(1).to(DEVICE)
        if state_batch.ndim == 1: state_batch = state_batch.unsqueeze(-1)
        if next_state_batch.ndim == 1: next_state_batch = next_state_batch.unsqueeze(-1)

        return state_batch, action_batch, next_state_batch, reward_batch, not_done_batch

    def __len__(self):
        return self.memory.__len__()
