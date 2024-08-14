
import torch.nn as nn
import torch
import numpy as np

class ConvNet(nn.Module):

    def __init__(self, obs:torch.Tensor, n_filters=16):

        super(ConvNet, self).__init__()
        img_shape = obs.shape[-2:] # Dim 0 will be reserved for the batch
        n_imgs = obs.shape[-3]

        k1 = 3
        s1 = 2
        img_shape1 = [(1 + (elem - k1) // (s1)) // 2 for elem in img_shape]
        img_shape2 = [elem // 2 for elem in img_shape]
        
        print(img_shape1)
        
        self.conv1 = nn.Conv2d(n_imgs, n_filters, kernel_size=k1, stride=s1)
        self.mp = nn.MaxPool2d(kernel_size=2)

    def forward(self, x) -> torch.Tensor:

        x = self.conv1(x)
        print(x.shape)
        x = self.mp(x)
        print(x.shape)
        return x

obs = torch.zeros((7,9)).unsqueeze(0)
t = ConvNet(obs=obs)

print(obs.shape)
print(t.forward(obs).shape[1:])

