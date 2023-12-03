import torch
from torch import nn
from torchvision import transforms

import random
import math


class GaussianNoise(nn.Module):
    """
    Add gaussian noise to hidden state. Noises are sampled with mean=0 and variance of a specified ratio of the
    variance of the corresponding channel in the hidden state.
    """

    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, hidden_state):
        shape = hidden_state.shape
        device = hidden_state.device
        mean = torch.zeros(shape, device=device)
        var = torch.var(hidden_state, (2, 3), keepdim=True) * self.ratio
        noise = torch.normal(mean, torch.sqrt(var))
        return hidden_state + noise


class RandomResizedCrop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_state):
        shape = hidden_state.shape
        return transforms.RandomResizedCrop(shape[-2:])(hidden_state)


class RandomRotation(nn.Module):
    def __init__(self, degree=30):
        super().__init__()
        self.degrees = degree

    def forward(self, hidden_state):
        return transforms.RandomRotation(self.degree)(hidden_state)


class RandomHorizontalFlip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_state):
        return transforms.RandomHorizontalFlip()(hidden_state)


class GridDropout(nn.Module):
    """
    Divide the feature map (last two dimensions only) into grids and randomly drop some of them.
    """

    def __init__(self, ratio=0.15, num_grids=8 * 8):
        super().__init__()
        self.ratio = ratio
        assert math.sqrt(num_grids) == int(
            math.sqrt(num_grids)
        ), "num_grids must be a perfect square"
        self.num_grids = num_grids

    def forward(self, hidden_state):
        n = hidden_state.shape[-1]
        grid_size = n // int(math.sqrt(self.num_grids))
        device = hidden_state.device
        mask = torch.ones_like(hidden_state, device=device)
        num_zeros = int(self.num_grids * self.ratio)
        zero_indices = random.sample(range(self.num_grids), num_zeros)
        for idx in zero_indices:
            row = (idx // (n // grid_size)) * grid_size
            col = (idx % (n // grid_size)) * grid_size
            mask[:, :, row : row + grid_size, col : col + grid_size] = 0
        return hidden_state * mask


