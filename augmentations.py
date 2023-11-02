import torch
from torch import nn
from torchvision import transforms


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
    # TODO: Fix this, this doesn't work
    def __init__(self, ratio=0.2):
        super().__init__()

    def forward(self, hidden_state):
        # Generate a mask
        pass

