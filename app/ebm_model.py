import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 2 * 2, 1)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.leaky_relu(self.conv3(h), 0.2)
        h = F.leaky_relu(self.conv4(h), 0.2)
        return self.fc(h.flatten(1))


def langevin_sample(model, x, steps=40, step_size=0.1):
    model.eval()
    x = x.clone().detach()
    for _ in range(steps):
        x.requires_grad_(True)
        energy = model(x).sum()
        grad = torch.autograd.grad(energy, x)[0]
        with torch.no_grad():
            x = x - 0.5 * step_size * grad + \
                step_size ** 0.5 * torch.randn_like(x)
            x = x.clamp(-1, 1).detach()
    model.train()
    return x
