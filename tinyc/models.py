"""Small benchmark models. They deliberately contain things a compiler should clean up:
BatchNorm after Conv (foldable), a constant sub-expression, a dead branch, and elementwise chains."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in=784, h=512, d_out=10):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(d_in, h), nn.Linear(h, h), nn.Linear(h, d_out)
        self.aux = nn.Linear(h, h)                         # dead branch (result unused)
        self.alpha = nn.Parameter(torch.tensor(0.5))       # learned residual gate

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        r = h
        dead = self.aux(h)                                  # noqa: F841  -> removed by DCE
        h = self.fc2(h)
        h = torch.relu(h * (torch.sigmoid(self.alpha) * 2.0) + r)  # const subexpr + mul/add/relu + residual
        return self.fc3(h)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16)
        self.c2, self.b2 = nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc1, self.fc2 = nn.Linear(32 * 7 * 7, 128), nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.b1(self.c1(x))))
        x = self.pool(F.relu(self.b2(self.c2(x))))
        x = torch.tanh(x * 0.5 + 0.1)                     # elementwise chain after a pool -> one fused loop
        x = torch.flatten(x, 1)
        return self.fc2(torch.sigmoid(self.fc1(x)) * 2.0 - 1.0)


def randomize_bn(m: nn.Module, seed=0):
    """Give BatchNorm non-trivial running stats so folding is actually exercised."""
    g = torch.Generator().manual_seed(seed)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.copy_(torch.randn(mod.num_features, generator=g) * 0.1)
            mod.running_var.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
            mod.weight.data.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
            mod.bias.data.copy_(torch.randn(mod.num_features, generator=g) * 0.1)
    return m.eval()


MODELS = {"mlp": (MLP, (784,)), "cnn": (SmallCNN, (1, 28, 28))}
