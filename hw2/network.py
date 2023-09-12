from torchsummary import summary
import torch
from torch import nn

net = nn.Sequential(nn.Conv2d(3, 8, 3, 1, 1),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2, 2, 0),
                    nn.BatchNorm2d(8),
                    nn.Conv2d(8, 16, 3, 1, 1),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2, 2, 0),
                    nn.Flatten(),
                    nn.Linear(1024, 10))
print(summary(net, (3, 32, 32)))