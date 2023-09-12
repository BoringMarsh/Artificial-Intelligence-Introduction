import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # 均匀分布的初始化器
                nn.init.constant_(m.bias, 0)  # 用0填充bias

    def forward(self, x):  # 向前传播函数
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x