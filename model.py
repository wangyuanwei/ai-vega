import torch
from torch import nn
import torchvision.models as models
from config import Common, Train

# 引入rest50模型
net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

class WeatherModel(nn.Module):
    def __init__(self, net):
        super(WeatherModel, self).__init__()
        # resnet50
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1000, Common.label_num)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        return x


model = WeatherModel(net)
