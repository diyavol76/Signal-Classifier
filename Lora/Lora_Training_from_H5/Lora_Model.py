import torch
from torch import nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class Network(nn.Module):
    def __init__(self, emb_dim=128):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

class Lora_Network(nn.Module):
    def __init__(self, emb_dim=512,input_bins=1):
        super(Lora_Network, self).__init__()
        self.conv1 =nn.Conv2d(input_bins,32,7,stride=2)

        self.conv2 = nn.Conv2d(32,32,3,padding=(1))

        self.conv3 = nn.Conv2d(32,64,3,padding=(1))

        self.conv4 = nn.Conv2d(64,64,3,padding=(1))

        self.conv5 = nn.Conv2d(32,64,1,padding=(0))

        self.fc1 = nn.Linear(64 * 4 * 7 * 7 * 19, emb_dim)

        self.pool = nn.MaxPool2d(2, 2)



        self.residual = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):


        x = F.relu(self.conv1(x))

        residual1 = x

        x = F.relu(self.conv2(x))
        x = self.conv2(x)
        x+=residual1
        x = F.relu(x)

        residual2 = x

        x = F.relu(self.conv2(x))
        x = self.conv2(x)
        x+=residual2
        x = F.relu(x)

        residual3 = x

        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        residual3 = self.conv5(residual3)
        x+=residual3
        x=F.relu(x)


        residual4 = x

        x = F.relu(self.conv4(x))
        x = self.conv4(x)
        x+=residual4
        x = F.relu(x)

        x = x.view(-1, 64 * 4 * 7*7*19)
        x = self.fc1(x)


        #x = nn.functional.normalize(x)
        return x

