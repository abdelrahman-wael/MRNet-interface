import torch.nn as nn
import torch
import torchvision.models as models
class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.alexnet(pretrained=True).features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported 
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x
