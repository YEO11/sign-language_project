import torch
import torch.nn as nn
import torchvision.models as models

class TSMModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TSMModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.temporal = nn.Conv1d(512, 512, kernel_size=3, padding=1, groups=512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.backbone(x)  # (B*T, 512)
        x = x.view(B, T, -1).permute(0, 2, 1)  # (B, 512, T)
        x = self.temporal(x)  # (B, 512, T)
        x = x.mean(dim=2)  # temporal avg pool
        x = self.classifier(x)
        return x
