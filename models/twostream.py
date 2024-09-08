import torch
from torch import nn

from models.slowfastnet import resnet50
from models.pytorch_i3d import InceptionI3d


class TwoStreamNet(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamNet, self).__init__()
        self.slowfast = resnet50(class_num=num_classes, dropout=.5)
        self.i3d = InceptionI3d(num_classes=num_classes, in_channels=2)
        self.fc_fusion = nn.Linear(2 * num_classes, num_classes)

    def forward(self, x_rgb, x_flow):
        rgb_features = self.slowfast(x_rgb)
        flow_features = self.i3d(x_flow)
        flow_features = torch.mean(flow_features, dim=2)

        combined_features = torch.cat((rgb_features, flow_features), dim=1)
        fused_features = self.fc_fusion(combined_features)

        return fused_features
