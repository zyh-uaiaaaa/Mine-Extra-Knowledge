
import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import trunc_normal_
from torch.autograd import Variable


class SwinTransFER(torch.nn.Module):

    def __init__(self, swin, swin_num_features=768, num_classes=7, cam=True):
        super().__init__()
        self.encoder = swin
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(swin_num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(swin_num_features, num_classes)
        self.cam = cam


    def forward(self, x):

        x = self.encoder.forward_features(x)
        x = self.norm(x)  # B L C

        feature = self.avgpool(x.transpose(1, 2))  # B C 1
        feature = torch.flatten(feature, 1)
        output = self.head(feature)

        if self.cam:

            fc_weights = self.head.weight
            fc_weights = fc_weights.view(1, self.num_classes, 768, 1, 1)
            fc_weights = Variable(fc_weights, requires_grad = False)

            # attention
            B, L, C = x.shape
            feat = x.transpose(1, 2).view(B, 1, C, 7, 7) # N * 1 * C * H * W
            hm = feat * fc_weights
            hm = hm.sum(2) # N * self.num_labels * H * W
            return output, hm
        
        else:
            return output

