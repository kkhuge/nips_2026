import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ProtoClassifier(nn.Module):
    """
    ETF prototype classifier for FedETF.
    """
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Build the ETF prototype matrix.
        self.register_buffer("proto", self.build_etf(feature_dim, num_classes))

        # Override the build_etf method in resnet_etf.py.

    def build_etf(self, d, c):
        # Key change: fix the seed so every client's ETF matrix has the same spatial orientation [cite: 167].
        # Otherwise the feature extractor may not converge during server aggregation because targets conflict.
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)

        # Build the centered simplex matrix M.
        M = torch.eye(c) - torch.ones(c, c) / c
        # Build the orthogonal matrix Q.
        A = torch.randn(d, d)
        Q, _ = torch.linalg.qr(A)
        Q = Q[:, :c]
        # Compute V = sqrt(c/(c-1)) * Q * M [cite: 142].
        V = math.sqrt(c / (c - 1)) * torch.matmul(Q, M)
        # Normalize each column [cite: 171].
        V = F.normalize(V, dim=0)

        torch.set_rng_state(rng_state)  # Restore the original random state.
        return V

    def forward(self, feature):
        """
        feature: [batch, feature_dim], already L2-normalized.
        proto:   [feature_dim, num_classes]
        """
        feature = F.normalize(feature, dim=1)
        logits = torch.matmul(feature, self.proto)  # [B, C]
        return logits


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_ETF(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Backbone output dimension = 512.
        self.linear_proto = nn.Linear(512, 128)      # Generate normalized features.
        self.proto_classifier = ProtoClassifier(128, num_classes)  # ETF prototype layer. ##16,128



        # Scaling can be learnable or fixed.
        self.scaling_train = nn.Parameter(torch.tensor(12.0))

    def _make_layer(self, block, planes, block_num, stride):
        strides = [stride] + [1]*(block_num-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        embedding = torch.flatten(x, 1)  # [B, 512]

        # FedETF features.
        feature = self.linear_proto(embedding)
        feature = F.normalize(feature, dim=1)

        # ETF logits
        logits_etf = self.proto_classifier(feature) * self.scaling_train


        return logits_etf, feature


def resnet18_etf(num_classes=100):
    return ResNet18_ETF(num_classes)
