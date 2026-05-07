import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ResNet18_UV(nn.Module):
    """
    完全兼容 FedUV 的 ResNet18：
    1. backbone -> avgpool -> flatten 得到 feature
    2. feature -> projector（2 层 MLP）
    3. projector 输出作为 encoder 表示（reps），供 hook 使用
    4. readout(logits) 用于分类
    """
    def __init__(self, num_classes=100, proj_dim=512):
        super(ResNet18_UV, self).__init__()
        self.in_planes = 64

        # ---------- Standard CIFAR version ----------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ---------- Projector（论文要求） ----------
        self.projector = nn.Sequential(
            nn.Linear(512, proj_dim)
        )

        # ---------- Readout（最终分类器） ----------
        self.readout = nn.Linear(proj_dim, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)       # [B, 512]

        # encoder (FedUV 所需)
        reps = self.projector(x)      # [B, proj_dim]

        # classifier
        logits = self.readout(reps)

        return logits


def resnet18_uv(num_classes=100):
    return ResNet18_UV(num_classes)
