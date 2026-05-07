import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import numpy as np

class Linear_Regression(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(Linear_Regression, self).__init__()
        k = 4096
        self.fc1 = nn.Linear(input_shape, k)
        self.fc2 = nn.Linear(k, k)
        self.fc3 = nn.Linear(k, out_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, np.sqrt(1.5 / k)) # 权重初始化为 N(0, 2/width)  100/4096
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.1)  # 偏置初始化为 N(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Logistic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logit = self.layer(x)
        return logit


class TwoHiddenLayerFc(nn.Module): #output=10
    def __init__(self, input_shape, out_dim):
        k = 128
        super(TwoHiddenLayerFc, self).__init__()
        self.fc1 = nn.Linear(input_shape, k)
        self.fc2 = nn.Linear(k, k)
        self.readout = nn.Linear(k, out_dim)


    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.readout(out)
        return out

class TwoHiddenLayerFcCifar10(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(TwoHiddenLayerFcCifar10, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, out_dim)

    def forward(self, x):
        # x = x.view(-1, 32*32) #用于cifar10数据集二分类问题
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class LeNet(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*5*5, 394)
        self.fc2 = nn.Linear(394, 192)
        self.readout = nn.Linear(192, out_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.readout(out)
        return out




class CifarCnn(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(CifarCnn, self).__init__()
        k=64
        self.conv1 = nn.Conv2d(input_shape[0], 16 * k, 3)
        self.conv2 = nn.Conv2d(16 * k, 16 * k, 3)
        self.fc1 = nn.Linear(16 * 6 * 6 * k, out_dim)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out



def choose_model(options):
    model_name = str(options['model']).lower()

    # ========= MLP / Linear 类模型 =========
    if model_name == '2nnc':
        return TwoHiddenLayerFcCifar10(options['input_shape'], options['num_class'])
    if model_name == 'linear_regression':
        return Linear_Regression(options['input_shape'], options['num_class'])
    elif model_name == '2nn':
        return TwoHiddenLayerFc(options['input_shape'], options['num_class'])
    elif model_name == 'ccnn':
        return CifarCnn(options['input_shape'], options['num_class'])
    elif model_name == 'lenet':
        return LeNet(options['input_shape'], options['num_class'])

    # ========= FedETF 专用模型 =========
    elif model_name.startswith('resnet18_etf'):
        # FedETF 版本 ResNet18 单独放在一个文件
        mod = importlib.import_module('src.models.resnet_etf')
        resnet_model = getattr(mod, 'resnet18_etf')
        return resnet_model(options['num_class'])

    # ========= FedUV 专用模型 =========
    elif model_name.startswith('resnet18_uv'):
        # FedUV 版本 ResNet18 放在 resnet_uv.py
        mod = importlib.import_module('src.models.resnet_uv')
        resnet_model = getattr(mod, 'resnet18_uv')  # 注意名称大小写一致
        return resnet_model(num_classes=options['num_class'])

    elif model_name.startswith('resnet50'):
        # FedUV 版本 ResNet18 放在 resnet_uv.py
        mod = importlib.import_module('src.models.resnet_50')
        resnet_model = getattr(mod, 'resnet50')  # 注意名称大小写一致
        return resnet_model(num_classes=options['num_class'])

    # ========= 普通 ResNet 系列模型 =========
    elif model_name.startswith('resnet18'):
        mod = importlib.import_module('src.models.resnet')
        resnet_model = getattr(mod, model_name)
        return resnet_model(options['num_class'])

    else:
        raise ValueError("Not support model: {}!".format(model_name))

