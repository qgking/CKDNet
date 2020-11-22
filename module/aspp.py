import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation, batchnorm):

        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding,
                                     dilation=dilation, bias=False)
        self.bn = batchnorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, backbone, output_stride, batchnorm):

        super(ASPP, self).__init__()
        if backbone == 'vgg16bn':
            inplanes = 512
        elif backbone == 'resnet101' or backbone == 'resnet152':
            inplanes = 2048
        elif backbone == 'resnet34':
            inplanes = 512
        else:
            raise Exception('Unknown backbone')

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], batchnorm=batchnorm)
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], batchnorm=batchnorm)
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], batchnorm=batchnorm)
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], batchnorm=batchnorm)

        self.global_average_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                                 nn.ReLU())
        self.bn_global_average_pool = batchnorm(256)

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = batchnorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_average_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x5 = self.bn_global_average_pool(x5)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SegASPPDecoder(nn.Module):
    def __init__(self, num_classes, backbone):
        super(SegASPPDecoder, self).__init__()
        if backbone == 'resnet101' or backbone == 'resnet152':
            low_level_inplanes = 256
        elif backbone == 'vgg16bn':
            low_level_inplanes = 256
        elif backbone == 'resnet34':
            low_level_inplanes = 64
        else:
            raise Exception('Unknown backbone')

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()

        # aspp always gives out 256 planes + 48 from conv1
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout2d(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):

        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        second_to_last_features = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(second_to_last_features)
        return x, second_to_last_features

    def _init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    aspp = ASPP(backbone='mobilenet', output_stride=8, batchnorm=nn.BatchNorm2d)
    input = torch.rand(1, 320, 32, 32)
    output = aspp(input)
    print("O/P Size: ", output.size())
