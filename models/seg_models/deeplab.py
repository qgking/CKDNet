# -*- coding: utf-8 -*-
# @Time    : 19/12/10 11:57
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    :
import torch.nn as nn
from models.gen_models.seg_branch import DeepLabDecoder
from module.gen_backbone import BACKBONE


class DeepLab_Aux(nn.Module):
    def __init__(self, backbone='resnet101', num_classes=1, return_features=False):
        super(DeepLab_Aux, self).__init__()
        self.return_features = return_features
        self.backbone = BACKBONE[backbone](backbone=backbone, pretrained=True)
        self.seg_branch = DeepLabDecoder(backbone=backbone, num_class=num_classes, return_features=return_features)

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        return small_lr_layers

    def forward(self, image, branch='cls'):
        backbone_out = self.backbone(image)
        if self.return_features:
            seg_out, features, aspp = self.seg_branch(backbone_out)
            return seg_out, backbone_out, features, aspp
        seg_out = self.seg_branch(backbone_out)
        return seg_out, backbone_out, None, None

