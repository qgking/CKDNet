from module.gen_backbone import resnet, vgg

BACKBONE = {
    'vgg16bn': vgg.vgg16_bn,
    'vgg16': vgg.vgg16,
    'vgg19bn': vgg.vgg19_bn,
    'vgg19': vgg.vgg19,
    'resnet101': resnet.ResNet,
    # 'resnet50': resnet.ResNet,
    # 'resnet152': resnet.ResNet,
    # 'resnet34': resnet.ResNet,
}
