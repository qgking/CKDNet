import torch
import torchvision.transforms as tvt
OPTIM = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = tvt.Compose([
    tvt.ToPILImage(),
    # no shear=[-10, 10],
    tvt.RandomAffine(0, translate=[0.01, 0.1], shear=[-10, 10],
                     scale=[0.7, 1.3]),
    # tvt.RandomAffine([-180, 180], translate=[0.1, 0.1], shear=[-10, 10],
    #                  scale=[0.7, 1.3]),
    # tvt.RandomRotation((-10, 10)),
    tvt.RandomRotation((-180, 180)),
    tvt.RandomHorizontalFlip(),
    tvt.RandomVerticalFlip(),
    tvt.RandomChoice([
        tvt.ColorJitter(brightness=0.1),
        tvt.ColorJitter(contrast=0.2),
        tvt.ColorJitter(saturation=0.1),
        tvt.ColorJitter(hue=0.15),
        tvt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        tvt.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
        tvt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ]),
    tvt.ToTensor(),
    tvt.Normalize(mean=mean,
                  std=std)
])

gt_transform = tvt.Compose([
    tvt.ToPILImage(),
    tvt.RandomAffine(0, translate=[0.01, 0.1], shear=[-10, 10],
                     scale=[0.7, 1.3]),
    # tvt.RandomAffine([-180, 180], translate=[0.1, 0.1], shear=[-10, 10],
    #                  scale=[0.7, 1.3]),
    # tvt.RandomRotation((-10, 10)),
    tvt.RandomRotation((-180, 180)),
    tvt.RandomHorizontalFlip(),
    tvt.RandomVerticalFlip(),
    tvt.ToTensor(),
])

active_transform = tvt.Compose([
    # tvt.ToPILImage(),
    # tvt.ColorJitter(0.02, 0.02, 0.02, 0.01),
    tvt.ToTensor(),
    tvt.Normalize(mean=mean,
                  std=std)
])
