from commons.utils import *
from tensorboardX import SummaryWriter
import torch
import matplotlib.pyplot as plt
from driver.base_train_helper import BaseTrainHelper
from driver import OPTIM
from grad_cam_resnet.utils import visualize_cam
from torchvision.utils import make_grid, save_image
from driver import mean, std
from torch.cuda import empty_cache
from data import ISIC2017, ISIC2018


class MUTUALHelper(BaseTrainHelper):
    def __init__(self, model, criterions, config):
        super(MUTUALHelper, self).__init__(model, criterions, config)
        print("Mutual Help")

    def out_put_summary(self):
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)
        # if not self.config.train:
        #     print(self.model)

    def reset_optim(self, branch='cls'):
        # optimizer = OPTIM[self.config.learning_algorithm](
        #     [{'params': self.model.backbone.parameters(), 'lr': self.config.learning_rate / 10},
        #      {'params': self.model.pcam.parameters(), 'lr': self.config.learning_rate},
        #      {'params': self.model.cls_branch.parameters(), 'lr': self.config.learning_rate}
        #      ],
        #     lr=self.config.learning_rate, weight_decay=1e-4)
        # if branch == 'cls':
        #         optimizer = OPTIM[self.config.learning_algorithm](self.model.parameters(),
        #                                                           lr=self.config.learning_rate,
        #                                                           weight_decay=1e-4)
        backbone_layer_id = [ii for m in self.model.get_backbone_layers() for ii in list(map(id, m.parameters()))]
        backbone_layer = filter(lambda p: id(p) in backbone_layer_id, self.model.parameters())
        rest_layer = filter(lambda p: id(p) not in backbone_layer_id, self.model.parameters())
        optimizer = OPTIM[self.config.learning_algorithm](
            [{'params': backbone_layer, 'lr': self.config.backbone_learning_rate},
             {'params': rest_layer, 'lr': self.config.learning_rate},
             ],
            lr=self.config.learning_rate,
            # momentum=0.9,
            weight_decay=1e-4)
        # return optimizer
        # elif branch == 'seg':
        #     if self.config.data_name == ISIC2017:
        #         optimizer = OPTIM[self.config.learning_algorithm](self.model.parameters(),
        #                                                           lr=self.config.learning_rate,
        #                                                           weight_decay=1e-4)
        #     if self.config.data_name == ISIC2018:
        #         optimizer = OPTIM[self.config.learning_algorithm](
        #             [{'params': self.model.backbone.parameters(), 'lr': self.config.learning_rate / 10},
        #              {'params': self.model.pcam.parameters(), 'lr': self.config.learning_rate},
        #              {'params': self.model.cls_branch.parameters(), 'lr': self.config.learning_rate}
        #              ],
        #             lr=self.config.learning_rate, weight_decay=1e-4)
        return optimizer

    def load_pretrained_coarse_seg_model(self, model_seg_coarse):
        # save_model_path = join(self.config.save_model_path, "coarse_seg_model.pt")
        # TODO change to upper code in formal env
        file_dir = '../../log/' + self.config.data_name + '_' + self.config.data_branch
        save_model_path = join(file_dir, "coarse_seg_model.pt")
        if not os.path.exists(save_model_path):
            raise FileExistsError('coarse seg model file %s not exits' % (save_model_path))
        else:
            print("loaded %s" % (save_model_path))
            weight_file = torch.load(save_model_path, map_location=('cuda:' + str(self.device)))
            model_seg_coarse.load_state_dict(weight_file)
        return model_seg_coarse

    def load_pretrained_cls_model(self, model_cls):
        file_dir = '../../log/' + self.config.data_name + '_' + self.config.data_branch
        save_model_path = join(file_dir, "Integrate_Model_Cls_Ensemble_CAM_Att.pt")
        if not os.path.exists(save_model_path):
            raise FileExistsError('cls model file %s not exits' % (save_model_path))
        else:
            print("loaded %s" % (save_model_path))
            weight_file = torch.load(save_model_path, map_location=('cuda:' + str(self.device)))
            model_cls.load_state_dict(weight_file)
        return model_cls
