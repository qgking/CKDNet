import matplotlib

matplotlib.use("Agg")
import sys

sys.path.extend(["../../", "../", "./"])
from commons.utils import *
import torch
import random
from driver.gen_driver.ConfigGEN import Configurable
from driver.gen_driver.train_waterfall_helper import MUTUALHelper
import configparser
from models.seg_models.deeplab import DeepLab_Aux
import pandas as pd
from models import MODELS
from driver import std, mean
from data import ISIC2018, ISIC2017
import time

config = configparser.ConfigParser()
import argparse


def main(config):
    model_seg_coarse = DeepLab_Aux(num_classes=1, return_features=True)
    model_cls = MODELS[config.model](backbone=config.backbone, num_classes=config.cls_classes)

    criterion = {
    }
    mutual_helper = MUTUALHelper(model_cls, criterion,
                                 config)
    mutual_helper.move_to_cuda()
    model_seg_coarse.to(mutual_helper.equipment)
    print("data name ", mutual_helper.config.data_name)

    test_loader_mtl_aug = mutual_helper.get_cls_test_aug_data_loader()
    predict_aug_fun = {
        ISIC2017: predict_isic17_aug,
        ISIC2018: predict_isic18_aug
    }
    try:
        model_seg_coarse = mutual_helper.load_pretrained_coarse_seg_model(model_seg_coarse)
    except FileExistsError as e:
        raise ValueError('file not exist')
    for nb_acl_iter in range(mutual_helper.config.nb_active_learning_iter):
        if mutual_helper.config.load_best_epoch:
            print("\n-----------load best state of model -----------")
            mutual_helper.load_best_state(iter=nb_acl_iter)
        predict_aug_fun[mutual_helper.config.data_name](mutual_helper, model_seg_coarse, test_loader_mtl_aug,
                                                        nb_acl_iter)
        mutual_helper.log.flush()
        exit(0)

    print("\n----------- TRAINING FINISHED -----------")
    mutual_helper.summary_writer.close()



def predict_isic18_aug(mutual_helper, model_seg_coarse, test_loader_mtl_aug, ac_iter):
    model_seg_coarse.eval()
    mutual_helper.model.eval()
    y_names = []
    y_score = []
    with torch.no_grad():
        for ii, (images, names) in enumerate(test_loader_mtl_aug, start=1):
            images = images.to(mutual_helper.equipment)[0]
            images_cls_logits, backbone_out, _, _ = model_seg_coarse(images)
            probs = torch.sigmoid(images_cls_logits)
            predictions_cls = mutual_helper.model(images, probs, backbone_out)
            scores = torch.mean(predictions_cls, dim=0)
            score = F.softmax(torch.unsqueeze(scores, dim=0), dim=1)
            score = score.cpu().numpy()
            y_names.extend(names)
            y_score.extend(score)
    y_prob = np.array(y_score)
    mutual_helper.create_isic_2018_cls_submit_csv(image_names=y_names, y_prob=y_prob, ac_iter=ac_iter, tag='aug')


def predict_isic17_aug(mutual_helper, model_seg_coarse, test_loader_mtl_aug, ac_iter):
    model_seg_coarse.eval()
    mutual_helper.model.eval()
    y = []
    y_score = []
    with torch.no_grad():
        for ii, (images, labels, _) in enumerate(test_loader_mtl_aug, start=1):
            images = images.to(mutual_helper.equipment)[0]
            if ii<= 10:
                start = time.time()
            images_cls_logits, backbone_out, _, _ = model_seg_coarse(images)
            probs = torch.sigmoid(images_cls_logits)
            predictions_cls = mutual_helper.model(images, probs, backbone_out)
            scores = torch.mean(predictions_cls, dim=0)
            score = F.softmax(torch.unsqueeze(scores, dim=0), dim=1)
            if ii<= 10:
                print('Time collapse for each img: %s' % (time.time() - start))
            score = score.cpu().numpy()
            label = labels.cpu().numpy()
            y.extend(label)
            y_score.extend(score)
    data_frame = pd.DataFrame({"y_score": np.argmax(y_score, axis=1), "y": y})
    data_frame.to_csv(join(mutual_helper.config.submission_dir, str(ac_iter) + '_arl_submit.csv'), index=False, sep=",")
    mutual_helper.evaluation_cls(y, y_score, ac_iter, type='arl')




if __name__ == '__main__':
    torch.manual_seed(6666)
    torch.cuda.manual_seed(6666)
    random.seed(6666)
    np.random.seed(6666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    torch.backends.cudnn.benchmark = True  # cudn
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='train_waterfall_cls.txt')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=False)

    args, extra_args = argparser.parse_known_args()
    file = 'configuration.txt'
    root = '../../log/ISIC2017_isicgen/Integrate_Model_Cls_Ensemble_CAM_Att_v1/none/budget_1/gen_0_run_0'
    config = Configurable(join(root, file), extra_args, isTrain=args.train)
    # config = Configurable(args.config_file, extra_args, isTrain=args.train)
    torch.set_num_threads(config.workers + 1)

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
