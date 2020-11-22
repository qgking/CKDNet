import matplotlib

matplotlib.use("Agg")
import sys

sys.path.extend(["../../", "../", "./"])
from commons.utils import *
import torch
from torch.cuda import empty_cache
import random
from driver.gen_driver.ConfigGEN import Configurable
from module.Critierion import DC_and_CE_loss, DC_and_Focal_loss
from driver.gen_driver.train_waterfall_helper import MUTUALHelper
import configparser
from driver import gt_transform, train_transforms
from torch.nn.utils import clip_grad_norm_
from models import MODELS
from commons.evaluation import correct_predictions, accuracy_check
from models.seg_models.deeplab import DeepLab_Aux
from driver import std, mean

config = configparser.ConfigParser()
import argparse


def main(config):
    model_seg_coarse = DeepLab_Aux(num_classes=1, return_features=True)
    model_cls = MODELS['Integrate_Model_Cls_Ensemble_CAM_Att'](backbone=config.backbone, num_classes=config.cls_classes)
    model_seg = MODELS[config.model](backbone=config.backbone, n_channels=3, num_classes=1)
    criterion = {
        # 'seg_loss': DC_and_CE_loss(),
        'seg_loss': DC_and_Focal_loss(),
    }
    mutual_helper = MUTUALHelper(model_seg, criterion,
                                 config)
    mutual_helper.move_to_cuda()

    model_cls.to(mutual_helper.equipment)
    print("data name ", mutual_helper.config.data_name)

    try:
        model_seg_coarse = mutual_helper.load_pretrained_coarse_seg_model(model_seg_coarse)
        model_seg_coarse.eval()
        model_seg_coarse.to(mutual_helper.equipment)
    except FileExistsError as e:
        print(e)
        exit(0)

    try:
        model_cls = mutual_helper.load_pretrained_cls_model(model_cls)
        model_cls.eval()
        model_cls.to(mutual_helper.equipment)
    except FileExistsError as e:
        print(e)
        exit(0)

    train_loader_mtl, vali_loader_mtl_nomal, _, _ = mutual_helper.get_data_loader(
        global_transform=train_transforms, gt_transform=gt_transform, output_size=mutual_helper.config.patch_x,
        default_label_size=mutual_helper.config.default_seg_label_size,
        train_batch_size=mutual_helper.config.train_seg_batch_size,
        test_batch_size=mutual_helper.config.test_seg_batch_size, task=1)
    _, \
    _, \
    data_sets_mtl, \
    index_split_mtl = mutual_helper.get_aug_data_loader(branch='seg',
                                                        online_aug='online',
                                                        vali_aug='normal',
                                                        default_label_size=mutual_helper.config.default_seg_label_size,
                                                        train_batch_size=mutual_helper.config.train_seg_batch_size,
                                                        test_batch_size=mutual_helper.config.test_seg_batch_size,
                                                        new_data=False)

    for nb_acl_iter in range(mutual_helper.config.nb_active_learning_iter):
        best_seg_acc = 0
        bad_step = 0
        optimizer_seg = mutual_helper.reset_optim()
        empty_cache()
        decay_epoch = [30, 80, 90, 100, 140]
        decay_next = 0
        decay_e = decay_epoch[decay_next]
        for epoch in range(mutual_helper.config.epochs):
            train_critics = {
                'train/cls_loss': 0,
                'train/cls_acc': 0,
                'train/seg_loss': 0,
                'train/seg_acc': 0,
            }
            train_critics_seg = train_seg(mutual_helper, model_seg_coarse, model_cls, train_loader_mtl, optimizer_seg,
                                          epoch)
            train_critics.update(train_critics_seg)
            # validation
            vali_critics_seg = test_seg(mutual_helper, model_seg_coarse, model_cls, vali_loader_mtl_nomal, epoch)
            # vali_critics_seg = test_seg(mutual_helper, model_seg_coarse, model_cls, vali_loader_mtl, epoch)
            vali_critics = {
                'vali/cls_loss': 0,
                'vali/cls_acc': 0,
                'vali/seg_loss': 0,
                'vali/seg_acc': 0,
            }
            vali_critics.update(vali_critics_seg)
            mutual_helper.log.flush()
            mutual_helper.plot_vali_loss(nb_acl_iter, epoch, vali_critics)
            mutual_helper.plot_train_loss(nb_acl_iter, epoch, train_critics)
            if vali_critics['vali/seg_acc'] >= best_seg_acc:
                print(" * Best vali seg acc: history = %.4f, current = %.4f" % (
                    best_seg_acc, vali_critics['vali/seg_acc']))
                best_seg_acc = vali_critics['vali/seg_acc']
                bad_step = 0
                mutual_helper.save_best_checkpoint(save_model=True, iter=nb_acl_iter)
            else:
                bad_step += 1
                if bad_step == 1:
                    mutual_helper.save_best_checkpoint(model_optimizer=optimizer_seg, iter=nb_acl_iter)
                if bad_step >= 20:
                    bad_step = 0
                    mutual_helper.load_best_state(iter=nb_acl_iter)
                    optimizer_seg = mutual_helper.load_best_optim(optimizer_seg, iter=nb_acl_iter)
                    for g in optimizer_seg.param_groups:
                        current_lr = max(g['lr'] * 0.1, mutual_helper.config.min_lrate)
                        print("Decaying the learning ratio to %.8f" % (current_lr))
                        g['lr'] = current_lr
                # if (epoch + 1) % decay_e == 0:
                #     for g in optimizer_seg.param_groups:
                #         current_lr = max(g['lr'] * 0.1, mutual_helper.config.min_lrate)
                #         print("Decaying the learning ratio to %.8f" % (current_lr))
                #         g['lr'] = current_lr
                #     decay_next += 1
                #     decay_e = decay_epoch[decay_next]
                #     print("Next decay will be in the %d th epoch" % (decay_e))

        if mutual_helper.config.load_best_epoch:
            print("\n-----------load best state of model -----------")
            mutual_helper.load_best_state(iter=nb_acl_iter)

        mutual_helper.log.flush()

    print("\n----------- TRAINING FINISHED -----------")
    mutual_helper.summary_writer.close()


def train_seg(mutual_helper, model_seg_coarse, model_cls, train_loader_seg, optimizer_seg, epoch):
    mutual_helper.model.train()
    loss_seg = Averagvalue()
    acc_seg = Averagvalue()
    optimizer_seg.zero_grad()
    batch_num = int(np.ceil(len(train_loader_seg.dataset) / float(mutual_helper.config.train_seg_batch_size)))
    for ee in range(1):
        for i, batch in enumerate(train_loader_seg):
            images, labels, _ = mutual_helper.generate_batch(batch)
            with torch.no_grad():
                images_cls_logits, seg_backbone_out, _, _ = model_seg_coarse(images)
                probs_cls = torch.sigmoid(images_cls_logits)
                cls_features_out = model_cls.get_backbone_out(images, probs_cls, seg_backbone_out)
            # cls_features_out = None
            # _, _, cam = mutual_helper.generate_cam_ex_batch(model_cls, images.detach(), probs_cls, seg_backbone_out)
            # grid = make_grid(images, nrow=4, padding=2)
            # grid = np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean
            # save_img = np.clip(grid * 255 + 0.5, 0, 255)
            # visualize(save_img,
            #           join(mutual_helper.config.tmp_dir,
            #                str(i) + "_images"))
            # grid = make_grid(labels, nrow=4, padding=2)
            # visualize(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)),
            #           join(mutual_helper.config.tmp_dir,
            #                str(i) + "_probs"))
            # grid = make_grid(probs_cls, nrow=4, padding=2)
            # visualize(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)),
            #           join(mutual_helper.config.tmp_dir,
            #                str(i) + "_label"))
            cam = None
            logits = mutual_helper.model(images, cam, cls_features_out, dua=False)
            probs = torch.sigmoid(logits)
            loss = mutual_helper.criterions['seg_loss'](probs, labels)
            acc = accuracy_check(probs, labels)
            loss.backward()
            loss_seg.update(loss.item())
            acc_seg.update(acc)
            if (i + 1) % mutual_helper.config.update_every == 0 or i == batch_num - 1:
                clip_grad_norm_(filter(lambda p: p.requires_grad, mutual_helper.model.parameters()), \
                                max_norm=mutual_helper.config.clip)
                optimizer_seg.step()
                optimizer_seg.zero_grad()
        empty_cache()
    print(
        "[Epoch %d] [%s loss: %f] [%s acc: %f]" % (
            epoch,
            'seg', loss_seg.avg, 'seg', acc_seg.avg
        )
    )
    return {
        'train/seg_loss': loss_seg.avg,
        'train/seg_acc': acc_seg.avg,
    }


def test_seg(mutual_helper, model_seg_coarse, model_cls, test_loader, epoch):
    mutual_helper.model.eval()
    losses = Averagvalue()
    acces = Averagvalue()
    for i, batch in enumerate(test_loader):
        images_seg, labels_seg, _ = mutual_helper.generate_batch(batch)
        with torch.no_grad():
            images_cls_logits, seg_backbone_out, _, _ = model_seg_coarse(images_seg)
            probs_cls = torch.sigmoid(images_cls_logits)
            cls_features_out = model_cls.get_backbone_out(images_seg, probs_cls, seg_backbone_out)
        # cls_features_out = None
        # _, _, cam = mutual_helper.generate_cam_ex_batch(model_cls, images_seg.detach(), probs_cls, seg_backbone_out)
        cam = None
        predictions_seg = mutual_helper.model(images_seg, cam, cls_features_out, dua=False)
        probs = torch.sigmoid(predictions_seg)
        # cam_masks_grid = make_grid(probs, nrow=4, padding=2)
        # visualize(np.transpose(cam_masks_grid.detach().cpu().numpy(), (1, 2, 0)),
        #           join(mutual_helper.config.submission_dir,
        #                'image_' + str(iter) + '_probs_batch_origin'))
        loss = mutual_helper.criterions['seg_loss'](probs, labels_seg)
        acc = accuracy_check(probs, labels_seg)
        acces.update(acc)
        losses.update(loss.item())
    empty_cache()
    # measure elapsed time
    info = 'Vali Epoch: [{0}/{1}]'.format(epoch, mutual_helper.config.epochs) + \
           ' Loss {loss.avg:f} '.format(loss=losses) + \
           ' Acc {acc.avg:f} '.format(acc=acces)
    print(info)
    return {
        'vali/seg_loss': losses.avg,
        'vali/seg_acc': acces.avg,
    }


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
    argparser.add_argument('--config_file', default='train_waterfall_seg.txt')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args, isTrain=args.train)
    torch.set_num_threads(config.workers + 1)

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
