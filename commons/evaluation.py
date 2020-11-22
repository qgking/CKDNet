import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils import compute_class_weight as sk_compute_class_weight
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import f1_score, average_precision_score
from data.preprocess.ISICPreprocess_2017 import classes, class_names
import math
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from scipy import interp


def calculate_metrics(y_test, y_pred):
    """Calculates the accuracy metrics"""

    accuracy = accuracy_score(y_test, y_pred)

    # Wrapping all the scoring function calls in a try & except to prevent
    # the following warning to result in a "TypeError: warnings_to_log()
    # takes 4 positional arguments but 6 were given" when sklearn calls
    # warnings.warn with an "UndefinedMetricWarning:Precision is
    # ill-defined and being set to 0.0 in labels with no predicted
    # samples." message on python 3.7.x
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)

    return [accuracy, precision, recall, f1score]


def get_confusion_matrix(y_true, y_pred, norm_cm=True, print_cm=True):
    true_class = np.argmax(y_true, axis=1)
    pred_class = np.argmax(y_pred, axis=1)

    cnf_mat = confusion_matrix(true_class, pred_class, labels=classes)

    total_cnf_mat = np.zeros(shape=(cnf_mat.shape[0] + 1, cnf_mat.shape[1] + 1), dtype=np.float)
    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    for i_row in range(cnf_mat.shape[0]):
        total_cnf_mat[i_row, -1] = np.sum(total_cnf_mat[i_row, 0:-1])

    for i_col in range(cnf_mat.shape[1]):
        total_cnf_mat[-1, i_col] = np.sum(total_cnf_mat[0:-1, i_col])

    if norm_cm:
        cnf_mat = cnf_mat / (cnf_mat.astype(np.float).sum(axis=1)[:, np.newaxis] + 0.001)

    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    return cnf_mat


def get_precision_recall(y_true, y_pred, print_pr=True):
    true_class = np.argmax(y_true, axis=1)
    pred_class = np.argmax(y_pred, axis=1)
    precision, recall, _, _ = precision_recall_fscore_support(y_true=true_class,
                                                              y_pred=pred_class,
                                                              labels=classes,
                                                              warn_for=())

    return precision, recall


def compute_class_weights(y, wt_type='balanced', return_dict=True):
    # need to check if y is one hot
    if len(y.shape) > 1:
        y = y.argmax(axis=-1)

    assert wt_type in ['ones', 'balanced', 'balanced-sqrt'], 'Weight type not supported'

    classes = np.unique(y)
    class_weights = np.ones(shape=classes.shape[0])

    if wt_type == 'balanced' or wt_type == 'balanced-sqrt':

        class_weights = sk_compute_class_weight(class_weight='balanced',
                                                classes=classes,
                                                y=y)
        if wt_type == 'balanced-sqrt':
            class_weights = np.sqrt(class_weights)

    if return_dict:
        class_weights = dict([(i, w) for i, w in enumerate(class_weights)])

    return class_weights


def jaccard(y_true, y_pred):
    intersect = np.sum(y_true * y_pred)  # Intersection points
    union = np.sum(y_true) + np.sum(y_pred)  # Union points
    return (float(intersect)) / (union - intersect + 1e-7)


def dice_coef(y_true, y_pred):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def custom_classes_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
            y_true.shape,
            y_pred.shape)
              )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    # Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels)

    avg = list(precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"],
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df


def compute_all_metric_for_seg(y_true, y_pred):
    batch, channel, width, height = y_true.shape
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    fpr, tpr, thresholds = roc_curve((y_true), y_pred)
    AUC_ROC = roc_auc_score(y_true, y_pred)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    # roc_curve = plt.figure()
    # plt.plot(fpr, tpr, '-', label=algorithm + '_' + dataset + '(AUC = %0.4f)' % AUC_ROC)
    # plt.title('ROC curve', fontsize=14)
    # plt.xlabel("FPR (False Positive Rate)", fontsize=15)
    # plt.ylabel("TPR (True Positive Rate)", fontsize=15)
    # plt.legend(loc="lower right")
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    confusion = confusion_matrix(y_true, y_pred)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    mean_jaccard, thresholded_jaccard = compute_jaccard(np.reshape(y_true, (batch, width, height)),
                                                        np.reshape(y_pred, (batch, width, height)))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    dice_score = dice_coef(y_true, y_pred)
    print("Area under the ROC curve: " + str(AUC_ROC)
          + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
          + "\nMean Jaccard similarity score: " + str(mean_jaccard)
          + "\nF1 score (F-measure): " + str(F1_score)
          + "\nACCURACY: " + str(accuracy)
          + "\nSENSITIVITY: " + str(sensitivity)
          + "\nSPECIFICITY: " + str(specificity)
          + "\nPRECISION: " + str(precision)
          + "\nDICE SCORE: " + str(dice_score)
          )


def compute_all_metric_for_single_seg(y_true, y_pred):
    tensor_y_pred = torch.from_numpy(y_pred).cuda().float()
    tensor_y_true = torch.from_numpy(y_true).cuda().float()
    accuracy = get_accuracy(tensor_y_pred, tensor_y_true)
    sensitivity = get_sensitivity(tensor_y_pred, tensor_y_true)
    specificity = get_specificity(tensor_y_pred, tensor_y_true)
    # dice_score = diceCoeff(tensor_y_pred, tensor_y_true)
    dice_score = get_DC(tensor_y_pred, tensor_y_true)
    mean_jaccard = get_JS(tensor_y_pred, tensor_y_true)
    F1_score = get_F1(tensor_y_pred, tensor_y_true)
    scores = {  # 'ROC': [],
        # 'Precision-Recall': [],
        'Jaccard': [],
        'F1': [], 'ACCURACY': [], 'SENSITIVITY': [], 'SPECIFICITY': [],
        # 'PRECISION': [],
        # 'DICEDIST': [],
        'DICESCORE': []}
    # scores['ROC'].append(AUC_ROC)
    scores['Jaccard'].append(mean_jaccard)
    scores['F1'].append(F1_score)
    scores['ACCURACY'].append(accuracy)
    scores['SENSITIVITY'].append(sensitivity)
    scores['SPECIFICITY'].append(specificity)
    # scores['PRECISION'].append(precision)
    scores['DICESCORE'].append(dice_score)
    return scores


def calculate_cls_metric(y_true, y_pred, task=1, thres=0.5):
    if len(np.unique(y_true)) == 3:
        task1_output_map = lambda x: 1 if x == 0 else 0
        task2_output_map = lambda x: 1 if x == 1 else 0
        task_output_map = task1_output_map if task == 1 else task2_output_map
        labels_task = list(map(task_output_map, y_true))
        AUC_ROC = roc_auc_score(labels_task, y_pred[:, task - 1])
        # preds_task = list(map(task_output_map, y_pred_hard))
        accuracy = accuracy_score(
            labels_task, np.where(y_pred[:, task - 1] >= thres, 1, 0))
        precision = average_precision_score(labels_task, y_pred[:, task - 1])
        conf_matrix = confusion_matrix(labels_task, y_pred[:, task - 1] >= thres)

    elif len(np.unique(y_true)) == 2:
        labels_task = y_true
        # preds_task = y_pred[:, task] >= 0.5
        AUC_ROC = roc_auc_score(labels_task, y_pred)
        accuracy = accuracy_score(
            labels_task, np.where(y_pred >= thres, 1, 0))
        precision = average_precision_score(labels_task, y_pred)
        conf_matrix = confusion_matrix(labels_task, y_pred >= thres)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    # confusion = confusion_matrix(labels_task, preds_task)
    # accuracy = 0
    # if float(np.sum(confusion)) != 0:
    #     accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    # specificity = 0
    # if float(confusion[0, 0] + confusion[0, 1]) != 0:
    #     specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    # sensitivity = 0
    # if float(confusion[1, 1] + confusion[1, 0]) != 0:
    #     sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    # precision = 0
    # if float(confusion[1, 1] + confusion[0, 1]) != 0:
    #     precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("----------Task : " + str(task) + " Metric----------"
          + "\nAUC: " + str(AUC_ROC)
          + "\nACCURACY: " + str(accuracy)
          + "\nSENSITIVITY: " + str(sensitivity)
          + "\nSPECIFICITY: " + str(specificity)
          + "\nPRECISION: " + str(precision)
          )
    return AUC_ROC, accuracy, sensitivity, specificity, precision


def compute_all_metric_for_class_wise_cls(y_true, y_pred, thres=0.5):
    labels_task = y_true
    preds_task = y_pred
    lesion_cls_metrics = {}
    AUC_ROC, accuracy, sensitivity, specificity, precision = calculate_cls_metric(labels_task, preds_task, task=1,
                                                                                  thres=thres)
    lesion_cls_metrics['task1'] = {
        'AUC': AUC_ROC,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision
    }
    AUC_ROC, accuracy, sensitivity, specificity, precision = calculate_cls_metric(labels_task, preds_task, task=2,
                                                                                  thres=thres)
    lesion_cls_metrics['task2'] = {
        'AUC': AUC_ROC,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision
    }
    return lesion_cls_metrics


def compute_jaccard(y_true, y_pred):
    mean_jaccard = 0.
    thresholded_jaccard = 0.

    for im_index in range(y_pred.shape[0]):
        current_jaccard = jaccard(y_true=y_true[im_index], y_pred=y_pred[im_index])

        mean_jaccard += current_jaccard
        thresholded_jaccard += 0 if current_jaccard < 0.65 else current_jaccard

    mean_jaccard = mean_jaccard / y_pred.shape[0]
    thresholded_jaccard = thresholded_jaccard / y_pred.shape[0]

    return mean_jaccard, thresholded_jaccard


import torch


def get_accuracy(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()
    corr = torch.sum(SR == GT)
    tensor_size = torch.prod(torch.tensor(SR.size()))
    acc = float(corr) / float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1).int() + (GT == 1).int()).int() == 2).int()
    FN = (((SR == 0).int() + (GT == 1).int()).int() == 2).int()

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0).int() + (GT == 0).int()).int() == 2).int()
    FP = (((SR == 1).int() + (GT == 0).int()).int() == 2).int()

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TP : True Positive
    # FP : False Positive
    TP = (((SR == 1).int() + (GT == 1).int()).int() == 2).int()
    FP = (((SR == 1).int() + (GT == 0).int()).int() == 2).int()

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    Inter = torch.sum((SR + GT) == 2).int()
    Union = torch.sum((SR + GT) >= 1).int()

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def diceCoeff(input, target):
    eps = 0.0001
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input).int() + torch.sum(target) + eps
    t = (2 * inter.float() + eps) / union.float()
    return float(t)


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    Inter = torch.sum((SR + GT) == 2).int()
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return out_classes, correct.item()


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.cpu().detach().numpy()
        np_ims.append(item)
    compare = np.equal(np.where(np_ims[0] > 0.5, 1, 0), np_ims[1])
    accuracy = np.sum(compare)
    return accuracy / len(np_ims[0].flatten())
