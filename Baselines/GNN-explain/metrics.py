import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

import numpy as np
def n_th_accuracy(i):
    def ret(pred, target):
        out = ((pred[:, i] > 0.5).to(bool) == target[:, i])
        return out.float().mean()
    return ret


def min_accuracy(x, y):
    return min([n_th_accuracy(i)(x, y) for i in range(75)])


def mean_accuracy(pred, target):
    return (pred - target).abs().mean()


def min_rank(pred: torch.Tensor, target: torch.Tensor):
    rank = torch.zeros(pred.shape[0])
    for i, (p, t) in enumerate(zip(pred, target)):
        min_val = p[t.bool()].min()
        rank[i] = (p > min_val).sum()
    return rank.mean()


def pr(pred: torch.Tensor, target):
    return 0


def max_rank(pred, target):
    rank = torch.zeros(pred.shape[0])
    for i, (p, t) in enumerate(zip(pred, target)):
        max_val = p[(1 - t).bool()].max()
        rank[i] = (p < max_val).sum()
    return rank.mean()


def precision(i):
    def ret(pred: torch.Tensor, target: torch.Tensor):
        attr = pred[:, i] > 0.5
        return (attr * target[:, i]).sum() / (attr == True).sum().to(float)

    return ret


def recall(i):
    def ret(pred, target):
        attr = pred[:, i] > 0.5
        return (attr * target[:, i]).sum() / (target == True).sum()

    return ret


def mse_rank(i):
    def ret(pred, target):
        return (pred[:, i] - target[:, i]).pow(2).mean()

    return ret


def ce_rank(i):
    loss = torch.nn.BCELoss()

    def ret(pred, target):
        return loss(pred[:, i], target[:, i])

    return ret


def binary_acc(pred, target):
    return (pred.argmax(1) == target).float().mean()


def bin_prec(pred: torch.Tensor, target: torch.Tensor):
    attr = pred.argmax(dim=1)
    return (attr * target).sum() / attr.sum().float()


def bin_rec(pred: torch.Tensor, target: torch.Tensor):
    attr = pred.argmax(dim=1)
    return (attr * target).sum() / (target == True).sum().float()


def bin_rev_prec(pred: torch.Tensor, target: torch.Tensor):
    attr = pred.argmin(dim=1)
    return (attr * (1 - target)).sum() / attr.logical_not().sum().to(float)


def bin_rev_rec(pred: torch.Tensor, target: torch.Tensor):
    attr = pred.argmin(dim=1)
    return (attr * (1 - target)).sum() / (target == False).sum().to(float)


def myloss(k, alpha=0.1):
    l1 = torch.nn.MSELoss()
    l2 = mse_rank(0)
    i = [k]

    def ret(pred, target):
        i[0] += 1

        return l1(pred, target) / (i[0] * alpha) + l2(pred, target)

    return ret

def mymseFun(pred: torch.Tensor, target: torch.Tensor):
    mse = torch.nn.MSELoss()
    tar = torch.nn.functional.one_hot(target, 2).to(pred.dtype)
    return mse(pred, tar)


def mymse():



    return mymseFun

"""def mymse():
    mse = torch.nn.MSELoss()

    def mymseFun(pred: torch.Tensor, target: torch.Tensor):
        tar = torch.nn.functional.one_hot(target, 2).to(pred.dtype)
        return mse(pred, tar)

    return ret"""


def au_roc(pred: torch.Tensor, target: torch.Tensor):
    return roc_auc_score(target, pred[:, 1])


def au_pr(pred: torch.Tensor, target: torch.Tensor):
    # return average_precision_score(target, pred[:,1])
    prec, rec, thresholds = precision_recall_curve(target, pred[:, 1])
    return auc(rec, prec)

def optimal_threshold(pred, target):
    fpr, tpr, thresholds = roc_curve(target, pred)
    # get the best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    #print('Best Threshold=%f' % (best_thresh))
    return best_thresh


def optimal_acc(pred, target):
    pred = np.array(pred[:,1])
    target = np.array(target)
    _, th = f1_score(pred, target)
    acc = ((pred > th).astype(bool) == target.astype(bool)).mean()
    return acc

def f1_score_metric(pred, target, threshold=None):
    pred = np.array(pred[:,1])
    target = np.array(target)
    return f1_score(pred, target)[0]

def f1_score(pred, target, threshold=None):
    if not threshold:
        pass
        precision, recall, thresholds = precision_recall_curve(target, pred)
        # print(precision+recall)
        fscore = (2 * precision * recall) / (precision + recall+1e-10)

        ix = np.argmax(fscore)

        return fscore[ix], thresholds[ix]
    else:

        pred2 = pred > threshold
        tp = (target*pred2).sum()
        tn = ((1-target)*(1-pred2)).sum()
        fp = ((1-target)*pred2).sum()
        fn = (target*(1-pred2)).sum()

        pr = tp/(tp+fp)
        re = tp/(tp+fn)

        return pr*re*2/(pr+re+1e-10), threshold
    #pr = tp/pred.sum()
    #re = tp/true.sum()"""