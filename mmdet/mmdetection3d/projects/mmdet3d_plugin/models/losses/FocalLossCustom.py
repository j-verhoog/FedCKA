# Created by Jolle Verhoog, 2026.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss

### NOW ONLY DOES AUTOFED FOR FALSE POSITIVES AND NOT FOR FALSE NEGATIVES
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    loss = _sigmoid_focal_loss(
        pred.contiguous(),
        target.contiguous(),
        gamma,
        alpha,
        None,
        'none'
    )

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FocalLossCustom(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False,
                 use_autofed=False,
                 p_th=0.9,
                 print_frequency=100):
        super(FocalLossCustom, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'

        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

        self.use_autofed = use_autofed
        self.p_th = p_th
        self.print_frequency = print_frequency
        self._step_counter = 0

    def _expand_weight(self, weight, pred):
        if weight is None:
            return None

        if weight.shape == pred.shape:
            return weight

        if weight.size(0) == pred.size(0):
            return weight.view(-1, 1).expand_as(pred)

        assert weight.numel() == pred.numel()
        return weight.view_as(pred)

    def _build_autofed_mask(self, pred, target):
        if self.activated:
            prob = pred
        else:
            prob = pred.sigmoid()

        if target.shape != pred.shape:
            num_classes = pred.size(1)
            target_safe = target.clone().long()
            target_safe[target_safe < 0] = num_classes
            target_safe[target_safe > num_classes] = num_classes
            target_one_hot = F.one_hot(
                target_safe, num_classes=num_classes + 1
            )[:, :num_classes]
            target_one_hot = target_one_hot.type_as(pred)
        else:
            target_one_hot = target.type_as(pred)

        is_negative = (target_one_hot == 0)
        trust_model_mask = (prob > self.p_th) & is_negative
        return trust_model_mask, is_negative

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.use_autofed:
            with torch.no_grad():
                trust_model_mask, is_negative = self._build_autofed_mask(pred, target)

                if self._step_counter % self.print_frequency == 0:
                    triggered_count = trust_model_mask.sum().item()
                    bg_count = is_negative.sum().item()
                    if bg_count > 0:
                        pct = 100.0 * triggered_count / bg_count
                        print(f'[AutoFed] step={self._step_counter} ignored={triggered_count}/{bg_count} background entries ({pct:.4f}%)')
                    else:
                        print(f'[AutoFed] step={self._step_counter} ignored=0/0 background entries (0.0000%)')

                self._step_counter += 1

            autofed_weight = (~trust_model_mask).type_as(pred)

            expanded_weight = self._expand_weight(weight, pred)
            if expanded_weight is None:
                weight = autofed_weight
            else:
                weight = expanded_weight * autofed_weight

        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor
            )
        else:
            raise NotImplementedError

        return loss_cls