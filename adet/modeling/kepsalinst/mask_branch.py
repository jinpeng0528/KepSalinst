from typing import Dict
import math

import torch
from torch import nn
import torch.nn.functional as F

from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec

from adet.layers import conv_with_kaiming_uniform
from adet.utils.comm import aligned_bilinear


INF = 100000000


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, norm, stride=1):
        super(CBAM, self).__init__()

        conv_block = conv_with_kaiming_uniform(norm, activation=True)
        self.conv1 = conv_block(in_channels, in_channels, kernel_size=3, dilation=2)
        self.conv2 = conv_block(in_channels, in_channels, kernel_size=3, dilation=2)

        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        out = self.relu(out)

        return out


class SaliencyPredictor(nn.Module):
    def __init__(self, n_channels, norm):
        super().__init__()
        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        self.conv_5_4 = conv_block(n_channels['p5'] + n_channels['p4'], n_channels['p4'], kernel_size=3, dilation=1)
        self.conv_4_3 = conv_block(n_channels['p4'] + n_channels['p3'], n_channels['p3'], kernel_size=3, dilation=1)
        self.conv_refine = conv_block(n_channels['p3'], n_channels['p3'], kernel_size=3, dilation=1)

        self.conv_final = nn.Conv2d(n_channels['p3'], 1, kernel_size=1)

        self.cbam = CBAM(n_channels['p5'], norm=norm)

    def forward(self, features):
        sizes = [x.shape[2:] for x in features.values()]

        guidance = self.cbam(features['p5'])

        y = features['p5']

        y = torch.cat([F.interpolate(y, sizes[-4]), features['p4']], dim=1)
        # y = self.conv_5_4(y) + F.interpolate(guidance, sizes[-4])
        y = self.conv_5_4(y) + aligned_bilinear(guidance, 2)

        y = torch.cat([F.interpolate(y, sizes[-5]), features['p3']], dim=1)
        # y = self.conv_4_3(y) + F.interpolate(guidance, sizes[-5])
        y = self.conv_4_3(y) + aligned_bilinear(guidance, 4)

        # y = torch.cat([F.interpolate(y, sizes[-4], mode='bilinear'), features['p4']], dim=1)
        # y = self.conv_5_4(y) + F.interpolate(guidance, sizes[-4], mode='bilinear')
        #
        # y = torch.cat([F.interpolate(y, sizes[-5], mode='bilinear'), features['p3']], dim=1)
        # y = self.conv_4_3(y) + F.interpolate(guidance, sizes[-5], mode='bilinear')

        y = self.conv_refine(y)
        y = self.conv_final(y)
        return y


def build_mask_branch(cfg, input_shape):
    return MaskBranch(cfg, input_shape)


class MaskBranch(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.sal_map_on = cfg.MODEL.KEPSALINST.SAL_MAP_ON

        self.in_features = cfg.MODEL.KEPSALINST.MASK_BRANCH.IN_FEATURES
        self.num_outputs = cfg.MODEL.KEPSALINST.MASK_BRANCH.OUT_CHANNELS
        norm = cfg.MODEL.KEPSALINST.MASK_BRANCH.NORM
        num_convs = cfg.MODEL.KEPSALINST.MASK_BRANCH.NUM_CONVS
        channels = cfg.MODEL.KEPSALINST.MASK_BRANCH.CHANNELS
        self.out_stride = input_shape[self.in_features[0]].stride

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature],
                channels, 3, 1
            ))

        tower = []
        for i in range(num_convs):
            tower.append(conv_block(
                channels, channels, 3, 1
            ))
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

        # share_tower = []
        # for i in range(share_convs):
        #     share_tower.append(conv_block(
        #         channels, channels, 3, 1
        #     ))
        # self.add_module('share_tower', nn.Sequential(*share_tower))
        #
        # keypoints_tower = []
        # for i in range(single_convs):
        #     keypoints_tower.append(conv_block(
        #         channels, channels, 3, 1
        #     ))
        # keypoints_tower.append(nn.Conv2d(
        #     channels, max(self.num_outputs, 1), 1
        # ))
        # self.add_module('keypoints_tower', nn.Sequential(*keypoints_tower))
        #
        # segm_tower = []
        # for i in range(single_convs):
        #     segm_tower.append(conv_block(
        #         channels, channels, 3, 1
        #     ))
        # segm_tower.append(nn.Conv2d(
        #     channels, max(self.num_outputs, 1), 1
        # ))
        # self.add_module('segm_tower', nn.Sequential(*segm_tower))

        # sal map
        # num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA

        # in_channels = feature_channels[self.in_features[0]]
        # self.seg_head = nn.Sequential(
        #     conv_block(in_channels, channels, kernel_size=3, stride=1),
        #     conv_block(channels, channels, kernel_size=3, stride=1)
        # )
        #
        # self.logits = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)

        # prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # torch.nn.init.constant_(self.logits.bias, bias_value)

        self.saliency_predictor = SaliencyPredictor(feature_channels, norm)
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.saliency_predictor.conv_final.bias, bias_value)

    def forward(self, features, gt_instances=None):
        # keypoints_feats = None
        logits_pred = None
        losses = {}

        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])

                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        mask_feats = self.tower(x)
        # share_feats = self.share_tower(x)
        # segm_feats = self.segm_tower(share_feats)

        # if self.extreme_points_on:
        #     keypoints_feats = self.keypoints_tower(share_feats)

        # auxiliary thing semantic loss
        if self.sal_map_on:
            logits_pred = self.saliency_predictor(features)
            if self.training:
                # compute semantic targets
                semantic_targets = []
                for per_im_gt in gt_instances:
                    h, w = per_im_gt.gt_bitmasks_full.size()[-2:]
                    areas = per_im_gt.gt_bitmasks_full.sum(dim=-1).sum(dim=-1)
                    areas = areas[:, None, None].repeat(1, h, w)
                    areas[per_im_gt.gt_bitmasks_full == 0] = INF
                    areas = areas.permute(1, 2, 0).reshape(h * w, -1)
                    min_areas, inds = areas.min(dim=1)
                    per_im_sematic_targets = per_im_gt.gt_classes[inds] + 1
                    per_im_sematic_targets[min_areas == INF] = 0
                    per_im_sematic_targets = per_im_sematic_targets.reshape(h, w)
                    semantic_targets.append(per_im_sematic_targets)

                semantic_targets = torch.stack(semantic_targets, dim=0)

                # resize target to reduce memory
                semantic_targets = semantic_targets[
                                   :, None, self.out_stride // 2::self.out_stride,
                                   self.out_stride // 2::self.out_stride
                                   ]

                # prepare one-hot targets
                num_classes = logits_pred.size(1)
                class_range = torch.arange(
                    num_classes, dtype=logits_pred.dtype,
                    device=logits_pred.device
                )[:, None, None]
                class_range = class_range + 1
                one_hot = (semantic_targets == class_range).float()
                num_pos = (one_hot > 0).sum().float().clamp(min=1.0)

                loss_sem = sigmoid_focal_loss_jit(
                    logits_pred, one_hot,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                ) / num_pos
                losses['loss_sem'] = loss_sem

                logits_pred = logits_pred.sigmoid()
                return mask_feats, logits_pred, losses

            else:
                logits_pred = logits_pred.sigmoid()
                return mask_feats, logits_pred, None


