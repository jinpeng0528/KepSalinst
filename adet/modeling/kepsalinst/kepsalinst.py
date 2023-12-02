# -*- coding: utf-8 -*-
import logging
import numpy as np
from skimage import io

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask

from .dynamic_mask_head import build_dynamic_mask_head
from .mask_branch import build_mask_branch

from adet.utils.comm import aligned_bilinear

__all__ = ["KepSalinst"]

logger = logging.getLogger(__name__)


def gaussian_kernel(x0, y0, sigma, width, height, device):
    x = torch.arange(0, width, 1, dtype=torch.float, device=device)  ## (width,)
    y = torch.arange(0, height, 1, dtype=torch.float, device=device)[:, None]  ## (height,1)
    return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def generate_heatmaps(height, width, peaks, sigma=3):
    n_peaks = len(peaks)
    hm = torch.zeros((height, width), dtype=torch.float, device=peaks.device)
    hm = torch.maximum(hm, gaussian_kernel(peaks[1], peaks[0], sigma, width, height, peaks.device))
    return hm


@META_ARCH_REGISTRY.register()
class KepSalinst(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())

        self.mask_out_stride = cfg.MODEL.KEPSALINST.MASK_OUT_STRIDE

        self.max_proposals = cfg.MODEL.KEPSALINST.MAX_PROPOSALS
        self.topk_proposals_per_im = cfg.MODEL.KEPSALINST.TOPK_PROPOSALS_PER_IM

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module

        self.generouter = nn.Conv2d(
            in_channels, self.mask_head.outer_map_num_params, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.generouter.weight, std=0.01)
        torch.nn.init.constant_(self.generouter.bias, 0)

        # self.controller = nn.Conv2d(
        #     in_channels, self.mask_head.center_segm_num_params,
        #     kernel_size=3, stride=1, padding=1
        # )
        self.controller = nn.Conv2d(
            in_channels, self.mask_head.center_segm_num_params + self.mask_head.extreme_segm_num_params,
            kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.iter = 0

    def forward(self, batched_inputs):
        self.iter += 1
        # print(self.iter)

        original_images = [x["image"].to(self.device) for x in batched_inputs]

        # normalize images
        images_norm = [self.normalizer(x) for x in original_images]
        images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)

        features = self.backbone(images_norm.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            self.add_bitmasks(gt_instances, images_norm.tensor.size(-2), images_norm.tensor.size(-1))
        else:
            gt_instances = None

        mask_feats, sal_pred, sem_losses = self.mask_branch(features, gt_instances)

        if self.training:
            results, proposal_losses, extras = self.proposal_generator(
                images_norm, features, gt_instances, self.controller, self.generouter
            )

            mask_losses = self._forward_mask_heads_train(results, extras, mask_feats, gt_instances)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update(mask_losses)
            return losses
        else:
            results, extras = self.proposal_generator(
                images_norm, features, gt_instances, self.controller, self.generouter
            )

            # pred_instances_w_masks = self._forward_mask_heads_test(results, extras, mask_feats, gt_instances)
            pred_instances_w_masks = self._forward_mask_heads_test(
                results, extras, mask_feats, sal_pred)

            padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images_norm.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                instances_per_im = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w
                )

                processed_results.append({
                    "instances": instances_per_im
                })

            return processed_results

    def _forward_mask_heads_train(self, proposals, extras, mask_feats, gt_instances):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]
        full_controller_feats = extras["controller_feats"]
        full_locations = extras["full_locations"]

        assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."
        if self.max_proposals != -1:
            if self.max_proposals < len(pred_instances):
                # inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
                inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
                logger.info("clipping proposals from {} to {}".format(
                    len(pred_instances), self.max_proposals
                ))
                pred_instances = pred_instances[inds[:self.max_proposals]]
        elif self.topk_proposals_per_im != -1:
            num_images = len(gt_instances)

            kept_instances = []
            for im_id in range(num_images):
                instances_per_im = pred_instances[pred_instances.im_inds == im_id]
                if len(instances_per_im) == 0:
                    kept_instances.append(instances_per_im)
                    continue

                unique_gt_inds = instances_per_im.gt_inds.unique()
                num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)

                for gt_ind in unique_gt_inds:
                    instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]

                    if len(instances_per_gt) > num_instances_per_gt:
                        scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
                        ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                        inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
                        instances_per_gt = instances_per_gt[inds]

                    kept_instances.append(instances_per_gt)

            pred_instances = Instances.cat(kept_instances)

        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            full_controller_feats, pred_instances, gt_instances, full_locations
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, extras, mask_feats, sal_pred):
        pred_instances = proposals["instances"]
        full_controller_feats = extras["full_controller_feats"]
        # full_locations = proposals["full_locations"]

        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(pred_instances):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(pred_instances)
        pred_instances.mask_head_params = pred_instances.controller_feats

        # pred_instances_w_masks = self.mask_head(
        #     mask_feats, self.mask_branch.out_stride,
        #     full_controller_feats, pred_instances, gt_instances
        # )
        pred_instances_w_masks = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            full_controller_feats, pred_instances, sal_pred=sal_pred
        )

        return pred_instances_w_masks

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                per_im_outer_bitmasks = []
                per_im_sigmas = []
                for ind, per_polygons in enumerate(polygons):

                    outer_bitmasks = torch.zeros(
                        [4, int(im_h / self.mask_out_stride), int(im_w / self.mask_out_stride)]) \
                        .to(self.device).float()

                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    one_index = torch.from_numpy(
                        np.argwhere(bitmask[start::self.mask_out_stride,
                                    start::self.mask_out_stride])).to(self.device)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)

                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                    area = (per_im_gt_inst.get("gt_boxes").tensor[ind][2] - per_im_gt_inst.get("gt_boxes").tensor[ind][
                        0]) * \
                           (per_im_gt_inst.get("gt_boxes").tensor[ind][3] - per_im_gt_inst.get("gt_boxes").tensor[ind][
                               1])
                    sigma = torch.sqrt(area) / 48
                    per_im_sigmas.append(sigma)

                    if one_index.sum():
                        leftmost = torch.min(one_index[:, 1])
                        rightmost = torch.max(one_index[:, 1])
                        upmost = torch.min(one_index[:, 0])
                        downmost = torch.max(one_index[:, 0])

                        outer_bitmasks[0, :, leftmost] = 1
                        outer_bitmasks[1, :, rightmost] = 1
                        outer_bitmasks[2, upmost] = 1
                        outer_bitmasks[3, downmost] = 1
                        outer_bitmasks = outer_bitmasks * bitmask

                        peaks = [torch.median(outer_bitmask.nonzero(), dim=0).values
                                 for outer_bitmask in outer_bitmasks]
                        heatmaps = []
                        for per_peaks in peaks:
                            heatmaps.append(
                                generate_heatmaps(outer_bitmasks.shape[1], outer_bitmasks.shape[2], per_peaks, sigma)[
                                None, :, :]
                            )
                        heatmaps = torch.vstack(heatmaps)
                        # heatmaps = heatmaps * bitmask
                    else:
                        heatmaps = torch.zeros_like(bitmask.unsqueeze(0).repeat(4, 1, 1))

                    # io.imsave('left.png', heatmaps[0].cpu())
                    # io.imsave('right.png', heatmaps[1].cpu())
                    # io.imsave('up.png', heatmaps[2].cpu())
                    # io.imsave('down.png', heatmaps[3].cpu())
                    # io.imsave('bitmask.png', bitmask.cpu())

                    per_im_outer_bitmasks.append(heatmaps)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)

                per_im_gt_inst.gt_outer_bitmasks = torch.stack(per_im_outer_bitmasks, dim=0)
                per_im_gt_inst.sigmas = torch.stack(per_im_sigmas)

            else:  # RLE format bitmask
                # TODO: Add RLE part
                # bitmasks = per_im_gt_inst.get("gt_masks").tensor
                # h, w = bitmasks.size()[1:]
                # # pad to new size
                # bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                # bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                # per_im_gt_inst.gt_bitmasks = bitmasks
                # per_im_gt_inst.gt_bitmasks_full = bitmasks_full
                print("RLE not supported!")

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results