import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear

from .blur_conv import GaussianBlurConv


def gaussian_kernel(x0, y0, sigma, width, height, device):
    x = torch.arange(0, width, 1, dtype=torch.float, device=device)  ## (width,)
    y = torch.arange(0, height, 1, dtype=torch.float, device=device)[:, None]  ## (height,1)
    return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def generate_heatmaps(height, width, peaks, sigma=3):
    n_peaks = len(peaks)
    hm = torch.zeros((height, width), dtype=torch.float, device=peaks.device)
    hm = torch.maximum(hm, gaussian_kernel(peaks[1], peaks[0], sigma, width, height, peaks.device))
    return hm

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def parse_dynamic_params(params, channels, weight_nums, bias_nums, out_channels, kernel_size):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, kernel_size, kernel_size)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * out_channels, -1, kernel_size, kernel_size)
            bias_splits[l] = bias_splits[l].reshape(num_insts * out_channels)

    return weight_splits, bias_splits

def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class FilterMerger(nn.Module):
    def __init__(self, param_length=169):
        super(FilterMerger, self).__init__()
        self.linear = nn.Linear(param_length * 5, param_length)
        self.ct_proj = nn.Linear(param_length, param_length, bias=False)
        self.ep_proj = nn.Linear(param_length, param_length, bias=False)
        self.weight_eps_linear = nn.Linear(param_length, param_length, bias=False)

    def forward(self, ct_param, eps_param):
        dists = (self.ep_proj(eps_param) - self.ct_proj(ct_param).unsqueeze(1)) ** 2
        # dists = (eps_param - ct_param.unsqueeze(1)) ** 2
        weight_eps = self.weight_eps_linear(dists)
        weight_eps = weight_eps.softmax(1)
        ep_param = (eps_param * weight_eps).sum(1)

        final_param = (ep_param + ct_param) / 2
        return final_param


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.sal_map_on = cfg.MODEL.KEPSALINST.SAL_MAP_ON

        self.channels = cfg.MODEL.KEPSALINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.KEPSALINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.KEPSALINST.MASK_OUT_STRIDE
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.disable_rel_coords = cfg.MODEL.KEPSALINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        self.outer_map_num_layers = cfg.MODEL.KEPSALINST.MASK_HEAD.OUTER_MAP_NUM_LAYERS
        self.outer_map_kernel_size = cfg.MODEL.KEPSALINST.MASK_HEAD.OUTER_MAP_KERNEL_SIZE
        self.outer_map_dilation = cfg.MODEL.KEPSALINST.MASK_HEAD.OUTER_MAP_DILATION
        self.outer_map_out_channels = 8

        self.center_map_num_layers = cfg.MODEL.KEPSALINST.MASK_HEAD.CENTER_MAP_NUM_LAYERS
        self.center_map_kernel_size = cfg.MODEL.KEPSALINST.MASK_HEAD.CENTER_MAP_KERNEL_SIZE
        self.center_map_dilation = cfg.MODEL.KEPSALINST.MASK_HEAD.CENTER_MAP_DILATION
        self.center_segm_out_channels = 1

        self.extreme_map_num_layers = cfg.MODEL.KEPSALINST.MASK_HEAD.EXTREME_MAP_NUM_LAYERS
        self.extreme_map_kernel_size = cfg.MODEL.KEPSALINST.MASK_HEAD.EXTREME_MAP_KERNEL_SIZE
        self.extreme_map_dilation = cfg.MODEL.KEPSALINST.MASK_HEAD.EXTREME_MAP_DILATION
        self.extreme_segm_out_channels = 1

        self.outer_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(self.channels, 4, kernel_size=3, stride=1, padding=2, dilation=2)
        )

        self.blur_conv = GaussianBlurConv(4, length=9, sigma=1.5)

        outer_map_weight_nums, outer_map_bias_nums = [], []
        if self.outer_map_num_layers == 1:
            if not self.disable_rel_coords:
                outer_map_weight_nums.append(
                    (self.in_channels + 2) * self.outer_map_out_channels * (self.outer_map_kernel_size ** 2))
            else:
                outer_map_weight_nums.append(
                    self.in_channels * self.outer_map_out_channels * (self.outer_map_kernel_size ** 2))
            outer_map_bias_nums.append(self.outer_map_out_channels)
        else:
            for i in range(self.outer_map_num_layers):
                if i == 0:
                    if not self.disable_rel_coords:
                        outer_map_weight_nums.append(
                            (self.in_channels + 2) * self.channels * (self.outer_map_kernel_size ** 2))
                    else:
                        outer_map_weight_nums.append(
                            self.in_channels * self.channels * (self.outer_map_kernel_size ** 2))
                    outer_map_bias_nums.append(self.channels)

                elif i == self.outer_map_num_layers - 1:
                    outer_map_weight_nums.append(
                        self.channels * self.outer_map_out_channels * (self.outer_map_kernel_size ** 2))
                    outer_map_bias_nums.append(self.outer_map_out_channels)
                else:
                    outer_map_weight_nums.append(
                        self.channels * self.channels * (self.outer_map_kernel_size ** 2))
                    outer_map_bias_nums.append(self.channels)

        center_segm_weight_nums, center_segm_bias_nums = [], []
        if self.center_map_num_layers == 1:
            if not self.disable_rel_coords:
                center_segm_weight_nums.append(
                    (self.in_channels + 2) * self.center_segm_out_channels * (self.center_map_kernel_size ** 2))
            else:
                center_segm_weight_nums.append(
                    self.in_channels * self.center_segm_out_channels * (self.center_map_kernel_size ** 2))
            center_segm_bias_nums.append(self.center_segm_out_channels)
        else:
            for i in range(self.center_map_num_layers):
                if i == 0:
                    if not self.disable_rel_coords:
                        center_segm_weight_nums.append(
                            (self.in_channels + 2) * self.channels * (self.center_map_kernel_size ** 2))
                    else:
                        center_segm_weight_nums.append(
                            self.in_channels * self.channels * (self.center_map_kernel_size ** 2))
                    center_segm_bias_nums.append(self.channels)
                elif i == self.center_map_num_layers - 1:
                    center_segm_weight_nums.append(
                        self.channels * self.center_segm_out_channels * (self.center_map_kernel_size ** 2))
                    center_segm_bias_nums.append(self.center_segm_out_channels)
                else:
                    center_segm_weight_nums.append(
                        self.channels * self.channels * (self.center_map_kernel_size ** 2))
                    center_segm_bias_nums.append(self.channels)

        extreme_segm_weight_nums, extreme_segm_bias_nums = [], []
        if self.extreme_map_num_layers == 1:
            if not self.disable_rel_coords:
                extreme_segm_weight_nums.append(
                    (self.in_channels + 2) * self.extreme_segm_out_channels * (self.extreme_map_kernel_size ** 2))
            else:
                extreme_segm_weight_nums.append(
                    self.in_channels * self.extreme_segm_out_channels * (self.extreme_map_kernel_size ** 2))
            extreme_segm_bias_nums.append(self.extreme_segm_out_channels)
        else:
            for i in range(self.extreme_map_num_layers):
                if i == 0:
                    if not self.disable_rel_coords:
                        extreme_segm_weight_nums.append(
                            (self.in_channels + 2) * self.channels * (self.extreme_map_kernel_size ** 2))
                    else:
                        extreme_segm_weight_nums.append(
                            self.in_channels * self.channels * (self.extreme_map_kernel_size ** 2))
                    extreme_segm_bias_nums.append(self.channels)
                elif i == self.extreme_map_num_layers - 1:
                    extreme_segm_weight_nums.append(
                        self.channels * self.extreme_segm_out_channels * (self.extreme_map_kernel_size ** 2))
                    extreme_segm_bias_nums.append(self.extreme_segm_out_channels)
                else:
                    extreme_segm_weight_nums.append(
                        self.channels * self.channels * (self.extreme_map_kernel_size ** 2))
                    extreme_segm_bias_nums.append(self.channels)

        self.outer_map_weight_nums = outer_map_weight_nums
        self.outer_map_bias_nums = outer_map_bias_nums
        self.outer_map_num_params = sum(outer_map_weight_nums) + sum(outer_map_bias_nums)

        self.center_segm_weight_nums = center_segm_weight_nums
        self.center_segm_bias_nums = center_segm_bias_nums
        self.center_segm_num_params = sum(center_segm_weight_nums) + sum(center_segm_bias_nums)

        self.extreme_segm_weight_nums = extreme_segm_weight_nums
        self.extreme_segm_bias_nums = extreme_segm_bias_nums
        self.extreme_segm_num_params = sum(extreme_segm_weight_nums) + sum(extreme_segm_bias_nums)

        self.register_buffer("_iter", torch.zeros([1]))

        self.filter_merger = FilterMerger(self.center_segm_num_params)

        self.test_iter = 0

    def mask_heads_forward(self, features, weights, biases, num_insts, dilation_rate=1):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding='same',
                dilation=dilation_rate,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances, mode, extreme_locs=None
    ):
        assert mode in ["outer", "center", "extreme"]

        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds

        if mode == "outer":
            mask_head_params = instances.generouter_feats
        else:
            mask_head_params = instances.controller_feats

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            if mode in ["outer", "center"]:
                instance_locations = instances.locations
                relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
                relative_coords = relative_coords.permute(0, 2, 1).float()
                soi = self.sizes_of_interest.float()[instances.fpn_levels]
                relative_coords = relative_coords / soi.reshape(-1, 1, 1)
                relative_coords = relative_coords.to(dtype=mask_feats.dtype)
                mask_head_inputs = torch.cat([
                    relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
                ], dim=1)
            elif mode == "extreme":
                pass
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        if mode == "outer":
            weights, biases = parse_dynamic_params(
                mask_head_params,
                self.channels, self.outer_map_weight_nums, self.outer_map_bias_nums,
                self.outer_map_out_channels, self.outer_map_kernel_size
            )
        elif mode == "center":
            weights, biases = parse_dynamic_params(
                mask_head_params,
                self.channels, self.center_segm_weight_nums, self.center_segm_bias_nums,
                self.center_segm_out_channels, self.center_map_kernel_size
            )
        elif mode == "extreme":
            pass

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)
        # mask_logits = mask_logits.reshape(-1, 1, H, W)
        if mode == "outer":
            mask_logits = mask_logits.reshape(n_inst, -1, mask_logits.size(2), mask_logits.size(3))
            mask_logits = self.outer_conv(mask_logits)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits_large = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits.reshape(n_inst, -1, mask_logits.size(2), mask_logits.size(3)), \
               mask_logits_large.reshape(n_inst, -1, mask_logits_large.size(2), mask_logits_large.size(3))

    def forward_all(
            self, mask_feats, mask_feat_stride, full_controller_feats,
            pred_instances, gt_outer_bitmasks=None, sal_pred=None
    ):
        # fig = sns.heatmap(data=mask_feats[0, :].mean(0).cpu())
        # fig_heatmap = fig.get_figure()
        # fig_heatmap.savefig('mask_feats.png', dpi=400)

        # fig = sns.heatmap(data=mask_feats[0, :].mean(0).cpu());fig_heatmap = fig.get_figure();fig_heatmap.savefig('mask_feats.png', dpi=400)

        tmp_instance = pred_instances[:]

        # outer map
        tmp_instance.generouter_feats = pred_instances.generouter_feats
        _, outer_logits_large = self.mask_heads_forward_with_coords(
            mask_feats, mask_feat_stride, tmp_instance, mode="outer")
        outer_score = outer_logits_large.sigmoid()

        if self.training:
            heatmaps = gt_outer_bitmasks
            # heatmaps = outer_score
            # heatmaps = self.blur_conv(heatmaps)
        else:
            # heatmaps = gt_outer_bitmasks
            heatmaps = outer_score
            # heatmaps = self.blur_conv(heatmaps)

        # center
        filter_ct = pred_instances.controller_feats[:, :self.center_segm_num_params]
        # extreme points
        filter_eps = []
        for i in range(len(pred_instances)):
            H, W = full_controller_feats[pred_instances[i].fpn_levels].shape[2:]
            per_hm = F.interpolate(heatmaps[i].unsqueeze(0), (H, W), mode='bilinear', align_corners=False)
            per_hm = per_hm / torch.clamp(per_hm.sum(-1, True).sum(-2, True), min=0.000001)

            ori_controller = full_controller_feats[pred_instances[i].fpn_levels]\
                                 [pred_instances[i].im_inds][:, self.center_segm_num_params:]
            weighted_controller = ori_controller * per_hm.permute(1, 0, 2, 3)
            filter_eps.append(weighted_controller.flatten(-2).sum(-1))
        filter_eps = torch.stack(filter_eps)

        filter_fn = self.filter_merger(filter_ct, filter_eps)
        tmp_instance.controller_feats = filter_fn
        # tmp_instance.controller_feats = filter_ct
        final_logits, final_logits_large = self.mask_heads_forward_with_coords(
            mask_feats, mask_feat_stride, tmp_instance, mode='center')
        final_score = final_logits_large.sigmoid()

        if self.training:
            return outer_score, final_score
        else:
            sal_score = None

            if self.sal_map_on:
                sal_score = (sal_pred * final_logits.sigmoid()).sum(-1).sum(-1) / final_logits.sigmoid().sum(-1).sum(-1)
                sal_score = torch.sqrt(sal_score.squeeze(-1))

            heatmaps_blur = self.blur_conv(heatmaps)
            max_ind_mask_out = heatmaps_blur.flatten(-2).max(-1).indices
            outer_loc_mask_out = torch.stack([torch.div(max_ind_mask_out, heatmaps_blur.shape[3], rounding_mode='floor'),
                                              max_ind_mask_out % heatmaps_blur.shape[3]], -1)
            outer_loc_mask_out = outer_loc_mask_out.permute(1, 0, 2)

            outer_loc_im = outer_loc_mask_out * self.mask_out_stride + 0.5 * self.mask_out_stride
            new_boxes = torch.vstack(
                [outer_loc_im[0, :, 1], outer_loc_im[2, :, 0], outer_loc_im[1, :, 1], outer_loc_im[3, :, 0]]).T

            # sal_pred_large = aligned_bilinear(sal_pred, int(mask_feat_stride / self.mask_out_stride))

            return final_score, new_boxes, sal_score
            # return sal_pred_large.repeat(final_score.shape[0], 1, 1, 1), new_boxes, sal_score

            ##### output extreme point maps ######
            # center_locations_mask_out = torch.round(pred_instances.locations / self.mask_out_stride)
            # peaks = center_locations_mask_out.flip(1)
            # # sigmas = pred_instances.sigmas / 2
            # sigmas = torch.ones(len(pred_instances)) * 2
            #
            # center_heatmaps = []
            # for sigma, peak in zip(sigmas, peaks):
            #     center_heatmaps.append(
            #         generate_heatmaps(heatmaps.shape[2], heatmaps.shape[3], peak, sigma)[None, :, :]
            #     )
            # center_heatmaps = torch.cat(center_heatmaps, dim=0).unsqueeze(1)
            #
            # heatmaps_out = torch.cat([outer_score, center_heatmaps], dim=1)
            # heatmaps_out = heatmaps_out - heatmaps_out.flatten(2).min(2).values.unsqueeze(2).unsqueeze(3)
            # heatmaps_out = heatmaps_out / heatmaps_out.flatten(2).max(2).values.unsqueeze(2).unsqueeze(3)
            # # remv = [1, 2, 3]
            # # heatmaps_out[:, remv] = 0
            # return heatmaps_out.sum(1, keepdim=True), new_boxes, sal_score
            ############################################################################################################

    def __call__(self, mask_feats, mask_feat_stride, full_controller_feats, pred_instances,
                 gt_instances=None, sal_pred=None):
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
            gt_outer_bitmasks = torch.cat([per_im.gt_outer_bitmasks for per_im in gt_instances])
            gt_outer_bitmasks = gt_outer_bitmasks[gt_inds].to(dtype=mask_feats.dtype)

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0

                losses["loss_outer"] = dummy_loss
                losses["loss_mask"] = dummy_loss * 5

            else:
                outer_score, final_mask_score = self.forward_all(
                    mask_feats, mask_feat_stride, full_controller_feats,
                    pred_instances, gt_outer_bitmasks, gt_instances
                )

                losses["loss_outer"] = dice_coefficient(outer_score, gt_outer_bitmasks).mean()
                losses["loss_mask"] = dice_coefficient(final_mask_score, gt_bitmasks).mean() * 5

            return losses
        else:
            self.test_iter += 1
            if len(pred_instances) > 0:

                final_mask_score, new_boxes, sal_score = \
                    self.forward_all(
                        mask_feats, mask_feat_stride, full_controller_feats,
                        pred_instances, sal_pred=sal_pred
                    )

                pred_instances.pred_global_masks = final_mask_score

                # if self.extreme_points_on:
                #     pred_instances.pred_boxes.tensor = new_boxes

                if self.sal_map_on:
                    pred_instances.scores = pred_instances.scores * sal_score

            return pred_instances
