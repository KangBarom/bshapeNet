from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
from modeling import ResNet
import nn as mynn
import utils.net as net_utils


# ---------------------------------------------------------------------------- #
# bshape R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

class bshape_rcnn_outputs(nn.Module):
    """bshape R-CNN specific outputs: either bshape logits or probs."""
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in

        n_classes = cfg.MODEL.NUM_CLASSES if cfg.BSHAPE.CLS_SPECIFIC_MASK else 1
        if cfg.BSHAPE.USE_FC_OUTPUT:
            # Predict bshapes with a fully connected layer
            self.classify = nn.Linear(dim_in, n_classes * cfg.BSHAPE.RESOLUTION**2)
        else:
            # Predict bshape using Conv
            self.classify = nn.Conv2d(dim_in, n_classes, 1, 1, 0)
            if cfg.BSHAPE.UPSAMPLE_RATIO > 1:
                self.upsample = mynn.BilinearInterpolation2d(
                    n_classes, n_classes, cfg.BSHAPE.UPSAMPLE_RATIO)
        self._init_weights()

    def _init_weights(self):
        if not cfg.BSHAPE.USE_FC_OUTPUT and cfg.BSHAPE.CLS_SPECIFIC_MASK and \
                cfg.BSHAPE.CONV_INIT=='MSRAFill':
            # Use GaussianFill for class-agnostic bshape prediction; fills based on
            # fan-in can be too large in this case and cause divergence
            weight_init_func = mynn.init.MSRAFill
        else:
            weight_init_func = partial(init.normal_, std=0.001)
        weight_init_func(self.classify.weight)
        init.constant_(self.classify.bias, 0)

    def detectron_weight_mapping(self):
        mapping = {
            'classify.weight': 'bshape_fcn_logits_w',
            'classify.bias': 'bshape_fcn_logits_b'
        }
        if hasattr(self, 'upsample'):
            mapping.update({
                'upsample.upconv.weight': None,  # don't load from or save to checkpoint
                'upsample.upconv.bias': None
            })
        orphan_in_detectron = []
        return mapping, orphan_in_detectron

    def forward(self, x):
        if not isinstance(x, list):
            x = self.classify(x)
        else:
            x[0] = self.classify(x[0])
            x[1] = x[1].view(-1, 1, cfg.BSHAPE.RESOLUTION, cfg.BSHAPE.RESOLUTION)
            x[1] = x[1].repeat(1, cfg.MODEL.NUM_CLASSES, 1, 1)
            x = x[0] + x[1]
        if cfg.BSHAPE.UPSAMPLE_RATIO > 1:
            x = self.upsample(x)
        if not self.training:
            x = F.sigmoid(x)
        return x
# def bshape_rcnn_losses(bshape_pred, rois_bshape, rois_label, weight):
#     n_rois, n_classes, _, _ = bshape_pred.size()
#     rois_bshape_label = rois_label[weight.data.nonzero().view(-1)]
#     # select pred bshape corresponding to gt label
#     if cfg.BSHAPE.MEMORY_EFFICIENT_LOSS:  # About 200~300 MB less. Not really sure how.
#         bshape_pred_select = Variable(
#             bshape_pred.data.new(n_rois, cfg.BSHAPE.RESOLUTION,
#                                cfg.BSHAPE.RESOLUTION))
#         for n, l in enumerate(rois_bshape_label.data):
#             bshape_pred_select[n] = bshape_pred[n, l]
#     else:
#         inds = rois_bshape_label.data + \
#           torch.arange(0, n_rois * n_classes, n_classes).long().cuda(rois_bshape_label.data.get_device())
#         bshape_pred_select = bshape_pred.view(-1, cfg.BSHAPE.RESOLUTION,
#                                           cfg.BSHAPE.RESOLUTION)[inds]
#     loss = F.binary_cross_entropy_with_logits(bshape_pred_select, rois_bshape)
#     return loss


def bshape_rcnn_losses(bshapes_pred, bshapes_int32):
    """bshape R-CNN specific losses."""
    n_rois, n_classes, _, _ = bshapes_pred.size()
    device_id = bshapes_pred.get_device()
    bshapes_gt = Variable(torch.from_numpy(bshapes_int32.astype('float32'))).cuda(device_id)
    weight = (bshapes_gt > -1).float()  # bshapes_int32 {1, 0, -1}, -1 means ignore
    loss = F.binary_cross_entropy_with_logits(
        bshapes_pred.view(n_rois, -1), bshapes_gt, weight, size_average=False)
    loss /= weight.sum()
    return loss * cfg.BSHAPE.WEIGHT_LOSS_MASK


# ---------------------------------------------------------------------------- #
# bshape heads
# ---------------------------------------------------------------------------- #

def bshape_rcnn_fcn_head_v1up4convs(dim_in, roi_xform_func, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2."""
    return bshape_rcnn_fcn_head_v1upXconvs(
        dim_in, roi_xform_func, spatial_scale, 4
    )


def bshape_rcnn_fcn_head_v1up4convs_gn(dim_in, roi_xform_func, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2, with GroupNorm"""
    return bshape_rcnn_fcn_head_v1upXconvs_gn(
        dim_in, roi_xform_func, spatial_scale, 4
    )
            
def bshape_rcnn_fcn_head_v1up4convs_gn_adp(dim_in, roi_xform_func, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2, with GroupNorm"""
    return bshape_rcnn_fcn_head_v1upXconvs_gn_adp(
        dim_in, roi_xform_func, spatial_scale, 4
    )

def bshape_rcnn_fcn_head_v1up4convs_gn_adp_ff(dim_in, roi_xform_func, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2, with GroupNorm"""
    return bshape_rcnn_fcn_head_v1upXconvs_gn_adp_ff(
        dim_in, roi_xform_func, spatial_scale, 4
    )

def bshape_rcnn_fcn_head_v1up(dim_in, roi_xform_func, spatial_scale):
    """v1up design: 2 * (conv 3x3), convT 2x2."""
    return bshape_rcnn_fcn_head_v1upXconvs(
        dim_in, roi_xform_func, spatial_scale, 2
    )


class bshape_rcnn_fcn_head_v1upXconvs(nn.Module):
    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale, num_convs):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.num_convs = num_convs

        dilation = cfg.BSHAPE.DILATION
        dim_inner = cfg.BSHAPE.DIM_REDUCED
        self.dim_out = dim_inner

        module_list = []
        for i in range(num_convs):
            module_list.extend([
                nn.Conv2d(dim_in, dim_inner, 3, 1, padding=1*dilation, dilation=dilation),
                nn.ReLU(inplace=True)
            ])
            dim_in = dim_inner
        self.conv_fcn = nn.Sequential(*module_list)

        # upsample layer
        self.upconv = nn.ConvTranspose2d(dim_inner, dim_inner, 2, 2, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if cfg.BSHAPE.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.001)
            elif cfg.BSHAPE.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                raise ValueError
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        for i in range(self.num_convs):
            mapping_to_detectron.update({
                'conv_fcn.%d.weight' % (2*i): '_[bshape]_fcn%d_w' % (i+1),
                'conv_fcn.%d.bias' % (2*i): '_[bshape]_fcn%d_b' % (i+1)
            })
        mapping_to_detectron.update({
            'upconv.weight': 'conv5_bshape_w',
            'upconv.bias': 'conv5_bshape_b'
        })

        return mapping_to_detectron, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='bshape_rois',
            method=cfg.BSHAPE.ROI_XFORM_METHOD,
            resolution=cfg.BSHAPE.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.BSHAPE.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.conv_fcn(x)
        return F.relu(self.upconv(x), inplace=True)


class bshape_rcnn_fcn_head_v1upXconvs_gn(nn.Module):
    """v1upXconvs design: X * (conv 3x3), convT 2x2, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale, num_convs):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.num_convs = num_convs

        dilation = cfg.BSHAPE.DILATION
        dim_inner = cfg.BSHAPE.DIM_REDUCED
        self.dim_out = dim_inner

        module_list = []
        for i in range(num_convs):
            module_list.extend([
                nn.Conv2d(dim_in, dim_inner, 3, 1, padding=1*dilation, dilation=dilation, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(dim_inner), dim_inner, eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = dim_inner
        self.conv_fcn = nn.Sequential(*module_list)

        # upsample layer
        self.upconv = nn.ConvTranspose2d(dim_inner, dim_inner, 2, 2, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if cfg.BSHAPE.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.001)
            elif cfg.BSHAPE.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                raise ValueError
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        for i in range(self.num_convs):
            mapping_to_detectron.update({
                'conv_fcn.%d.weight' % (3*i): '_bshape_fcn%d_w' % (i+1),
                'conv_fcn.%d.weight' % (3*i+1): '_bshape_fcn%d_gn_s' % (i+1),
                'conv_fcn.%d.bias' % (3*i+1): '_bshape_fcn%d_gn_b' % (i+1)
            })
        mapping_to_detectron.update({
            'upconv.weight': 'conv5_bshape_w',
            'upconv.bias': 'conv5_bshape_b'
        })

        return mapping_to_detectron, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='bshape_rois',
            method=cfg.BSHAPE.ROI_XFORM_METHOD,
            resolution=cfg.BSHAPE.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.BSHAPE.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.conv_fcn(x)
        return F.relu(self.upconv(x), inplace=True)


class bshape_rcnn_fcn_head_v0upshare(nn.Module):
    """Use a ResNet "conv5" / "stage5" head for bshape prediction. Weights and
    computation are shared with the conv5 box head. Computation can only be
    shared during training, since inference is cascaded.

    v0upshare design: conv5, convT 2x2.
    """
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = cfg.BSHAPE.DIM_REDUCED
        self.SHARE_RES5 = True
        assert cfg.MODEL.SHARE_RES5

        self.res5 = None  # will be assigned later
        dim_conv5 = 2048
        self.upconv5 = nn.ConvTranspose2d(dim_conv5, self.dim_out, 2, 2, 0)

        self._init_weights()

    def _init_weights(self):
        if cfg.BSHAPE.CONV_INIT == 'GaussianFill':
            init.normal_(self.upconv5.weight, std=0.001)
        elif cfg.BSHAPE.CONV_INIT == 'MSRAFill':
            mynn.init.MSRAFill(self.upconv5.weight)
        init.constant_(self.upconv5.bias, 0)

    def share_res5_module(self, res5_target):
        """ Share res5 block with box head on training """
        self.res5 = res5_target

    def detectron_weight_mapping(self):
        detectron_weight_mapping, orphan_in_detectron = \
          ResNet.residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
        # Assign None for res5 modules, do not load from or save to checkpoint
        for k in detectron_weight_mapping:
            detectron_weight_mapping[k] = None

        detectron_weight_mapping.update({
            'upconv5.weight': 'conv5_bshape_w',
            'upconv5.bias': 'conv5_bshape_b'
        })
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rpn_ret, roi_has_bshape_int32=None):
        if self.training:
            # On training, we share the res5 computation with bbox head, so it's necessary to
            # sample 'useful' batches from the input x (res5_2_sum). 'Useful' means that the
            # batch (roi) has corresponding bshape groundtruth, namely having positive values in
            # roi_has_bshape_int32.
            inds = np.nonzero(roi_has_bshape_int32 > 0)[0]
            inds = Variable(torch.from_numpy(inds)).cuda(x.get_device())
            x = x[inds]
        else:
            # On testing, the computation is not shared with bbox head. This time input `x`
            # is the output features from the backbone network
            x = self.roi_xform(
                x, rpn_ret,
                blob_rois='bshape_rois',
                method=cfg.BSHAPE.ROI_XFORM_METHOD,
                resolution=cfg.BSHAPE.ROI_XFORM_RESOLUTION,
                spatial_scale=self.spatial_scale,
                sampling_ratio=cfg.BSHAPE.ROI_XFORM_SAMPLING_RATIO
            )
            x = self.res5(x)
        x = self.upconv5(x)
        x = F.relu(x, inplace=True)
        return x


class bshape_rcnn_fcn_head_v0up(nn.Module):
    """v0up design: conv5, deconv 2x2 (no weight sharing with the box head)."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = cfg.BSHAPE.DIM_REDUCED

        self.res5, dim_out = ResNet_roi_conv5_head_for_bshapes(dim_in)
        self.upconv5 = nn.ConvTranspose2d(dim_out, self.dim_out, 2, 2, 0)

        # Freeze all bn (affine) layers in resnet!!!
        self.res5.apply(
            lambda m: ResNet.freeze_params(m)
            if isinstance(m, mynn.AffineChannel2d) else None)
        self._init_weights()

    def _init_weights(self):
        if cfg.BSHAPE.CONV_INIT == 'GaussianFill':
            init.normal_(self.upconv5.weight, std=0.001)
        elif cfg.BSHAPE.CONV_INIT == 'MSRAFill':
            mynn.init.MSRAFill(self.upconv5.weight)
        init.constant_(self.upconv5.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping, orphan_in_detectron = \
          ResNet.residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
        detectron_weight_mapping.update({
            'upconv5.weight': 'conv5_bshape_w',
            'upconv5.bias': 'conv5_bshape_b'
        })
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='bshape_rois',
            method=cfg.BSHAPE.ROI_XFORM_METHOD,
            resolution=cfg.BSHAPE.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.BSHAPE.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.res5(x)
        # print(x.size()) e.g. (128, 2048, 7, 7)
        x = self.upconv5(x)
        x = F.relu(x, inplace=True)
        return x


def ResNet_roi_conv5_head_for_bshapes(dim_in):
    """ResNet "conv5" / "stage5" head for predicting bshapes."""
    dilation = cfg.BSHAPE.DILATION
    stride_init = cfg.BSHAPE.ROI_XFORM_RESOLUTION // 7  # by default: 2
    module, dim_out = ResNet.add_stage(dim_in, 2048, 512, 3, dilation, stride_init)
    return module, dim_out





class bshape_rcnn_fcn_head_v1upXconvs_gn_adp(nn.Module):
    """v1upXconvs design: X * (conv 3x3), convT 2x2, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale, num_convs):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.num_convs = num_convs

        dilation = cfg.BSHAPE.DILATION
        dim_inner = cfg.BSHAPE.DIM_REDUCED
        self.dim_out = dim_inner

        module_list = []
        for i in range(num_convs - 1):
            module_list.extend([
                nn.Conv2d(dim_in, dim_inner, 3, 1, padding=1*dilation, dilation=dilation, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(dim_inner), dim_inner, eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = dim_inner
        self.conv_fcn = nn.Sequential(*module_list)

        self.bshape_conv1 = nn.ModuleList()
        num_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
        for i in range(num_levels):
            self.bshape_conv1.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_inner, 3, 1, padding=1*dilation, dilation=dilation, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(dim_inner), dim_inner, eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ))


        # upsample layer
        self.upconv = nn.ConvTranspose2d(dim_inner, dim_inner, 2, 2, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if cfg.BSHAPE.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.001)
            elif cfg.BSHAPE.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                raise ValueError
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        for i in range(self.num_convs):
            mapping_to_detectron.update({
                'conv_fcn.%d.weight' % (3*i): '_bshape_fcn%d_w' % (i+1),
                'conv_fcn.%d.weight' % (3*i+1): '_bshape_fcn%d_gn_s' % (i+1),
                'conv_fcn.%d.bias' % (3*i+1): '_bshape_fcn%d_gn_b' % (i+1)
            })
        mapping_to_detectron.update({
            'upconv.weight': 'conv5_bshape_w',
            'upconv.bias': 'conv5_bshape_b'
        })

        return mapping_to_detectron, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='bshape_rois',
            method=cfg.BSHAPE.ROI_XFORM_METHOD,
            resolution=cfg.BSHAPE.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.BSHAPE.ROI_XFORM_SAMPLING_RATIO,
            panet=True
        )
        for i in range(len(x)):
            x[i] = self.bshape_conv1[i](x[i])
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]
        x = self.conv_fcn(x)
        return F.relu(self.upconv(x), inplace=True)



class bshape_rcnn_fcn_head_v1upXconvs_gn_adp_ff(nn.Module):
    """v1upXconvs design: X * (conv 3x3), convT 2x2, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale, num_convs):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.num_convs = num_convs

        dilation = cfg.BSHAPE.DILATION
        dim_inner = cfg.BSHAPE.DIM_REDUCED
        self.dim_out = dim_inner

        module_list = []
        for i in range(2):
            module_list.extend([
                nn.Conv2d(dim_in, dim_inner, 3, 1, padding=1*dilation, dilation=dilation, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(dim_inner), dim_inner, eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = dim_inner
        self.conv_fcn = nn.Sequential(*module_list)

        self.bshape_conv1 = nn.ModuleList()
        num_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
        for i in range(num_levels):
            self.bshape_conv1.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_inner, 3, 1, padding=1*dilation, dilation=dilation, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(dim_inner), dim_inner, eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ))

        self.bshape_conv4 = nn.Sequential(
                nn.Conv2d(dim_in, dim_inner, 3, 1, padding=1*dilation, dilation=dilation, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(dim_inner), dim_inner, eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True))

        self.bshape_conv4_fc = nn.Sequential(
                nn.Conv2d(dim_in, dim_inner, 3, 1, padding=1*dilation, dilation=dilation, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(dim_inner), dim_inner, eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True))

        self.bshape_conv5_fc = nn.Sequential(
                nn.Conv2d(dim_in, int(dim_inner / 2), 3, 1, padding=1*dilation, dilation=dilation, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(dim_inner), int(dim_inner / 2), eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True))

        self.bshape_fc = nn.Sequential(
                nn.Linear(int(dim_inner / 2) * (cfg.BSHAPE.ROI_XFORM_RESOLUTION) ** 2, cfg.BSHAPE.RESOLUTION ** 2, bias=True),
                nn.ReLU(inplace=True))



        # upsample layer
        self.upconv = nn.ConvTranspose2d(dim_inner, dim_inner, 2, 2, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if cfg.BSHAPE.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.001)
            elif cfg.BSHAPE.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                raise ValueError
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.01)
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        for i in range(self.num_convs):
            mapping_to_detectron.update({
                'conv_fcn.%d.weight' % (3*i): '_bshape_fcn%d_w' % (i+1),
                'conv_fcn.%d.weight' % (3*i+1): '_bshape_fcn%d_gn_s' % (i+1),
                'conv_fcn.%d.bias' % (3*i+1): '_bshape_fcn%d_gn_b' % (i+1)
            })
        mapping_to_detectron.update({
            'upconv.weight': 'conv5_bshape_w',
            'upconv.bias': 'conv5_bshape_b'
        })

        return mapping_to_detectron, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='bshape_rois',
            method=cfg.BSHAPE.ROI_XFORM_METHOD,
            resolution=cfg.BSHAPE.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.BSHAPE.ROI_XFORM_SAMPLING_RATIO,
            panet=True
        )
        for i in range(len(x)):
            x[i] = self.bshape_conv1[i](x[i])
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]
        x = self.conv_fcn(x)
        batch_size = x.size(0)
        x_fcn = F.relu(self.upconv(self.bshape_conv4(x)), inplace=True)
        x_ff = self.bshape_fc(self.bshape_conv5_fc(self.bshape_conv4_fc(x)).view(batch_size, -1))

        return [x_fcn, x_ff]
