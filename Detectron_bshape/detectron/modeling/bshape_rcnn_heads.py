# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Various network "heads" for predicting masks in Mask R-CNN.

The design is as follows:

... -> RoI ----\
                -> RoIFeatureXform -> mask head -> mask output -> loss
... -> Feature /
       Map

The mask head produces a feature representation of the RoI for the purpose
of mask prediction. The mask output module converts the feature representation
into real-valued (soft) masks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_bshape_rcnn_outputs(model, blob_in, dim):
    """Add Mask R-CNN specific outputs: either mask logits or probs."""
    num_cls = cfg.MODEL.NUM_CLASSES if cfg.BSHAPE.CLS_SPECIFIC_MASK else 1

    if cfg.BSHAPE.USE_FC_OUTPUT:
        # Predict masks with a fully connected layer (ignore 'fcn' in the blob
        # name)
        dim_fc = int(dim * (cfg.BSHAPE.RESOLUTION / cfg.BSHAPE.UPSAMPLE_RATIO)**2)
        blob_out = model.FC(
            blob_in,
            'bshape_fcn_logits',
            dim_fc,
            num_cls * cfg.BSHAPE.RESOLUTION**2,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )
    else:
        # Predict mask using Conv

        # Use GaussianFill for class-agnostic mask prediction; fills based on
        # fan-in can be too large in this case and cause divergence
        fill = (
            cfg.BSHAPE.CONV_INIT
            if cfg.BSHAPE.CLS_SPECIFIC_MASK else 'GaussianFill'
        )
        blob_out = model.Conv(
            blob_in,
            'bshape_fcn_logits',
            dim,
            num_cls,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(fill, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )

        if cfg.BSHAPE.UPSAMPLE_RATIO > 1:
            blob_out = model.BilinearInterpolation(
                'bshape_fcn_logits', 'bshape_fcn_logits_up', num_cls, num_cls,
                cfg.BSHAPE.UPSAMPLE_RATIO
            )

    if not model.train:  # == if test
        blob_out = model.net.Sigmoid(blob_out, 'bshape_fcn_probs')

    return blob_out


def add_bshape_rcnn_losses(model, blob_mask):
    """Add Mask R-CNN specific losses."""
    loss_mask = model.net.SigmoidCrossEntropyLoss(
        [blob_mask, 'bshapes_int32'],
        'loss_bshape',
        scale=model.GetLossScale() * cfg.BSHAPE.WEIGHT_LOSS_MASK
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_mask])
    model.AddLosses('loss_bshape')
    return loss_gradients


# ---------------------------------------------------------------------------- #
# bshape heads
# ---------------------------------------------------------------------------- #

def bshape_rcnn_fcn_head_v1up4convs(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2."""
    return bshape_rcnn_fcn_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 4
    )


def bshape_rcnn_fcn_head_v1up4convs_gn(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2, with GroupNorm"""
    return bshape_rcnn_fcn_head_v1upXconvs_gn(
        model, blob_in, dim_in, spatial_scale, 4
    )


def bshape_rcnn_fcn_head_v1up(model, blob_in, dim_in, spatial_scale):
    """v1up design: 2 * (conv 3x3), convT 2x2."""
    return bshape_rcnn_fcn_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 2
    )


def bshape_rcnn_fcn_head_v1upXconvs(
    model, blob_in, dim_in, spatial_scale, num_convs
):
    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_[bshape]_roi_feat',
        blob_rois='bshape_rois',
        method=cfg.BSHAPE.ROI_XFORM_METHOD,
        resolution=cfg.BSHAPE.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BSHAPE.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.BSHAPE.DILATION
    dim_inner = cfg.BSHAPE.DIM_REDUCED

    for i in range(num_convs):
        current = model.Conv(
            current,
            '_[bshape]_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            kernel=3,
            dilation=dilation,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.BSHAPE.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_bshape',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.BSHAPE.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_bshape', 'conv5_bshape')

    return blob_mask, dim_inner


def bshape_rcnn_fcn_head_v1upXconvs_gn(
    model, blob_in, dim_in, spatial_scale, num_convs
):
    """v1upXconvs design: X * (conv 3x3), convT 2x2, with GroupNorm"""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_bshape_roi_feat',
        blob_rois='bshape_rois',
        method=cfg.BSHAPE.ROI_XFORM_METHOD,
        resolution=cfg.BSHAPE.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BSHAPE.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.BSHAPE.DILATION
    dim_inner = cfg.BSHAPE.DIM_REDUCED

    for i in range(num_convs):
        current = model.ConvGN(
            current,
            '_bshape_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            group_gn=get_group_gn(dim_inner),
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.BSHAPE.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_bshape',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.BSHAPE.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_bshape', 'conv5_bshape')

    return blob_mask, dim_inner


def bshape_rcnn_fcn_head_v0upshare(model, blob_in, dim_in, spatial_scale):
    """Use a ResNet "conv5" / "stage5" head for mask prediction. Weights and
    computation are shared with the conv5 box head. Computation can only be
    shared during training, since inference is cascaded.

    v0upshare design: conv5, convT 2x2.
    """
    # Since box and mask head are shared, these must match
    assert cfg.BSHAPE.ROI_XFORM_RESOLUTION == cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    if model.train:  # share computation with bbox head at training time
        dim_conv5 = 2048
        blob_conv5 = model.net.SampleAs(
            ['res5_2_sum', 'roi_has_bshape_int32'],
            ['_[bshape]_res5_2_sum_sliced']
        )
    else:  # re-compute at test time
        blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_bshapes(
            model,
            blob_in,
            dim_in,
            spatial_scale
        )

    dim_reduced = cfg.BSHAPE.DIM_REDUCED

    blob_mask = model.ConvTranspose(
        blob_conv5,
        'conv5_bshape',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.BSHAPE.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=const_fill(0.0)
    )
    model.Relu('conv5_bshape', 'conv5_bshape')

    return blob_mask, dim_reduced


def bshape_rcnn_fcn_head_v0up(model, blob_in, dim_in, spatial_scale):
    """v0up design: conv5, deconv 2x2 (no weight sharing with the box head)."""
    blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_bshapes(
        model,
        blob_in,
        dim_in,
        spatial_scale
    )

    dim_reduced = cfg.BSHAPE.DIM_REDUCED

    model.ConvTranspose(
        blob_conv5,
        'conv5_bshape',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=('GaussianFill', {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_bshape', 'conv5_bshape')

    return blob_mask, dim_reduced


def add_ResNet_roi_conv5_head_for_bshapes(model, blob_in, dim_in, spatial_scale):
    """Add a ResNet "conv5" / "stage5" head for predicting masks."""
    model.RoIFeatureTransform(
        blob_in,
        blob_out='_[bshape]_pool5',
        blob_rois='bshape_rois',
        method=cfg.BSHAPE.ROI_XFORM_METHOD,
        resolution=cfg.BSHAPE.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BSHAPE.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.BSHAPE.DILATION
    stride_init = int(cfg.BSHAPE.ROI_XFORM_RESOLUTION / 7)  # by default: 2

    s, dim_in = ResNet.add_stage(
        model,
        '_[bshape]_res5',
        '_[bshape]_pool5',
        3,
        dim_in,
        2048,
        512,
        dilation,
        stride_init=stride_init
    )

    return s, 2048
