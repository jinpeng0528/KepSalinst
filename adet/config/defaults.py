from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True
_C.INPUT.CROP.CROP_INSTANCE = True

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# The options for the quality of box prediction
# It can be "ctrness" (as described in FCOS paper) or "iou"
# Using "iou" here generally has ~0.4 better AP on COCO
# Note that for compatibility, we still use the term "ctrness" in the code
_C.MODEL.FCOS.BOX_QUALITY = "ctrness"

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0

# The normalizer of the classification loss
# The normalizer can be "fg" (normalized by the number of the foreground samples),
# "moving_fg" (normalized by the MOVING number of the foreground samples),
# or "all" (normalized by the number of all samples)
_C.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
_C.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0

_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
_C.MODEL.FCOS.YIELD_PROPOSAL = False
_C.MODEL.FCOS.YIELD_BOX_FEATURES = False

# ---------------------------------------------------------------------------- #
# KepSalinst Options
# ---------------------------------------------------------------------------- #
_C.MODEL.KEPSALINST = CN()
_C.MODEL.KEPSALINST.SAL_MAP_ON = False

# the downsampling ratio of the final instance masks to the input image
_C.MODEL.KEPSALINST.MASK_OUT_STRIDE = 4
_C.MODEL.KEPSALINST.BOTTOM_PIXELS_REMOVED = -1

# if not -1, we only compute the mask loss for MAX_PROPOSALS random proposals PER GPU
_C.MODEL.KEPSALINST.MAX_PROPOSALS = -1
# if not -1, we only compute the mask loss for top `TOPK_PROPOSALS_PER_IM` proposals
# PER IMAGE in terms of their detection scores
_C.MODEL.KEPSALINST.TOPK_PROPOSALS_PER_IM = -1

_C.MODEL.KEPSALINST.MASK_HEAD = CN()
_C.MODEL.KEPSALINST.MASK_HEAD.CHANNELS = 8
_C.MODEL.KEPSALINST.MASK_HEAD.NUM_LAYERS = 3
_C.MODEL.KEPSALINST.MASK_HEAD.OUTER_MAP_NUM_LAYERS = 3
_C.MODEL.KEPSALINST.MASK_HEAD.SEGM_MAP_NUM_LAYERS = 3
_C.MODEL.KEPSALINST.MASK_HEAD.USE_FP16 = False
_C.MODEL.KEPSALINST.MASK_HEAD.DISABLE_REL_COORDS = False

_C.MODEL.KEPSALINST.MASK_HEAD.OUTER_MAP_NUM_LAYERS = 3
_C.MODEL.KEPSALINST.MASK_HEAD.OUTER_MAP_KERNEL_SIZE = 1
_C.MODEL.KEPSALINST.MASK_HEAD.OUTER_MAP_DILATION = 1
_C.MODEL.KEPSALINST.MASK_HEAD.CENTER_MAP_NUM_LAYERS = 3
_C.MODEL.KEPSALINST.MASK_HEAD.CENTER_MAP_KERNEL_SIZE = 1
_C.MODEL.KEPSALINST.MASK_HEAD.CENTER_MAP_DILATION = 1
_C.MODEL.KEPSALINST.MASK_HEAD.EXTREME_MAP_NUM_LAYERS = 3
_C.MODEL.KEPSALINST.MASK_HEAD.EXTREME_MAP_KERNEL_SIZE = 1
_C.MODEL.KEPSALINST.MASK_HEAD.EXTREME_MAP_DILATION = 1

_C.MODEL.KEPSALINST.MASK_BRANCH = CN()
_C.MODEL.KEPSALINST.MASK_BRANCH.OUT_CHANNELS = 8
_C.MODEL.KEPSALINST.MASK_BRANCH.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.KEPSALINST.MASK_BRANCH.CHANNELS = 128
_C.MODEL.KEPSALINST.MASK_BRANCH.NORM = "BN"
_C.MODEL.KEPSALINST.MASK_BRANCH.NUM_CONVS = 4