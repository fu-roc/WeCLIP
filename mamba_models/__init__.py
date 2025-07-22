import os
from functools import partial
import torch
from omegaconf import OmegaConf

from .vmamba import VSSM, Backbone_VSSM
import yaml
from yacs.config import CfgNode as CN

def build_vssm_model(config, **kwargs):
    config = merge_configs( get_config(), config)
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = Backbone_VSSM(
            out_indices=(2,3), pretrained='pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth',
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            # ===================
            posembed=config.MODEL.VSSM.POSEMBED,
            imgsize=config.DATA.IMG_SIZE,
        )
        return model.cuda()

    return None


def build_model(config, is_pretrain=False):
    model = None
    if model is None:
        model = build_vssm_model(config)
    if model is None:
        from .simvmamba import simple_build
        model = simple_build(config.MODEL.TYPE)
    return model




def get_config():
    """Get a yacs CfgNode object for config"""
    _C = CN()

    # Base config files
    _C.BASE = ['']

    # -----------------------------------------------------------------------------
    # Data settings
    # -----------------------------------------------------------------------------
    _C.DATA = CN()
    # Batch size for a single GPU, could be overwritten by command line argument
    _C.DATA.BATCH_SIZE = 128
    # Path to dataset, could be overwritten by command line argument
    _C.DATA.DATA_PATH = ''
    # Dataset name
    _C.DATA.DATASET = 'imagenet'
    # Input image size
    _C.DATA.IMG_SIZE = 224
    # Interpolation to resize image (random, bilinear, bicubic)
    _C.DATA.INTERPOLATION = 'bicubic'
    # Use zipped dataset instead of folder dataset
    # could be overwritten by command line argument
    _C.DATA.ZIP_MODE = False
    # Cache Data in Memory, could be overwritten by command line argument
    _C.DATA.CACHE_MODE = 'part'
    # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
    _C.DATA.PIN_MEMORY = True
    # Number of data loading threads
    _C.DATA.NUM_WORKERS = 8

    # [SimMIM] Mask patch size for MaskGenerator
    _C.DATA.MASK_PATCH_SIZE = 32
    # [SimMIM] Mask ratio for MaskGenerator
    _C.DATA.MASK_RATIO = 0.6

    # -----------------------------------------------------------------------------
    # Model settings
    # -----------------------------------------------------------------------------
    _C.MODEL = CN()
    # Model type
    _C.MODEL.TYPE = 'vssm'
    # Model name
    _C.MODEL.NAME = 'vssm_tiny_224'
    # Pretrained weight from checkpoint, could be imagenet22k pretrained weight
    # could be overwritten by command line argument
    _C.MODEL.PRETRAINED = ''
    # Checkpoint to resume, could be overwritten by command line argument
    _C.MODEL.RESUME = ''
    # Number of classes, overwritten in data preparation
    _C.MODEL.NUM_CLASSES = 1000
    # Dropout rate
    _C.MODEL.DROP_RATE = 0.0
    # Drop path rate
    _C.MODEL.DROP_PATH_RATE = 0.1
    # Label Smoothing
    _C.MODEL.LABEL_SMOOTHING = 0.1

    # MMpretrain models for test
    _C.MODEL.MMCKPT = False

    # VSSM parameters
    _C.MODEL.VSSM = CN()
    _C.MODEL.VSSM.PATCH_SIZE = 4
    _C.MODEL.VSSM.IN_CHANS = 3
    _C.MODEL.VSSM.DEPTHS = [2, 2, 9, 2]
    _C.MODEL.VSSM.EMBED_DIM = 96
    _C.MODEL.VSSM.SSM_D_STATE = 16
    _C.MODEL.VSSM.SSM_RATIO = 2.0
    _C.MODEL.VSSM.SSM_RANK_RATIO = 2.0
    _C.MODEL.VSSM.SSM_DT_RANK = "auto"
    _C.MODEL.VSSM.SSM_ACT_LAYER = "silu"
    _C.MODEL.VSSM.SSM_CONV = 3
    _C.MODEL.VSSM.SSM_CONV_BIAS = True
    _C.MODEL.VSSM.SSM_DROP_RATE = 0.0
    _C.MODEL.VSSM.SSM_INIT = "v0"
    _C.MODEL.VSSM.SSM_FORWARDTYPE = "v2"
    _C.MODEL.VSSM.MLP_RATIO = 4.0
    _C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
    _C.MODEL.VSSM.MLP_DROP_RATE = 0.0
    _C.MODEL.VSSM.PATCH_NORM = True
    _C.MODEL.VSSM.NORM_LAYER = "ln"
    _C.MODEL.VSSM.DOWNSAMPLE = "v2"
    _C.MODEL.VSSM.PATCHEMBED = "v2"
    _C.MODEL.VSSM.POSEMBED = False
    _C.MODEL.VSSM.GMLP = False

    # -----------------------------------------------------------------------------
    # Training settings
    # -----------------------------------------------------------------------------
    _C.TRAIN = CN()
    _C.TRAIN.START_EPOCH = 0
    _C.TRAIN.EPOCHS = 300
    _C.TRAIN.WARMUP_EPOCHS = 20
    _C.TRAIN.WEIGHT_DECAY = 0.05
    _C.TRAIN.BASE_LR = 5e-4
    _C.TRAIN.WARMUP_LR = 5e-7
    _C.TRAIN.MIN_LR = 5e-6
    # Clip gradient norm
    _C.TRAIN.CLIP_GRAD = 5.0
    # Auto resume from latest checkpoint
    _C.TRAIN.AUTO_RESUME = True
    # Gradient accumulation steps
    # could be overwritten by command line argument
    _C.TRAIN.ACCUMULATION_STEPS = 1
    # Whether to use gradient checkpointing to save memory
    # could be overwritten by command line argument
    _C.TRAIN.USE_CHECKPOINT = False

    # LR scheduler
    _C.TRAIN.LR_SCHEDULER = CN()
    _C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
    # Epoch interval to decay LR, used in StepLRScheduler
    _C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
    # LR decay rate, used in StepLRScheduler
    _C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
    # warmup_prefix used in CosineLRScheduler
    _C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
    # [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
    _C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
    _C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

    # Optimizer
    _C.TRAIN.OPTIMIZER = CN()
    _C.TRAIN.OPTIMIZER.NAME = 'adamw'
    # Optimizer Epsilon
    _C.TRAIN.OPTIMIZER.EPS = 1e-8
    # Optimizer Betas
    _C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
    # SGD momentum
    _C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

    # [SimMIM] Layer decay for fine-tuning
    _C.TRAIN.LAYER_DECAY = 1.0

    # MoE
    _C.TRAIN.MOE = CN()
    # Only save model on master device
    _C.TRAIN.MOE.SAVE_MASTER = False
    # -----------------------------------------------------------------------------
    # Augmentation settings
    # -----------------------------------------------------------------------------
    _C.AUG = CN()
    # Color jitter factor
    _C.AUG.COLOR_JITTER = 0.4
    # Use AutoAugment policy. "v0" or "original"
    _C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
    # Random erase prob
    _C.AUG.REPROB = 0.25
    # Random erase mode
    _C.AUG.REMODE = 'pixel'
    # Random erase count
    _C.AUG.RECOUNT = 1
    # Mixup alpha, mixup enabled if > 0
    _C.AUG.MIXUP = 0.8
    # Cutmix alpha, cutmix enabled if > 0
    _C.AUG.CUTMIX = 1.0
    # Cutmix min/max ratio, overrides alpha and enables cutmix if set
    _C.AUG.CUTMIX_MINMAX = None
    # Probability of performing mixup or cutmix when either/both is enabled
    _C.AUG.MIXUP_PROB = 1.0
    # Probability of switching to cutmix when both mixup and cutmix enabled
    _C.AUG.MIXUP_SWITCH_PROB = 0.5
    # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
    _C.AUG.MIXUP_MODE = 'batch'

    # -----------------------------------------------------------------------------
    # Testing settings
    # -----------------------------------------------------------------------------
    _C.TEST = CN()
    # Whether to use center crop when testing
    _C.TEST.CROP = True
    # Whether to use SequentialSampler as validation sampler
    _C.TEST.SEQUENTIAL = False
    _C.TEST.SHUFFLE = False

    # -----------------------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------------------
    # [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
    _C.ENABLE_AMP = False

    # Enable Pytorch automatic mixed precision (amp).
    _C.AMP_ENABLE = True
    # [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
    _C.AMP_OPT_LEVEL = ''
    # Path to output folder, overwritten by command line argument
    _C.OUTPUT = ''
    # Tag of experiment, overwritten by command line argument
    _C.TAG = 'default'
    # Frequency to save checkpoint
    _C.SAVE_FREQ = 1
    # Frequency to logging info
    _C.PRINT_FREQ = 10
    # Fixed random seed
    _C.SEED = 0
    # Perform evaluation only, overwritten by command line argument
    _C.EVAL_MODE = False
    # Test throughput only, overwritten by command line argument
    _C.THROUGHPUT_MODE = False
    # Test traincost only, overwritten by command line argument
    _C.TRAINCOST_MODE = False
    # for acceleration
    _C.FUSED_LAYERNORM = False

    return _C


def cfgnode_to_dict(cfg_node):
    if not isinstance(cfg_node, CN):
        return cfg_node
    config_dict = {}
    for key, value in cfg_node.items():
        config_dict[key] = cfgnode_to_dict(value)
    return config_dict

def merge_configs(default_cfg, other_cfg):
    default_dict = cfgnode_to_dict(default_cfg)
    return OmegaConf.merge(default_dict, other_cfg)