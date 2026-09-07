import os
import sys
import torch
import warnings
import cv2
import numpy as np

# 抑制 numpy _ARRAY_API 警告
warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy: _ARRAY_API not found.*")

# 运行时配置
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
CODE_ROOT = os.path.join(PROJECT_ROOT, '02_code')
MODEL_ROOT = os.path.join(PROJECT_ROOT, '03_models')

# 添加必要的路径
ROI_V2_CODE_DIR = os.path.join(CODE_ROOT, '02_models', 'roi_locator_V2')
BRANCH_V2_CODE_DIR = os.path.join(CODE_ROOT, '02_models', 'branch_seg_csnet_v2')

if ROI_V2_CODE_DIR not in sys.path:
    sys.path.append(ROI_V2_CODE_DIR)
if BRANCH_V2_CODE_DIR not in sys.path:
    sys.path.append(BRANCH_V2_CODE_DIR)

from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
import segmentation_models_pytorch as smp
from model import CSNet  # branch_seg_csnet_v2 里的模型


def _first_existing_path(*candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("未找到可用文件，候选路径如下:\n" + "\n".join(candidates))

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            cls._instance.roi_model = None
            cls._instance.branch_model = None
            cls._instance.tape_model = None
            cls._instance.bud_global_model = None
            cls._instance.bud_local_model = None
        return cls._instance

    def load_roi_model(self):
        if self.roi_model is None:
            roi_config = os.path.join(CODE_ROOT, '02_models', 'roi_locator_V2', 'configs', 'mask_rcnn_r50_fpn_roi_v2.py')
            roi_checkpoint = _first_existing_path(
                os.path.join(MODEL_ROOT, 'roi', 'epoch_12.pth'),
                os.path.join(MODEL_ROOT, 'roi', 'epoch_11.pth'),
                os.path.join(MODEL_ROOT, 'roi', 'epoch_10.pth'),
            )
            register_all_modules()
            self.roi_model = init_detector(roi_config, roi_checkpoint, device=self.device)
            self.roi_model.eval()
        return self.roi_model

    def load_branch_model(self):
        if self.branch_model is None:
            seg_checkpoint = _first_existing_path(
                os.path.join(MODEL_ROOT, 'branch_segmentation', 'best_dice_model.pth'),
                os.path.join(MODEL_ROOT, 'branch_segmentation', 'best_iou_model.pth'),
                os.path.join(MODEL_ROOT, 'branch_segmentation', 'best_loss_model.pth'),
            )
            self.branch_model = CSNet(in_channels=3, n_classes=1)
            ckpt = torch.load(seg_checkpoint, map_location=self.device)
            self.branch_model.load_state_dict(ckpt['model_state_dict'])
            self.branch_model.to(self.device)
            self.branch_model.eval()
        return self.branch_model

    def load_tape_model(self):
        if self.tape_model is None:
            model_path = _first_existing_path(
                os.path.join(MODEL_ROOT, 'Tape_Segmentation_V2_outputs', 'best_model.pth'),
                os.path.join(MODEL_ROOT, 'Tape_Segmentation_outputs', 'best_model.pth'),
            )
            self.tape_model = smp.Unet(
                encoder_name="mobilenet_v2",
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )
            self.tape_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.tape_model.to(self.device)
            self.tape_model.eval()
        return self.tape_model

    def load_bud_global_model(self):
        if self.bud_global_model is None:
            bud_config = os.path.join(CODE_ROOT, '02_models', 'bud_final', 'mask-rcnn_hrnetv2p-w32-1x_coco.py')
            bud_checkpoint = _first_existing_path(
                os.path.join(MODEL_ROOT, 'compag', 'bud_baseline_deloc_patch', 'best.pth'),
                os.path.join(MODEL_ROOT, 'cherry_bud_ui_release_bud_hrnet_ROI_bud_min_v2', 'best_coco_segm_mAP_epoch_15.pth'),
                os.path.join(MODEL_ROOT, 'cherry_bud_ui_release_bud_hrnet_ROI_bud_min_v2', 'best_coco_segm_mAP_epoch_40.pth'),
            )
            register_all_modules()
            self.bud_global_model = init_detector(bud_config, bud_checkpoint, device=self.device)
            self.bud_global_model.eval()
            for module in self.bud_global_model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()
        return self.bud_global_model

    def load_bud_local_model(self):
        if self.bud_local_model is None:
            config = os.path.join(CODE_ROOT, '02_models', 'bud_final', 'mask-rcnn_hrnetv2p-w32-1x_coco.py')
            checkpoint = _first_existing_path(
                os.path.join(MODEL_ROOT, 'compag', 'bud_baseline_deloc_patch', 'best.pth'),
                os.path.join(MODEL_ROOT, 'cherry_bud_ui_release_bud_hrnet_ROI_bud_min_v2', 'best_coco_segm_mAP_epoch_15.pth'),
                os.path.join(MODEL_ROOT, 'cherry_bud_ui_release_bud_hrnet_ROI_bud_min_v2', 'best_coco_segm_mAP_epoch_40.pth'),
            )
            register_all_modules()
            self.bud_local_model = init_detector(config, checkpoint, device=self.device)
            self.bud_local_model.eval()
            for module in self.bud_local_model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()
        return self.bud_local_model
