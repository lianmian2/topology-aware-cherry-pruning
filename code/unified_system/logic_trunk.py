import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
TRUNK_CODE_DIR = os.path.join(PROJECT_ROOT, '02_code', '01_data_processing', 'trunk_extraction')

if TRUNK_CODE_DIR not in sys.path:
    sys.path.append(TRUNK_CODE_DIR)

from trunk_extractor import extract_trunk_dynamic

def run_trunk_extraction(branch_mask, tape_mask=None):
    """
    输入 branch_mask (二值化图像)
    输出 trunk_mask, remaining_branch_mask
    """
    trunk_mask, branch_mask_refined = extract_trunk_dynamic(branch_mask, tape_mask)
    return trunk_mask, branch_mask_refined
