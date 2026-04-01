

import os, io, math, json, random, requests, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw as PILDraw, ImageFont
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

if torch.cuda.is_available():
    GPU_ID = 0
    DEVICE = torch.device(f"cuda:{GPU_ID}")
    torch.cuda.set_device(GPU_ID)
else:
    GPU_ID = -1
    DEVICE = torch.device("cpu")


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    gpu_name = torch.cuda.get_device_name(GPU_ID)
    vram_gb  = torch.cuda.get_device_properties(GPU_ID).total_memory / 1024**3
    free_gb  = torch.cuda.mem_get_info(GPU_ID)[0] / 1024**3
    print(f" GPU {GPU_ID}: {gpu_name}  |  VRAM: {vram_gb:.1f} GB  |  Free: {free_gb:.1f} GB")
    if free_gb < 8:
        print(f"  Low VRAM: {free_gb:.1f} GB")
else:
    print(" CPU mode")

USE_AMP = torch.cuda.is_available()
scaler  = torch.cuda.amp.GradScaler(enabled=USE_AMP)


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "quickdraw_data")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")

RUN_TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", RUN_TS)
SEQ_DIR    = os.path.join(OUTPUT_DIR, "sequential")

for d in (DATA_DIR, CKPT_DIR, OUTPUT_DIR, SEQ_DIR):
    os.makedirs(d, exist_ok=True)

EDA_PATH      = os.path.join(CKPT_DIR, "eda_params.json")
PRETRAIN_PATH = os.path.join(CKPT_DIR, "pt_best.pt")

def finetune_path(cls_name: str) -> str:
    return os.path.join(CKPT_DIR, f"gen_{cls_name.replace(' ','_')}.pt")

print(f"  Output dir: {OUTPUT_DIR}")












CLASSES = [
    "airplane", "bus", "canoe", "car", "helicopter",
    "hot air balloon", "motorbike", "sailboat", "submarine", "train"
]
N_TRAIN_PER_CLASS    = 5000
N_VAL_PER_CLASS      = 2000
N_TEST_PER_CLASS     = 2000
N_PRETRAIN_PER_CLASS = 5000




N_PRIMITIVES   = 16
PRIM_LENGTH    = 0.01


TOKEN_BOS = 0; TOKEN_EOS = 1; TOKEN_SEP = 2; TOKEN_PAD = 3
SPECIAL_TOKENS = 4
VOCAB_SIZE     = SPECIAL_TOKENS + N_PRIMITIVES


N_LAYERS = 8; N_HEADS = 8; D_MODEL = 512
D_FF     = D_MODEL * 4
MAX_SEQ  = 256; DROPOUT = 0.1

MAX_SEQ_HARD_LIMIT = 512




PRETRAIN_BATCH      = 64
PRETRAIN_EPOCHS     = 15
PRETRAIN_LR         = 1e-3
GRAD_CLIP           = 1.0
EARLY_STOP_PATIENCE = 3
FINETUNE_GEN_EPOCHS = 10
FINETUNE_GEN_LR     = 1e-4



TEMPERATURE    = 1.0
TOP_K          = 10
PROMPT_RATIO   = 0.5

MIN_NEW_TOKENS = 20
