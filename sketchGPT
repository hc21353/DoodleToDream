"""
SketchGPT https://arxiv.org/abs/2405.03099 중 generation(completion) 부분만 구현

"""

# 1. 임포트
import os
import io
import math
import json
import random
import requests
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 서버 headless 환경: 화면 없이 파일로만 저장
# 로컬 GUI 환경이면 "TkAgg"로 변경하면 창으로 뜸
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw   # 256×256 sequential stroke 저장

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm                   
from sklearn.metrics import classification_report, confusion_matrix

# 2. 장치 설정 — RTX 3090
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

assert torch.cuda.is_available(), "CUDA not found. Please check: nvidia-smi"
DEVICE = torch.device("cuda")

# 메모리 파편화 방지 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# RTX 3090 최적화
torch.backends.cuda.matmul.allow_tf32 = True   # matmul TF32 가속
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True    # 입력 크기 고정 시 자동 최적 커널

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
free_gb  = torch.cuda.mem_get_info(0)[0] / 1024**3
print(f" {gpu_name}  |  VRAM: {vram_gb:.1f} GB  |  Free: {free_gb:.1f} GB")
if free_gb < 8:
    print(f"  Low VRAM: {free_gb:.1f} GB free — kill other processes: nvidia-smi")

# AMP (Automatic Mixed Precision): fp16 forward + fp32 optimizer
USE_AMP = True
scaler  = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# 3. 경로 설정
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))   # 스크립트 위치
DATA_DIR    = os.path.join(BASE_DIR, "quickdraw_data")     # ndjson 캐시
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints")        # 모델 가중치
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")            # 생성 이미지

for d in (DATA_DIR, CKPT_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

EDA_PATH      = os.path.join(CKPT_DIR, "eda_params.json")
PRETRAIN_PATH = os.path.join(CKPT_DIR, "pt_best.pt")
FINETUNE_PATH = os.path.join(CKPT_DIR, "gen_best.pt")


# 4. 하이퍼파라미터
#  논문 명시값 — 변경 불가
#   모델 : L=8, A=8, H=512  (Table 3 optimal)
#   D_FF : 2048 (H×4)
#   Train: 5K / Val: 2.5K / Test: 2.5K per class 
#   Pre-train: 5K / class 
#   Temperature: 1.0~1.4 
#   특수토큰: BOS, EOS, SEP, PAD
#
#   논문 미명시값 — EDA로 결정 또는 RTX 3090 기준 설정
#   N_PRIMITIVES, MAX_SEQ, PRIM_LENGTH, Optimizer, LR, Scheduler, Batch, Dropout

# 데이터 — 논문 기준값
# 10클래스 × (5K train + 2K val + 2K test) = 90K 총 샘플
CLASSES = [
    "airplane", "bus", "canoe", "car", "helicopter",
    "hot air balloon", "motorbike", "sailboat", "submarine", "train"
]
N_TRAIN_PER_CLASS    = 5000   # 논문 4.5절 기준 (5K)
N_VAL_PER_CLASS      = 2000   # 검증용 (논문 2.5K에서 소폭 축소)
N_TEST_PER_CLASS     = 2000
N_PRETRAIN_PER_CLASS = 5000

# Primitive: 16개(22.5° 간격); 해당 값 변경 시 재학습 필요 (vocab_size 변경됨)
#   - pre-train은 클래스 label을 직접 쓰지 않으므로 기존 가중치로 시작 가능
#   - fine-tune은 새 데이터로 재학습 권장
N_PRIMITIVES = 16
PRIM_LENGTH  = 0.01

# 특수 토큰 
TOKEN_BOS = 0; TOKEN_EOS = 1; TOKEN_SEP = 2; TOKEN_PAD = 3
SPECIAL_TOKENS = 4
VOCAB_SIZE = SPECIAL_TOKENS + N_PRIMITIVES

# 모델 구조 
N_LAYERS = 8
N_HEADS  = 8
D_MODEL  = 512
D_FF     = D_MODEL * 4    # 2048
MAX_SEQ  = 256
DROPOUT  = 0.1
# EDA 결과 95th=572 → 512로 올려서 복잡한 클래스도 충분히 표현
MAX_SEQ_HARD_LIMIT = 512

# 학습 — loss 그래프 기반 조정
# pre-train: val이 12~13ep에서 flat → 15ep + patience=3으로 충분
# fine-tune: val이 처음부터 flat → 10ep + patience=3
PRETRAIN_BATCH      = 64     # 10클래스 5K = 50K samples, batch 64 적합
PRETRAIN_EPOCHS     = 15     # 손실 그래프상 12~13ep 수렴 확인
PRETRAIN_LR         = 1e-3
GRAD_CLIP           = 1.0
EARLY_STOP_PATIENCE = 3      # val flat 빠르게 감지

FINETUNE_GEN_EPOCHS = 10
FINETUNE_GEN_LR     = 1e-4

# 생성
# temperature 낮출수록 더 결정론적 (덜 랜덤), 높을수록 다양하지만 노이즈
TEMPERATURE = 1.0    # 논문 best range 1.0~1.4 중 하한 (곡선 표현에 유리)
TOP_K       = 10

# 5. DataLoader 헬퍼
# RTX 3090: num_workers=4, pin_memory=True
def make_loader(dataset, batch_size, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=4,           # RTX 3090 서버: 4 worker 적합
        pin_memory=True,         # CPU→GPU 비동기 전송
        drop_last=drop_last,
        persistent_workers=True, # worker 재생성 오버헤드 제거
    )


# 6. 데이터 다운로드 & 전처리
def download_ndjson(class_name: str, max_n: int,
                    recognized_only: bool = True) -> list:
    """
    QuickDraw simplified ndjson 다운로드 
    recognized=True: 게임 AI가 인식 성공한 스케치만 사용 
    """
    url_name  = class_name.replace(" ", "%20")
    url       = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{url_name}.ndjson"
    save_path = os.path.join(DATA_DIR, f"{class_name.replace(' ','_')}.ndjson")

    if not os.path.exists(save_path):
        print(f"  다운로드: {class_name} ...")
        r = requests.get(url, stream=True); r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)
    else:
        print(f"  캐시: {class_name}")

    sketches = []; total_seen = 0
    with open(save_path) as f:
        for line in f:
            total_seen += 1
            data = json.loads(line)
            if recognized_only and not data.get("recognized", True):
                continue
            sketches.append(data["drawing"])
            if len(sketches) >= max_n: break

    print(f"  로드: {len(sketches)}개  (검토 {total_seen}개, recognized 필터)")
    return sketches


def drawing_to_stroke3(drawing: list) -> np.ndarray:
    """drawing([[x들],[y들]]) → stroke-3: (Δx, Δy, pen_lift)"""
    pts, prev_x, prev_y = [], 0, 0
    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        for i in range(len(xs)):
            pts.append([xs[i]-prev_x, ys[i]-prev_y, 1 if i==len(xs)-1 else 0])
            prev_x, prev_y = xs[i], ys[i]
    return np.array(pts, dtype=np.float32)


def normalize_stroke3(s3: np.ndarray) -> np.ndarray:
    """논문 3.1절: min-max 정규화 → [0,1]  (delta → 절대 → 정규화 → delta)"""
    abs_xy = np.cumsum(s3[:, :2], axis=0)
    mn, mx = abs_xy.min(0), abs_xy.max(0)
    denom  = np.where(mx-mn < 1e-8, 1.0, mx-mn)
    norm   = (abs_xy - mn) / denom
    delta  = np.diff(norm, axis=0, prepend=norm[:1])
    out = s3.copy(); out[:, :2] = delta
    return out

# 7. Stroke → Primitive 매핑 & 토크나이저 
def build_primitives(n: int) -> np.ndarray:
    """등간격 방향 단위벡터 집합 (논문 3.1절 Fig.2)"""
    angles = np.linspace(0, 2*math.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

PRIMITIVES = build_primitives(N_PRIMITIVES)


def prim_id(dx: float, dy: float) -> int:
    """수식 1,2: 코사인 유사도 → argmax → primitive 인덱스"""
    vec = np.array([dx, dy])
    n   = np.linalg.norm(vec)
    if n < 1e-8: return 0
    return int(np.argmax(PRIMITIVES @ (vec / n)))


def scale_factor(dx: float, dy: float) -> int:
    """수식 3: T(p_i, s_i) = ⌈ m(s_i) / m(p_i) ⌉  (ceiling)"""
    L = math.sqrt(dx**2 + dy**2)
    if L < 1e-8: return 1
    return max(1, min(8, math.ceil(L / PRIM_LENGTH)))


def tokenize(s3: np.ndarray) -> list:
    """
    수식 4,5: stroke → primitive 토큰 시퀀스
    T = [BOS, p1*T1, SEP, p2*T2, SEP, ..., pn*Tn, EOS]
    """
    tokens = [TOKEN_BOS]
    for i, (dx, dy, lift) in enumerate(s3):
        tok   = SPECIAL_TOKENS + prim_id(dx, dy)
        scale = scale_factor(dx, dy)
        tokens.extend([tok] * scale)
        if lift == 1 and i < len(s3)-1:
            tokens.append(TOKEN_SEP)
    tokens.append(TOKEN_EOS)
    return tokens

# 8. EDA — N_PRIMITIVES, MAX_SEQ, PRIM_LENGTH 데이터 기반 결정
def run_eda(classes: list, n_sample: int = 500) -> tuple:
    """
    데이터 분포 분석 후 3개 파라미터 결정:
      N_PRIMITIVES : stroke 방향 분포 → 16 (QuickDraw 균일 분포)
      MAX_SEQ      : 토큰 길이 95th percentile → 2의 제곱수 올림
      PRIM_LENGTH  : 정규화 후 stroke 길이 중앙값 (scale_factor 의미 있게)
    """
    global PRIMITIVES

    print("\n" + "="*55)
    print("  EDA: N_PRIMITIVES / MAX_SEQ / PRIM_LENGTH")
    print("="*55)

    all_angles, all_lengths, all_stroke_lens = [], [], []

    for cls_name in classes:
        drawings = download_ndjson(cls_name, n_sample*3, recognized_only=True)
        count = 0
        for d in drawings:
            s3 = drawing_to_stroke3(d)
            if len(s3) < 5: continue
            s3 = normalize_stroke3(s3)
            for dx, dy, _ in s3:
                L = math.sqrt(dx**2 + dy**2)
                if L > 1e-8:
                    all_angles.append(math.atan2(dy, dx))
                    all_stroke_lens.append(L)
            all_lengths.append(len(tokenize(s3)))
            count += 1
            if count >= n_sample: break

    all_angles      = np.array(all_angles)
    all_lengths     = np.array(all_lengths)
    all_stroke_lens = np.array(all_stroke_lens)

    # 시각화 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(np.degrees(all_angles), bins=36, range=(-180,180),
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axhline(len(all_angles)/36, color='red', ls='--', label='Uniform baseline')
    axes[0].set(xlabel='Direction (deg)', ylabel='Count', title='Stroke Direction Distribution')
    axes[0].legend(); axes[0].grid(alpha=.3)

    p50_l = np.percentile(all_stroke_lens, 50)
    axes[1].hist(all_stroke_lens, bins=60,
                 range=(0, np.percentile(all_stroke_lens,99)),
                 color='coral', edgecolor='white', alpha=0.8)
    axes[1].axvline(p50_l, color='red', ls='--', lw=2,
                    label=f'median={p50_l:.4f} -> PRIM_LENGTH')
    axes[1].set(xlabel='Stroke length (normalized)', title='Stroke Length Distribution')
    axes[1].legend(); axes[1].grid(alpha=.3)

    p95 = np.percentile(all_lengths, 95)
    axes[2].hist(all_lengths, bins=50, color='mediumpurple',
                 edgecolor='white', alpha=0.8)
    axes[2].axvline(p95, color='red', ls='--', lw=2,
                    label=f'95th={p95:.0f} -> MAX_SEQ')
    axes[2].set(xlabel='Token length', title='Token Length Distribution')
    axes[2].legend(); axes[2].grid(alpha=.3)

    plt.suptitle(f"EDA: {', '.join(classes)}")
    plt.tight_layout()
    _save_and_show("eda_analysis.png")

    # N_PRIMITIVES: EDA로 결정하는 값이 아니라 설계 결정값 사용
    # 16 → 22.5° 간격 (각짐), 32 → 11.25° 간격 (부드러운 곡선 표현 가능)
    rec_n_prim   = N_PRIMITIVES   # 전역 설정값 32 그대로 사용 
    rec_prim_len = float(p50_l)
    raw_max      = int(p95)
    rec_max_seq  = min(2**math.ceil(math.log2(max(raw_max, 64))), MAX_SEQ_HARD_LIMIT)

    scales = [max(1, min(8, math.ceil(l/rec_prim_len)))
              for l in all_stroke_lens[:2000]]
    print(f"\n  N_PRIMITIVES  = {rec_n_prim}")
    print(f"  PRIM_LENGTH   = {rec_prim_len:.5f}")
    print(f"  MAX_SEQ       = {rec_max_seq}  (95th={raw_max})")
    print(f"  scale_factor 분포: { {i:scales.count(i) for i in range(1,6)} }")

    return rec_n_prim, rec_max_seq, rec_prim_len

# 9. Dataset
class SketchDataset(Dataset):
    """Pre-training용: 전체 시퀀스"""
    def __init__(self, tokens_list, labels, max_seq):
        self.items = []
        for toks, lbl in zip(tokens_list, labels):
            toks = list(toks[:max_seq-1])
            if TOKEN_EOS not in toks: toks.append(TOKEN_EOS)
            toks += [TOKEN_PAD] * (max_seq - len(toks))
            self.items.append((torch.tensor(toks, dtype=torch.long), int(lbl)))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


class PartialSketchDataset(Dataset):
    """
    Fine-tuning용: 논문 3.4절 "partially drawn sketches"
    매 호출마다 시퀀스의 min_ratio~max_ratio 사이 랜덤 위치에서 자름
    → 모델이 앞부분 보고 나머지 완성하는 능력 학습
    """
    def __init__(self, tokens_list, labels, max_seq,
                 min_ratio=0.1, max_ratio=0.9):
        self.max_seq   = max_seq
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.raw = []
        for toks, lbl in zip(tokens_list, labels):
            toks = list(toks[:max_seq-1])
            if TOKEN_EOS not in toks: toks.append(TOKEN_EOS)
            eos_idx = toks.index(TOKEN_EOS)
            self.raw.append((toks[:eos_idx+1], int(lbl)))

    def __len__(self): return len(self.raw)

    def __getitem__(self, i):
        toks, lbl = self.raw[i]
        n = len(toks)
        if n < 4:
            cut = n
        else:
            min_cut = max(2, int(n * self.min_ratio))
            max_cut = max(min_cut+1, int(n * self.max_ratio))
            cut = random.randint(min_cut, max_cut)
        partial = toks[:cut]
        if TOKEN_EOS not in partial: partial.append(TOKEN_EOS)
        partial += [TOKEN_PAD] * (self.max_seq - len(partial))
        return torch.tensor(partial[:self.max_seq], dtype=torch.long), lbl


def build_datasets(classes, n_train, n_val, n_test, n_pretrain, max_seq):
    """
    논문 4.5절: 클래스당 고정 절댓값 (5K/2.5K/2.5K)
    recognized=True 필터 후 3배 여유분 다운로드
    """
    per_cls = []

    for cls_idx, cls_name in enumerate(classes):
        needed = n_train + n_val + n_test
        drawings = download_ndjson(cls_name, needed*3, recognized_only=True)

        cls_toks = []
        for d in drawings:
            s3 = drawing_to_stroke3(d)
            if len(s3) < 5: continue
            s3 = normalize_stroke3(s3)
            toks = tokenize(s3)
            if len(toks) < 5: continue
            cls_toks.append(toks)
            if len(cls_toks) >= needed: break

        if len(cls_toks) < needed:
            print(f"    {cls_name}: {len(cls_toks)}개 (목표 {needed})")
        per_cls.append(cls_toks)

    tr_toks, tr_labs = [], []
    va_toks, va_labs = [], []
    te_toks, te_labs = [], []
    cls_counts       = []

    for ci, cls_toks in enumerate(per_cls):
        random.shuffle(cls_toks)
        n = len(cls_toks)
        t1 = min(n_train, int(n*0.5))
        t2 = min(n_val,   int(n*0.25))

        tr_toks.extend(cls_toks[:t1]);         tr_labs.extend([ci]*t1)
        va_toks.extend(cls_toks[t1:t1+t2]);   va_labs.extend([ci]*t2)
        te_toks.extend(cls_toks[t1+t2:t1+t2*2]); te_labs.extend([ci]*t2)
        cls_counts.append(t1)
        print(f"  {classes[ci]}: train={t1}, val={t2}, test={t2}")

    # 클래스 균형 가중치
    cls_w  = [1.0/c for c in cls_counts]
    sw     = [cls_w[l] for l in tr_labs]

    # 셔플
    def shuffle_pair(a, b):
        c = list(zip(a,b)); random.shuffle(c)
        return zip(*c) if c else ([],[])

    tr_t, tr_l = shuffle_pair(tr_toks, tr_labs)
    va_t, va_l = shuffle_pair(va_toks, va_labs)
    te_t, te_l = shuffle_pair(te_toks, te_labs)

    tr_t2, tr_l2 = list(tr_t), list(tr_l)
    sw = [cls_w[l] for l in tr_l2]

    train_ds   = SketchDataset(tr_t2, tr_l2, max_seq)
    val_ds     = SketchDataset(list(va_t), list(va_l), max_seq)
    test_ds    = SketchDataset(list(te_t), list(te_l), max_seq)
    partial_tr = PartialSketchDataset(tr_t2, tr_l2, max_seq)
    partial_va = PartialSketchDataset(list(va_t), list(va_l), max_seq)

    print(f"\n  [분할] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    lens = [len(t) for t in tr_toks]
    print(f"  [토큰] 평균={np.mean(lens):.0f} 최대={np.max(lens)} "
          f">{max_seq}: {np.mean(np.array(lens)>max_seq):.1%}")

    return train_ds, val_ds, test_ds, partial_tr, partial_va, sw

# 10. 모델 (논문 수식 6,7,8 — Table 3 optimal: L=8/A=8/H=512)
class CausalSelfAttention(nn.Module):
    """
    수식 6: MaskedAttention(X) = softmax(X·W_Q·(X·W_K)^T ⊙ Mask / √d_k) · X·W_V
    수식 7: MultiHead(X) = Concat(head_1,...,head_h) · W_O
    """
    def __init__(self, d_model, n_heads, max_seq, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.qkv  = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model,   bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq, max_seq))
        self.register_buffer("causal_mask", mask.view(1,1,max_seq,max_seq))

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)

        def to_heads(t):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1,2)
        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.causal_mask[:,:,:L,:L]==0, float('-inf'))
        w      = self.attn_drop(torch.softmax(scores, dim=-1))
        out    = (w @ v).transpose(1,2).contiguous().view(B, L, D)
        return self.resid_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-LN GPT-2 스타일 블록"""
    def __init__(self, d_model, n_heads, d_ff, max_seq, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SketchGPT(nn.Module):
    """
    논문 3.2절: GPT-2 inspired decoder-only transformer
    Generation only (cls_head 없음)
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 d_ff, max_seq, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 수식 8
        self.apply(self._init_weights)

        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [SketchGPT] L={n_layers}/A={n_heads}/H={d_model}  "
              f"params={n_p:,} ({n_p/1e6:.1f}M)")

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, tokens):
        """수식 8: → (B, L, vocab_size)"""
        B, L = tokens.shape
        pos  = torch.arange(L, device=tokens.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        for blk in self.blocks: x = blk(x)
        return self.lm_head(self.ln_f(x))

# 11. 손실 & 평가
def lm_loss(logits, tokens):
    """수식 8: NLL = CrossEntropy(logits[:,:-1], tokens[:,1:])  PAD 무시"""
    return F.cross_entropy(
        logits[:,:-1].contiguous().view(-1, logits.size(-1)),
        tokens[:,1:].contiguous().view(-1),
        ignore_index=TOKEN_PAD
    )

@torch.no_grad()
def eval_lm(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for toks, _ in loader:
        toks = toks.to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            total += lm_loss(model(toks), toks).item()
        n += 1
    return total / max(n, 1)

# 12. Pre-training (논문 3.3절, 수식 8)
def pretrain(model, train_ds, val_ds, device, sample_weights=None,
             epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LR,
             batch=PRETRAIN_BATCH, save_path=PRETRAIN_PATH):
    """비지도 NTP 사전학습: 전체 시퀀스, 모든 클래스 혼합"""
    print("\n" + "="*55)
    print(f"  PHASE 1: Pre-training  (epochs={epochs}, batch={batch}, lr={lr})")
    print("="*55)

    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.float),
            len(train_ds), replacement=True
        )
        print("  WeightedRandomSampler 적용")

    tr = make_loader(train_ds, batch, sampler=sampler,
                     shuffle=(sampler is None), drop_last=True)
    va = make_loader(val_ds, batch)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9,0.95))
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.1)

    tr_ls, va_ls, best, no_improve = [], [], float('inf'), 0

    for ep in range(1, epochs+1):
        model.train(); ep_loss = 0.0
        bar = tqdm(tr, desc=f"PT {ep}/{epochs}", ncols=90)
        for toks, _ in bar:
            toks = toks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = lm_loss(model(toks), toks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            lr=f"{sch.get_last_lr()[0]:.5f}")

        sch.step()
        va_loss = eval_lm(model, va, device)
        tr_loss = ep_loss / len(tr)
        tr_ls.append(tr_loss); va_ls.append(va_loss)
        ppl = math.exp(min(va_loss, 20))
        print(f"  Ep {ep:3d} | train={tr_loss:.4f} val={va_loss:.4f} "
              f"ppl={ppl:.1f} lr={sch.get_last_lr()[0]:.5f}")

        if va_loss < best:
            best = va_loss; no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  ⏹ Early stopping (ep={ep})")
                break

    _plot_loss(tr_ls, va_ls, "Pre-training Loss", "pretrain_loss.png")
    model.load_state_dict(torch.load(save_path, map_location=device,
                                     weights_only=True))
    print(f" Pre-training 완료  best_val={best:.4f}  저장: {save_path}")
    return model

# 13. Fine-tuning (논문 3.4절 — partial sketch completion)
def finetune_gen(model, partial_train_ds, partial_val_ds, device,
                 sample_weights=None,
                 epochs=FINETUNE_GEN_EPOCHS, lr=FINETUNE_GEN_LR,
                 save_path=FINETUNE_PATH):
    """
    논문 3.4절: "feeding the model with partially drawn sketches"
    PartialSketchDataset: 매 step 랜덤 비율로 잘린 시퀀스 → completion 학습
    """
    print("\n" + "="*55)
    print(f"  PHASE 2: Fine-tuning Generation  (epochs={epochs}, lr={lr})")
    print("="*55)

    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.float),
            len(partial_train_ds), replacement=True
        )

    tr = make_loader(partial_train_ds, PRETRAIN_BATCH,
                     sampler=sampler, shuffle=(sampler is None), drop_last=True)
    va = make_loader(partial_val_ds, PRETRAIN_BATCH)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.1)

    tr_ls, va_ls, best, no_improve = [], [], float('inf'), 0

    for ep in range(1, epochs+1):
        model.train(); ep_loss = 0.0
        bar = tqdm(tr, desc=f"GenFT {ep}/{epochs}", ncols=90)
        for toks, _ in bar:
            toks = toks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = lm_loss(model(toks), toks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        sch.step()
        va_loss = eval_lm(model, va, device)
        tr_loss = ep_loss / len(tr)
        tr_ls.append(tr_loss); va_ls.append(va_loss)
        print(f"  Ep {ep:3d} | train={tr_loss:.4f} val={va_loss:.4f} "
              f"lr={sch.get_last_lr()[0]:.5f}")

        if va_loss < best:
            best = va_loss; no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  ⏹ Early stopping (ep={ep})")
                break

    _plot_loss(tr_ls, va_ls, "Generation Fine-tuning Loss", "gen_finetune_loss.png")
    model.load_state_dict(torch.load(save_path, map_location=device,
                                     weights_only=True))
    print(f" Fine-tuning 완료  best_val={best:.4f}  저장: {save_path}")
    return model

# 14. 자기회귀 생성 (논문 4.2절, 4.6절)
@torch.no_grad()
def generate(model, device, prompt=None, max_new=None,
             temperature=TEMPERATURE, top_k=TOP_K,
             min_new_tokens=20):
    """
    논문 4.6절 ablation: temperature 1.0~1.4 최적.

    min_new_tokens: 최소 이 수만큼 생성하기 전까지 EOS 금지.
    → 모델이 너무 일찍 EOS를 내서 그림을 미완성으로 끝내는 문제 방지.
    """
    if max_new is None:
        max_new = MAX_SEQ
    model.eval()
    toks = list(prompt) if prompt else [TOKEN_BOS]
    generated = 0   # prompt 이후 새로 생성한 토큰 수

    for _ in range(max_new):
        ctx   = torch.tensor([toks[-MAX_SEQ:]], dtype=torch.long, device=device)
        logit = model(ctx)[0, -1]
        logit[TOKEN_PAD] = float('-inf')

        # 최소 토큰 수 미달 시 EOS 금지
        if generated < min_new_tokens:
            logit[TOKEN_EOS] = float('-inf')

        logit /= temperature
        if top_k > 0:
            v, _ = logit.topk(min(top_k, logit.size(-1)))
            logit[logit < v[-1]] = float('-inf')
        tok = torch.multinomial(F.softmax(logit, -1), 1).item()
        toks.append(tok)
        generated += 1
        if tok == TOKEN_EOS: break
    return toks

# 15. 시각화 유틸
def toks_to_strokes(toks):
    """
    토큰 시퀀스 → stroke 좌표 리스트 (시각화용).

    step_size=0.03:
      - 0.05는 너무 커서 stroke가 캔버스를 빠르게 벗어나 클리핑됨
      - 0.03이면 원본 QuickDraw 스케치와 비슷한 크기/밀도로 그려짐
      - 매 primitive 토큰마다 점 하나씩 찍고 SEP/EOS에서 획 구분
    """
    step_size = 0.03
    polylines   = []
    current_pts = []
    x, y = 0.5, 0.5   # 캔버스 중앙 시작

    for t in toks:
        if t in (TOKEN_BOS, TOKEN_PAD):
            continue
        if t == TOKEN_EOS:
            if len(current_pts) >= 2:
                polylines.append(current_pts)
            break
        if t == TOKEN_SEP:
            if len(current_pts) >= 2:
                polylines.append(current_pts)
            current_pts = [(x, y)]
            continue

        pid = t - SPECIAL_TOKENS
        if 0 <= pid < N_PRIMITIVES:
            d = PRIMITIVES[pid]
            if not current_pts:
                current_pts.append((x, y))
            x = float(np.clip(x + d[0] * step_size, 0.0, 1.0))
            y = float(np.clip(y + d[1] * step_size, 0.0, 1.0))
            current_pts.append((x, y))   # 매 토큰마다 점 추가

    if len(current_pts) >= 2:
        polylines.append(current_pts)
    return polylines


def draw(polylines, ax, title="", color="black", smooth=True):
    """
    폴리라인을 ax에 그림. 첫 번째 코드 방식: 고정 [0,1] 범위.
    smooth=True: scipy 스플라인으로 꺾인 선을 부드럽게 (곡선 표현 향상).
    """
    if not polylines:
        ax.axis('off'); ax.set_title(title, fontsize=8); return

    for pts in polylines:
        if len(pts) < 2: continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        if smooth and len(pts) >= 4:
            try:
                from scipy.interpolate import splprep, splev
                tck, u = splprep([xs, ys], s=0, k=min(3, len(pts)-1))
                u_new  = np.linspace(0, 1, len(pts) * 5)
                xs_s, ys_s = splev(u_new, tck)
                ax.plot(xs_s, ys_s, color=color, lw=1.5,
                        solid_capstyle='round', solid_joinstyle='round')
            except Exception:
                ax.plot(xs, ys, color=color, lw=1.5,
                        solid_capstyle='round', solid_joinstyle='round')
        else:
            ax.plot(xs, ys, color=color, lw=1.5,
                    solid_capstyle='round', solid_joinstyle='round')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.invert_yaxis(); ax.set_aspect('equal')
    ax.axis('off'); ax.set_title(title, fontsize=8)


def draw_original_quickdraw(drawing, ax, title="", color="black"):
    """QuickDraw 원본 drawing을 역변환 없이 직접 그림."""
    all_x, all_y = [], []
    for s in drawing: all_x.extend(s[0]); all_y.extend(s[1])
    if not all_x: return
    xr = max(all_x)-min(all_x) or 1; yr = max(all_y)-min(all_y) or 1
    PAD = 0.05
    for s in drawing:
        xs = [(x-min(all_x))/xr*(1-2*PAD)+PAD for x in s[0]]
        ys = [(y-min(all_y))/yr*(1-2*PAD)+PAD for y in s[1]]
        ax.plot(xs, ys, color=color, lw=1.5,
                solid_capstyle='round', solid_joinstyle='round')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.invert_yaxis()
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=8)


def _save_and_show(fname: str):
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [저장] {path}")
    plt.close()


def _plot_loss(tr, va, title, fname):
    fig, ax = plt.subplots(figsize=(7, 3))
    ep = range(1, len(tr)+1)
    ax.plot(ep, tr, 'b-o', ms=3, label='Train')
    ax.plot(ep, va, 'r-o', ms=3, label='Val')
    ax.set(xlabel='Epoch', ylabel='NLL Loss', title=title)
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    _save_and_show(fname)


def show_raw_samples(n=4):
    """
    QuickDraw 원본 데이터 직접 시각화 (역변환 없음).
    모델/데이터셋 없이 독립 실행 가능.
    """
    # 클래스 수에 맞게 자동으로 색상 생성
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10', len(CLASSES))
    COLS = [matplotlib.colors.to_hex(cmap(i)) for i in range(len(CLASSES))]

    fig, axes = plt.subplots(len(CLASSES), n,
                             figsize=(n*2.8, len(CLASSES)*2.8))
    for ci, cname in enumerate(CLASSES):
        drawings = download_ndjson(cname, n*5, recognized_only=True)
        samples  = random.sample(drawings, min(n, len(drawings)))
        for j, d in enumerate(samples):
            draw_original_quickdraw(
                d, axes[ci][j],
                title=f"{cname}\n(original)", color=COLS[ci]
            )
    plt.suptitle("QuickDraw 원본 데이터 (역변환 없음)", y=1.01)
    plt.tight_layout()
    _save_and_show("raw_samples.png")


def _ax_to_pil(ax, img_size=256):
    """
    matplotlib axes를 PIL Image로 추출.
    axes에 그려진 것을 그대로 뽑으므로 show_generated와 100% 동일.
    """
    import io
    fig = ax.get_figure()
    buf = io.BytesIO()
    # ax 영역만 bbox로 잘라서 저장
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    bbox_inch = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(buf, format='png', dpi=150,
                bbox_inches=bbox_inch, facecolor='white')
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((img_size, img_size),
                                                Image.LANCZOS)
    return img


def show_generated(model, train_ds, device, n=4,
                   save_sequential=True, img_size=256):
    """
    원본(위) vs 생성(아래) 비교 그리드.
    save_sequential=True: 생성 이미지를 sequential stroke 단위로도 저장.
    generated_sketches.png와 sequential 이미지가 완전히 동일한 렌더링.
    """
    PROMPT_RATIO   = 0.2
    MIN_NEW_TOKENS = 40
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10', len(CLASSES))
    COLS = [matplotlib.colors.to_hex(cmap(i)) for i in range(len(CLASSES))]

    SEQ_DIR = os.path.join(OUTPUT_DIR, "sequential")

    cls_idx_map = {i: [] for i in range(len(CLASSES))}
    for idx, (_, lbl) in enumerate(train_ds.items):
        cls_idx_map[lbl].append(idx)

    fig, axes = plt.subplots(len(CLASSES)*2, n,
                             figsize=(n*2.8, len(CLASSES)*5.6))
    # renderer를 미리 확보
    fig.canvas.draw()

    for ci, cname in enumerate(CLASSES):
        sample_idxs = random.sample(cls_idx_map[ci],
                                    min(n, len(cls_idx_map[ci])))
        for j, idx in enumerate(sample_idxs):
            toks_tensor, _ = train_ds.items[idx]
            full_toks = toks_tensor.tolist()
            real_len  = next((i for i, t in enumerate(full_toks)
                              if t == TOKEN_EOS), len(full_toks))

            # ── 원본 행 ──
            ax_orig = axes[ci*2][j]
            orig_strokes = toks_to_strokes(full_toks)
            draw(orig_strokes, ax_orig,
                 f"{cname}\n(original)", COLS[ci])

            # ── 생성 행 ──
            ax_gen = axes[ci*2+1][j]
            prompt_len = max(5, int(real_len * PROMPT_RATIO))
            prompt     = [t for t in full_toks[:prompt_len]
                          if t not in (TOKEN_EOS, TOKEN_PAD)]
            gen_toks   = generate(model, device, prompt=prompt,
                                  min_new_tokens=MIN_NEW_TOKENS)
            gen_polylines = toks_to_strokes(gen_toks)
            draw(gen_polylines, ax_gen,
                 f"{cname}\n(generated #{j+1})", COLS[ci])

            # ── Sequential 저장 (생성 ax 기준) ──────────────
            if save_sequential and gen_polylines:
                sample_dir = os.path.join(
                    SEQ_DIR, cname.replace(" ", "_"), f"sample_{j:02d}")
                os.makedirs(sample_dir, exist_ok=True)

                # 전체 완성 그림 → ax를 렌더링 후 PIL 저장
                # 누적 stroke는 동일 draw() 파이프라인으로 별도 figure 생성
                for k in range(1, len(gen_polylines) + 1):
                    dpi = 100
                    sz  = img_size / dpi
                    tmp_fig, tmp_ax = plt.subplots(figsize=(sz, sz), dpi=dpi)
                    tmp_fig.patch.set_facecolor('white')
                    draw(gen_polylines[:k], tmp_ax, title="", color="black")
                    tmp_fig.tight_layout(pad=0)
                    import io
                    buf = io.BytesIO()
                    tmp_fig.savefig(buf, format='png', dpi=dpi,
                                    bbox_inches='tight', pad_inches=0.02,
                                    facecolor='white')
                    plt.close(tmp_fig)
                    buf.seek(0)
                    pil_img = Image.open(buf).convert("RGB").resize(
                        (img_size, img_size), Image.LANCZOS)
                    pil_img.save(os.path.join(sample_dir,
                                              f"stroke_{k:03d}.png"))

                # stroke_all = 전체 (마지막 누적과 동일)
                dpi = 100; sz = img_size / dpi
                tmp_fig, tmp_ax = plt.subplots(figsize=(sz, sz), dpi=dpi)
                tmp_fig.patch.set_facecolor('white')
                draw(gen_polylines, tmp_ax, title="", color="black")
                tmp_fig.tight_layout(pad=0)
                buf = io.BytesIO()
                tmp_fig.savefig(buf, format='png', dpi=dpi,
                                bbox_inches='tight', pad_inches=0.02,
                                facecolor='white')
                plt.close(tmp_fig)
                buf.seek(0)
                pil_all = Image.open(buf).convert("RGB").resize(
                    (img_size, img_size), Image.LANCZOS)
                pil_all.save(os.path.join(sample_dir, "stroke_all.png"))

                # JSON + NPY
                stroke_data = [
                    {"stroke_id": i,
                     "points": [[float(x), float(y)] for x, y in pl]}
                    for i, pl in enumerate(gen_polylines)
                ]
                with open(os.path.join(sample_dir, "strokes.json"), "w") as f:
                    json.dump({"class": cname, "sample_idx": j,
                               "n_strokes": len(gen_polylines),
                               "img_size": img_size,
                               "strokes": stroke_data}, f, indent=2)

                npy_records = []
                for sid, pl in enumerate(gen_polylines):
                    for pid, (x, y) in enumerate(pl):
                        npy_records.append((sid, pid, x, y))
                npy_arr = np.array(npy_records,
                                   dtype=[('stroke', np.int32),
                                          ('point',  np.int32),
                                          ('x',      np.float32),
                                          ('y',      np.float32)])
                np.save(os.path.join(sample_dir, "strokes.npy"), npy_arr)

                print(f"  {cname} sample_{j:02d}: "
                      f"{len(gen_polylines)} strokes saved -> {sample_dir}")

    plt.suptitle("Original (odd rows) vs Generated (even rows)", y=1.01)
    plt.tight_layout()
    _save_and_show("generated_sketches.png")

def _polylines_to_pil(polylines, img_size=256, smooth=True):
    """
    toks_to_strokes 결과 폴리라인 → PIL Image (256×256, 흰 배경, 검은 선).
    show_generated의 draw()와 완전히 동일한 렌더링 파이프라인 사용.
    matplotlib → numpy array → PIL 변환으로 일관성 보장.
    """
    import io
    dpi = 100
    px  = img_size / dpi
    fig, ax = plt.subplots(figsize=(px, px), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    draw(polylines, ax, title="", color="black", smooth=smooth)

    # 여백 제거 후 numpy 변환
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi,
                bbox_inches='tight', pad_inches=0,
                facecolor='white')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((img_size, img_size))
    return img


def save_sequential_strokes(model, train_ds, device,
                            n_per_class: int = 3,
                            img_size: int = 256):
    """
    클래스별 생성 스케치를 stroke 누적 단위로 저장.

    저장 구조:
      outputs/sequential/<class_name>/sample_<si>/
        stroke_001.png   ← stroke 1개 누적
        stroke_002.png   ← stroke 1+2 누적
        ...
        stroke_all.png   ← 전체 완성
        strokes.json     ← raw 좌표 (class, n_strokes, points per stroke)
        strokes.npy      ← structured numpy array (stroke, point, x, y)

    렌더링: show_generated와 동일한 파이프라인 (_polylines_to_pil)
    이미지: 256×256 RGB, 흰 배경, 검은 선
    """
    SEQ_DIR = os.path.join(OUTPUT_DIR, "sequential")

    cls_idx_map = {i: [] for i in range(len(CLASSES))}
    for idx, (_, lbl) in enumerate(train_ds.items):
        cls_idx_map[lbl].append(idx)

    print(f"\n[Sequential] Saving to: {SEQ_DIR}")

    for ci, cname in enumerate(CLASSES):
        avail = cls_idx_map[ci]
        if not avail:
            continue
        sample_idxs = random.sample(avail, min(n_per_class, len(avail)))

        for si, idx in enumerate(sample_idxs):
            # 생성 (show_generated와 동일 파라미터)
            toks_tensor, _ = train_ds.items[idx]
            full_toks  = toks_tensor.tolist()
            real_len   = next((i for i, t in enumerate(full_toks)
                               if t == TOKEN_EOS), len(full_toks))
            prompt_len = max(5, int(real_len * 0.2))
            prompt     = [t for t in full_toks[:prompt_len]
                          if t not in (TOKEN_EOS, TOKEN_PAD)]
            gen_toks   = generate(model, device, prompt=prompt,
                                  min_new_tokens=40)

            # ── toks → polylines (show_generated와 동일 함수) ──
            polylines = toks_to_strokes(gen_toks)
            if not polylines:
                continue

            # 저장 경로 
            sample_dir = os.path.join(SEQ_DIR,
                                      cname.replace(" ", "_"),
                                      f"sample_{si:02d}")
            os.makedirs(sample_dir, exist_ok=True)

            # JSON 저장 
            stroke_data = [
                {"stroke_id": i,
                 "points": [[float(x), float(y)] for x, y in pl]}
                for i, pl in enumerate(polylines)
            ]
            with open(os.path.join(sample_dir, "strokes.json"), "w") as f:
                json.dump({"class": cname, "sample_idx": si,
                           "n_strokes": len(polylines),
                           "img_size": img_size,
                           "strokes": stroke_data}, f, indent=2)

            # NPY 저장 
            npy_records = []
            for sid, pl in enumerate(polylines):
                for pid, (x, y) in enumerate(pl):
                    npy_records.append((sid, pid, x, y))
            npy_arr = np.array(npy_records,
                               dtype=[('stroke', np.int32),
                                      ('point',  np.int32),
                                      ('x',      np.float32),
                                      ('y',      np.float32)])
            np.save(os.path.join(sample_dir, "strokes.npy"), npy_arr)

            # 이미지 누적 저장 
            for k in range(1, len(polylines) + 1):
                img = _polylines_to_pil(polylines[:k], img_size)
                img.save(os.path.join(sample_dir, f"stroke_{k:03d}.png"))

            # 전체 완성본
            img_all = _polylines_to_pil(polylines, img_size)
            img_all.save(os.path.join(sample_dir, "stroke_all.png"))

            print(f"  {cname} sample_{si:02d}: "
                  f"{len(polylines)} strokes saved")

    print(f"[Sequential] Done: {SEQ_DIR}")
def main(
    skip_eda      = False,
    skip_pretrain = False,
    skip_finetune = False,
    reuse_pretrain_weights = False,  # 기존 pt_best.pt로 시작 (새 데이터로 fine-tune만)
):
    """
    단계별 실행:
      main()                                              # 처음부터 전부
      main(skip_eda=True)                                 # EDA 재사용
      main(skip_eda=True, skip_pretrain=True)             # fine-tune만
      main(skip_eda=True, skip_pretrain=True,
           skip_finetune=True)                            # 시각화만
      main(reuse_pretrain_weights=True,
           skip_eda=True)                                 # 기존 pt 가중치로 fine-tune만

    클래스 추가 시:
      - N_PRIMITIVES=16 유지 → vocab_size=20 동일 → pre-train 가중치 구조 호환
      - 단, 새 클래스 데이터 분포가 포함되지 않아 fine-tune은 재학습 필요
      - reuse_pretrain_weights=True로 기존 가중치를 시작점으로 fine-tune 가능
    """
    global N_PRIMITIVES, VOCAB_SIZE, MAX_SEQ, PRIM_LENGTH, PRIMITIVES

    print("="*60)
    print("  SketchGPT — RTX 3090 로컬")
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  클래스 : {len(CLASSES)}개")
    print(f"  모델   : L={N_LAYERS}/A={N_HEADS}/H={D_MODEL}")
    print(f"  저장   : {CKPT_DIR}")
    print("="*60)

    # ── Step 0: EDA ─────────────────────────────────────
    if skip_eda and os.path.exists(EDA_PATH):
        print("\n▶ Step 0: EDA 로드")
        with open(EDA_PATH) as f: eda = json.load(f)
        if eda["n_primitives"] != N_PRIMITIVES:
            raise ValueError(
                f"N_PRIMITIVES 불일치: 저장={eda['n_primitives']}, "
                f"현재={N_PRIMITIVES}. checkpoints/ 삭제 후 main()으로 재실행."
            )
        N_PRIMITIVES = eda["n_primitives"]
        MAX_SEQ      = eda["max_seq"]
        PRIM_LENGTH  = eda["prim_length"]
        VOCAB_SIZE   = SPECIAL_TOKENS + N_PRIMITIVES
        PRIMITIVES   = build_primitives(N_PRIMITIVES)
        print(f"  N_PRIMITIVES={N_PRIMITIVES}  MAX_SEQ={MAX_SEQ}  "
              f"PRIM_LENGTH={PRIM_LENGTH:.5f}")
    else:
        print("\n▶ Step 0: EDA 실행")
        rec_n_prim, rec_max_seq, rec_prim_len = run_eda(CLASSES, n_sample=500)
        N_PRIMITIVES = rec_n_prim
        VOCAB_SIZE   = SPECIAL_TOKENS + N_PRIMITIVES
        MAX_SEQ      = min(rec_max_seq, MAX_SEQ_HARD_LIMIT)
        PRIM_LENGTH  = rec_prim_len
        PRIMITIVES   = build_primitives(N_PRIMITIVES)
        with open(EDA_PATH, "w") as f:
            json.dump({"n_primitives": N_PRIMITIVES,
                       "max_seq": MAX_SEQ,
                       "prim_length": PRIM_LENGTH,
                       "classes": CLASSES}, f, indent=2)
        print(f"  EDA 저장: {EDA_PATH}")

    # ── Step 1: 데이터 ───────────────────────────────────
    print("\n▶ Step 1: 데이터 준비")
    train_ds, val_ds, test_ds, partial_tr, partial_va, sw = build_datasets(
        CLASSES,
        N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, N_TEST_PER_CLASS,
        N_PRETRAIN_PER_CLASS, MAX_SEQ
    )

    # ── Step 2: 모델 초기화 ──────────────────────────────
    print("\n▶ Step 2: 모델 초기화")
    model = SketchGPT(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq=MAX_SEQ, dropout=DROPOUT
    ).to(DEVICE)

    # ── Step 3: Pre-training ─────────────────────────────
    if skip_pretrain and os.path.exists(PRETRAIN_PATH):
        print(f"\n▶ Step 3: Pre-training 체크포인트 로드")
        model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE,
                                         weights_only=True))
        print(f"  로드: {PRETRAIN_PATH}")
    elif reuse_pretrain_weights and os.path.exists(PRETRAIN_PATH):
        # 기존 가중치로 시작해서 새 데이터로 pre-train 이어서
        print(f"\n▶ Step 3: 기존 Pre-train 가중치로 warm-start")
        model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE,
                                         weights_only=True))
        print(f"  기존 가중치 로드 후 새 데이터로 추가 학습")
        model = pretrain(model, train_ds, val_ds, DEVICE,
                         sample_weights=sw,
                         epochs=5,       # warm-start: 5 epoch만
                         lr=PRETRAIN_LR * 0.3)  # 낮은 LR로 조정
    else:
        print("\n▶ Step 3: Pre-training 실행")
        model = pretrain(model, train_ds, val_ds, DEVICE,
                         sample_weights=sw)

    # ── Step 4: Fine-tuning ──────────────────────────────
    if skip_finetune and os.path.exists(FINETUNE_PATH):
        print(f"\n▶ Step 4: Fine-tuning 체크포인트 로드")
        model.load_state_dict(torch.load(FINETUNE_PATH, map_location=DEVICE,
                                         weights_only=True))
        print(f"  로드: {FINETUNE_PATH}")
    else:
        print("\n▶ Step 4: Fine-tuning 실행")
        model = finetune_gen(model, partial_tr, partial_va, DEVICE,
                             sample_weights=sw)

    # ── Step 5: 시각화 + Sequential 저장 ────────────────
    print("\n[Step 5-A] Raw QuickDraw samples")
    show_raw_samples(n=4)

    print("\n[Step 5-B] Generated sketches (original vs generated)")
    show_generated(model, train_ds, DEVICE, n=4,
               save_sequential=False, img_size=256)

    print("\n[Step 5-C] Sequential stroke images (10 per class, 100 total)")
    save_sequential_strokes(model, train_ds, DEVICE,
                        n_per_class=10, img_size=256)

    print(f"\n Done!")
    print(f"   checkpoints/     : model weights")
    print(f"   outputs/         : visualization images")
    print(f"   outputs/sequential/ : per-stroke images + JSON + NPY")


if __name__ == "__main__":
    # 실행 옵션: 원하는 라인만 주석을 해제하여 실행

    # 1. 처음부터 전부 실행
    main()

    # 2. EDA 재사용, pre-train + fine-tune
    # main(skip_eda=True)

    # 3. EDA + pre-train 재사용, fine-tune만
    # main(skip_eda=True, skip_pretrain=True)

    # 4. 학습 완료 후 시각화만 수행
    # main(skip_eda=True, skip_pretrain=True, skip_finetune=True)

    # 5. 원본 데이터만 바로 확인
    # show_raw_samples(n=4)
