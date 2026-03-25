# ============================================================
#  KID (Kernel Inception Distance) 평가 
#
#  실제 이미지: KID폴더/KID용/{class}/real_00.png ~ real_09.png
#  생성 이미지: outputs/20260323_033904/sequential/{class}/sample_XX/stroke_all.png
# ============================================================

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchmetrics.image.kid import KernelInceptionDistance

# 경로 설정: 경로는 환경에 맞게 수정

# KID 이미지 폴더 
REAL_ROOT = Path("/home/myid/hc21353/evaluation_metrics/KID")

# 생성 이미지 루트 
SEQ_ROOT  = Path("/home/myid/hc21353/outputs/20260323_033904/sequential")

# 평가할 클래스 목록 (이미지 폴더명 기준)
# real 이미지 폴더명과 sequential 폴더명이 다를 경우 아래 매핑 활용
USER_CLASSES = [
    'airplane', 'bus', 'canoe', 'car', 'helicopter',
    'hot_air_balloon', 'motorbike', 'sailboat', 'submarine', 'train'
]

# KID용 폴더명 → sequential 폴더명 매핑
# (이름이 다를 시 수정)
KID_TO_SEQ = {
    'airplane':        'airplane',
    'bus':             'bus',
    'canoe':           'canoe',
    'car':             'car',
    'helicopter':      'helicopter',
    'hot_air_balloon': 'hot_air_balloon',
    'motorbike':       'motorbike',
    'sailboat':        'sailboat',
    'submarine':       'submarine',
    'train':           'train',
}

# KID subset_size: 총 샘플 수(real+gen 합계)보다 작아야 함
# real 10장 × 10class = 100장, gen 샘플 수에 따라 조정
KID_SUBSET_SIZE = 10   # 클래스당 real 10장이므로 per-class KID는 10으로 설정


# 이미지 로더

def load_images_to_tensor(img_paths, invert=False, device="cpu"):
    """
    이미지 경로 리스트 → uint8 텐서 [B, 3, H, W]
    invert=True : 흑백 반전 (검은배경/흰선 → 흰배경/검은선)
    """
    tensors = []
    for p in img_paths:
        img = Image.open(p).convert("RGB").resize((256, 256))
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # C, H, W
        if invert:
            t = 255 - t
        tensors.append(t)
    if len(tensors) == 0:
        return None
    return torch.stack(tensors).to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"  KID 평가  (device: {device})")
    print("=" * 60)

    # 이미지 로드 
    print("\n[1] 이미지 로드 중...")
    real_paths_per_class = {}
    for cls in USER_CLASSES:
        cls_dir = REAL_ROOT / cls
        if not cls_dir.exists():
            print(f"    real 폴더 없음: {cls_dir}")
            continue
        paths = sorted(cls_dir.glob("real_*.png"))
        if len(paths) == 0:
            # jpg도 시도
            paths = sorted(cls_dir.glob("real_*.jpg"))
        real_paths_per_class[cls] = paths
        print(f"  {cls:<20}: {len(paths)}장")

    # 생성 이미지 경로 수집 
    print("\n[2] 생성(generated) 이미지 경로 수집 중...")
    gen_paths_per_class = {}
    for cls in USER_CLASSES:
        seq_cls = KID_TO_SEQ.get(cls, cls)
        cls_dir = SEQ_ROOT / seq_cls
        if not cls_dir.exists():
            print(f"    generated 폴더 없음: {cls_dir}")
            gen_paths_per_class[cls] = []
            continue
        paths = sorted([
            sample_dir / "stroke_all.png"
            for sample_dir in sorted(cls_dir.iterdir())
            if sample_dir.is_dir() and sample_dir.name.startswith("sample_")
            and (sample_dir / "stroke_all.png").exists()
        ])
        gen_paths_per_class[cls] = paths
        print(f"  {cls:<20}: {len(paths)}장")

    # KID 계산 (전체 통합) 
    print("\n[3] KID 계산 중 (전체 통합)...")
    all_real = []
    all_gen  = []
    for cls in USER_CLASSES:
        all_real.extend(real_paths_per_class.get(cls, []))
        all_gen.extend(gen_paths_per_class.get(cls, []))

    if len(all_real) == 0 or len(all_gen) == 0:
        print("  실제 이미지 또는 생성 이미지가 없어 KID를 계산할 수 없습니다.")
        return

    # subset_size는 min(real수, gen수) 이하여야 함
    subset_size = min(len(all_real), len(all_gen), 50)
    print(f"  real={len(all_real)}장  gen={len(all_gen)}장  subset_size={subset_size}")

    kid_global = KernelInceptionDistance(subset_size=subset_size).to(device)

    real_tensor = load_images_to_tensor(all_real, invert=False, device=device)
    gen_tensor  = load_images_to_tensor(all_gen,  invert=False, device=device)

    # 생성 이미지가 검은배경/흰선인 경우 invert=True로 변경
    # gen_tensor = load_images_to_tensor(all_gen, invert=True, device=device)

    kid_global.update(real_tensor, real=True)
    kid_global.update(gen_tensor,  real=False)
    kid_mean, kid_std = kid_global.compute()

    print(f"\n   전체 KID: {kid_mean.item():.5f} ± {kid_std.item():.5f}")

    # KID 계산 (클래스별) 
    print("\n[4] KID 계산 중 (클래스별)...")
    class_scores = {}
    for cls in USER_CLASSES:
        r_paths = real_paths_per_class.get(cls, [])
        g_paths = gen_paths_per_class.get(cls, [])

        if len(r_paths) == 0 or len(g_paths) == 0:
            print(f"  [{cls}] 이미지 부족, 건너뜁니다.")
            continue

        ss = min(len(r_paths), len(g_paths), KID_SUBSET_SIZE)
        if ss < 2:
            print(f"  [{cls}] subset_size < 2, 건너뜁니다.")
            continue

        kid_cls = KernelInceptionDistance(subset_size=ss).to(device)
        r_t = load_images_to_tensor(r_paths, invert=False, device=device)
        g_t = load_images_to_tensor(g_paths, invert=False, device=device)

        kid_cls.update(r_t, real=True)
        kid_cls.update(g_t, real=False)
        m, s = kid_cls.compute()
        class_scores[cls] = (m.item(), s.item())
        print(f"  {cls:<20}: KID = {m.item():.5f} ± {s.item():.5f}")

    # 최종 요약
    print("\n" + "=" * 60)
    print("  최종 KID 스코어 요약 (낮을수록 좋음)")
    print("=" * 60)
    print(f"  {'전체':20}: {kid_mean.item():.5f} ± {kid_std.item():.5f}")
    for cls, (m, s) in class_scores.items():
        print(f"  {cls:<20}: {m:.5f} ± {s:.5f}")


if __name__ == "__main__":
    main()
