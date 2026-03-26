# SketchGPT

quickDraw 데이터를 활용한 stroke 기반 스케치 생성 및 분류 모델 학습 및 시각화 프로젝트

## 환경 및 설치
- **GPU:** NVIDIA GeForce RTX 3090 (VRAM 24GB)
- **IDE:** VSCode + Python 가상환경

```bash
# PyTorch (CUDA 12.1) 설치
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 필요한 라이브러리 설치
pip install numpy matplotlib tqdm requests scikit-learn
```

## 실행 방법
가상환경이 활성화된 상태에서 `sketchGPT.py`를 실행

```bash
python sketchgpt_local.py
```

기본적으로 `main()` 함수가 실행되며, 특정 단계를 스킵하고 싶다면 파이썬 파일 내부의 `if __name__ == "__main__":` 블록에서 인자 수정 (상세 옵션은 코드 내부 주석 참고)
