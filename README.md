# DoodleToDream

quickDraw 데이터를 활용한 stroke 기반 스케치 생성 및 분류 모델 학습과 다각도 성능 평가(KID, Accuracy, Early Recognition) 프로젝트
- **데이터셋**: Google QuickDraw (10개 클래스: airplane, bus, canoe, car, helicopter, hot air balloon, motorbike, sailboat, submarine, train)

## 📂 프로젝트 구조 (Project Structure)
현재 깃허브 저장소의 디렉토리 구조는 다음과 같습니다.

```text
DoodleToDream/
├── evaluation_metrics/       # 성능 평가 및 분석 소스 코드 폴더
│   ├── KID/                  # 실제 이미지 데이터 (real_00.png ~ real_09.png)
│   ├── classification_metrics.py  
│   └── evaluation_KID.py     
├── VQ-SGEN/       
│   ├── VQ-SGEN/              # VQ-SGEN 논문 기반 이미지 생성 실행 파일                  
├── model_files/              # 학습된 가중치 및 설정 파일 (best_model.pth, thresholds.json 등)
├── outputs/                  # 생성된 이미지 시퀀스 결과물
├── README.md                 # 프로젝트 통합 가이드
└── sketchGPT.py              # sketchGPT 논문 기반 이미지 생성 실행 파일

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

### 1. sketchGPT 학습 및 이미지 생성
가상환경이 활성화된 상태에서 `sketchGPT.py`를 실행

```bash
python sketchgpt_local.py
```

기본적으로 `main()` 함수가 실행되며, 특정 단계를 스킵하고 싶다면 파이썬 파일 내부의 `if __name__ == "__main__":` 블록에서 인자 수정 (상세 옵션은 코드 내부 주석 참고)

### 2. 성능 평가
생성된 이미지의 품질과 분류 성능을 측정하기 위해 `evaluation_metrics` 폴더의 스크립트 실행

2-1. KID (Kernel Inception Distance) 측정
실제 스케치 데이터와 생성된 데이터 사이의 유사도를 계산 (점수가 낮을수록 생성 품질이 우수)

```bash
python evaluation_metrics/evaluation_KID.py
```

2-2. 분류 정확도 및 Early Recognition 평가
MobileNetV2 기반 분류기를 통해 완성된 이미지의 정확도와, 스케치 도중 모델이 얼마나 빨리 정답을 맞히는지 평가

```bash
python evaluation_metrics/classification_metrics.py
```

- **지표 2:** 완성 이미지(`stroke_all.png`)의 Top-1 Accuracy 측정
- **지표 3:** 누적 픽셀 비중 및 획수 기반의 Early Recognition Score 계산
- **지표 2:** 실행 후 상위 폴더에 `classification_results.csv` 파일이 생성되어 상세 분석 결과 확인 가능
