# Sleep Stage Classification from PSG Audio

PSG(수면다원검사) 오디오 데이터를 기반으로 수면 단계를 자동 분류하는 딥러닝 파이프라인입니다.

## 개요

이 프로젝트는 수면다원검사(Polysomnography) 장비에서 녹음된 **Mic/Snore 채널 오디오**를 분석하여 수면 단계(Wake, N1, N2, N3, REM)를 자동으로 분류합니다.

### 주요 특징

- EDF(European Data Format) 파일에서 오디오 채널 자동 추출
- RML(Respironics XML) 파일에서 수면 단계 라벨 파싱
- Mel-spectrogram 기반 특성 추출
- CNN/CRNN/Transformer 모델 지원
- 8:2 Train/Test 분리 및 자동 검증

---

## 파이프라인 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT DATA                                │
│  APNEA_EDF/{subject_id}/*.edf    APNEA_RML/{subject_id}.rml     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  [1] DATA LOADING                                                │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │   EDF Reader    │    │   RML Parser    │                     │
│  │  (edf_reader.py)│    │  (rml_parser.py)│                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                               │
│           ▼                      ▼                               │
│    Mic/Snore 채널           Sleep Stage                         │
│    오디오 신호              라벨 (30초 단위)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  [2] FEATURE EXTRACTION (audio_features.py)                      │
│                                                                  │
│  오디오 신호 (500Hz, ~4시간)                                      │
│       │                                                          │
│       ▼                                                          │
│  30초 에폭으로 분할 (epoch_duration=30)                           │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────┐                    │
│  │         Mel-Spectrogram 변환             │                    │
│  │  • n_mels: 128 (주파수 대역)              │                    │
│  │  • n_fft: 512 (FFT 윈도우)               │                    │
│  │  • hop_length: 128                       │                    │
│  │                                          │                    │
│  │  Input: 30초 오디오 (15,000 samples)     │                    │
│  │  Output: (128, 117) 스펙트로그램          │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
│  결과: features.shape = (N_epochs, 128, 117)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  [3] LABEL PREPROCESSING                                         │
│                                                                  │
│  원본 라벨: Wake(0), NonREM1(1), NonREM2(2), NonREM3(3), REM(4)  │
│       │                                                          │
│       ▼                                                          │
│  연속 라벨로 재매핑 (데이터에 없는 클래스 제외)                     │
│  예: {0, 1, 2, 4} → {0, 1, 2, 3}                                 │
│       │                                                          │
│       ▼                                                          │
│  NotScored(-1) 샘플 제거                                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  [4] DATA SPLIT (8:2)                                            │
│                                                                  │
│  전체 데이터 (495 epochs)                                         │
│       │                                                          │
│       ├──▶ Train Set (80%): 396 samples                         │
│       │         │                                                │
│       │         ├──▶ Training (85%): 336 samples                │
│       │         └──▶ Validation (15%): 60 samples               │
│       │                                                          │
│       └──▶ Test Set (20%): 99 samples (평가 전용)                │
│                                                                  │
│  ※ random_seed=42 고정으로 재현성 보장                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  [5] MODEL (classifier.py)                                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    SleepStageCNN                         │    │
│  │                                                          │    │
│  │  Input: (batch, 1, 128, 117)                            │    │
│  │           │                                              │    │
│  │           ▼                                              │    │
│  │  ┌─────────────────────────────────────┐                │    │
│  │  │ ConvBlock 1: Conv2d(1→32) + BN + ReLU + MaxPool     │    │
│  │  │ ConvBlock 2: Conv2d(32→64) + BN + ReLU + MaxPool    │    │
│  │  │ ConvBlock 3: Conv2d(64→128) + BN + ReLU + MaxPool   │    │
│  │  │ ConvBlock 4: Conv2d(128→256) + BN + ReLU + MaxPool  │    │
│  │  └─────────────────────────────────────┘                │    │
│  │           │                                              │    │
│  │           ▼                                              │    │
│  │  AdaptiveAvgPool2d → (batch, 256, 4, 4)                 │    │
│  │           │                                              │    │
│  │           ▼                                              │    │
│  │  ┌─────────────────────────────────────┐                │    │
│  │  │ Flatten                              │                │    │
│  │  │ Linear(4096 → 128) + ReLU + Dropout │                │    │
│  │  │ Linear(128 → 64) + ReLU + Dropout   │                │    │
│  │  │ Linear(64 → num_classes)            │                │    │
│  │  └─────────────────────────────────────┘                │    │
│  │           │                                              │    │
│  │           ▼                                              │    │
│  │  Output: (batch, num_classes) - 각 클래스 확률           │    │
│  │                                                          │    │
│  │  총 파라미터: 921,732개                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  대안 모델: CRNN (CNN+LSTM), Transformer                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  [6] TRAINING (train.py)                                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Loss Function: CrossEntropyLoss (class_weight 적용)      │    │
│  │   - 클래스 불균형 보정을 위한 가중치                       │    │
│  │   - weight = total / (n_classes * class_count)           │    │
│  │                                                          │    │
│  │ Optimizer: AdamW                                         │    │
│  │   - learning_rate: 1e-4                                  │    │
│  │   - weight_decay: 1e-5                                   │    │
│  │                                                          │    │
│  │ Scheduler: CosineAnnealingLR                             │    │
│  │   - 학습률 점진적 감소                                    │    │
│  │                                                          │    │
│  │ Early Stopping: patience=10                              │    │
│  │   - Val Loss가 10 에폭 동안 개선 없으면 중단              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  학습 루프:                                                       │
│  for epoch in range(num_epochs):                                 │
│      1. Train: forward → loss → backward → optimizer.step()     │
│      2. Validate: forward → metrics 계산                         │
│      3. Best model 저장 (F1 기준)                                │
│      4. Early stopping 체크                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  [7] EVALUATION                                                  │
│                                                                  │
│  Test Set (학습에 사용되지 않은 데이터)으로 평가                    │
│                                                                  │
│  평가 지표:                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Accuracy: 전체 정답률                                    │    │
│  │                                                          │    │
│  │ F1 Score (macro): 클래스별 F1의 평균                     │    │
│  │   - 클래스 불균형에 강건                                  │    │
│  │                                                          │    │
│  │ F1 Score (weighted): 샘플 수 가중 평균                   │    │
│  │                                                          │    │
│  │ Cohen's Kappa: 우연 일치를 보정한 일치도                  │    │
│  │   - < 0.20: Poor                                         │    │
│  │   - 0.21-0.40: Fair                                      │    │
│  │   - 0.41-0.60: Moderate                                  │    │
│  │   - 0.61-0.80: Substantial                               │    │
│  │   - > 0.80: Almost Perfect                               │    │
│  │                                                          │    │
│  │ Confusion Matrix: 예측 vs 실제 분포                      │    │
│  │ Classification Report: 클래스별 Precision/Recall/F1     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                  │
│                                                                  │
│  ./output/                                                       │
│  ├── sleep_stage_cnn.pt      # 학습된 모델 가중치                 │
│  ├── evaluation_data.npz     # 데이터 분할 정보                   │
│  └── results.png             # 평가 결과 시각화                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 프로젝트 구조

```
sleep-stage-classifier/
├── requirements.txt              # Python 의존성
├── .gitignore                    # Git 제외 파일
│
├── run_pipeline.py               # 메인 실행 스크립트 (학습 + 검증)
├── predict.py                    # 학습된 모델로 추론
├── evaluate_model.py             # 상세 모델 평가
├── visualize_data.py             # 데이터 시각화
│
└── sleep_stage_classifier/       # 핵심 모듈
    ├── __init__.py
    ├── config.py                 # 설정 (AudioConfig, ModelConfig, TrainConfig)
    ├── train.py                  # 학습 로직 (Trainer, EarlyStopping)
    │
    ├── data/
    │   ├── __init__.py
    │   ├── edf_reader.py         # EDF 파일 읽기, Mic 채널 추출
    │   ├── rml_parser.py         # RML 파일에서 수면 단계 라벨 파싱
    │   └── dataset.py            # PyTorch Dataset, DataLoader
    │
    ├── features/
    │   ├── __init__.py
    │   └── audio_features.py     # Mel-spectrogram, MFCC 추출
    │
    └── models/
        ├── __init__.py
        └── classifier.py         # CNN, CRNN, Transformer 모델
```

---

## 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd sleep-stage-classifier

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터 디렉토리 구조
mkdir -p APNEA_EDF APNEA_RML

# EDF 파일 배치: APNEA_EDF/{subject_id}/{subject_id}[00x].edf
# RML 파일 배치: APNEA_RML/{subject_id}.rml
```

**데이터 형식:**

| 파일 | 형식 | 설명 |
|------|------|------|
| `*.edf` | European Data Format | PSG 다채널 신호 (Mic/Snore 채널 필수) |
| `*.rml` | Respironics XML | 수면 단계 라벨 (30초 에폭 단위) |

### 3. 학습 실행

```bash
# 기본 실행 (CNN, 30 에폭, 8:2 분할)
python run_pipeline.py

# 옵션 지정
python run_pipeline.py \
    --edf_dir ./APNEA_EDF \
    --rml_dir ./APNEA_RML \
    --model cnn \
    --epochs 50 \
    --test_ratio 0.2 \
    --batch_size 16
```

**사용 가능한 모델:**
- `cnn`: 기본 CNN (권장)
- `crnn`: CNN + Bidirectional LSTM
- `transformer`: CNN + Transformer Encoder

### 4. 추론

```bash
# 저장된 모델로 새 데이터 예측
python predict.py
```

### 5. 상세 평가

```bash
# Test Set 상세 평가 및 시각화
python evaluate_model.py
```

---

## 주요 설정값

### AudioConfig (오디오 처리)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `epoch_duration` | 30 | 수면 단계 에폭 길이 (초) |
| `n_mels` | 128 | Mel 필터뱅크 개수 |
| `n_fft` | 512 | FFT 윈도우 크기 (원본 SR에 맞게 조정됨) |
| `hop_length` | 128 | STFT 홉 길이 |

### TrainConfig (학습)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `batch_size` | 16 | 배치 크기 |
| `learning_rate` | 1e-4 | 학습률 |
| `weight_decay` | 1e-5 | L2 정규화 |
| `early_stopping_patience` | 10 | 조기 종료 인내심 |

### 수면 단계 라벨

| 라벨 | 클래스 | 설명 |
|------|--------|------|
| Wake | 0 | 각성 |
| NonREM1 (N1) | 1 | 얕은 수면 |
| NonREM2 (N2) | 2 | 중간 수면 |
| NonREM3 (N3) | 3 | 깊은 수면 |
| REM | 4 | 렘 수면 |

---

## 출력 파일

학습 완료 후 `./output/` 디렉토리에 생성:

| 파일 | 설명 |
|------|------|
| `sleep_stage_cnn.pt` | 학습된 모델 체크포인트 |
| `evaluation_data.npz` | 데이터 분할 정보 (재현성) |
| `results.png` | 평가 결과 시각화 |

---

## 하드웨어 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| RAM | 8GB | 16GB+ |
| GPU | 불필요 (CPU 가능) | GTX 1650 4GB+ |
| 저장공간 | 10GB | 50GB+ (500명 데이터) |

**참고:** 현재 모델(92만 파라미터)은 매우 가볍습니다. RTX 2060 Super 1대로 충분합니다.

---

## 알려진 제한사항

1. **클래스 불균형**: 일반적으로 N2가 가장 많고, REM이 적음
2. **단일 피험자 학습 시 낮은 성능**: 500명+ 데이터로 개선 예상
3. **동적 클래스 처리**: 피험자별로 존재하는 수면 단계가 다를 수 있음 (예: 낮잠 데이터에 N3 없음) → 라벨 자동 재매핑으로 처리

---

## 참고 자료

- [AASM Sleep Scoring Manual](https://aasm.org/) - 수면 단계 기준
- [EDF Format Specification](https://www.edfplus.info/specs/edf.html)
- [Mel-Spectrogram 설명](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html)
# ai_analyze
