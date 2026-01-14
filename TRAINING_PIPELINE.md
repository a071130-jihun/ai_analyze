# Sleep Stage Classification - Training Pipeline

수면다원검사(PSG) 오디오 데이터를 활용한 수면 단계 자동 분류 딥러닝 파이프라인입니다.

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT DATA                                      │
│                                                                              │
│    APNEA_EDF/{subject_id}/*.edf        APNEA_RML/{subject_id}.rml           │
│    (PSG 다채널 신호)                     (수면 단계 라벨)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Data Loading                                                        │
│  ├── EDF Reader: Mic/Snore 채널 추출                                         │
│  └── RML Parser: 30초 단위 수면 단계 라벨 파싱                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Feature Extraction                                                  │
│  └── Mel-Spectrogram 변환 (128 mels × 117 time frames per 30s epoch)        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Data Preprocessing                                                  │
│  ├── Label Remapping (연속 정수로 변환)                                       │
│  ├── Subject-level Train/Test Split (Data Leakage 방지)                     │
│  └── Feature Normalization (Robust Scaling)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Data Augmentation                                                   │
│  ├── SpecAugment (Frequency/Time Masking)                                   │
│  ├── SNR Noise Injection                                                     │
│  ├── Time/Frequency Shift                                                    │
│  └── Mixup Training                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Model Training                                                      │
│  ├── Model: CNN / CRNN / Transformer / DeepResNet                           │
│  ├── Loss: Focal Loss + Consistency Training                                │
│  ├── Optimizer: AdamW + Cosine Annealing LR                                 │
│  └── Early Stopping + Best Model Checkpointing                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Evaluation                                                          │
│  ├── Metrics: Accuracy, F1 (macro/weighted), Cohen's Kappa                  │
│  └── Confusion Matrix & Classification Report                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                          │
│                                                                              │
│    ./output/sleep_stage_{model}.pt       ./output/results.png               │
│    (학습된 모델 가중치)                   (평가 결과 시각화)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 상세 단계별 설명

### 2.1 Data Loading

#### EDF Reader (`edf_reader.py`)
```python
# EDF 파일에서 Mic/Snore 채널 자동 탐색 및 추출
class EDFReader:
    mic_channel_names = ["Mic", "MSnore", "Snore", "Audio", "Sound"]
    
    def read_mic_channel(edf_path) -> (audio_signal, sample_rate)
```

- **입력**: EDF 파일 (PSG 다채널 신호)
- **출력**: Mic 채널 오디오 신호 (np.float32), 샘플레이트
- **처리**: 여러 EDF 파일이 있는 경우 시간순 정렬 후 연결

#### RML Parser (`rml_parser.py`)
```python
# Respironics XML 형식의 수면 스코어링 파일 파싱
class RMLParser:
    def get_labels_at_intervals(epoch_duration=30) -> List[stage_type]
```

- **입력**: RML 파일 (수면 전문가가 스코어링한 결과)
- **출력**: 30초 단위 수면 단계 라벨 리스트
- **수면 단계**: Wake, NonREM1(N1), NonREM2(N2), NonREM3(N3), REM

---

### 2.2 Feature Extraction

#### Audio Feature Extractor (`audio_features.py`)
```python
# 오디오 신호 → Mel-Spectrogram 변환
class AudioFeatureExtractor:
    sample_rate = 16000  # 리샘플링 타겟
    n_mels = 128         # Mel 필터뱅크 개수
    n_fft = 512          # FFT 윈도우 크기
    hop_length = 128     # STFT 홉 길이
    
    def extract_epoch_features(audio, epoch_duration=30):
        # 30초 에폭 → (128, 117) Mel-spectrogram
```

**처리 과정:**
```
30초 오디오 (15,000 samples @ 500Hz)
        │
        ▼ Normalize (zero-mean, unit-max)
        │
        ▼ STFT (n_fft=512, hop=128)
        │
        ▼ Mel Filterbank (128 bands)
        │
        ▼ Log Compression
        │
Output: (128, 117) Mel-spectrogram
```

---

### 2.3 Data Preprocessing

#### Label Processing
```python
# 원본 라벨 → 연속 정수 변환
SLEEP_STAGE_MAP = {
    "Wake": 0,
    "NonREM1": 1,
    "NonREM2": 2,
    "NonREM3": 3,
    "REM": 4,
    "NotScored": -1  # 제거됨
}

# 3-stage 옵션 (Wake/NREM/REM)
convert_to_3stage: 0→0(Wake), 1,2,3→1(NREM), 4→2(REM)
```

#### Subject-level Split (Data Leakage 방지)
```python
def split_data_by_subject(features_per_subject, labels_per_subject, test_ratio=0.2):
    """
    같은 피험자의 모든 에포크는 Train 또는 Test 중 하나에만 속함
    → 모델이 '개인의 뇌파 지문'을 암기하는 것을 방지
    """
```

| Split | 비율 | 용도 |
|-------|------|------|
| Train | 68% (80% × 85%) | 모델 학습 |
| Validation | 12% (80% × 15%) | 조기 종료, 하이퍼파라미터 튜닝 |
| Test | 20% | 최종 성능 평가 (학습에 미사용) |

#### Feature Normalization
```python
def standardize_features(train_features, robust=True):
    # Robust Scaling: 이상치에 강건
    p5, p50, p95 = percentile(train_features, [5, 50, 95])
    normalized = (features - p50) / (p95 - p5)
    normalized = clip(normalized, -3, 3)
```

---

### 2.4 Data Augmentation

#### SpecAugment (`augmentation.py`)
```python
class SpecAugment:
    """시간/주파수 마스킹으로 모델 일반화 향상"""
    freq_mask_param = 20   # 최대 주파수 마스킹 폭
    time_mask_param = 25   # 최대 시간 마스킹 폭
    n_freq_masks = 2       # 주파수 마스크 개수
    n_time_masks = 2       # 시간 마스크 개수
```

#### 추가 증강 기법
| 기법 | 설명 | 강도 (medium) |
|------|------|---------------|
| SNR Noise | 신호 대 잡음비 기반 노이즈 주입 | SNR 5~20 dB |
| Time Shift | 시간축 이동 | ±10 frames |
| Frequency Shift | 주파수축 이동 | ±3 bands |
| Random Gain | 볼륨 변화 | 0.85~1.15x |
| Mixup | 두 샘플 선형 보간 | alpha=0.2 |

**증강 강도 옵션:**
- `none`: 증강 없음
- `light`: 약한 증강 (과적합 적을 때)
- `medium`: 기본값 (권장)
- `strong`: 강한 증강
- `aggressive`: 매우 강한 증강 (대규모 데이터셋)

---

### 2.5 Model Architecture

#### Available Models

| 모델 | 파라미터 | 특징 | 권장 사용 |
|------|---------|------|----------|
| `cnn` | ~1M | 기본 CNN, 빠른 학습 | 기본값 |
| `crnn` | ~2M | CNN + BiLSTM, 시퀀스 모델링 | 시간 의존성 중요 시 |
| `transformer` | ~2M | CNN + Transformer Encoder | 장거리 의존성 |
| `deep_transformer` | ~5M | 깊은 Transformer | 대규모 데이터셋 |
| `deep_resnet` | ~3M | ResNet + Inception + SE Block | 복잡한 패턴 |
| `deep_resnet_large` | ~10M | 대형 ResNet | Multi-GPU 학습 |
| `sequence` | ~4M | Multi-Scale CNN + BiLSTM + Attention | 다중 에폭 컨텍스트 |

#### CNN Architecture (기본)
```
Input: (batch, 1, 128, 117)
        │
        ▼ ConvBlock 1: Conv2d(1→32) + BN + ReLU + MaxPool
        ▼ ConvBlock 2: Conv2d(32→64) + BN + ReLU + MaxPool
        ▼ ConvBlock 3: Conv2d(64→128) + BN + ReLU + MaxPool
        │
        ▼ AdaptiveAvgPool2d → (batch, 128, 4, 4)
        │
        ▼ Flatten → Linear(2048→128) → ReLU → Dropout
        ▼ Linear(128→num_classes)
        │
Output: (batch, num_classes)
```

#### Sequence Model Architecture (시퀀스)
```
Input: (batch, seq_len, 1, 128, 117)  # seq_len개 연속 에폭
        │
        ▼ Multi-Scale CNN Backbone (각 에폭 독립 처리)
        │   ├── Branch 1x1: 세부 특징
        │   ├── Branch 3x3: 지역 패턴
        │   ├── Branch 5x5: 중간 패턴
        │   └── Branch 7x7: 광역 패턴
        │
        ▼ Concat + SE Attention
        │
        ▼ BiLSTM (2 layers, hidden=256)
        │
        ▼ Center Epoch Selection (중앙 에폭 예측)
        │
        ▼ Classifier (LayerNorm + GELU + Dropout)
        │
Output: (batch, num_classes)
```

---

### 2.6 Training Configuration

#### Loss Functions

**Focal Loss (기본)**
```python
class FocalLoss:
    """클래스 불균형 문제 해결"""
    gamma = 2.0  # 쉬운 샘플 가중치 감소
    alpha = class_weights  # 클래스별 가중치
    
    # CE_loss * (1 - p_t)^gamma
```

**Consistency Training (옵션)**
```python
class JSDConsistencyLoss:
    """증강된 버전들의 예측 일관성 유지"""
    # Original + Aug1 + Aug2 의 예측 분포가 유사하도록 학습
```

#### Optimizer & Scheduler
```python
# Optimizer
optimizer = AdamW(
    lr=3e-4,
    weight_decay=1e-5
)

# Scheduler (옵션)
- "cosine": CosineAnnealingLR (기본)
- "plateau": ReduceLROnPlateau
- "onecycle": OneCycleLR
- "warmup_cosine": Warmup + Cosine Decay
```

#### Training Loop
```python
for epoch in range(num_epochs):
    # 1. Train
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = focal_loss(outputs, batch_y)
        
        if use_consistency:
            aug1, aug2 = augment(batch_x)
            consistency_loss = JSD(outputs, model(aug1), model(aug2))
            loss += consistency_weight * consistency_loss
        
        loss.backward()
        optimizer.step()
    
    # 2. Validate
    val_metrics = evaluate(val_loader)
    
    # 3. Best Model Checkpointing (F1 기준)
    if val_metrics['f1'] > best_f1:
        save_model()
    
    # 4. Early Stopping (patience=15)
    if no_improvement_for(15_epochs):
        break
```

---

### 2.7 Evaluation Metrics

| 지표 | 설명 | 해석 |
|------|------|------|
| **Accuracy** | 전체 정답률 | 클래스 불균형 시 과대평가 가능 |
| **F1 (macro)** | 클래스별 F1의 평균 | 불균형에 강건, 주요 지표 |
| **F1 (weighted)** | 샘플 수 가중 평균 | 다수 클래스 성능 반영 |
| **Cohen's Kappa** | 우연 일치 보정 일치도 | <0.4 Poor, 0.4~0.6 Moderate, >0.8 Excellent |

---

## 3. 실행 방법

### 기본 실행
```bash
python run_pipeline.py
```

### 옵션 지정
```bash
python run_pipeline.py \
    --model cnn \              # 모델 타입
    --epochs 30 \              # 학습 에폭
    --batch_size 16 \          # 배치 크기
    --stages 5 \               # 수면 단계 수 (3 or 5)
    --aug medium \             # 증강 강도
    --focal \                  # Focal Loss 사용
    --focal_gamma 2.0 \        # Focal Loss gamma
    --consistency \            # Consistency Training
    --multi_gpu                # Multi-GPU 사용
```

### 모델별 권장 설정

| 모델 | seq_len | batch_size | epochs | 비고 |
|------|---------|------------|--------|------|
| cnn | 1 | 16~32 | 30 | 기본값 |
| crnn | 5~11 | 8~16 | 40 | 시퀀스 컨텍스트 |
| sequence | 5~11 | 8~16 | 40 | 권장 |
| deep_resnet | 1 | 16~32 | 50 | 대규모 데이터 |

---

## 4. 프로젝트 구조

```
sleep-stage-classifier/
├── run_pipeline.py               # 메인 실행 스크립트
├── predict.py                    # 추론 스크립트
├── evaluate_model.py             # 평가 스크립트
│
└── sleep_stage_classifier/       # 핵심 모듈
    ├── config.py                 # 설정 (AudioConfig, TrainConfig 등)
    ├── train.py                  # 학습 로직 (Trainer, Loss, Scheduler)
    ├── augmentation.py           # 데이터 증강 기법
    │
    ├── data/
    │   ├── edf_reader.py         # EDF 파일 읽기
    │   ├── rml_parser.py         # RML 파일 파싱
    │   └── dataset.py            # PyTorch Dataset, DataLoader
    │
    ├── features/
    │   └── audio_features.py     # Mel-spectrogram, MFCC 추출
    │
    └── models/
        ├── classifier.py         # CNN, CRNN, Transformer
        └── sequence_model.py     # Sequence Model, DeepResNet
```

---

## 5. 핵심 설계 결정

### 5.1 Subject-level Split
- **문제**: 같은 피험자의 에폭이 Train/Test에 섞이면 모델이 수면 단계가 아닌 개인 특성을 학습
- **해결**: 피험자 단위로 분할하여 Data Leakage 방지

### 5.2 Robust Scaling
- **문제**: 오디오 데이터에 이상치가 많음
- **해결**: Percentile 기반 정규화 (5th~95th) + Clipping (±3σ)

### 5.3 Focal Loss + Class Weights
- **문제**: 수면 단계 불균형 (N2 >> REM, N1)
- **해결**: Focal Loss로 쉬운 샘플 가중치 감소 + 클래스별 가중치

### 5.4 Consistency Training
- **문제**: 증강된 샘플의 예측 불일치
- **해결**: JSD Loss로 원본/증강 예측 분포 일관성 유지

---

## 6. 출력 파일

| 파일 | 위치 | 설명 |
|------|------|------|
| 학습된 모델 | `./output/sleep_stage_{model}.pt` | 모델 가중치 + 설정 |
| 체크포인트 | `./output/checkpoints/` | 학습 중 최적 모델 |
| 결과 시각화 | `./output/results.png` | Confusion Matrix, 메트릭 차트 |
| 캐시 데이터 | `./cache/` | 전처리된 특징 (재사용) |

---

## 7. 하드웨어 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| RAM | 8GB | 16GB+ |
| GPU | 불필요 | GTX 1650 4GB+ |
| 저장공간 | 10GB | 50GB+ |

**Multi-GPU 지원**: `--multi_gpu` 옵션으로 DataParallel 자동 활성화

---

## 8. 참고 자료

- [AASM Sleep Scoring Manual](https://aasm.org/) - 수면 단계 기준
- [SpecAugment Paper](https://arxiv.org/abs/1904.08779) - 데이터 증강
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002) - 클래스 불균형 해결
