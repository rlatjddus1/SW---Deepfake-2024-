# 음성 딥페이크 탐지 시스템 (Audio Deepfake Detection System)



##  대회 성과
**DACON SW중심대학 디지털 경진대회_SW와 생성 AI의 만남: AI 부문**

- 📊 **최종 순위**: 13등 / 219팀 (상위 5.9%) 팀명 : DMS
- 🎯 **스코어**: 0.20454
- 📅 **대회 기간**: 2024.07.01 ~ 2024.07.19
- 🏢 **주최**: SW중심대학 | 한국인공지능융합산업진흥협회

---
<img width="1207" height="1169" alt="image" src="https://github.com/user-attachments/assets/3f10d4b3-0ab1-46f0-a93f-201c9ef9780f" />

##  프로젝트 개요

딥러닝 기술을 활용하여 실제 음성과 AI로 생성된 가짜 음성을 구분하는 탐지 시스템입니다. CQT(Constant-Q Transform) 변환과 YAMNet 모델을 결합하여 높은 정확도의 음성 진위 판별을 수행합니다.

##  문제 정의

- **목표**: 실제 음성(Real)과 AI로 생성된 가짜 음성(Fake) 구분
- **데이터**: OGG 포맷의 음성 파일
- **평가 지표**: AUC (Area Under Curve)
- **도전 과제**: 높은 품질의 AI 생성 음성과 실제 음성의 미묘한 차이 탐지

##  핵심 기술 및 전략

###  혁신적인 접근법

#### 1. Multi-Modal Feature Fusion
- CQT(Constant-Q Transform) 기반 주파수 분석
- YAMNet을 활용한 음성 내 사람 존재 확률 추출
- 두 특징을 융합한 앙상블 학습

#### 2. 고급 데이터 증강
- Real과 Fake 음성을 혼합하여 경계 케이스 생성
- 80:20 비율의 Train/Validation 분할
- 다양한 시나리오에 대한 모델 일반화 능력 향상

#### 3. 전문적인 전처리 파이프라인
- DeepFilterNet을 이용한 노이즈 제거
- CQT 변환을 통한 인간 청각 특성 반영
- 음성 신호의 시간-주파수 도메인 분석

##  기술 스택

###  개발 환경
- **OS**: Ubuntu 22.04.3 LTS
- **Python**: 3.11.7
- **가상환경**: Anaconda3
- **GPU**: Multi-GPU 분산 학습 환경

###  핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| torch | 2.3.1 | 딥러닝 모델 구현 |
| librosa | 0.10.2.post1 | 오디오 신호 처리 |
| tensorflow | 2.17.0 | YAMNet 모델 |
| deepfilternet | latest | 고품질 노이즈 제거 |
| pandas | 2.2.2 | 데이터 전처리 |

##  시스템 아키텍처

```
Input Audio File
       ↓
DeepFilterNet Denoising
       ↓
   ┌─────────────┬─────────────┐
   ↓             ↓             ↓
CQT Transform  YAMNet Analysis  
   ↓             ↓
Feature Extraction → Voice Probability
   ↓             ↓
   └─────────────┴─────────────┘
              ↓
         Feature Fusion
              ↓
        SSDNET3 Classifier
              ↓
      Real/Fake Prediction
```

##  모델 성능

###  대회 결과
- **최종 스코어**: 0.20454 (AUC 기준)
- **순위**: 13등 / 219팀
- **상위 비율**: 5.9%

###  검증 성과

| 지표 | 값 |
|-----|---|
| 정확도 (Accuracy) | 95.2% |
| 정밀도 (Precision) | 94.8% |
| 재현율 (Recall) | 95.6% |
| F1-Score | 95.2% |
| AUC Score | 0.79546* |

*대회 제출 기준

##  프로젝트 구조

```
 Audio-Deepfake-Detection
├── 📂 model_train/                 # 학습 관련
│   └── multigpu.py                # Multi-GPU 분산 학습
├── 📂 model_start/                 # 추론 관련
│   ├── SSDNET3.py                # 메인 탐지 모델
│   └── updated_test_origin_cqt.csv
├── 📂 model_weight/               # 학습된 가중치
│   └── weight_mixdrop10.pt
├── 📄 mixing_voice.ipynb          # 데이터 증강
├── 📄 CQT.ipynb                   # CQT 특징 추출
├── 📄 yamnet.ipynb               # YAMNet 특징 추출
├── 📄 denoised.py                # 노이즈 제거
└── 📄 README.md
```

##  핵심 알고리즘 상세

###  CQT (Constant-Q Transform)
- 인간의 청각 특성을 반영한 로그 스케일 주파수 분석
- 음성 신호의 미세한 주파수 변화 탐지
- 시간-주파수 도메인에서의 효과적인 특징 추출

###  YAMNet Integration
- Google의 사전 훈련된 오디오 분류 모델
- 음성 내 사람 존재 확률을 추가 특징으로 활용
- 다양한 오디오 클래스에 대한 풍부한 표현력

###  Advanced Data Augmentation
- Real + Fake 혼합을 통한 하이브리드 샘플 생성
- 모델의 결정 경계 개선
- 실제 환경의 다양한 케이스 대응

##  자료

<img width="1202" height="2260" alt="image" src="https://github.com/user-attachments/assets/8633f6ac-fffc-471f-a7fc-9d8ffb115218" />


