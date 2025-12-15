---
layout: single
title:  "Qwen VLs vs DeepSeek-OCR: 멀티모달 LLM 아키텍처, 목적, 학습 전략 심층 리뷰"
categories: Study-concept
tag: [VLM, OCR, Qwen-VL, DeepSeek-OCR, Architecture]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

최근 공개된 Qwen 2.5 VL, Qwen 3 VL, 그리고 DeepSeek-OCR은 각기 다른 기술적 혁신과 명확한 목표를 가지고 멀티모달 인공지능(AI)의 영역을 확장하고 있다. Qwen 시리즈가 범용성, 에이전트 기능, 그리고 초장문맥 이해에 집중한다면, DeepSeek-OCR은 OCR 도메인을 통해 ‘컨텍스트 광학 압축’이라는 근본적인 효율성 문제를 해결하고자 한다.

이 글에서는 세 모델의 아키텍처, 핵심 목표, 그리고 학습 전략을 상세한 수치를 포함하여 비교 분석한다.

---

# 1. Background & Positioning

Qwen 계열 모델들은 기존 Qwen2-VL의 성공적인 공개 이후, 개발자들의 피드백을 수용하며 성능을 지속적으로 강화했다. Qwen 2.5 VL은 시각적 에이전트 기능과 장시간 비디오 이해 능력을 핵심적으로 개선한 모델로 등장했으며, Qwen 3 VL은 여기서 더 나아가 **인지(Cognition), 추론(Reasoning), 실행(Execution)**의 모든 단계에 걸쳐 능력을 통합하고 확장한 모델이다. Qwen 3 VL은 Dense 아키텍처와 MoE 아키텍처를 모두 제공하여 다양한 배포 환경을 지원한다.

반면, DeepSeek-OCR은 일반적인 VLM 성능 향상이 아닌, LLM 중심 관점에서 시각 모달리티를 텍스트 정보의 효율적인 압축 매체로 활용하여 장문맥 처리의 계산적 한계를 극복하고자 하는 개념 증명(Proof-of-Concept) 모델이다.

# 2. Purpose & Goals

세 모델은 추구하는 궁극적인 목표가 명확하게 구분된다.

| 모델 | 핵심 목적 및 방향성 | 구체적인 강화 능력 및 수치 |
| :--- | :--- | :--- |
| **Qwen 2.5 VL** | Fine-grained Perception 확립 및 Agentic Amplifier 역할 | 1시간 이상의 비디오를 이해하고 이벤트 세그먼트를 초 단위로 정밀하게 위치 파악. 바운딩 박스/포인트 형태의 정확한 객체 지역화(localization). 인보이스, 양식, 테이블 등의 내용을 구조화된 JSON 출력으로 추출하여 금융/상업 분야에 활용. |
| **Qwen 3 VL** | 초장문맥, 3D 공간 추론 및 시각 코딩 도약 | Native 256K 컨텍스트 길이 지원 및 100만 토큰까지 확장 가능. 32개 언어를 지원하는 OCR 성능 강화. UI 설계 초안으로부터 Draw.io/HTML/CSS/JS 코드를 생성하는 Visual Coding 능력 강화. 시점, 가림 관계, 3D Grounding을 통한 고급 공간 인지. |
| **DeepSeek-OCR** | 컨텍스트 광학 압축 (Contexts Optical Compression) 개념 검증 | 9-10배 텍스트 압축에서 96%+ OCR 디코딩 정밀도 달성. 20배 압축에서도 약 **60%**의 정확도 유지. 초고해상도 입력을 가장 적은 비전 토큰으로 처리하며 OmniDocBench에서 SOTA 성능 달성. |

# 3. Architecture Analysis

## 3-1. Vision Encoder & Input Processing

| 세부 사항 | Qwen 2.5 VL | Qwen 3 VL | DeepSeek-OCR |
| :--- | :--- | :--- | :--- |
| **Vision Encoder 기반** | 재설계된 ViT | ViT 기반 (Qwen3VLVisionModel) | DeepEncoder (SAM-base + CLIP-large) |
| **Vision MLP 활성화** | SwiGLU (Qwen 2.5 LLM 구조와 통일) | GeLU | GeLU |
| **Vision 정규화** | RMSNorm | Layer Norm | Layer Norm |
| **Attention 메커니즘** | Window Attention 전략적 도입 (4개 레이어만 Full Self-Attention 사용) | - | SAM (Window Attention) + CLIP (Global Attention)의 직렬 조합 |
| **압축 기술** | MLP 기반 Merger (4개 패치 그룹화) | DeepStack 구조 | 16배 Convolutional Compressor |
| **패치 임베딩** | Conv3d (비디오/동적 해상도 지원) | Conv3d | Conv2d |
| **해상도 처리** | 동적 해상도 및 동적 FPS 샘플링 | Navit/동적 해상도 | Tiny, Small, Base, Large, Gundam 모드 (Tiling) 지원 |

DeepSeek-OCR의 DeepEncoder는 고해상도 이미지를 처리하면서도 낮은 활성화 메모리를 유지하도록 설계되었다. 약 380M 파라미터로 구성된 DeepEncoder는 4096개의 패치 토큰을 16배 압축하여 256개의 토큰으로 줄이는 핵심적인 역할을 수행한다.

## 3-2. LLM Decoder & VL Fusion

| 세부 사항 | Qwen 2.5 VL | Qwen 3 VL | DeepSeek-OCR |
| :--- | :--- | :--- | :--- |
| **LLM 디코더** | Qwen 2.5 LLM (3B, 7B, 72B) | Qwen 3 LLM (Dense 및 MoE) | DeepSeek3B-MoE (570M 활성화 파라미터) |
| **Text Attention** | nn.Linear * 3, GQA | nn.Linear * 3, GQA | MLA (Multi-head Linear Attention) |
| **VL 융합 방식** | Vision 토큰을 Text Input 앞/중간에 concat | DeepStack: ViT 다층적 특징을 LLM의 여러 계층에 재주입 | DeepEncoder 토큰을 `<image>` 토큰 위치에 Interleave |
| **위치 인코딩** | Absolute Time 정렬 MRoPE | Interleaved-MRoPE | Absolute pos (CLIP) + RoPE (LLM) |
| **특이 사항** | - | DeepStack, Text–Timestamp Alignment | OCR 특화 옵션 (no_repeat_ngram_size, 압축 비율 디버깅) |

Qwen 3 VL의 DeepStack은 전통적인 단일 계층 삽입 방식의 한계를 극복하고 ViT의 **다층적 특징(multi-level features)**을 LLM의 여러 계층에 걸쳐 주입하여 미세한 시각적 특징과 고차원 언어 표현을 정교하게 결합하는 구조적 혁신을 달성했다.

# 4. Training Strategy & Scale

## 4-1. Qwen 2.5 VL: 3-Stage Pre-training

Qwen 2.5 VL은 총 4.1조 토큰으로 학습되었으며, 장기 컨텍스트 처리를 위해 최대 시퀀스 길이 32,768 토큰까지 확장한 것이 특징이다.

| 단계 | 목적 | 학습 모듈 | 토큰 수 | 시퀀스 길이 |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** (Visual Pre-training) | ViT를 통한 비전 표현 정렬 | ViT만 학습 (LLM Frozen) | 1.5T | 8,192 |
| **Stage 2** (Multimodal Pre-training) | 멀티모달 추론 강화 | ViT & LLM 모두 학습 | 2.0T | 8,192 |
| **Stage 3** (Long-Context Pre-training) | 긴 문맥 처리 능력 학습 | ViT & LLM 모두 학습 | 0.6T | 32,768 |
| **Post-training** (SFT) | Instruction-following 성능 향상 | LLM 파라미터 학습 (ViT Frozen) | 약 200만 데이터 항목 | - |

SFT 단계에서는 50% 순수 텍스트와 50% 멀티모달 데이터(이미지-텍스트, 비디오-텍스트)를 활용하여 Instruction-following 능력을 훈련시켰다.

## 4-2. DeepSeek-OCR: 2-Stage Training

DeepSeek-OCR의 학습은 DeepEncoder의 압축 능력과 LLM의 디코딩 능력을 분리하여 최적화했다.

| 단계 | 목적 | 학습 모듈 | 데이터 구성 | 주요 수치 |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** (DeepEncoder Training) | Next-token prediction 기반 DeepEncoder 단독 학습 | DeepEncoder 전체 학습 | OCR 1.0, OCR 2.0, LAION 일반 데이터 100M | Sequence length: 4,096. Batch size: 1280. |
| **Stage 2** (DeepSeek-OCR Full Training) | OCR 및 일반 비전 Joint Training | SAM/Compressor Frozen, CLIP/LLM Trainable | 70% OCR, 20% 일반 비전, 10% 텍스트 전용 | Sequence length: 8,192. Global batch: 640. 160개의 A100 40GB GPU 사용. |

특히 Stage 2에서는 SAM과 Compressor를 고정(Frozen) 상태로 유지하고 CLIP과 MoE 디코더만 학습함으로써, DeepEncoder가 달성한 광학 압축 비율을 손상시키지 않고 LLM이 이 압축된 정보를 효율적으로 디코딩하도록 유도했다.

# 5. Summary & Insights

Qwen VLs와 DeepSeek-OCR은 멀티모달 AI의 진화를 이끄는 두 가지 주요 경로를 보여준다.

1. **Qwen VLs (범용성 및 인지적 통합)**: Qwen 시리즈는 DeepStack과 Interleaved-MRoPE와 같은 구조적 개선을 통해 시각 정보를 LLM의 깊은 추론 과정에 무손실로 통합하는 데 성공했다. 이는 모델이 256K 문맥 길이에서 100% 정확도를 기록하고, 심지어 100만 토큰의 초장문맥 비디오에서도 **99.5%**의 검색 정확도를 유지하는 놀라운 기억력과 추론 능력으로 이어졌다. Qwen VLs는 단순히 정보를 인식하는 수준을 넘어, 실제 세상의 사건과 관계를 깊이 이해하고 실행하는 에이전트형 모델로 진화하고 있다.
2. **DeepSeek-OCR (효율성 및 광학 압축)**: DeepSeek-OCR의 접근 방식은 LLM의 근본적인 장문맥 문제를 해결하기 위한 독특한 제안이다. 시각 모달리티를 고압축률의 메모리 매체로 사용하여 텍스트 토큰을 획기적으로 줄였다. 이 연구는 단순히 OCR 성능을 넘어서, 인간의 **기억 소멸 메커니즘(memory decay)**을 시각적 해상도 저하(압축률 증가)를 통해 시뮬레이션할 수 있는 새로운 연구 방향을 제시하며, 미래의 이론적으로 무제한적인 컨텍스트 아키텍처 구축의 가능성을 열었다.

이 두 모델군은 각각 '무엇이든 할 수 있는 강력한 인공지능'과 '극도로 효율적인 인지 시스템'이라는 두 가지 비전을 동시에 보여주며 멀티모달 연구의 지평을 넓히고 있다.
