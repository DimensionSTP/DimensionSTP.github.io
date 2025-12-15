---
layout: single
title:  "고전 Transformer 구조의 진화: LLM과 ViT 아키텍처 비교 분석"
categories: Code
tag: [Transformer, LLM, ViT, Architecture]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

최근 LLM 아키텍처는 2017년의 초기 Transformer(Attention Is All You Need)의 **Encoder–Decoder** 뼈대 위에, **깊은 학습의 안정성**, **추론 효율성**, 그리고 **극단적인 용량 확장**이라는 목표를 중심으로 미세 튜닝된 형태이다. GPT-2 이후 대부분의 LLM이 **Decoder-only** 구조를 표준으로 삼았다.

이 과정에서 트랜스포머는 텍스트(LLM)와 이미지(ViT)라는 이질적인 두 도메인에서 완전히 다른 **구조적 선택**을 하며 분화되었다. 특히 LLM은 **Pre-LayerNorm, RMSNorm, RoPE, MoE** 조합으로 발전하며 속도와 용량을 극대화한 반면, ViT는 **LayerNorm, Bias=True, GeLU** 조합으로 픽셀 기반의 통계적 불안정성(Mean-shift, Covariance-shift)을 잡는 데 주력했다.

이 글은 초기 Transformer부터 현대 LLM(Qwen3, DeepSeek V3)에 이르기까지, 그리고 LLM과 ViT 아키텍처가 왜 다르게 설계되었는지를 **구조적, 통계적 이유**와 함께 상세히 분석한다.

---

# 1. Modern LLM Evolution

초기 Transformer 구조는 이제 하나의 레거시(Legacy)가 되었으며, 현대 LLM은 이를 **대규모 스케일링**에 적합하도록 재설계했다. 핵심은 **Pre-Norm을 통한 학습 안정화**와 **MoE/GQA를 통한 효율적인 대형화**이다.

## 1-1. Normalization Layer Evolution

### 1) Post-LN에서 Pre-LN으로: 깊이 스케일링의 필수 조건

| 구분 | Post-LN (초기 Transformer, BERT) | Pre-LN (현대 LLM 표준: Llama, Qwen) | 발전 이유 및 구조적 근거 |
| :--- | :--- | :--- | :--- |
| **구조** | $\dots \to \text{Attn}(x) + x \to \text{LayerNorm} \to \dots$ | $\dots \to x + \text{Attn}(\text{Norm}(x)) \to \dots$ | **안정성 (Stability)** |
| **문제점** | 초기화 시 출력 레이어 근처의 **예상 경사도(Expected Gradients)가 매우 큼**. | LayerNorm이 잔차 경로($x$)를 건드리지 않아 **항등 경로($+I$)**가 항상 보장됨. | **학습 속도/비용** |
| **해결책** | **학습률 웜업(Warm-up) 단계가 필수적**. 웜업 없이는 학습 발산. | **웜업 없이도 안정적으로 학습 가능**. Loss Decay가 더 빠름. | - |

Post-LN 모델은 LayerNorm이 잔차 합산 **후(Post)**에 적용되기 때문에, 깊은 네트워크에서 경사도 폭발이나 소실 문제가 쉽게 발생한다. 이론적 분석에 따르면, Post-LN 모델은 출력 레이어 근처 파라미터의 경사도 스케일이 매우 커지기 때문에, 학습률을 극도로 낮게 시작해야 하는 웜업 과정이 필수적이다.

반면, Pre-LN은 **LayerNorm을 잔차 연결 안쪽(Pre)**에 배치하여, 경사도 계산 시 항등 행렬($I$)이 항상 더해지는 경로를 확보한다. 이 구조 덕분에 Pre-LN은 경사도가 레이어 깊이($L$)에 관계없이 일정하게 유지되어, 수백 레이어의 LLM을 **안정적**으로 훈련할 수 있게 된다.

### 2) LayerNorm $\to$ RMSNorm 및 QK-Norm: 효율과 수치 안정성

| 구분 | LayerNorm (LN) | RMSNorm | QK-Norm (Qwen3, OLMo 2) |
| :--- | :--- | :--- | :--- |
| **Centering** | **평균 제거** ($\mu=0$) | 평균 미제거 | Q, K에 적용되는 추가 RMSNorm |
| **도입 이유** | 안정화 | 계산 효율성 | **수치 안정성 및 Long Context** |
| **작동 원리** | 분산 $\sigma$로 정규화 | RMS (크기)로만 정규화 | Q, K 벡터의 스케일을 1 근처로 안정화 |

* **RMSNorm 채택의 구조적 이유:** Pre-LN 구조 하에서 깊은 모델의 안정성이 확보되자, LayerNorm이 수행하는 **평균 제거(Centering)** 연산은 계산 효율을 저해하는 불필요한 단계로 간주되었다. RMSNorm은 평균 제거 없이 벡터의 크기(RMS)만으로 정규화하여 **계산이 단순하고 빠르다**. LayerNorm과 성능 차이가 미미하거나 비슷하면서 속도에서 이점을 가져, Llama, Qwen 계열 LLM의 표준이 되었다.
* **QK-Norm의 역할:** DeepSeek V2, Qwen3, OLMo 2와 같은 현대 LLM은 **QK-Norm**을 Multi-Head Attention 내부의 Query($Q$)와 Key($K$) 벡터에 적용한다. 이는 **긴 컨텍스트(Long Context) 처리** 및 **FP16 저정밀도 학습** 환경에서 $Q K^T$ 내적 값의 스케일 폭발을 방지하여 Softmax 포화 및 Overflow를 막는 **수치 안정성(Numerical Stability)** 강화책이다.

## 1-2. Positional Encoding: Absolute to RoPE

| 구분 | Sinusoidal Absolute PE (초기 Transformer) | RoPE (Rotary Position Embedding) | 발전 이유 |
| :--- | :--- | :--- | :--- |
| **방식** | Embedding에 절대 위치 벡터를 더함 | Q, K 벡터에 위치별 회전(Rotation)을 적용 | **길이 일반화(Length Generalization)** |
| **적용 시점** | 입력(Embedding) 시 단 한 번 | **모든 Transformer Layer의 Attention 직전** | - |
| **적용 대상** | 토큰 임베딩 전체 ($E_{tok} + E_{pos}$) | **Q와 K 벡터에만** 적용, V는 제외 | - |

**RoPE 채택의 구조적 이유:** 초기 Sinusoidal PE는 훈련 시 접하지 않은 더 긴 시퀀스 길이로 **외삽(Extrapolation)하는 능력이 부족했다**. RoPE는 Q와 K에 위치 $i$와 $j$에 따른 회전 행렬 $R(i), R(j)$를 적용하여, 최종 Attention Score $\langle \tilde{q}_i, \tilde{k}_j \rangle$가 **상대 위치 $(j-i)$의 함수**가 되도록 설계되었다. 이를 통해 모델이 길이 변화에 강건해졌고, Qwen3의 128K 컨텍스트 지원 등 LLM의 긴 컨텍스트 처리를 가능케 했다.

**왜 Q, K에만 적용하는가?:** 위치 정보가 필요한 것은 **Attention 가중치 $\alpha_{ij}$**를 계산하는 단계이다. 이 가중치는 $Q$와 $K$의 내적으로 결정되므로, **내용(Value, $V$)**이 아닌 **방향/관점(Query, $Q$)**과 **접근점(Key, $K$)**에만 회전을 적용하는 것이 논리적이다.

## 1-3. FFN & Attention Innovation

### 1) MoE를 통한 용량 확장 (Conditional Computation)

현대 LLM의 가장 큰 구조적 특징은 **MoE(Mixture-of-Experts)**의 채택이다. LLM의 성능은 모델 크기(Total Parameter)에 비례한다는 **스케일링 법칙**을 따르지만, Dense 모델은 학습 및 추론 비용이 파라미터 수에 비례한다.

* **MoE의 원리:** Transformer 블록 내의 **FFN 레이어**를 다수의 독립적인 **Expert Network** (각각이 FFN)로 대체한다. Gating Network($G$)가 입력 토큰마다 가장 적합한 $k$개의 Expert만 선택하여 활성화한다 (**Sparse MoE**).
* **MoE의 이점:** 전체 파라미터 수(**용량**)는 수백억~수조 개로 확장하되 (예: DeepSeek V3 671B, Grok-1 314B, Mixtral 47B), 실제 추론 시의 활성 파라미터 수(**계산 비용, FLOPs**)는 Dense 모델과 비슷하게 유지된다. Mixtral-8x7B는 13B 활성 파라미터로 Llama-2-70B와 유사하거나 더 나은 성능을 달성했다.
* **MoE의 주된 위치:** FFN 레이어. FFN 레이어가 Self-Attention 레이어보다 희소성이 높고 도메인 특이성(Domain Specificity)이 낮다는 가설에 근거하며, 실제로 DeepSeekMoE 분석에서 FFN 레이어에서 현저한 희소성(20% 활성)이 관찰되었다.

### 2) Attention 효율화: GQA/MLA 및 Long Context 기법

| 구분 | MHSA (초기 Transformer) | GQA (Llama, Qwen3) | MLA (DeepSeek V3/R1) |
| :--- | :--- | :--- | :--- |
| **목적** | 높은 표현력 | **추론 시 KV 캐시 메모리 절감** | **KV 캐시 메모리 절감 + 성능 유지** |
| **구조** | $Q, K, V$ 헤드 수 동일 ($H_Q=H_{KV}$) | $Q$ 헤드 $\gg K, V$ 헤드 | $K, V$ 벡터를 저차원으로 압축 |
| **특징** | 메모리 복잡도 $\propto L \cdot H_Q$ | 메모리 복잡도 $\propto L \cdot H_{KV}$ (대폭 감소) | 압축/복원 행렬 곱셈 추가되지만, DeepSeek에서 MHA/GQA보다 성능 우위 |

* **Sliding Window Attention (SWA):** Gemma 3와 OLMo 3 등에서 채택된 기법으로, 각 토큰이 전체 시퀀스가 아닌 일정 크기의 윈도우 내에서만 Attention을 계산하게 한다. 이를 통해 **긴 컨텍스트 처리 시 메모리 사용량을 크게 줄인다**. Gemma 3는 SWA와 전체 Attention을 5:1 비율로 혼합하여 효율성을 극대화했다.

## 1-4. FFN Activation: SwiGLU

초기 Transformer의 **ReLU**는 GeLU를 거쳐 현대 LLM의 표준인 **SwiGLU**로 발전했다.

* **SwiGLU 메커니즘:** $\text{SwiGLU}(x) = (xW_1) \odot \text{Swish}(xW_2)$ 형태의 Gated Activation. 여기서 $xW_2$ 부분은 게이트 역할을 수행하며, 정보의 흐름을 **선택적으로(selective)** 조절한다.
* **발전 이유:** 텍스트 데이터의 **이산적/희소적(discrete/sparse) 의미 구조**에 맞춰, SwiGLU의 게이팅 메커니즘이 ReLU나 GeLU보다 **더 풍부한 비선형성**과 **표현력**을 제공함이 실험적으로 확인되었기 때문이다.

---

# 2. LLM vs ViT: Structural Divergence

LLM과 ViT는 같은 Pre-Norm 구조를 사용함에도 불구하고, 입력 데이터의 통계적 속성이 극도로 다르기 때문에 정규화, 활성화, 바이어스 설정이 상반된다.

## 2-1. Normalization

### 1) LayerNorm vs. RMSNorm: DC Shift 제거 여부

| 구분 | LLM (RMSNorm) | ViT (LayerNorm) | 구조적 근거: 입력 신호의 통계적 특성 |
| :--- | :--- | :--- | :--- |
| **입력** | **Discrete Symbolic Tokens**. 통계 안정. | **Dense/Continuous Pixels**. 노이즈 많음. | - |
| **Covariance** | 낮음. | **매우 높음** (Spatial/Channel Correlation). | - |
| **Mean Shift** | 작음. RMSNorm이 mean을 통과시키나 누적 위험 낮음. | **매우 큼**. 조명/명암 차이로 Patch Mean이 크게 요동침. | - |
| **선택 이유** | 계산 효율성/속도 (Centering 불필요). | **강력한 안정화.** $\mu$를 0으로 맞추어 DC Shift와 Covariance Shift를 제어. | - |

* **Covariance Shift:** 레이어 통과 시 특징 벡터들의 **공분산 구조($\Sigma$)가 요동치는 현상**. 이미지 Patch는 인접 픽셀 간 공간 상관성(Spatial Correlation) 및 RGB 채널 간 채널 공분산(Channel Covariance)이 강하여, 선형/비선형 레이어를 거치며 이 공분산 구조가 크게 변한다.
* **DC Shift:** 이미지 Patch의 **평균값(DC Component, 밝기)**이 데이터마다, 또는 Patch마다 크게 변동하는 현상. ViT의 입력은 조명 밝기(Illumination), 명암(Contrast), 배경 오프셋(Background Offset) 등 픽셀 기반의 **노이즈**를 포함하므로, Patch 평균이 크게 요동친다.
* **LN의 역할:** LayerNorm은 $\hat{x} = \frac{x-\mu}{\sigma}$ 연산을 통해 평균($\mu$)을 제거, 즉 **DC Shift를 제거**하고 분산($\sigma$)을 정규화한다. 이는 통계적으로 불안정한 이미지 입력에서 Covariance Shift와 Mean Shift를 초기부터 안정시키는 데 필수적이다. RMSNorm은 이 DC Shift를 제거하지 못하므로, Mean Shift가 Residual 경로를 통해 누적될 위험이 있다.

## 2-2. Bias Usage Differences

| 구분 | LLM (Bias=False) | ViT (Bias=True) | 구조적 근거 |
| :--- | :--- | :--- | :--- |
| **Bias 역할** | 효율성/안정성 위해 제거. | **학습되는 Offset**을 통해 DC Component를 보정. | - |
| **ViT Bias ($b$)의 작동** | Linear $y=xW+b$에서 $b$는 각 출력 차원($d$)마다 존재하는 **학습 가능한 벡터**. | **밝기/명암 보정:** 밝은 Patch에서 발생하는 평균 Shift를 $\approx -cW$로 상쇄/보정하는 역할을 학습. | - |
| **LLM Bias 제거 이유** | Pre-LN + RMSNorm 조합이 안정성을 확보하므로 Bias 기여도 ↓. | 토큰은 **Discrete Symbolic**이며, 조명/노이즈에 의한 Mean Shift가 없어 Bias가 불필요. | - |

Bias는 **Global Constant Shift**가 아니라, 각 출력 차원의 **학습 가능한 오프셋**이다. ViT에서 Bias를 유지하는 이유는, 초기 Patch Projection 단계에서 이미지의 **국소적인 조명 밝기, 명암 대비, 배경 오프셋** 등의 Baseline Shift를 **학습적으로 보정**하는 중요한 역할을 수행하기 위함이다. LLM의 토큰 임베딩은 이러한 연속적인 노이즈/평균 이동 문제가 없으므로, Bias를 제거하여 모델을 더 간결하게 만들고, RoPE의 긴 컨텍스트 외삽 안정성을 확보한다.

## 2-3. Activation

### FFN Activation: SwiGLU vs. GeLU

| 구분 | LLM (SwiGLU) | ViT (GeLU) | 구조적 근거: Feature Density |
| :--- | :--- | :--- | :--- |
| **구조** | Gated Activation (Gating) | Smooth Non-linearity (Soft Gating) | - |
| **FFN 특징** | **Sparse Symbolic**. 선택적 활성화(Gating)가 큰 이득. | **Dense Continuous**. Gating 이득 작음. | - |
| **안정성** | Gradient Smoothness 개선. | **Noise/Variance 흡수**. Conv-like 구조와 궁합 좋음. | - |

**GeLU 채택 이유:** ViT의 입력은 픽셀 기반이라 노이즈(Sensor noise, Texture micro-pattern)와 분산(Variance)이 많다. GeLU는 작은 입력값($x \approx 0$)을 억제하고 큰 값만 통과시키는 **Soft Gating** 특성이 있어, **작은 잡음 값을 부드럽게 0에 가깝게** 만들어 **노이즈를 효과적으로 흡수**하고, 활성화 폭발 없이 안정적인 동작을 보장한다. 반면 SwiGLU의 복잡한 Gating 구조는 Dense Feature를 처리하는 Vision 도메인에서는 추가적인 성능 이득이 미미하거나, 오히려 불안정해질 수 있다.

---

# 3. Summary & Insights

## 3-1. Three Key Drivers of Evolution

| 동인 | 초기 구조 (2017) | 현대 LLM 구조 (2024~) | 핵심 원리 |
| :--- | :--- | :--- | :--- |
| **1. 학습 안정화** | Post-LN, Warm-up 필수 | Pre-LN, RMSNorm, QK-Norm | **잔차 항등 경로** 보장, **내부 스케일 안정화**. |
| **2. 용량 및 성능** | Dense FFN, ReLU | Sparse **MoE**, SwiGLU | **조건부 계산(Conditional Computation)**을 통한 FLOPs 대비 용량 극대화. |
| **3. 효율적인 추론** | MHSA, Absolute PE | GQA/MLA, RoPE, SWA | **KV Cache 메모리 절감**, **긴 컨텍스트 외삽** 능력 확보. |

**MoE의 영향:** MoE는 LLM의 스케일링을 혁신한 핵심 기술이다. LLM의 성능 향상이 모델 크기에 의해 좌우될 때, MoE는 FFN 레이어를 다수의 전문가 네트워크로 분할하고(Sparse MoE), 토큰당 일부만 활성화함으로써, **총 파라미터 수(지식 저장 용량)**를 수천억~수조 개로 늘리면서도 **실제 추론 비용**은 Dense 모델 수준으로 유지한다. DeepSeek V3와 Qwen3 모두 이 MoE 전략을 채택하고 있다.

## 3-2. Insights on Divergence

LLM과 ViT의 구조적 차이점은 데이터 도메인의 통계적 특성에 대한 **최적화 전략의 극명한 대립**을 보여준다.

| 구조적 선택 | LLM (텍스트) | ViT (이미지) | 인사이트 |
| :--- | :--- | :--- | :--- |
| **Normalization** | RMSNorm | LayerNorm | LLM은 **효율성**을 위해 Mean Centering을 생략했지만, ViT는 **노이즈/DC Shift 안정성**을 위해 Mean Centering을 포기하지 못함. |
| **Bias** | Bias=False | Bias=True | LLM은 **안정성과 효율**을 위해 Bias를 제거했지만, ViT는 **도메인 노이즈(조명, 명암) 보정**을 위해 학습 가능한 Offset(Bias)을 유지함. |
| **Activation** | SwiGLU | GeLU | LLM은 **희소성**에 맞는 Gating(선택적 활성화)을 선택, ViT는 **Dense Feature의 안정적인 Noise 흡수**를 선택. |

**후기:** Transformer는 하나의 고정된 모델이 아니라, 어떤 데이터를 다루는지에 따라 **정규화 계층, 바이어스, 활성화 함수**의 미세한 디테일을 조정하여 **학습의 안정성**과 **도메인의 통계적 요구사항**을 만족시키도록 끊임없이 진화하는 **프레임워크**이다. LLM이 성능과 효율성을 극단적으로 추구하며 MoE와 GQA로 나아갈 때, ViT는 근본적인 데이터의 불안정성(픽셀 노이즈, Covariance Shift)을 제어하기 위해 전통적인 LayerNorm과 Bias를 고집하는 모습은, 딥러닝 아키텍처 설계가 **성능뿐만 아니라 환경 적응(Environmental Adaptation)**의 문제임을 명확히 보여준다.
