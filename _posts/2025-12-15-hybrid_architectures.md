---
layout: single
title:  "차세대 LLM 아키텍처 혁신: 효율성을 위한 하이브리드 전략 (Nemotron-H, Qwen3-Next, Kimi Linear 분석)"
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

**모던 LLM의 근본적 한계와 패러다임의 전환**

기존의 모던 LLM 아키텍처는 **트랜스포머(Transformer)**의 자체 어텐션(Self-Attention, SA) 메커니즘에 기반하고 있다. 이는 시퀀스 길이가 길어질수록 계산량과 메모리 사용량이 시퀀스 길이의 제곱 함수($O(n^2)$)로 증가하는 근본적인 병목 현상을 야기했다. 이러한 제약은 장문맥(long context) 처리 시 성능과 메모리 비용을 비선형적으로 증가시켜, LLM의 광범위한 상업적 배포 및 에이전틱 지능(Agentic Intelligence)으로의 발전을 저해하는 주요 원인이었다.

Nemotron-H, Qwen3-Next, Kimi Linear와 같은 차세대 모델들은 이러한 트랜스포머의 정통성을 깨고, **'더 똑똑한 모델'을 넘어 '더 효율적이고, 빠르며, 저렴한 모델'**을 구축하는 방향으로 AI 개발 경쟁의 패러다임을 전환시키고 있다. 이들은 모두 하이브리드(Hybrid) 아키텍처를 전략적으로 도입하여, $O(n^2)$ 복잡도를 선형 시간 복잡도 모듈로 대체하거나 보완함으로써 확장성과 경제성 문제를 해결하고자 한다.

---

# 1. Nemotron-H: Mamba-Transformer Hybrid

NVIDIA 연구원들이 개발한 Nemotron-H 제품군은 트랜스포머의 자체 어텐션 레이어 대부분을 Mamba 상태 공간 모델(SSM)로 대체하여 추론 시간과 메모리 비용을 획기적으로 줄이도록 설계된 모델이다.

## 1-1. Architecture & Mamba SSM

Nemotron-H는 성능을 위해 선택적으로 배치된 자체 어텐션 레이어를 유지하는 것이 특징이다.

* **하이브리드 구성**:
  * Nemotron-H-56B-Base는 총 118개의 레이어로 구성되어 있으며, Mamba-2 레이어 54개, MLP 레이어 54개, 자체 어텐션 레이어 10개를 포함한다.
  * Nemotron-H-8B-Base는 총 52개의 레이어로 구성되며, Mamba-2 레이어 24개, MLP 레이어 24개, 자체 어텐션 레이어 4개를 포함한다.
  * 전체 레이어 중 약 **8%**만 자체 어텐션을 사용하며, 이 레이어들은 모델 전체에 걸쳐 균등하게 분산된다.
* **Mamba SSM 믹싱의 핵심**: Mamba 및 Mamba-2 레이어는 토큰당 일정한 시간 계산(constant time computation)과 고정된 메모리를 제공하는 상태 공간 모델이다. 이는 토큰을 생성하는 추론(generation) 단계에서 계산량과 메모리가 시퀀스 길이와 선형적으로 증가하는 트랜스포머의 단점을 해소한다.
* **구조적 원칙**: 자체 어텐션 레이어는 항상 FFN 레이어 앞에 위치하며, 모델의 첫 번째 레이어는 Mamba-2 레이어, 마지막 레이어는 FFN 레이어이다.

## 1-2. Efficiency Techniques

Nemotron-H는 단순히 아키텍처를 변경하는 것을 넘어, FP8 훈련과 MiniPuzzle 압축이라는 혁신적인 효율성 기술을 통합하여 상업적 배포 가능성을 높였다.

* **FP8 훈련 정밀도**: Nemotron-H-56B-Base는 20조(20T) 토큰의 대규모 데이터셋을 사용하여 FP8 정밀도로 완전히 사전 학습된 최초의 Nemotron 모델이다.
  * **기술**: 텐서당 전류 스케일링(per-tensor current scaling)이라는 조잡한(coarse grained) 양자화 방식을 사용하여, BF16 성능에 필적하는 정확도를 유지한다. 이를 통해 더 빠른 훈련 속도와 낮은 하드웨어 요구 사항을 달성하여 훈련 비용을 크게 절감할 수 있다.
* **MiniPuzzle 모델 압축**: MiniPuzzle은 가지치기(pruning)와 증류(distillation)를 결합한 하드웨어 인식 압축 프레임워크이다.
  * **목적**: 56B 모델을 47B 파라미터로 축소하여, 단일 32GiB GPU 환경(예: RTX 5090)에서 FP4 정밀도로 약 100만 토큰에 달하는 긴 컨텍스트 추론을 가능하게 한다.
  * **효율성**: 47B 모델은 56B 모델과 유사한 정확도를 유지하면서도 추론 속도가 1.2배 향상된다.

## 1-3. Benchmark Analysis

Nemotron-H 모델은 기존의 최신 트랜스포머 모델(예: Qwen-2.5-72B, Llama-3.1-70B)과 비교하여 정확도와 효율성 측면에서 경쟁 우위를 보인다.

* **추론 처리량**: 긴 컨텍스트(65536 입력 길이, 1024 출력 토큰)에서 NVIDIA H100 GPU를 사용하여 측정했을 때, Nemotron-H-56B-Base는 경쟁 모델 대비 최대 3배 더 높은 추론 처리량을 기록했다. 47B 압축 모델은 2.9배의 속도 향상을 보였다.
* **정확도 경쟁**: Nemotron-H-56B-Base는 17개 평가 작업 중 9개에서 Qwen-2.5-72B-Base 및 Llama-3.1-70B-Base보다 가장 높은 정확도를 달성했다. 특히 장기 추론 작업(Long-thought reasoning)인 MATH (R1-style) 벤치마크에서는 Llama-3.1-405B와 DeepSeek-V3-671B와 같은 훨씬 큰 모델과 비교해도 경쟁력 있는 성능을 보였다.

---

# 2. Qwen3-Next: Ultra-Sparse MoE

Qwen3-Next는 **Ultra-Sparse MoE (Mixture-of-Experts)**와 하이브리드 어텐션을 결합하여, 전체 파라미터 규모를 늘리면서도 실제 연산량(FLOPs)을 극단적으로 낮추고 초장문맥을 효율적으로 모델링하는 것을 목표로 한다.

## 2-1. Hybrid Attention & GDN

Qwen3-Next는 트랜스포머의 표준 어텐션을 **Gated DeltaNet (GDN)**과 Gated Attention의 하이브리드 조합으로 대체한다.

* **Gated DeltaNet (GDN)**: GDN은 선형 어텐션의 일종으로, 긴 문맥 모델링을 효율적으로 수행한다. GDN은 델타 규칙(Delta Rule) 기반의 업데이트 방식을 사용하여 연관 메모리(associative memory)의 안정적인 학습을 가능하게 한다.
* **하이브리드 레이아웃**: Qwen3-Next-80B-A3B는 총 48개 레이어를 가지며, `12×(3×(Gated DeltaNet→MoE)→1×(Gated Attention→MoE))`의 패턴으로 배열된다. 이는 전체 레이어의 75%에 GDN이 적용되고, 나머지 25%에 Gated Attention(표준 어텐션)이 적용됨을 의미한다.
* **표현력 강화**: 표준 Attention 레이어에는 출력 게이팅 메커니즘을 도입하여 안정성을 높였고, Attention Head의 차원을 256으로 확장하여 표현력을 강화했다. 또한 RoPE(Rotary Position Embedding)를 전체 포지션 차원의 25%에만 적용하여 장문 입력에 대한 일반화 능력을 개선했다.

## 2-2. Ultra-Sparse MoE Design

Qwen3-Next의 Ultra-Sparse MoE 구조는 모델의 전체 용량과 실제 추론 비용을 분리하는 핵심 전략이다.

* **희소성 비율**: 전체 800억 개의 파라미터 중, 실제 추론 시 활성화되는 파라미터는 **약 37억 개 (3B)**로, 전체의 **3.7%**에 불과하다.
* **전문가 풀**: 512명의 전문가(Experts) 풀을 보유하며, 매 토큰마다 10명의 전문가와 **1명의 공유 전문가(Shared Expert)**만을 활성화한다. 이는 Qwen3의 MoE 설계(128명 중 8명 활성화) 대비 전문가 풀 규모를 확장하면서도 연산 효율성을 극대화한 것이다.

## 2-3. Training Stability & MTP

Qwen3-Next는 효율적인 추론을 위한 기술과 안정적인 학습을 위한 설계를 모두 통합했다.

* **Multi-Token Prediction (MTP)**: 한 번에 여러 토큰을 생성할 수 있는 MTP 기능을 기본적으로 지원하며, 이는 Speculative Decoding과 결합될 때 추론 효율성을 극대화하고 지연 시간을 줄인다.
* **학습 안정성**: 대규모 모델 학습 시 발생하는 불안정성(Attention Sink 등)을 완화하기 위해 Zero-Centered RMSNorm을 채택하고, 출력 게이팅 메커니즘을 도입하여 Attention 출력의 안정성을 보장했다.
* **초장문맥**: 네이티브로 262,144 토큰까지 지원하며, YaRN 기법을 통해 1,010,000 토큰까지 확장 가능하다.
* **성능 이점**: 32K 토큰 이상의 긴 문맥에서 기존 Qwen3-32B 대비 10배 이상의 추론 처리량을 달성했다.

---

# 3. Kimi Linear: Delta Attention

Kimi Linear는 Moonshot AI의 모델로, 에이전틱 지능(agentic intelligence)과 긴 궤적(extended trajectories) 처리에서 요구되는 높은 효율성을 목표로 개발된 하이브리드 선형 어텐션 아키텍처이다.

## 3-1. KDA & Fine-grained Gating

Kimi Linear의 핵심은 **Kimi Delta Attention (KDA)**으로, 이는 Gated DeltaNet (GDN)을 개선한 선형 어텐션 모듈이다.

* **델타 규칙 확장**: KDA는 GDN과 마찬가지로 **델타 규칙(Delta Rule)**을 사용하여 메모리 상태를 업데이트한다. 이 업데이트는 재구성 손실에 대한 온라인 경사 하강법으로 해석되며, 상태 S에 랭크-1 교정 업데이트가 포함된다.
* **미세 조정된 게이팅 (Fine-grained Gating)**: KDA는 GDN의 스칼라(scalar) 붕괴 게이트($\alpha_t$)와 달리, **채널별(channel-wise) 대각 행렬 게이트 $\text{Diag}(\alpha_t)$**를 도입한다.
  * **장점**: 이 게이트는 각 피처 차원이 독립적인 망각률을 유지하도록 하여, 유한 상태 RNN 메모리(finite-state RNN memory)에 대한 더욱 정밀한 제어를 가능하게 한다. 합성 작업(예: Palindrome, MQAR)에서 GDN보다 빠르고 높은 정확도를 달성하는 근본적인 이유이다.

## 3-2. Hardware Efficiency

KDA는 구현 수준에서도 하드웨어 효율성을 극대화했다.

* **DPLR (Diagonal-Plus-Low-Rank) 제약 변형**: KDA는 일반적인 DPLR 행렬의 제약된 변형을 사용하여 전이 동역학(transition dynamics)을 매개변수화한다.
* **Chunkwise 알고리즘**: KDA는 이 DPLR 제약 변형을 위해 맞춤형 청크 단위 병렬 알고리즘(bespoke chunkwise-parallel algorithm)을 사용하며, 이는 일반적인 DPLR 공식 대비 계산량을 실질적으로 줄여 커널 레벨에서 거의 2배의 속도 향상을 달성한다.
* **효율성 결과**: Kimi Linear는 장문맥 추론 시 **KV 캐시 사용량을 최대 75%**까지 절감하며, 100만 토큰 컨텍스트에서 최대 6.3배의 디코딩 처리량(decoding throughput)을 달성했다.

## 3-3. NoPE Strategy

Kimi Linear는 하이브리드 구조를 최적화하여 성능과 효율성 사이의 균형점을 찾았다.

* **하이브리드 비율**: KDA 레이어와 전역 어텐션(Full MLA, Multi-Head Latent Attention) 레이어를 3:1 비율로 교차하여 배치하는 인터-레이어(inter-layer) 하이브리드 디자인을 채택했다. 이 비율은 최상의 품질-처리량 트레이드오프를 제공하는 것으로 확인되었다.
* **NoPE (No Position Encoding)**: 모든 Full MLA 레이어에 **위치 인코딩(NoPE)**을 적용하지 않고, 모든 위치 정보 인코딩 및 시간적 편향(recency bias) 처리의 책임을 KDA 레이어에 위임한다. 이는 RoPE의 고정된 주파수가 야기하는 문맥 창 확장 문제를 회피하고, 모델이 장문 범위에서 더 견고하게 외삽(extrapolation)할 수 있도록 돕는 전략이다.

---

# 4. Deep Dive: SSM vs Delta Rule

Nemotron-H의 Mamba-2 SSM과 Qwen3-Next 및 Kimi Linear의 DeltaNet 계열 (GDN/KDA)은 모두 선형 시간 복잡도를 갖는 모델이지만, 메모리를 갱신하고 정보를 통합하는 기반 원리에서 뚜렷한 차이를 보인다.

| 특징 | Mamba SSM (Mamba-2) | Gated DeltaNet (GDN) | Kimi Delta Attention (KDA) |
| :--- | :--- | :--- | :--- |
| **근본 원리** | Structured State Space Model (SSM). Hebbian-유사 학습. | Fast Weight Programmers / 델타 규칙. | 델타 규칙의 Fine-grained 확장. |
| **업데이트 목표** | correlation/energy 손실 최소화. | 재구성 손실($\|S^\top k_t - v_t\|^2$)에 대한 온라인 경사 하강. | 재구성 손실에 대한 SGD 및 정밀 제어. |
| **메모리 업데이트** | $S_t = \alpha_t S_{t-1} + \beta_t k_t v_t^\top$ (단순 곱셈적 붕괴). | $S_t = \alpha_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ (붕괴 + 랭크-1 교정). | $S_t = (I - \beta_t k_t k_t^\top) \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top$ (채널별 붕괴 + 랭크-1 교정). |
| **게이팅 정밀도** | 데이터 종속적 스칼라 붕괴 게이트. | 스칼라 붕괴 게이트 (거친 제어). | 채널별 대각 행렬 게이트 (미세 조정 제어). |

### 핵심 철학적 차이: '저장과 붕괴' vs. '교정과 수정'

1. **Mamba (SSM): 저장 및 붕괴**: Mamba는 기본적으로 이전 상태에 붕괴($\alpha_t$)를 적용하고, 새로운 키-값 쌍을 추가하여 정보를 저장한다. 이는 시간이 지남에 따라 정보가 자연스럽게 잊히도록 (붕괴되도록) 설계된 방식이다.
2. **GDN/KDA (델타 규칙): 능동적 교정**: 델타넷 계열은 메모리 $S$를 학습 가능한 연관 메모리로 간주한다. 이들은 랭크-1 교정 업데이트($I - \beta_t k_t k_t^\top$)를 통해, 새로운 입력 $k_t$가 원하는 출력 $v_t$를 생성하도록 메모리 상태를 능동적으로 수정하고 교정한다.

특히 KDA의 미세 조정 게이팅은 메모리 제어에 있어 가장 정교한 접근 방식을 제공한다. GDN이나 Mamba의 스칼라 게이팅이 전체 메모리에 걸쳐 동일한 망각률을 적용하는 반면, KDA는 각 피처 차원(feature dimension)이 독립적으로 망각률을 조절할 수 있게 하여, 정확한 기억 검색 능력을 극대화한다.

---

# 5. Summary & Insights

## 5-1. Goal Achievement

Nemotron-H, Qwen3-Next, Kimi Linear는 모두 트랜스포머 아키텍처의 비용, 속도, 접근성 문제를 해결하기 위해 $O(n^2)$ 복잡도를 선형 시간 복잡도 모듈로 대체하는 하이브리드 전략을 채택했다.

| 모델 | 선형 모듈 | 하이브리드 비율 (선형:전역) | 핵심 효율성 기술 | 최대 효율성 이점 |
| :--- | :--- | :--- | :--- | :--- |
| **Nemotron-H** | Mamba-2 SSM | 약 9:1 (10개 SA 레이어만 유지) | FP8 훈련, MiniPuzzle 압축. | 추론 속도 최대 3배 향상, 단일 32GiB GPU 배포 가능. |
| **Qwen3-Next** | Gated DeltaNet (GDN) | 3:1 (75% GDN) | Ultra-Sparse MoE (3.7% 활성화), MTP. | 장문맥(>32K) 추론 처리량 10배 증가. |
| **Kimi Linear** | Kimi Delta Attention (KDA) | 3:1 (75% KDA) | Fine-grained Gating, NoPE 전략. | KV 캐시 75% 절감, 1M 컨텍스트 디코딩 6.3배 가속. |

## 5-2. Intelligent Efficiency Era

이러한 차세대 모델 아키텍처의 등장은 AI 산업이 '무차별적인 확장(brute-force scaling)' 시대에서 '지능적인 아키텍처 설계' 시대로 완전히 전환되었음을 보여준다. 단순히 파라미터 수를 늘리는 것이 아니라, 특정 하드웨어 제약과 서비스 환경(실시간 응답, 에이전트 지능)에 최적화된 구조를 찾는 것이 새로운 경쟁 우위(competitive moat)가 될 것이다.

1. **효율성 메커니즘의 검증**: Mamba SSM, GDN, KDA와 같은 선형 시간 복잡도 모듈은 이제 순수한 트랜스포머에 대등하거나 더 우수한 성능을 보일 수 있음이 경험적으로 입증되었다. 특히, Nemotron-H의 비교 실험은 하이브리드 모델이 동일 데이터셋으로 학습된 순수 트랜스포머 모델과 동등하거나 높은 정확도에 도달할 수 있음을 확인시켜주었다.
2. **Delta Rule의 정교함**: Kimi Linear의 KDA는 선형 어텐션의 한계로 지적되던 정밀한 기억 검색 능력을 델타 규칙의 **능동적 교정(corrective update)**과 **채널별 게이팅(fine-grained gating)**을 통해 극복했다. KDA가 단순한 SSA(State Space Architecture)의 붕괴 기반 업데이트를 넘어, 학습된 연관 메모리를 능동적으로 수정한다는 점은 장기 기억 모델링의 새로운 가능성을 열었다.
3. **하이브리드 전략의 안정성**: 모든 모델이 전역 어텐션을 완전히 제거하지 않고 약 8%~25%의 비율을 유지하는 것은, 선형 모듈이 여전히 해결하기 어려운 전역적인 정보 통합과 복잡한 추론 흐름을 위한 안전장치의 역할을 수행하는 것으로 해석할 수 있다. 이 전략적 절충은 성능 희생 없이 효율성을 극대화하는 현실적인 방법이다.

결론적으로, Nemotron-H의 FP8 훈련 및 압축 기술, Qwen3-Next의 극단적 희소성 MoE, Kimi Linear의 고효율 KDA는 모두 LLM의 **총 소유 비용(TCO)**을 낮추고 배포 환경을 민주화하는 데 기여하고 있다. 차세대 모델은 하드웨어 제약 조건(예: 단일 32GiB GPU)을 만족시키면서도 플래그십 성능에 근접하는 '맞춤형 효율 엔진'을 탑재하는 방향으로 진화하고 있다.
