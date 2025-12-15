---
layout: single
title:  "Think-Fusion의 다양한 구현 방식 비교 리뷰: Qwen3, DeepSeek V3.1, Llama Nemotron을 중심으로"
categories: Study-concept
tag: [LLM, CoT, Reasoning, Hybrid-thinking, Think-fusion]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

2025년 LLM 필드의 핵심 혁신 동력은 'Reasoning'(추론) 능력이었다. Test-time-computing을 통해 LLM은 복잡한 문제 해결 과정을 명시적으로 수행하며 벤치마크 성능을 극적으로 끌어올렸다. 하지만 이 과정은 단순한 질문에 대해서도 불필요한 대기 시간을 발생시켜 일반적인 대화 환경에서는 오히려 사용자 경험을 저해하는 요인이 되었다.

이에 대한 해결책으로 Think-fusion 방식이 제안되었는데, 이는 단일 모델(unified model)이 복잡한 추론 모드(Thinking mode)와 효율적인 일반 대화 모드(Non-thinking mode)를 모두 지원하는 기술이다.

본 글에서는 현재 시장을 주도하고 있는 세 가지 대표적인 하이브리드 싱킹 모델인 Qwen3, DeepSeek V3.1, Llama Nemotron Ultra가 이 Think-fusion을 구현한 방식(Case 1, 2, 3)을 비교 분석하고, 각 방식의 학습 전략 및 잠재적 위협 요소에 대해 심층적으로 리뷰하고자 한다.

---

# 1. Hybrid Thinking Implementation

LLM은 일반적으로 assistant 토큰 뒤에 응답을 생성하도록 학습되며, Think-fusion 모델은 추론 모드 활성화 시 `<think>[reasoning]</think>[response]` 형태의 답변을 생성하도록 훈련된다. Non-think 모드에서 모델이 어떤 템플릿을 사용하는지에 따라 세 가지 구현 방식의 차이가 발생한다.

| Case | 모델 | Think ON 템플릿 | Think OFF 템플릿 | 제어 방식 |
| :--- | :--- | :--- | :--- | :--- |
| Case 1 | Llama Nemotron | `<think>[reasoning]</think>[response]` | `[response]` (태그 생략) | 시스템 프롬프트 (detailed thinking on/off) |
| Case 2 | Qwen3 | `<think>[reasoning]</think>[response]` | `<think></think>[response]` (빈 추론 강제 삽입) | enable_thinking 파라미터 또는 /think, /no_think 명령어 |
| Case 3 | DeepSeek V3.1 | `<think>[reasoning]</think>[response]` | `</think>[response]` (닫는 태그로 시작) | thinking 파라미터 |

## 1-1. Case 1: Llama Nemotron Ultra 253B

Llama Nemotron Ultra 253B는 시스템 프롬프트를 통해 추론 모드를 제어하는 가장 직관적인 방식을 채택한다.

* **Think ON**: 시스템 프롬프트에 `detailed thinking on`을 설정하고, Temperature=0.6, Top P=0.95와 같은 샘플링 파라미터를 권장한다. 모델은 추론 과정을 `<think>...</think>` 블록에 포함한다. 추론이 불필요한 경우에도 `<think></think>` 태그가 포함될 수 있으며, 이는 예상된 동작(expected behaviour)이다.
* **Think OFF**: 시스템 프롬프트에 `detailed thinking off`를 설정하며, 탐욕적 디코딩(greedy decoding, Temperature 0) 사용을 권장하여 효율성을 극대화한다. 이 모드에서는 추론 태그 없이 최종 답변만 생성된다.

## 1-2. Case 2: Qwen3

Qwen3는 Reasoning을 하드 스위치(`enable_thinking` 파라미터)와 소프트 스위치(`/think`, `/no_think` 명령어) 두 가지 방법으로 제어한다.

* **하드 스위치 (`enable_thinking=False`)**: `tokenizer.apply_chat_template` 사용 시, `enable_thinking=False`가 설정되면 챗 템플릿(`tokenizer_config.json`)에 정의된 대로 `<think>\n\n</think>\n\n`와 같은 빈 `<think></think>` reasoning 파트를 Assistant Indicator 뒤에 강제로 붙여 넣어 추론이 없는 답변을 생성하도록 한다. 이는 추론 동작을 엄격하게 비활성화하면서도, 모델이 항상 '생각' 단계를 거치도록 템플릿 구조의 일관성을 유지하려는 시도이다.
* **소프트 스위치 (`/no_think`)**: `enable_thinking=True` 상태에서 `/no_think`를 사용하면 모델은 항상 `<think>...</think>` 블록을 출력하지만, 그 내용은 비어 있을 수 있다.

## 1-3. Case 3: DeepSeek V3.1

DeepSeek V3.1은 챗 템플릿 자체를 변경하여 Think/Non-think 모드를 지원하며, 두 모드의 응답 분포를 명확히 분리하려는 특징을 가진다.

* **Think ON (`thinking=True`)**: 어시스턴트 접두사가 `<｜Assistant｜><think>`로 끝나며, 모델은 `<think>` 토큰 이후에 추론을 시작한다.
* **Think OFF (`thinking=False`)**: 어시스턴트 접두사가 `<｜Assistant｜></think>`로 끝나며, `</think>` 토큰으로 바로 응답을 유도하여 추론 과정 없이 최종 답변이 생성된다.
* **특이점**: DeepSeek V3.1은 멀티턴 대화 시, 이전 턴의 응답에 `</think>{response}<｜end of sentence｜>`가 포함되지만, 현재 턴의 접두사에는 오직 `<think>` 또는 `</think>` 토큰만 붙어 모드를 명확히 분리한다.

# 2. Training Methodology

하이브리드 싱킹 모델들은 각기 다른 기반 아키텍처와 훈련 전략을 사용하여 복잡한 추론 능력을 확보했다.

## 2-1. Qwen3 Strategy

Qwen3는 총 6개의 모델(MoE 모델 2개, Dense 모델 6개)로 구성된 포괄적인 모델 스위트이다.

* **데이터 및 아키텍처**: Qwen3는 사전 학습(Pretraining) 및 후처리 학습(Post-training)을 거쳤다. 가장 큰 모델(MoE)을 학습한 뒤, strong-to-weak distillation을 통해 나머지 모델들을 학습했으며, Dense 모델들은 pruning 기법을 통해 사이즈를 줄인 것으로 추정된다.
* **Hybrid Mode 학습**: Qwen3는 스위치 토큰(`</think>`, `</no_think>`)을 학습에 사용하여 Reasoning On/Off를 제어하도록 훈련되었다.
* **장문 처리**: Qwen3는 기본적으로 32,768 토큰의 컨텍스트 길이를 지원하며, YaRN 기법을 사용하여 최대 131,072 토큰까지 컨텍스트 길이를 확장할 수 있음을 검증했다.

## 2-2. DeepSeek V3.1 Architecture

DeepSeek V3.1은 DeepSeek-V3.1-Base를 기반으로 포스트 트레이닝되었으며, 최적의 효율성에 중점을 둔 아키텍처와 학습 기법을 사용했다.

* **데이터 확장**: V3 베이스 체크포인트 위에 2단계 장문 컨텍스트 확장 방식을 적용했다. 32K 확장 단계는 630B 토큰으로 10배, 128K 확장 단계는 209B 토큰으로 3.3배 확장되었다.
* **아키텍처 혁신 (DeepSeek V3 기반)**:
  * **Multi-head Latent Attention (MLA)**: 기존 MHA와 달리 LoRA 기반의 경량화된 Q, K, V 행렬을 생성하고, 특히 K와 V를 하나의 projection에서 생성하여 파라미터 효율성과 메모리 사용량을 절감했다.
  * **MoE 개선**: Auxiliary‑Loss Free Load Balancing과 Bias‑Corrected Top‑k Gating Strategy를 도입하여, 추가적인 손실 함수 없이도 전문가 간 부하를 동적으로 균형 있게 분산시켰다. DeepSeek V3의 MoE 레이어는 1개의 공유 전문가와 256개의 라우팅 전문가로 구성되며, 토큰당 8개의 전문가가 활성화된다.
  * **Multi-Token Prediction (MTP)**: 학습 시 LM Head를 공유하고 Transformer 블록 내에서 재귀적으로 MTP를 적용하는 독자적인 방식을 사용하여 모델 크기 증가 없이 MTP를 활용했다.
* **혼합 정밀도 (Mixed Precision)**: DeepSeek V3.1은 Nvidia Hopper 아키텍처의 강점을 활용하기 위해 UE8M0 FP8 scale 데이터 형식을 모델 가중치와 활성화에 사용하여 훈련되었다. 이는 FP8의 빠른 연산 속도와 BF16/FP32의 안정성을 동시에 확보하기 위함이다.

## 2-3. Llama Nemotron NAS & RL

Llama Nemotron은 Meta Llama-3.1-405B-Instruct의 파생 모델로, 아키텍처 최적화와 다단계 강화 학습에 중점을 두었다.

* **아키텍처 (NAS)**: Neural Architecture Search (NAS) 접근 방식을 통해 모델 아키텍처가 맞춤화되었으며, 이로 인해 Skip attention, Variable FFN, FFN Fusion 등 비표준적이고 비반복적인 블록이 사용되어 메모리 사용량 및 지연 시간이 개선되었다.
* **다단계 포스트 트레이닝**: 성능 복구를 위해 지식 증류(KD, 650억 토큰)와 지속적인 사전 학습(CPT, 880억 토큰)을 거쳤다. 이후 수학, 코드, 추론, 채팅, 도구 호출을 위한 지도 미세 조정(SFT) 단계와, Group Relative Policy Optimization (GRPO) 알고리즘을 사용한 다단계 RL(강화 학습) 단계를 거쳤다.
* **Hybrid Mode 학습**: 모델이 두 모드를 구별하도록 훈련시키기 위해, Reasoning On/Off 모드에 대한 응답을 모두 포함하는 합성적으로 생성된 응답이 포함된 프롬프트를 학습에 사용했다.

# 3. Pros, Cons & Risks

하이브리드 싱킹 모델의 잠재적 위협은 기본적으로 LLM이 다음 토큰 예측(Next Token Prediction) 기반 모델이기 때문에 발생한다. 모델이 확률적으로 의도하지 않은 모드의 토큰을 생성할 가능성이 상존한다.

## 3-1. Think True Mode

| 모델 | 장점 (Reasoning ON) | 잠재적 위협 (Think True) | 권장 파라미터 |
| :--- | :--- | :--- | :--- |
| Qwen3 (Case 2) | 복잡한 논리적 추론, 수학, 코딩 등에서 성능 향상. 추론 과정이 `<think>...</think>` 블록으로 명확히 구분된다. | **모델 결정 위험**: 모델이 `<think>` 토큰을 생성한 뒤, 바로 `</think>`를 생성하여 원치 않게 Non-think 모드가 되어버릴 수 있다. | T=0.6, P=0.95, K=20. 탐욕적 디코딩 금지. |
| DeepSeek V3.1 (Case 3) | Thinking 모드에서 Non-thinking 모드 대비 크게 높은 성능을 보임 (예: AIME 2024 Pass@1 93.1, LiveCodeBench Pass@1 74.8). 템플릿의 명확한 분리로 모드 전환이 안정적이다. | **학습 효율성 저하 가능성**: 두 모드가 완전히 다른 분포를 가지게 되어 모델 학습 시 sample efficiency가 떨어질 수 있다. | (별도 명시된 파라미터 없음). |
| Llama Nemotron (Case 1) | 추론 능력이 크게 강화됨 (예: AIME25 Pass@1 16.67% → 72.50%, MATH500 Pass@1 80.40% → 97.00%). | **불필요한 태그 생성**: 추론이 불필요한 경우에도 `<think></think>` 태그가 포함되어 오버헤드가 발생 가능 (예상된 동작). | T=0.6, P=0.95. |

## 3-2. Instruct True Mode

| 모델 | 장점 (Non-think / Instruct) | 잠재적 위협 (Instruct True) | 권장 파라미터 |
| :--- | :--- | :--- | :--- |
| Qwen3 (Case 2) | `enable_thinking=False` 하드 스위치는 추론 동작을 엄격하게 비활성화하여 효율성을 높임. 구조적 일관성을 위해 빈 `<think></think>` 강제 삽입. | **강제 추론 위험**: `enable_thinking=False`일지라도, 확률적으로 두 번의 thinking이 나와 의도치 않은 추론이 발생할 가능성. | T=0.7, P=0.8, K=20. |
| DeepSeek V3.1 (Case 3) | `</think>` 토큰으로 바로 응답을 시작하여 추론 과정을 강력하게 생략하고 빠른 응답을 보장함. 모드가 명확히 분리되어 모델이 자발적으로 모드를 변경할 위험이 상대적으로 낮음. | (Case 3는 모드 분리가 명확하여 Case 1/2의 자발적 모드 변경 위험이 적음). | (별도 명시된 파라미터 없음). |
| Llama Nemotron (Case 1) | 추론 과정을 완전히 생략하고 최종 응답만 생성하여 효율적이고 빠른 일반 대화에 적합함. **탐욕적 디코딩(Temperature 0)**을 사용하여 효율성을 극대화할 것을 권장함. | **자발적 추론 위험**: Non-think 모드에서 inference를 수행할 때, 학습 시 `<think>` 토큰이 포함된 시퀀스를 보았기 때문에 모델이 확률적으로 `<think>` 토큰을 생성하고 의도치 않게 Think 모드로 전환될 수 있음. | Greedy Decoding (T=0). |

# 4. Summary & Insights

2025년 상반기를 이끌었던 Think-fusion은 단일 모델로 성능과 사용성을 모두 잡으려는 혁신적인 방법론이었다. 그러나 그 구현 방식에 따라 모델의 예측 특성으로 인한 잠재적 위험을 내포하고 있다.

## 4-1. Fundamental Drawbacks

최근 GPT-5가 Router를 통해 Reasoning 모델과 Instruct 모델을 아예 분리하거나, Alibaba가 Qwen3의 일부 모델을 Instruct-only로 변경하는 등 Think-fusion을 회피하는 경향이 나타나는 것은 다음과 같은 근본적인 한계 때문이다.

1. **성능 저하 및 학습 불균형**: 하나의 모델이 복잡한 추론(Reasoning)과 간결한 답변(Instruct)이라는 두 가지 상충되는 태스크를 모두 수행하도록 학습될 경우, 특정 태스크에서 성능 저하를 가져올 수 있다. 특히 Reasoning은 Attention 파라미터에, 일반 Instruct 데이터는 MLP 파라미터에 업데이트가 집중되어 학습 불균형이 발생할 수 있다.
2. **훈련의 복잡성**: Think 데이터와 Non-think 데이터의 적절한 비율을 찾는 작업(데이터 mixture)이 매우 어렵고, 비율 조절 실패 시 서비스 속도 저하나 성능 저하가 발생한다.
3. **효율성 저하 (Serving 측면)**: Think-fusion 모델은 응답 길이가 천차만별이어서 vLLM과 같은 환경에서 배치(batch) 처리 효율성이 떨어진다. 차라리 모델을 분리하여 따로 서빙하는 것이 효율적일 수 있다.

## 4-2. Implementation Insights

* **Case 1 (Llama Nemotron)**: 가장 단순하지만, Non-think 모드에서 모델이 자발적으로 Think 토큰을 생성할 위험이 있어, 효율성을 위해 탐욕적 디코딩을 강제해야 한다.
* **Case 2 (Qwen3)**: 챗 템플릿에 빈 `<think></think>`을 강제 삽입하여 템플릿 일관성을 유지하려 했으나, 모델의 자발적인 모드 전환 위험(원치 않는 빈 추론 또는 의도치 않은 실제 추론)이 존재한다.
* **Case 3 (DeepSeek V3.1)**: `<think>`와 `</think>` 토큰을 통해 두 모드의 응답 분포를 명확히 분리하려는 시도로, 모드 전환의 예측 불확실성을 가장 크게 낮추었다. 이는 현재까지 공개된 Think-fusion 방식 중 가장 안정적인 제어를 제공하는 구조이다.

결론적으로, 초기 Think-fusion은 단일 모델로 두 기능을 통합하려는 시도였지만, DeepSeek V3.1이 템플릿 분리를 통해 두 모드의 분포를 분리하려는 시도나, GPT-5 및 Qwen3의 모델 분리 전략은 모델 학습의 안정성과 서비스 효율성을 극대화하기 위해 각 태스크에 최적화된 구조를 분리하는 방향으로 LLM 개발의 흐름이 전환되고 있음을 보여준다.
