---
layout: single
title: "Do Transformers Need Three Projections? Systematic Study of QKV Variants Review"
categories: Study-concept
tag: [LLM, Transformer, Attention, QKV, EdgeAI]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.04032)

[Code link](https://github.com/Brainchip-Inc/Do-Transformers-Need-3-Projections)

이 논문은 Transformer attention에서 너무 당연하게 받아들인 설계를 다시 묻는다. 표준 self-attention은 입력 hidden state에서 query, key, value를 각각 다른 projection으로 만든다. 대부분의 구현과 최적화는 이 구조를 기본값으로 두고, head 수, cache layout, positional encoding, kernel optimization 쪽을 손본다. 그런데 이 논문은 더 앞단의 질문을 던진다.

"정말 Q, K, V projection이 모두 따로 필요할까"

겉으로 보면 단순한 weight tying ablation처럼 보일 수 있다. 하지만 내가 보기엔 이 논문의 핵심은 parameter 수를 조금 줄이는 것이 아니라, **KV cache memory를 줄이는 projection-level attention design space** 를 체계적으로 여는 데 있다. 특히 headline variant인 Q-K=V는 key와 value projection을 공유한다. 그러면 inference 때 저장해야 하는 KV cache가 K cache와 V cache 두 개가 아니라 하나로 줄어든다.

> 한 줄 요약: 이 논문은 QKV attention에서 projection sharing을 체계적으로 비교하고, key와 value를 공유하는 Q-K=V가 품질 손실을 작게 유지하면서 KV cache를 50% 줄일 수 있으며, GQA/MQA와 결합하면 cache reduction이 87.5%에서 96.9%까지 커질 수 있음을 보인 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- long context, edge deployment, on-device inference에서는 parameter 수보다 **KV cache memory** 가 더 직접적인 병목이 되는 경우가 많다.
- Q-K=V는 GQA/MQA와 경쟁하는 방식이 아니라, 그 위에 추가로 얹을 수 있는 projection sharing 방식이다.
- synthetic, vision, language modeling까지 넓게 비교해서 Q, K, V의 역할을 실험적으로 분해한다.
- Q=K-V와 Q=K=V가 왜 어려운지도 같이 보여주기 때문에, 단순히 "projection을 줄여도 된다"가 아니라 어떤 sharing은 되고 어떤 sharing은 깨지는지 설명한다.

이 논문은 efficient attention paper라기보다 **attention factorization paper** 에 가깝다. 기존 연구가 head dimension, head count, cache quantization, sparsity를 주로 건드렸다면, 이 논문은 attention을 만드는 projection 자체가 얼마나 독립적이어야 하는지 묻는다.

# 1. Problem Setting

## 1-1. Problem definition

표준 Transformer attention은 같은 hidden state $X$에서 Q, K, V를 각각 만든다.

$$
Q = X W_Q, K = X W_K, V = X W_V
$$

그리고 attention output은 보통 다음처럼 계산된다.

$$
A = softmax(Q K^T / sqrt(d_h))
$$

$$
Y = A V
$$

이 구조에서 Q는 현재 token이 무엇을 찾는지, K는 과거 token이 어떤 주소로 검색될지, V는 검색된 뒤 실제로 섞일 내용을 담당한다. 이런 분리는 직관적으로 자연스럽다. 하지만 inference memory 관점에서는 K와 V가 모두 cache에 남는다. sequence length가 길어질수록 cache memory는 layer 수, head 수, head dimension, token 수에 비례해서 커진다.

단순화하면 standard MHA의 per-layer cache는 다음 형태로 볼 수 있다.

$$
M_{cache} = 2 * L * H * d_h
$$

여기서 factor 2는 K와 V 두 cache를 의미한다. 즉 K와 V를 하나로 묶을 수 있다면, 이론적으로 KV cache의 절반을 바로 줄일 수 있다. 이 논문이 겨냥하는 문제는 바로 이 지점이다.

**Q, K, V의 기능적 분리가 실제로 항상 필요한가.**

더 구체적으로는 아래 세 가지 sharing constraint를 비교한다.

| Variant | Projection constraint | Attention directionality | Cache implication |
| --- | --- | --- | --- |
| Q-K=V | Q separate, K and V shared | asymmetric score 가능 | K와 V cache를 하나로 통합 |
| Q=K-V | Q and K shared, V separate | raw score가 symmetric해짐 | cache는 크게 줄지 않음 |
| Q=K=V | Q, K, V 모두 shared | raw score가 symmetric해짐 | projection은 가장 단순하지만 표현력 위험 큼 |

## 1-2. Why previous approaches are insufficient

기존 efficient attention 계열은 주로 다음 방향으로 KV cache를 줄여왔다.

- GQA는 여러 query head가 더 적은 수의 key-value head를 공유하게 만든다.
- MQA는 모든 query head가 하나의 key-value head를 공유하게 만든다.
- KV cache quantization은 cache precision을 낮춘다.
- sparse attention은 일부 token만 보게 만든다.
- sliding window나 retrieval cache는 context 사용 범위를 제한한다.

이 접근들은 모두 중요하지만, 대부분 **head sharing** 이나 **cache representation** 을 건드린다. 반면 이 논문은 projection sharing을 본다. 즉 K와 V가 애초에 같은 projection에서 나와도 되는지 묻는다.

이 차이는 작지 않다. GQA/MQA는 head axis를 줄이는 방식이다. Q-K=V는 projection role axis를 줄이는 방식이다. 그래서 둘은 같은 축에서 경쟁하지 않는다. 논문이 특히 강조하는 지점도 여기에 있다. Q-K=V는 GQA-4나 MQA와 결합할 수 있고, 그 경우 cache reduction이 더 커진다.

기존 접근의 빈칸은 다음 한 문장으로 요약된다.

"KV cache를 줄이는 가장 직접적인 방법은 V cache를 안 따로 저장하는 것일 수 있다"

물론 이 가정은 위험하다. K는 attention score를 만들고, V는 content mixing을 담당한다. 두 역할이 같아지면 score를 잘 만들기 위한 표현과 content를 잘 전달하기 위한 표현이 충돌할 수 있다. 이 논문은 이 위험이 실제로 얼마나 큰지 여러 task에서 재는 실험으로 볼 수 있다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 크게 네 가지다.

1. **QKV projection sharing variants를 체계적으로 정의하고 비교한다.**
   - Q-K=V, Q=K-V, Q=K=V 세 가지 constraint를 synthetic, vision, language modeling에서 비교한다.

2. **Q-K=V가 가장 실용적인 variant라는 결론을 제시한다.**
   - key와 value를 공유하면 KV cache를 50% 줄인다.
   - language modeling에서는 Q-K=V가 only 3.1% perplexity degradation을 보인다고 보고한다.

3. **projection sharing과 head sharing이 complementary하다는 점을 보인다.**
   - Q-K=V + GQA-4는 87.5% cache reduction을 보고한다.
   - Q-K=V + MQA는 96.9% cache reduction을 보고한다.

4. **왜 Q-K=V는 되고 Q=K-V는 어려운지 해석한다.**
   - Q-K=V는 Q와 K가 분리되어 attention score directionality를 유지한다.
   - Q=K-V와 Q=K=V는 raw attention map이 symmetric해지는 문제가 있다.
   - 논문은 asymmetric attention을 회복하기 위해 2D positional encoding도 탐색한다.

## 2-2. Design intuition

이 논문의 설계 직관은 두 단계로 이해할 수 있다.

첫째, K와 V는 생각보다 가까운 representational space를 쓸 수 있다. K는 retrieval address이고 V는 retrieved content라고 설명하지만, 실제 trained transformer에서는 둘이 완전히 독립적인 의미 공간일 필요가 없을 수 있다. 논문은 Q-K=V가 품질을 보존하는 이유로 keys and values can occupy similar representational spaces와 low-rank attention regime을 제시한다.

둘째, Q와 K를 묶는 것은 훨씬 더 위험하다. Q=K가 되면 score matrix는 기본적으로 self-similarity가 된다.

$$
S = Q Q^T
$$

이 score는 positional term이나 mask를 제외하면 symmetric한 구조를 가진다. 하지만 autoregressive attention에서 중요한 것은 token 간 방향성이다. 현재 token이 과거 token을 어떻게 보는지와, 과거 token이 현재 token을 어떻게 보는지는 같은 문제가 아니다. Q=K-V는 이 directionality를 약화시킨다.

이 차이 때문에 Q-K=V는 cache reduction과 quality preservation 사이에서 가장 좋은 trade-off가 된다.

**핵심은 K와 V를 묶는 것은 content 저장 방식을 줄이는 것이고, Q와 K를 묶는 것은 attention score의 방향성을 건드리는 것이다.**

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Q, K, V projection 중 어떤 것을 공유해도 되는지 체계적으로 평가 |
| Main variant | Q-K=V, separate Q with shared K and V |
| Alternative variants | Q=K-V, Q=K=V |
| Core benefit | KV cache memory reduction without changing attention API too much |
| Key experiments | synthetic tasks, vision tasks, 300M and 1.2B language modeling |
| Main practical claim | Q-K=V is complementary to GQA and MQA |
| Code release | language modeling experiments for 300M and 1.2B settings 공개 |

## 3-2. Module breakdown

### 1) Standard QKV baseline

표준 baseline은 Q, K, V를 모두 따로 projection한다.

$$
Q = X W_Q, K = X W_K, V = X W_V
$$

이 구조의 장점은 역할 분리가 가장 명확하다는 점이다. Q는 query role, K는 address role, V는 content role을 각각 학습한다. 단점은 inference 때 K와 V를 모두 저장해야 하므로 long context에서 cache memory가 커진다는 점이다.

### 2) Q-K=V

Q-K=V는 이 논문의 headline result다.

$$
Q = X W_Q, K = V = X W_{KV}
$$

이 방식은 Q를 따로 유지하기 때문에 attention score는 여전히 asymmetric한 형태를 가질 수 있다.

$$
S = Q K^T
$$

하지만 K와 V가 같은 tensor이므로 cache에는 하나의 representation만 저장하면 된다. 이것이 50% KV cache reduction의 직접적인 이유다. 구현 관점에서도 attention의 전체 구조는 크게 바꾸지 않는다. 다만 value mixing에 쓰는 tensor가 key tensor와 같아진다.

이 variant가 중요한 이유는 단순하다. **KV cache를 줄이면서 attention의 검색 방향성은 보존한다.**

### 3) Q=K-V

Q=K-V는 query와 key를 공유하고 value는 따로 둔다.

$$
Q = K = X W_{QK}, V = X W_V
$$

이 경우 value cache는 남기 때문에 cache memory 이득은 Q-K=V보다 직접적이지 않다. 더 큰 문제는 score다.

$$
S = Q Q^T
$$

이 score는 self-similarity 기반이 되므로 raw attention map이 symmetric해진다. causal mask가 적용되면 실제 사용 가능한 영역은 triangular해지지만, score를 만드는 기본 함수가 방향성을 충분히 표현하지 못할 수 있다. 그래서 논문은 이 variant가 Q-K=V보다 불리하다고 해석한다.

### 4) Q=K=V

Q=K=V는 하나의 projection으로 query, key, value를 모두 만든다.

$$
Q = K = V = X W
$$

가장 극단적인 weight tying이다. parameter 관점에서는 가장 단순하지만, score와 content가 모두 같은 표현을 공유한다. 이 경우 model이 attention address, attention query, propagated content를 하나의 vector space에서 모두 처리해야 하므로 표현력 위험이 가장 크다.

이 variant는 연구적으로는 중요하다. projection 역할을 끝까지 묶었을 때 어디서 깨지는지 보여주기 때문이다. 하지만 실용적인 cache reduction recipe로는 Q-K=V가 더 낫다.

### 5) 2D positional encoding for symmetric variants

논문은 Q=K-V와 Q=K=V가 만드는 symmetric attention 문제를 완화하기 위해 2D positional encoding도 탐색한다. 직관은 query position과 key position을 score 계산에 다르게 넣어 raw content similarity만으로 attention이 결정되지 않게 하는 것이다.

다만 이 부분은 main recipe라기보다 diagnostic tool에 가깝다. 핵심 결론은 Q=K를 묶은 variant를 어떻게든 살리는 것보다, Q를 분리하고 K=V를 공유하는 것이 더 안정적이라는 쪽이다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 세 범주의 실험을 사용한다.

1. Synthetic tasks
   - projection sharing이 단순한 memorization이나 local pattern에서만 되는지 확인하기 위한 작은 task들이다.

2. Vision tasks
   - MNIST, CIFAR, TinyImageNet, anomaly detection, segmentation 등에서 비교한다.
   - vision setting에서는 attention map symmetry와 positional encoding의 효과를 비교하기 좋다.

3. Language modeling
   - 300M 및 1.2B parameter model을 사용한다.
   - 10B tokens setting에서 standard QKV, Q-K=V, GQA, MQA, combined variants를 비교한다.
   - GitHub README 기준 language modeling data는 SlimPajama를 사용한다.

## 4-2. Training strategy

공개 code repository는 language modeling experiment를 재현하기 위한 script를 variant별로 분리해 둔다. 중요한 점은 notation과 file alias가 조금 다르다는 것이다.

| Paper notation | Description | Practical note |
| --- | --- | --- |
| QKV | standard baseline | separate Q, K, V |
| Q-K=V | shared K and V | headline variant |
| Q=K-V | shared Q and K | 300M scale only |
| Q=K=V | all shared | 300M scale only |
| GQA-4 / GQA-8 | grouped query attention | head sharing baseline |
| MQA | multi-query attention | strongest cache sharing baseline |
| Q-GQA | Q-K=V plus GQA | projection sharing plus head sharing |
| Q-MQA | Q-K=V plus MQA | largest cache reduction |

GitHub README에는 paper experiments가 8 x NVIDIA A100 40GB, DDP, bfloat16 mixed precision에서 수행됐다고 적혀 있다. 300M setting은 약 24 hours, 1.2B setting은 약 3 days로 안내된다. 이 수치는 reproduce planning 관점에서 유용하지만, 논문의 주된 claim은 training speed보다 inference cache memory에 있다.

## 4-3. Engineering notes

실무 관점에서 이 논문을 읽을 때 가장 중요한 engineering point는 세 가지다.

### 1) Q-K=V는 cache format을 직접 바꾼다

Standard attention에서는 token마다 K와 V를 모두 저장한다. Q-K=V에서는 K와 V가 같은 tensor이므로 cache entry를 하나로 줄일 수 있다. 이론적으로는 per-layer KV cache의 factor 2가 factor 1이 된다.

$$
M_{QKV} = 2 * L * H * d_h
$$

$$
M_{Q-K=V} = 1 * L * H * d_h
$$

### 2) GQA/MQA와 같은 축이 아니다

GQA/MQA는 head 수를 줄인다. Q-K=V는 role 수를 줄인다. 그래서 둘을 곱처럼 결합할 수 있다. 논문이 보고한 cache reduction도 이 구조와 맞는다.

- Q-K=V alone: 50% reduction
- Q-K=V + GQA-4: 87.5% reduction
- Q-K=V + MQA: 96.9% reduction

### 3) 기존 checkpoint에 바로 붙이는 compression은 아니다

이 논문은 trained QKV model의 K/V weight를 사후에 평균내는 식의 plug-in compression이 아니다. variant별로 architecture를 바꿔 학습하고 비교한다. 따라서 이미 학습된 model에 무손실로 적용하는 방법이라고 읽으면 안 된다.

**재사용하려면 pretraining 또는 continued training recipe에 들어가야 한다.**

# 5. Evaluation

## 5-1. Main results

논문의 main result는 크게 네 줄로 정리할 수 있다.

1. Q-K=V는 여러 task에서 standard QKV와 비슷하거나 일부 경우 더 좋은 성능을 보인다.
2. Language modeling에서는 50% KV cache reduction을 얻으면서 only 3.1% perplexity degradation을 보고한다.
3. Q-K=V와 GQA-4를 결합하면 87.5% cache reduction을 보고한다.
4. Q-K=V와 MQA를 결합하면 96.9% cache reduction을 보고한다.

여기서 가장 중요한 결과는 2번만이 아니다. 3번과 4번이 더 실무적일 수 있다. 이미 많은 inference stack이 GQA/MQA를 쓰고 있기 때문에, projection sharing이 그 위에 추가로 얹힌다면 long context memory budget을 더 크게 줄일 수 있다.

## 5-2. What really matters in the experiments

이 논문의 실험에서 진짜 봐야 하는 것은 absolute score ranking이 아니라 failure mode다.

### 1) Q-K=V는 directionality를 유지한다

Q-K=V는 Q와 K를 분리한다. 그래서 token i가 token j를 보는 score와 token j가 token i를 보는 score가 같은 self-similarity 함수에 갇히지 않는다. 이 점이 Q=K-V와 가장 큰 차이다.

### 2) Q=K-V와 Q=K=V의 문제는 cache가 아니라 score다

이 variant들은 projection 수를 줄이지만 attention map symmetry를 유발한다. 이 결과는 important negative result다. 모든 projection sharing이 같은 것은 아니며, 어떤 role을 묶는지가 훨씬 중요하다.

### 3) Low-rank attention 해석은 실용적이다

논문은 attention이 low-rank regime에서 작동한다는 관찰과 K/V representational proximity를 통해 Q-K=V가 버틸 수 있는 이유를 설명한다. 이 해석이 맞다면, future model design에서는 K/V projection을 완전히 분리하는 대신 shared base plus small adapter 같은 중간 설계도 가능하다.

### 4) Edge deployment claim은 cache 기준으로 읽어야 한다

논문은 on-device inference를 중요한 motivation으로 둔다. 다만 여기서 직접적인 근거는 KV cache memory reduction이다. 실제 end-to-end latency, kernel support, memory bandwidth, batching, quantization과 결합했을 때의 결과는 별도 engineering validation이 필요하다.

# 6. Limitations

1. **Training-from-scratch cost가 있다.**
   - Q-K=V는 기존 trained QKV checkpoint에 바로 붙이는 lossless compression이 아니다. architecture variant로 학습해야 한다.

2. **Language modeling scale은 frontier scale이 아니다.**
   - 300M과 1.2B, 10B tokens 결과는 설계 방향을 보기에는 충분하지만, 10B+ parameter 혹은 trillion-token regime에서 같은 trade-off가 유지되는지는 추가 확인이 필요하다.

3. **Perplexity degradation 3.1%는 작은 값이지만 zero-cost는 아니다.**
   - long-context deployment에서는 50% cache reduction이 더 중요할 수 있지만, quality-sensitive task에서는 이 손실이 downstream behavior에 어떻게 나타나는지 봐야 한다.

4. **GQA/MQA와 결합한 실제 runtime은 kernel 구현에 달려 있다.**
   - cache memory가 줄어도 framework가 shared K/V cache를 잘 활용하지 못하면 latency 이득은 제한될 수 있다.

5. **Q=K variants의 2D positional encoding 결과는 main recipe로 보기 어렵다.**
   - symmetric attention을 완화하는 방향은 흥미롭지만, 논문의 실용적 결론은 여전히 Q-K=V 중심이다.

6. **Vision과 language의 요구사항이 다를 수 있다.**
   - vision task에서 projection sharing이 잘 작동해도 autoregressive long-context LM에서 필요한 directionality와 cache behavior는 다르다. 두 결과를 같은 의미로 읽으면 안 된다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 long-context efficiency를 보는 관점을 조금 바꾼다. 보통 KV cache를 줄인다고 하면 quantization, eviction, sparsity, retrieval, head sharing을 먼저 떠올린다. 그런데 Q-K=V는 더 구조적인 질문을 한다.

"왜 우리는 V를 K와 따로 저장해야 한다고 믿었나"

이 질문은 실무적으로 중요하다. long-context serving에서는 prefill보다 decode 단계의 cache residency가 계속 비용으로 남는다. 특히 batch size가 커지거나 context length가 길어지면, model weight보다 KV cache가 먼저 memory budget을 압박할 수 있다. Q-K=V는 이런 상황에서 cache format 자체를 줄이는 선택지다.

또 하나 중요한 점은 Q-K=V가 MQA와 경쟁하지 않는다는 것이다. 이미 MQA/GQA를 쓰는 model design에서도 K/V projection sharing을 추가로 고민할 수 있다. 즉 이 논문은 efficient attention의 새 replacement라기보다, 기존 recipe에 추가할 수 있는 orthogonal knob으로 보는 편이 맞다.

## 7-2. Reuse potential

실제로 재사용한다면 아래 방향을 먼저 떠올릴 수 있다.

1. Small on-device LM pretraining
   - memory budget이 빡빡하고 quality target이 modest한 모델에서 Q-K=V를 baseline으로 넣어볼 만하다.

2. Long-context specialist model
   - 128K 이상 context를 겨냥할 때 cache reduction이 직접적인 serving capacity로 이어질 수 있다.

3. GQA/MQA plus projection sharing ablation
   - 기존 GQA/MQA architecture에서 K/V projection을 공유하는 ablation을 추가하면 실험 비용 대비 signal이 클 수 있다.

4. Shared base plus residual K/V adapter
   - 완전 K=V보다 부드러운 중간 설계도 가능하다.

$$
K = X W_{base} + X W_{K,res}
$$

$$
V = X W_{base} + X W_{V,res}
$$

이런 구조는 cache benefit을 일부 잃을 수 있지만, K/V role conflict를 줄이는 방향으로 볼 수 있다. 다만 이는 논문에 없는 follow-up idea이므로, 실제 효과는 별도 검증이 필요하다.

## 7-3. Follow-up papers

- Multi-Query Attention 관련 논문
- Grouped-Query Attention 관련 논문
- KV cache quantization and eviction 관련 논문
- Multi-head Latent Attention 계열 논문
- Long-context serving system 논문

특히 이 논문은 MLA나 GQA처럼 cache를 줄이는 다른 architecture와 같이 읽으면 좋다. 공통 질문은 하나다.

"attention quality를 얼마나 유지하면서 cache state를 얼마나 압축할 수 있는가"

# 8. Summary

- 이 논문은 Q, K, V projection을 모두 따로 둬야 하는지 체계적으로 묻는다.
- 가장 실용적인 variant는 Q를 분리하고 K와 V를 공유하는 Q-K=V다.
- Q-K=V는 KV cache를 50% 줄이고, language modeling에서 only 3.1% perplexity degradation을 보고한다.
- GQA/MQA와 결합할 수 있기 때문에 cache reduction은 87.5% 또는 96.9%까지 커진다.
- 핵심 해석은 K/V sharing은 directionality를 보존하지만, Q/K sharing은 attention score symmetry 문제를 만든다는 점이다.
