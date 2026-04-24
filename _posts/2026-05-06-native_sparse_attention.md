---
layout: single
title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention Review"
categories: Study-concept
tag: [LLM, SparseAttention, LongContext]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2502.11089)

Native Sparse Attention, 줄여서 NSA는 "sparse attention으로 attention 연산량을 줄였다" 정도가 핵심이 아니다. 이 논문이 진짜로 겨냥하는 문제는 단순한 FLOPs 감소가 아니라, **long-context LLM에서 sparse attention을 학습부터 inference까지 실제로 쓸 수 있는 architecture / kernel / training unit으로 만드는 것**이다.

기존 sparse attention 논문들은 대개 두 방향 중 하나에 가까웠다. 하나는 pretrained Full Attention 모델 위에서 inference 시점에 KV cache를 덜 읽거나 token을 pruning하는 방식이고, 다른 하나는 sliding window나 fixed pattern처럼 구조적으로 sparse한 attention을 쓰는 방식이다. 전자는 학습 시점에는 여전히 full attention을 쓰기 때문에 sparse pattern이 model 내부에 native하게 자리 잡기 어렵고, 후자는 long-range information을 충분히 유연하게 읽기 어렵다. NSA는 이 둘 사이에서 꽤 명확한 설계를 제안한다. **전역 정보는 압축해서 훑고, 중요한 block은 다시 fine-grained하게 읽고, local context는 sliding window가 맡는다.**

내가 보기엔 이 논문의 가장 좋은 점은 "sparsity pattern을 어떻게 고를 것인가?"만 다루지 않는다는 점이다. NSA는 sparse attention이 실제로 빠르려면 contiguous block access, GQA/MQA와의 memory sharing, Tensor Core 활용, backward operator, prefill/decoding phase의 arithmetic intensity까지 같이 봐야 한다고 주장한다. 즉 이 논문은 attention algorithm 논문이면서 동시에 kernel/system 논문에 가깝다.

> 한 줄 요약: NSA는 compression / selection / sliding-window 세 branch를 결합해 global context awareness와 local precision을 동시에 유지하고, blockwise sparse kernel과 native training recipe를 통해 long-context attention을 학습과 inference 전 구간에서 효율화하려는 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Long-context LLM에서 attention bottleneck은 단순히 이론적 복잡도 문제가 아니라 **training, prefilling, decoding마다 다른 hardware bottleneck**으로 나타난다.
- NSA는 sparse attention을 post-hoc inference trick이 아니라 **pretraining부터 들어가는 model architecture**로 다룬다.
- compression branch를 coarse global scanner로 쓰고, 그 score를 selection branch의 block routing signal로 재사용하는 설계가 깔끔하다.
- GQA/MQA 환경에서 head별로 제각각 KV block을 고르면 memory access union이 커진다는 문제를 명확히 짚고, group-level selection으로 hardware alignment를 맞춘다.
- DeepSeek 계열 long-context / MoE / efficient serving stack을 이해할 때, NSA는 "attention sparsity를 실제 system으로 내리는 방법"을 보여주는 좋은 reference가 된다.

내 해석은 이렇다. **NSA의 핵심은 sparse attention 자체가 아니라, sparse attention이 빠르게 동작할 수밖에 없는 access pattern을 architecture 단계에서 강제하는 것**이다. 이 논문은 "어떤 token이 중요한가?"보다 "중요한 token을 GPU가 잘 읽을 수 있는 방식으로 고르는가?"를 더 중요하게 본다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 long-context LLM에서 Full Attention의 비용을 줄이면서도 model capability를 유지하는 것이다.

일반적인 causal self-attention에서 query token $q_t$는 이전 token들의 key/value $k_{:t}, v_{:t}$ 전체를 본다. 이 방식은 가장 직접적이고 강력하지만, sequence length가 길어질수록 attention이 전체 latency와 memory traffic의 핵심 병목이 된다. 논문은 특히 64k context decoding에서 softmax attention이 전체 latency의 70~80%를 차지할 수 있다고 설명한다.

여기서 중요한 점은 attention bottleneck이 phase마다 다르게 생긴다는 것이다.

- **Training / prefilling**에서는 batched matrix multiplication이 많아 compute-bound에 가깝다. 따라서 계산량 자체를 줄이는 것이 중요하다.
- **Autoregressive decoding**에서는 한 token씩 생성하면서 전체 KV cache를 반복해서 읽어야 하므로 memory-bandwidth-bound에 가깝다. 따라서 KV cache memory access를 줄이는 것이 중요하다.

즉 long-context sparse attention이 정말 실용적이려면 다음 두 조건을 동시에 만족해야 한다.

1. 학습과 prefilling에서는 compute reduction이 실제 wall-clock speedup으로 이어져야 한다.
2. decoding에서는 KV cache를 덜 읽고, 덜 읽는 방식이 GPU memory access에 유리해야 한다.

NSA는 이 문제를 **native sparse attention** 문제로 다시 정의한다. pretrained full attention model을 inference에서 억지로 sparse하게 만드는 것이 아니라, 처음부터 sparse attention pattern을 model이 학습하게 만들자는 것이다.

## 1-2. Why previous approaches are insufficient

논문이 기존 sparse attention 방법들을 비판하는 포인트는 꽤 현실적이다.

첫째, 많은 방법이 **phase-restricted sparsity**에 머문다. 예를 들어 어떤 방법은 decoding에서 KV cache를 줄이는 데 초점을 맞추지만 prefilling에서는 dense attention과 유사한 비용을 내고, 어떤 방법은 prefilling을 가속하지만 autoregressive decoding에서는 충분한 이득을 못 낸다. 긴 문서 요약이나 code completion처럼 prefilling 비중이 큰 workload와, long chain-of-thought reasoning처럼 decoding 비중이 큰 workload는 병목이 다르기 때문에 한쪽만 빠른 sparse attention은 실제 serving 관점에서 제한적이다.

둘째, 기존 sparse attention은 최신 attention architecture와 잘 맞지 않을 수 있다. 특히 MQA/GQA에서는 여러 query head가 KV cache를 공유한다. 그런데 head마다 독립적으로 KV block을 고르면, 실제 memory access는 각 head 선택의 합집합이 된다. 이 경우 계산량은 줄어도 KV cache read volume은 크게 줄지 않을 수 있다. NSA가 GQA group 내에서 같은 selected block을 공유하려는 이유가 여기에 있다.

셋째, inference-only sparsification은 model의 pretraining trajectory와 어긋날 수 있다. Full Attention으로 학습된 모델은 attention을 dense하게 쓸 수 있다는 전제 아래 representation과 retrieval head를 형성한다. 이를 inference에서 갑자기 pruning하면 중요한 long-range path가 잘릴 수 있다. 논문은 top 20% attention이 전체 attention score의 70%만 cover한다는 기존 관찰을 인용하며, post-hoc pruning의 위험을 지적한다.

넷째, trainable sparse attention으로 가려 해도 구현 난도가 크다. clustering, hashing, sampling 기반 방식은 discrete operation이 많아 gradient flow가 끊기거나, token-granular random access 때문에 FlashAttention류 blockwise kernel 최적화와 잘 맞지 않는다. 결국 이론적으로는 sparse해도 실제 GPU에서는 느릴 수 있다.

정리하면 기존 접근의 한계는 단순히 "sparse pattern이 별로다"가 아니다. 문제는 **sparse pattern, training graph, memory access pattern, kernel schedule이 서로 따로 논다는 것**이다. NSA는 이 네 가지를 하나의 설계 문제로 묶는다.

# 2. Core Idea

## 2-1. Main contribution

NSA의 핵심 아이디어는 original KV 전체를 그대로 attention하지 않고, 각 query에 대해 더 작고 정보 밀도가 높은 representation KV set을 동적으로 구성하는 것이다.

논문은 attention output을 대략 다음처럼 재정의한다.

$$
\tilde{K}_t = f_K(q_t, k_{:t}, v_{:t}), \quad \tilde{V}_t = f_V(q_t, k_{:t}, v_{:t})
$$

$$
o_t^* = \text{Attn}(q_t, \tilde{K}_t, \tilde{V}_t)
$$

여기서 NSA는 $\tilde{K}, \tilde{V}$를 하나의 방식으로 만들지 않는다. 세 가지 branch를 병렬로 둔다.

| Branch | 역할 | 설계 직관 |
| --- | --- | --- |
| Compression | 긴 context를 block 단위로 압축해 coarse-grained global information을 제공 | 전체를 대략 훑는 global scanner |
| Selection | compression attention score를 활용해 중요한 block을 고르고, 해당 block의 원본 token을 fine-grained하게 읽음 | global scanner가 찾은 후보를 정밀하게 재조회 |
| Sliding window | 최근 token window를 따로 처리 | local context와 recency pattern을 안정적으로 담당 |

최종 output은 세 branch의 attention output을 learned gate로 합친다.

$$
o_t^* = \sum_{c \in \{cmp, slc, win\}} g_t^c \cdot \text{Attn}(q_t, \tilde{K}_t^c, \tilde{V}_t^c)
$$

여기서 $g_t^c$는 input feature에서 MLP와 sigmoid를 통해 얻는 branch별 gate다. 중요한 점은 NSA가 sparse attention을 하나의 hard pattern으로 고정하지 않는다는 것이다. compression, selected fine-grained token, local window라는 서로 다른 information source를 분리하고, model이 token별로 어느 branch를 얼마나 쓸지 학습하게 한다.

## 2-2. Design intuition

NSA의 설계 직관은 세 단계로 볼 수 있다.

첫째, **attention score는 token 단위로 완전히 흩어져 있지 않고 blockwise continuity를 가진다**는 관찰이다. 논문은 Full Attention 모델의 attention map을 시각화하며, 가까운 key들이 비슷한 importance를 보이는 blockwise clustering pattern을 보여준다. 그렇다면 token 하나하나를 random하게 고르는 것보다, contiguous block 단위로 고르는 것이 성능과 hardware 효율을 동시에 만족시킬 가능성이 높다.

둘째, **compression branch를 selection branch의 routing signal로 재사용한다**는 점이다. 단순히 block importance를 예측하기 위해 별도 predictor를 만들면 auxiliary loss, 추가 parameter, 추가 kernel overhead가 생긴다. NSA는 이미 compression attention에서 계산한 score를 활용해 selection block의 importance를 유도한다. 이 선택은 꽤 영리하다. global summary를 만들기 위해 어차피 계산한 score를 fine-grained retrieval의 index로 재사용하기 때문이다.

셋째, **local pattern을 별도 branch로 분리한다**는 점이다. LLM attention에서 local context는 강하고 빠르게 학습되는 signal이다. 이 local signal이 compression/selection branch까지 지배하면, 장거리 branch가 제대로 학습되기 전에 local shortcut에 끌려갈 수 있다. 그래서 NSA는 sliding window branch를 별도로 두고, compression/selection branch가 각각 global / sparse retrieval 역할을 배우도록 만든다.

내가 보기엔 NSA의 구조는 attention 내부에 작은 retrieval pipeline을 넣은 것처럼 읽을 수 있다.

- compression = coarse indexing
- selection = top-k block retrieval
- selected attention = retrieved block에 대한 precise reading
- sliding window = recency prior
- gate = query-dependent source fusion

이렇게 보면 NSA는 RAG와도 닮아 있다. 차이는 retrieval 대상이 외부 문서가 아니라 **현재 context의 KV cache**라는 점이고, retrieval 단위가 text chunk가 아니라 **hardware-friendly token block**이라는 점이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Full Attention 수준의 capability를 유지하면서 long-context attention의 training / prefilling / decoding 비용을 줄이는 것 |
| Key idea | 전체 KV를 직접 보지 않고 compression / selection / sliding-window branch로 remapped KV set을 구성 |
| Core branches | compressed attention, selected attention, sliding attention |
| Selection signal | compression branch의 attention score를 block importance로 재사용 |
| Hardware alignment | token-level random selection 대신 contiguous block selection 사용 |
| GQA/MQA compatibility | GQA group 내 query heads가 같은 sparse KV block을 공유하도록 importance score를 aggregate |
| Kernel strategy | Triton 기반 sparse selection attention kernel, group-centric query loading, contiguous selected KV block fetching |
| Training support | backward operator까지 포함한 end-to-end sparse attention training |
| Difference from prior work | inference-only pruning이 아니라 sparse attention 자체를 pretraining trajectory 안에 넣음 |

## 3-2. Module breakdown

### 1) Overall framework: 세 branch의 gated sum

NSA는 query $q_t$마다 세 종류의 remapped KV를 만든다.

$$
\mathcal{C} = \{cmp, slc, win\}
$$

- $cmp$: compressed branch
- $slc$: selected branch
- $win$: sliding-window branch

각 branch는 자기 방식으로 $\tilde{K}_t^c, \tilde{V}_t^c$를 만들고, query는 branch별 attention을 수행한다. 이후 branch output을 gate로 합산한다.

이 구조에서 중요한 목표는 $N_t \ll t$를 유지하는 것이다. 여기서 $N_t$는 query가 실제로 attention하는 remapped key/value의 총 개수다. 즉 긴 context 전체를 그대로 읽지 않고, 작은 representation set만 읽게 만든다.

하지만 $N_t$를 줄이는 것만으로는 부족하다. selected token들이 memory에 흩어져 있으면 GPU에서는 random access가 많아지고, 실제 latency는 기대만큼 줄지 않는다. 그래서 NSA는 selection 단위를 token이 아니라 contiguous block으로 잡는다.

### 2) Token Compression: 긴 context를 block-level memory로 바꾼다

Compression branch는 key/value sequence를 sequential block으로 나누고, 각 block을 하나의 compressed key/value representation으로 변환한다.

논문 notation으로는 block length를 $l$, stride를 $d$로 두고, learnable MLP $\phi$와 intra-block position encoding을 사용해 block 안의 key들을 하나의 compressed key로 mapping한다.

직관적으로는 다음과 같다.

$$
\tilde{k}^{cmp}_i = \phi(k_{id+1:id+l})
$$

여기서 stride $d$는 보통 block length $l$보다 작게 둔다. 즉 compression block이 서로 overlap된다. 논문은 이를 information fragmentation을 완화하기 위한 선택으로 설명한다.

이 branch의 역할은 두 가지다.

1. 긴 context 전체를 coarse-grained하게 훑는다.
2. 이후 selected branch가 어떤 block을 다시 읽을지 결정하는 importance score를 제공한다.

compression만 쓰면 중요한 fine-grained token을 잃을 수 있다. 하지만 selection만 쓰려면 어떤 block이 중요한지 알아야 한다. NSA는 compression을 단순한 요약 branch가 아니라 **sparse retrieval의 router**로도 사용한다.

### 3) Token Selection: compressed score로 중요한 contiguous block을 다시 읽는다

Selection branch는 NSA에서 가장 중요한 부분이다.

먼저 compression branch에서 query와 compressed keys 사이의 attention score를 얻는다.

$$
p_t^{cmp} = \text{Softmax}(q_t^T \tilde{K}_t^{cmp})
$$

이 score는 compressed block이 query와 얼마나 관련 있는지를 나타낸다. NSA는 이 score를 selection block의 importance score로 변환하고, importance가 높은 top-$n$ block을 고른다. 선택된 block 안에서는 원본 key/value token을 그대로 가져와 attention한다.

즉 selection branch는 다음 두 단계를 거친다.

1. compressed score로 중요한 block을 찾는다.
2. 그 block 안의 원본 token을 fine-grained하게 attention한다.

이 설계의 장점은 중요하다. 별도 neural router를 두지 않고도 query-aware block selection을 만들 수 있고, selected token들이 contiguous block으로 유지되므로 kernel 입장에서도 읽기 쉽다.

GQA/MQA compatibility도 이 부분에서 나온다. GQA에서는 여러 query head가 같은 KV head group을 공유한다. 만약 head마다 다른 block을 선택하면, 실제 KV cache memory access는 head별 선택의 union이 되어 커진다. NSA는 같은 GQA group 안의 head들에 대해 block importance를 aggregate해 **같은 selected block set을 공유**하게 만든다. 이 선택은 expressiveness 일부를 포기하는 대신, decoding memory access를 크게 줄이려는 hardware-aware trade-off다.

### 4) Sliding Window: local shortcut을 별도 branch에 격리한다

Sliding window branch는 최근 $w$개 token의 KV를 그대로 유지한다.

$$
\tilde{K}_t^{win} = k_{t-w:t}, \quad \tilde{V}_t^{win} = v_{t-w:t}
$$

이 branch가 단순해 보이지만, NSA에서는 꽤 중요한 안정화 장치다. LLM에서 local context는 강력하다. 다음 token 예측에서 가까운 token은 대체로 중요하고, model은 이 패턴을 빨리 학습한다. 만약 local context까지 compression/selection branch가 모두 담당하게 만들면, long-range branch가 local shortcut에 의해 지배될 수 있다.

그래서 NSA는 local context를 sliding window branch가 담당하게 하고, compression/selection branch는 장거리 정보와 sparse retrieval을 학습하게 한다. 논문은 세 branch에 independent key/value를 제공해 branch 간 shortcut learning과 gradient interference를 줄이려 한다고 설명한다.

내 해석으로는 이 branch 분리는 NSA의 성능 안정성에 중요하다. Sparse attention에서 흔한 문제는 long-range retrieval을 넣었지만 실제로는 model이 local window에만 의존하는 것이다. NSA는 local path를 명시적으로 인정하고, 대신 long-range path가 자기 역할을 배울 공간을 만든다.

### 5) Gated aggregation: 세 정보원을 query별로 섞는다

NSA는 branch output을 단순 평균하지 않는다. input feature로부터 branch별 gate $g_t^c$를 만들고, 이를 통해 세 branch의 output을 합친다.

이 gate는 중요한 flexibility를 준다. 어떤 token은 local context만으로 충분할 수 있고, 어떤 token은 긴 문서 앞부분의 압축 정보가 필요할 수 있으며, 어떤 token은 selected block의 fine-grained 정보가 필요할 수 있다. gate는 이 세 정보원의 상대적 비중을 token별로 조정한다.

여기서 주의할 점은 gate가 sparse block selection 자체를 완전히 differentiable하게 만드는 것은 아니라는 점이다. Top-$n$ block 선택은 여전히 discrete operation이다. 다만 NSA는 sparse operator 전체를 training graph 안에 넣고 backward를 지원함으로써, sparse attention을 post-hoc inference trick이 아니라 native trainable component로 만든다.

### 6) Kernel design: sparse pattern이 아니라 sparse access pattern을 설계한다

NSA의 kernel design은 논문에서 매우 중요한 부분이다.

일반적인 FlashAttention 방식처럼 temporally continuous query block을 SRAM에 올리면, sparse selection에서는 query마다 필요한 KV block이 달라질 수 있어 memory access가 비효율적이다. NSA는 GQA group 내 head들이 selected KV block을 공유한다는 점을 이용한다.

핵심은 다음과 같다.

1. **Group-centric data loading**  
   하나의 query position $t$에 대해 GQA group 안의 모든 query head를 함께 SRAM에 올린다. 이 head들은 같은 selected KV block index를 공유한다.

2. **Shared KV fetching**  
   selected block index에 해당하는 contiguous KV block을 순차적으로 SRAM에 load한다. token 단위 random access가 아니라 block 단위 contiguous access를 만든다.

3. **Outer loop on Triton grid**  
   selected block count $n$이 query block마다 거의 동일하므로, query/output loop를 Triton grid scheduler에 맡기고 inner loop에서 selected KV block을 처리한다.

이 설계의 목적은 명확하다. Sparse attention에서 흔히 생기는 "계산은 줄였는데 memory access와 scheduling이 망가져서 느린" 문제를 피하는 것이다. NSA는 algorithm 단계에서부터 GPU가 잘 먹는 blockwise access pattern을 만들고, kernel에서 그 구조를 그대로 활용한다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 NSA와 Full Attention baseline을 동일한 backbone과 training recipe에서 비교한다. 데이터 mixture 자체의 세부 구성은 논문에 충분히 공개되어 있지 않다. 따라서 corpus 구성이나 filtering recipe는 원문에서 추가 확인 필요하다.

확실하게 확인되는 training setup은 다음과 같다.

| Item | Value |
| --- | --- |
| Backbone | GQA + MoE Transformer |
| Total parameters | 27B |
| Active parameters | 3B |
| Layers | 30 |
| Hidden dimension | 2560 |
| GQA groups | 4 |
| Attention heads | 64 |
| Query/key dimension | 192 |
| Value dimension | 128 |
| MoE | DeepSeekMoE |
| Routed experts | 72 |
| Shared experts | 2 |
| Top-k experts | 6 |
| First layer | MoE 대신 SwiGLU MLP 사용 |

NSA hyperparameter는 다음과 같다.

| NSA component | Value |
| --- | --- |
| Compression block size $l$ | 32 |
| Compression stride $d$ | 16 |
| Selection block size $l'$ | 64 |
| Selected block count $n$ | 16 |
| Fixed selected blocks | initial block 1개 + local block 2개 포함 |
| Sliding window size $w$ | 512 |

논문 본문에는 pretraining token 수와 관련해 주의할 부분이 있다. Introduction 쪽에서는 27B backbone을 260B tokens로 pretraining했다고 설명하지만, Pretraining Setup section에서는 270B tokens의 8k-length text로 pretraining했다고 적는다. 블로그 게시 전에는 PDF 기준으로 이 260B / 270B 표기 차이를 다시 확인해야 한다. 이 초안에서는 training setup section의 구체 값인 **270B tokens**를 우선 사용한다.

## 4-2. Training strategy

학습은 크게 세 단계로 볼 수 있다.

1. **8k-length text pretraining**  
   NSA와 Full Attention baseline을 같은 backbone에서 학습한다. 논문은 두 모델 모두 full convergence까지 학습했다고 설명한다.

2. **32k long-context adaptation**  
   이후 32k-length text에 대해 continued training과 supervised fine-tuning을 수행하고, YaRN을 사용해 long-context adaptation을 진행한다.

3. **Chain-of-thought reasoning post-training**  
   CoT reasoning 실험에서는 DeepSeek-R1로부터 distillation한 32k-length mathematical reasoning traces 10B tokens로 SFT를 수행한다. 이렇게 만든 모델을 Full Attention-R, NSA-R로 비교한다.

여기서 중요한 점은 NSA가 별도의 auxiliary loss를 핵심으로 삼지 않는다는 것이다. 논문의 주장은 sparse attention operator 자체가 native하게 training graph에 들어가야 한다는 것이다. 즉 "나중에 pruning해도 된다"가 아니라, model이 pretraining 단계부터 compression / selection / window branch를 사용하는 방식에 적응해야 한다는 관점이다.

## 4-3. Engineering notes

NSA의 engineering note에서 가장 중요한 개념은 **arithmetic intensity**다. 같은 sparse ratio라도 training/prefilling과 decoding에서 병목이 다르기 때문에 최적화 목표가 다르다.

- training/prefilling: compute-bound에 가까우므로 attention FLOPs와 kernel utilization이 중요하다.
- decoding: memory-bound에 가까우므로 KV cache read volume을 줄이는 것이 중요하다.

논문은 8-GPU A100 환경에서 Triton 기반 NSA kernel을 Triton 기반 FlashAttention-2와 비교한다. 같은 backend에서 비교하려는 의도다.

Efficiency analysis 설정은 다음과 같다.

| Item | Value |
| --- | --- |
| GPUs | 8 x A100 |
| GQA group $g$ | 4 |
| Heads per group $h$ | 16 |
| Query/key dimension $d_k$ | 192 |
| Value dimension $d_v$ | 128 |
| NSA block/window setting | Section 4와 동일 |

Decoding에서는 각 step마다 전체 KV cache를 읽지 않고 다음만 읽는다.

- compression tokens: roughly $\lfloor (s-l)/d \rfloor$
- selected tokens: $n l'$
- neighbor tokens: $w$

여기서 $s$는 cached sequence length다. 이 때문에 context가 길수록 Full Attention 대비 memory access 감소폭이 커진다. 논문 Table 4 기준 64k context에서 Full Attention은 65536 token-equivalent를 읽는 반면, NSA는 5632 token-equivalent를 읽고, expected speedup은 11.6x로 제시된다.

# 5. Evaluation

## 5-1. Main results

논문 평가는 세 축으로 구성된다.

1. general benchmark
2. long-context benchmark
3. chain-of-thought reasoning benchmark
4. 별도 efficiency analysis

핵심 결과만 압축하면 다음과 같다.

| Evaluation | Full Attention | NSA | Comment |
| --- | ---: | ---: | --- |
| General benchmark average | 0.443 | 0.456 | NSA가 평균 기준 우세. 단, MMLU와 MBPP는 Full Attention이 높음 |
| LongBench average | 0.437 | 0.469 | NSA가 H2O / InfLLM / Quest / Exact-Top / Full Attention보다 평균 높음 |
| AIME 24, 8k generation limit | 0.046 | 0.121 | DeepSeek-R1 distillation SFT 이후 Full Attention-R vs NSA-R 비교 |
| AIME 24, 16k generation limit | 0.092 | 0.146 | 긴 reasoning context에서도 NSA-R 우세 |
| 64k forward speedup | 1.0x | 9.0x | Triton NSA kernel vs Triton FlashAttention-2 기준 |
| 64k backward speedup | 1.0x | 6.0x | context가 길수록 speedup 증가 |
| 64k decoding expected speedup | 1.0x | 11.6x | memory access volume 기반 expected speedup |

General benchmark에서는 NSA가 평균 0.456으로 Full Attention 0.443보다 높다. 세부적으로는 MMLU에서 Full Attention이 0.567, NSA가 0.565로 거의 비슷하지만 Full Attention이 약간 높고, MBPP에서도 Full Attention이 0.482, NSA가 0.466으로 높다. 반면 MMLU-PRO, CMMLU, BBH, GSM8K, MATH, DROP, HumanEval에서는 NSA가 더 높다. 따라서 "모든 지표에서 압도"가 아니라, **평균 및 다수 지표에서 우세**라고 읽는 것이 맞다.

LongBench에서는 차이가 더 분명하다. 논문은 sparse attention baseline으로 H2O, InfLLM, Quest, Exact-Top을 비교한다. token budget은 모든 sparse method가 query당 2560 activated tokens가 되도록 맞춘다. 평균 점수는 H2O 0.303, InfLLM 0.383, Quest 0.392, Exact-Top 0.423, Full Attention 0.437, NSA 0.469다.

여기서 재미있는 점은 Exact-Top보다 NSA가 높다는 것이다. Exact-Top은 full attention score를 먼저 계산하고 top score key를 선택하는, 실제 efficiency 측면에서는 비현실적인 비교군에 가깝다. 그럼에도 NSA가 LongBench 평균에서 더 높게 나온 것은 post-hoc top selection보다 native sparse pretraining이 task-optimal pattern을 더 잘 학습했을 가능성을 보여준다.

Needle-in-a-Haystack에서는 64k context 전 위치에서 perfect retrieval accuracy를 보고한다. 이 결과는 compression branch가 global context를 훑고 selection branch가 fine-grained token을 보존하는 구조가 long-context retrieval에 맞는다는 주장을 뒷받침한다.

CoT reasoning 평가도 흥미롭다. 논문은 DeepSeek-R1에서 distill한 32k-length math reasoning traces 10B tokens로 SFT를 수행한 뒤 AIME 24를 평가한다. generation token limit 8192에서 Full Attention-R은 0.046, NSA-R은 0.121이고, 16384에서는 Full Attention-R 0.092, NSA-R 0.146이다. 절대 점수만 보면 높은 수준은 아니지만, 같은 backbone / 같은 SFT 조건에서 sparse attention이 long reasoning trace를 다루는 데 불리하지 않았다는 점이 중요하다.

## 5-2. What really matters in the experiments

이 논문에서 benchmark 숫자보다 더 중요하게 봐야 하는 것은 **lifecycle consistency**다.

NSA는 sparse attention이 다음 세 구간 모두에서 동작해야 한다고 본다.

1. pretraining / backward
2. prefilling / forward
3. decoding / KV cache reading

많은 sparse attention 방법은 한 구간에서만 강하다. 예를 들어 decoding KV cache eviction은 decoding에는 유리하지만 training cost를 낮추지 못할 수 있고, prefilling 전용 sparse attention은 long CoT decoding에서 이득이 작을 수 있다. NSA의 강점은 세 구간을 모두 같은 sparse architecture로 연결하려 했다는 점이다.

또 하나 봐야 할 점은 **kernel 공정성**이다. 논문은 NSA kernel과 Full Attention baseline을 모두 Triton backend에서 비교한다고 설명한다. FlashAttention-2의 highly optimized CUDA/Triton 구현과 실제 production stack 사이에는 차이가 있을 수 있으므로, absolute latency 숫자는 환경 의존적일 수 있다. 하지만 같은 backend에서 context length별 speedup trend를 비교한 것은 의미가 있다.

세 번째로 중요한 점은 **native training effect**다. NSA가 LongBench에서 Exact-Top보다 높은 평균을 보인 것은 단순히 "더 좋은 token을 골랐다"라고만 설명하기 어렵다. Exact-Top은 full attention score에서 top key를 뽑기 때문에 selection 자체는 강력하다. 그런데 NSA는 training 과정에서 sparse branch와 나머지 model component가 함께 적응한다. 논문은 이 synchronized adaptation이 long-context 성능에 기여한다고 해석한다.

다만 이 해석은 조심해야 한다. LongBench subset 일부는 낮은 점수 때문에 제외되었고, 데이터 mixture도 상세히 공개되어 있지 않다. 따라서 결과를 재사용하려면 같은 benchmark protocol과 excluded subset을 원문에서 다시 확인해야 한다.

# 6. Limitations

1. **Training data mixture가 충분히 공개되어 있지 않다**  
   NSA와 Full Attention baseline의 비교 자체는 같은 recipe라 의미가 있지만, 데이터 mixture / filtering / domain ratio가 자세히 공개되어 있지 않다. 다른 조직이 그대로 재현하기는 어렵다.

2. **260B / 270B token 표기 차이를 확인해야 한다**  
   논문 Introduction 쪽에는 260B tokens가 보이고, Pretraining Setup에서는 270B tokens로 적힌다. 최종 게시 전 PDF와 버전 기준으로 정확히 확인해야 한다.

3. **DeepSeek-style MoE + GQA setting에 강하게 맞춰져 있다**  
   실험 backbone은 27B total / 3B active MoE, GQA, DeepSeekMoE 구조다. dense Transformer나 다른 attention head configuration에서도 같은 trade-off가 유지되는지는 별도 검증이 필요하다.

4. **A100/Triton 기준 kernel result의 portability가 필요하다**  
   논문 efficiency 결과는 8-GPU A100과 Triton implementation 기준이다. H100, MI300, TPU, production inference server 등에서 같은 speedup이 나오는지는 별도 kernel engineering과 profiling이 필요하다.

5. **Top-n block selection은 여전히 discrete decision이다**  
   NSA는 sparse attention을 training graph 안에 넣고 backward operator를 제공하지만, top-n block selection 자체가 완전히 smooth differentiable해지는 것은 아니다. 따라서 "natively trainable"을 "selection index까지 완전히 연속적으로 학습된다"로 읽으면 안 된다.

6. **모든 general benchmark에서 일관되게 이긴 것은 아니다**  
   평균은 NSA가 높지만 MMLU와 MBPP에서는 Full Attention이 조금 더 높다. 논문 주장은 "대부분의 benchmark와 평균에서 유지 또는 개선"으로 읽는 것이 적절하다.

7. **구현 복잡도가 높다**  
   NSA는 단순히 PyTorch에서 top-k block을 뽑는 수준의 module이 아니다. compression branch, selected branch, sliding branch, GQA group-level block sharing, sparse selection kernel, backward operator까지 맞아야 한다. 실제 서비스 적용에는 상당한 kernel/system 비용이 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

내가 보기엔 NSA의 가장 큰 의미는 sparse attention을 **algorithmic sparsity**가 아니라 **system-native sparsity**로 다뤘다는 점이다.

LLM 논문을 읽다 보면 sparse attention은 자주 나온다. 하지만 실제로 serving에 올리려고 하면 다음 질문들이 바로 나온다.

- 선택된 token들이 memory에 흩어져 있으면 어떻게 읽을 것인가?
- GQA/MQA에서 head별 선택이 다르면 KV cache sharing 이득이 사라지지 않는가?
- prefill은 빨라졌는데 decode는 그대로 느린 것 아닌가?
- backward는 지원되는가?
- long-context training에서도 이득이 있는가?

NSA는 이 질문들을 정면으로 다룬다. 특히 blockwise selection과 GQA group-shared selection은 실무적으로 중요하다. sparse attention은 "몇 개 token만 보겠다"보다 "그 token들을 어떻게 연속적으로 읽겠다"가 더 중요할 때가 많다.

RAG나 agent workload 관점에서도 흥미롭다. 긴 context 안에서 중요한 block을 찾아 다시 읽는 구조는 외부 retrieval 없이 context 내부 retrieval을 수행하는 것과 비슷하다. 특히 repository-level code generation, 긴 문서 QA, multi-turn agent memory처럼 context 안에 많은 정보가 들어가 있지만 매 token마다 전체를 볼 필요는 없는 workload에서 이 구조는 자연스럽다.

## 7-2. Reuse potential

NSA를 바로 재사용하려면 full kernel stack이 필요하지만, 아이디어 차원에서는 몇 가지를 가져올 수 있다.

1. **Coarse-to-fine context reading**  
   긴 context를 먼저 압축 representation으로 훑고, 높은 score block만 fine-grained하게 읽는 방식은 attention뿐 아니라 retrieval / document parsing / multimodal memory에도 응용 가능하다.

2. **Local branch 분리**  
   local context는 강력한 prior이므로 별도로 처리하고, long-range branch가 shortcut에 먹히지 않게 분리하는 설계는 long-context architecture 전반에서 유용하다.

3. **GQA-aware sparse selection**  
   sparse KV selection을 할 때 head별 expressiveness만 보지 말고, 실제 memory access union이 얼마나 커지는지 봐야 한다. GQA/MQA에서는 group-level selection이 더 실용적일 수 있다.

4. **Benchmark보다 kernel profile 먼저 보기**  
   sparse attention 논문을 평가할 때는 LongBench 점수뿐 아니라 forward / backward / prefill / decode 각각의 latency, memory access, kernel utilization을 같이 봐야 한다.

5. **Native training 여부 확인**  
   inference-only sparse method는 쉽게 붙일 수 있지만, model이 그 sparsity pattern에 적응하지 않았을 수 있다. long-context 품질까지 보려면 sparse pattern을 pretraining 또는 at least continued training 안에 넣는 실험이 필요하다.

실제로 내가 small-scale reproduction을 한다면 다음 순서로 볼 것 같다.

- 1단계: dense baseline과 NSA-like branch를 1B 이하 모델에서 같은 data로 비교
- 2단계: compression / selection / window branch ablation
- 3단계: token-level selection vs block-level selection의 kernel profile 비교
- 4단계: GQA group-shared selection과 head-wise selection의 KV cache memory access 비교
- 5단계: LongBench / Needle / repository QA / long CoT workload로 phase별 효과 확인

## 7-3. Follow-up papers

- **FlashAttention / FlashAttention-2**  
  NSA kernel design을 이해하려면 blockwise attention kernel과 SRAM/HBM traffic 관점이 필요하다.

- **MQA / GQA**  
  NSA의 GQA group-level block sharing을 이해하려면 KV sharing 구조 자체를 먼저 봐야 한다.

- **H2O / StreamingLLM / SnapKV / Quest / InfLLM / MInference**  
  NSA가 비교하거나 비판하는 inference-time sparse / KV-cache selection 계열이다.

- **SeerAttention / HashAttention / ClusterKV**  
  trainable 또는 query-aware sparse selection을 어떻게 설계했는지 비교하기 좋다.

- **DeepSeekMoE / DeepSeek-V2 / DeepSeek-V3 계열**  
  NSA 실험이 DeepSeekMoE와 GQA/MoE stack 위에서 이뤄지므로, backbone의 design assumption을 이해하는 데 도움이 된다.

- **YaRN**  
  논문에서 32k long-context adaptation에 사용한 context extension 방법이다.

# 8. Summary

- NSA는 sparse attention을 post-hoc inference trick이 아니라 native training architecture로 다루는 논문이다.
- 핵심 구조는 compression / selection / sliding-window 세 branch이며, compression score를 selection block routing signal로 재사용한다.
- blockwise contiguous selection과 GQA group-shared selection을 통해 sparse attention이 실제 GPU memory access와 맞도록 설계한다.
- 27B total / 3B active MoE backbone에서 Full Attention 대비 general / long-context / reasoning 평가를 유지하거나 개선하고, 64k context에서 큰 forward / backward / decoding speedup을 보고한다.
- 다만 데이터 mixture, 260B/270B token 표기, hardware portability, discrete top-n selection, 구현 복잡도는 최종 리뷰에서 반드시 같이 확인해야 한다.
