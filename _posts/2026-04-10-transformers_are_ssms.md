---
layout: single
title: "Transformers are SSMs Review"
categories: Study-concept
tag: [Mamba-2, SSM, Attention]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2405.21060)

Transformers are SSMs는 **Mamba-2 소개 논문**이면서 동시에, attention과 state-space model을 **공통 algebra 위에서 다시 묶어낸 논문**으로 읽는 편이 맞다. 이 논문이 흥미로운 이유는 단순히 "Mamba를 더 빠르게 만들었다"가 아니라, **SSM, linear attention, structured matrices**를 하나의 언어로 정리하고, 그 언어 위에서 알고리즘과 아키텍처를 같이 구성했기 때문이다.

요즘 efficient attention, SSM, hybrid long-context architecture를 읽다 보면 결국 같은 질문으로 돌아오게 된다. **왜 어떤 sequence mixer는 recurrent form과 quadratic form을 동시에 갖는가? 왜 어떤 구조는 이론적으로는 선형인데 실제 하드웨어에서는 느린가? 왜 pure recurrent 모델은 retrieval에서 자꾸 attention을 다시 부르나?** 이 논문은 그 질문들에 대한 좋은 idea map을 제공한다.

> 한 줄 요약: 이 논문은 SSM과 attention을 semiseparable matrix라는 공통 구조 위에서 재해석해 Structured State Space Duality(SSD) 프레임워크를 만들고, 그 결과로 더 hardware-friendly한 Mamba-2와 SSD 알고리즘까지 제안한 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- SSM과 attention을 "대체 관계"가 아니라 **같은 계산을 다른 관점에서 보는 관계**로 다시 정리해 준다.
- Mamba-2의 속도 향상이 단순 kernel trick이 아니라 **이론 -> 알고리즘 -> 시스템 최적화**로 이어지는 흐름 속에서 나온다는 점이 중요하다.
- 최근 long-context / efficient mixer 논문들을 읽을 때 필요한 공통 vocabulary를 준다. 즉, **무슨 recurrence를 썼는가**보다 **어떤 linear form과 quadratic form을 동시에 갖는가**로 보게 만든다.

내가 보기엔 이 논문은 "Mamba-2가 빠르다"보다, **sequence mixer 이해의 지평을 넓혀주는 논문**이다. attention과 SSM을 따로 보는 대신, structured matrix multiplication 관점으로 보면 이후의 hybrid architecture들도 훨씬 쉽게 읽힌다.

# 1. Problem Setting

## 1-1. Problem definition
- 이 논문이 겨냥하는 표면적 문제는 Transformer attention의 비효율이고, 더 본질적인 문제는 **attention과 SSM이 서로 다른 생태계에서 발전해 왔다는 점**이다.
- Transformer는 강력하지만 training에서는 sequence length에 대해 quadratic cost를 갖고, autoregressive inference에서는 KV cache가 길이에 따라 계속 커진다.
- 반대로 structured SSM은 training에서 선형 스케일링, inference에서 상수 크기의 상태(state)를 갖는 장점이 있지만, Transformer 진영에서 축적된 이론, 알고리즘, 시스템 최적화의 혜택을 충분히 공유하지 못했다.
- 따라서 이 논문의 목표는 "SSM이 Transformer보다 빠르다"를 보이는 것이 아니라, **둘 사이의 이론적 연결고리를 만들어 SSM도 Transformer 수준의 최적화와 스케일링 전략을 쓸 수 있게 만드는 것**이다.
- 특히 selective SSM인 Mamba는 정보 밀도가 높은 언어 데이터에서 잘 작동했지만, 핵심 계산이 scan 기반이라 matmul unit을 직접 활용하기 어렵고, 그 결과 modern GPU가 좋아하는 계산 패턴과는 약간 어긋나 있었다.

## 1-2. Why previous approaches are insufficient
- linear attention은 이미 attention과 recurrence를 연결하는 관점을 보여줬지만, 그 연결은 주로 **특정 kernelized attention family**에 국한되어 있었다.
- SSM 쪽은 독립적으로 발전해 오면서, 이론적으로도 "attention과 어떤 관계인가"가 명확하지 않았고, 시스템적으로도 tensor parallelism이나 sequence parallelism 같은 Transformer 중심 최적화를 바로 가져오기 어려웠다.
- Mamba 같은 selective SSM은 분명 강했지만, **하드웨어 친화성**이라는 관점에서는 여전히 scan implementation에 많이 의존했다.
- 또 pure recurrent 모델은 구조적으로 **finite-state memory**를 가지기 때문에, associative recall이나 in-context retrieval처럼 "과거의 특정 토큰을 다시 정확히 꺼내는 문제"에서 attention보다 불리할 수 있다.
- 결국 기존 접근의 한계는 "attention이냐 recurrence냐"의 이분법에 있었다. 이 논문은 그 프레임 자체를 바꾸려 한다.

# 2. Core Idea

## 2-1. Main contribution
- 첫 번째 핵심 기여는 **SSM = semiseparable matrix transformation**이라는 연결을 명확히 보여준 것이다. 즉, state-space recurrence를 structured matrix multiplication 관점으로 다시 쓸 수 있다.
- 두 번째는 linear attention을 더 넓게 일반화한 **Structured Masked Attention (SMA)** 개념이다. 이로써 causal mask 기반 linear attention을 더 일반적인 structured mask로 확장한다.
- 세 번째는 SSM과 SMA가 만나는 교집합을 **Structured State Space Duality (SSD)** 로 정리한 것이다. 이 클래스의 모델은 **SSM 같은 linear form**과 **attention 같은 quadratic form**을 동시에 가진다.
- 네 번째는 이 duality를 실제 계산으로 끌고 내려온 **SSD 알고리즘**이다. 이 알고리즘은 recurrence의 장점과 blockwise quadratic computation의 장점을 같이 활용해, 계산의 대부분을 matrix multiplication으로 바꾼다.
- 다섯 번째는 이 이론을 실제 아키텍처 설계로 연결한 **Mamba-2**다. block design, head structure, tensor parallelism, sequence parallelism까지 Transformer 생태계의 설계 원리를 SSM 쪽으로 옮겨온다.

## 2-2. Design intuition
- 이 논문의 핵심 직관은 단순하다. **recurrence와 attention은 완전히 다른 계산이 아닐 수 있다.** 같은 구조를 어떤 순서로 계산하느냐에 따라 linear mode와 quadratic mode로 보일 수 있다.
- SSM 관점에서는 "작은 state를 recurrence로 업데이트한다"로 보이고, attention 관점에서는 "토큰 간 pairwise interaction을 계산한다"로 보이지만, structured matrix 관점에서는 둘 다 같은 변환의 다른 계산 순서일 수 있다.
- 이 관점이 중요한 이유는, 어떤 sequence mixer가 이론적으로 선형이라고 해서 실제로 빠른 것은 아니기 때문이다. **하드웨어는 asymptotic complexity보다 matmul-friendly한 계산을 더 좋아한다.**
- 그래서 SSD는 pure recurrent scan 하나로 밀어붙이지 않는다. 필요한 부분에서는 quadratic/blockwise form을 쓰고, chunk 간 연결은 recurrence로 처리한다.
- 또 Mamba-2는 기존 Mamba의 일반 diagonal SSM보다 약간 더 제한된 구조를 택한다. 대신 그 제약을 이용해 **하드웨어 효율, 병렬화, 구현 단순성**을 얻는다. 즉, 이 논문은 이론적 일반성보다 **좋은 operating point**를 찾는 데 더 가깝다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | attention과 SSM을 하나의 구조로 해석하고, 그 위에서 더 빠르고 스케일 가능한 SSM을 설계하는 것 |
| Key abstraction | semiseparable matrices + structured masked attention + structured state space duality |
| Core algorithm | block decomposition 기반 SSD algorithm |
| Output architecture | Mamba-2 block + TP/SP-friendly systems design |
| Difference from prior work | 단순한 새 recurrence나 attention approximation이 아니라, **이론, 알고리즘, 시스템을 한 번에 연결**한다는 점 |

## 3-2. Module breakdown
### 1) SSM as semiseparable matrix
- 이 논문의 뼈대는 Section 3이다. 저자들은 SSM sequence transformation이 사실상 **semiseparable matrix multiplication**과 같다는 점을 보인다.
- 직관적으로 보면, SSM의 state size `N`은 sequence transformation matrix의 특정 하삼각 submatrix rank를 제한하는 역할을 한다.
- 즉, recurrence는 "시간축을 따라 state를 넘기는 계산"이면서 동시에, 다른 관점에서는 **특정 저차 구조를 가진 sequence mixing matrix**다.
- 이 관점이 중요한 이유는, scan / state passing / block decomposition 같은 여러 알고리즘을 모두 **structured matrix multiplication algorithm**으로 해석할 수 있게 만들기 때문이다.
- 내가 보기엔 이 paper의 가장 큰 공헌은 사실 Mamba-2보다 이 부분이다. 이후 섹션들이 전부 여기서 파생되기 때문이다.

### 2) Structured Masked Attention (SMA)
- Section 4에서는 linear attention을 텐서 contraction 관점으로 다시 유도하고, 이를 **Structured Masked Attention**으로 일반화한다.
- 보통 linear attention은 causal mask 아래에서 cumsum recurrence로 설명되는데, 여기서는 causal mask를 더 일반적인 structured mask로 바꾼다.
- 이때 가장 중요한 special case가 **1-semiseparable structured masked attention (1-SS SMA)** 다.
- 1-SS SMA는 linear attention의 자연스러운 일반화이면서, 동시에 diagonal SSM의 special case로도 볼 수 있다.
- 더 강한 결과는 Theorem 5.2다. 저자들은 **빠른 recurrent form을 갖는 efficient autoregressive attention은 결국 semiseparable structured attention이어야 한다**고 보인다. 이건 "linear attention류가 왜 recurrence로 떨어지는가"를 꽤 강하게 설명하는 문장이다.

### 3) Structured State Space Duality (SSD)
- Section 5의 SSD는 이 논문의 제목과 직결되는 핵심이다.
- 요지는 SSM과 attention이 전부 같다는 게 아니라, **둘이 만나는 큰 교집합이 존재한다**는 것이다.
- 더 정확히 말하면, scalar-identity 구조를 가진 structured SSM과 1-semiseparable mask를 가진 structured masked attention이 서로 dual이 된다.
- 그래서 같은 모델을 **linear-time recurrence**로 볼 수도 있고, **attention-like quadratic form**으로 볼 수도 있다.
- 이 부분을 과장해서 "모든 Transformer가 SSM이다"라고 읽으면 안 된다. 오히려 안전한 독해는, **일부 중요한 efficient sequence model family는 같은 구조를 공유한다**는 쪽이다.

### 4) SSD algorithm
- Section 6은 이 paper가 진짜 실전형이 되는 지점이다.
- 저자들은 semiseparable matrix를 chunk 단위 block으로 나누고, 이를 **대각 블록(intra-chunk)** 과 **비대각 블록(inter-chunk)** 으로 분해한다.
- 대각 블록은 작은 subproblem이므로 quadratic form으로 계산해도 부담이 적고, 비대각 블록은 low-rank factorization을 이용해 chunk boundary state를 전달한다.
- 구현 관점에서 보면 크게 세 단계다.
  - 각 chunk 내부의 출력을 계산한다.
  - 각 chunk의 최종 state를 계산한다.
  - chunk 간 recurrence를 돌려 올바른 state를 전달한 뒤, 이를 다시 각 chunk output에 반영한다.
- 중요한 건 이 계산의 대부분이 **matrix multiplication**으로 바뀐다는 점이다. 그래서 scan 기반 selective SSM보다 GPU tensor core를 더 잘 활용할 수 있다.
- 이 논문이 보여주는 핵심 메시지는 명확하다. **선형 복잡도 자체보다, 그 선형 복잡도를 어떤 primitive로 구현하느냐가 더 중요하다.**

### 5) Mamba-2 block design
- Section 7의 Mamba-2는 SSD를 실제 네트워크 블록 설계로 연결한다.
- 가장 눈에 띄는 변화는 **sequential projection 제거**다. Mamba-1에서는 SSM input을 만든 뒤 그로부터 추가 파라미터를 순차적으로 생성했는데, Mamba-2는 필요한 data-dependent projection을 블록 앞쪽에서 병렬로 만든다.
- 이 변경은 단순한 코드 단순화가 아니다. parameter 수를 조금 줄이고, 더 중요한 쪽으로는 **tensor parallelism에 잘 맞는 블록 구조**를 만든다.
- 여기에 NormFormer 스타일의 추가 normalization을 넣어 안정성을 높인다.
- 또 head structure를 attention vocabulary로 다시 설명한다. Mamba의 selective SSM은 저자 표현으로 **MIS / MVA (multi-input SSM / multi-value attention)** 패턴에 가깝고, Mamba-2도 이 관점을 유지한다.
- 이 부분이 흥미로운 이유는, SSM 설계를 "state dimension, A/B/C parameterization"로만 보지 않고 **head sharing pattern**이라는 Transformer 친화적 언어로 다시 읽게 해주기 때문이다.

### 6) Systems optimization: TP / sequence parallel / variable length
- Section 8은 개인적으로 봤을 때, 의외로 중요했다.
- 많은 SSM 논문이 이론과 layer 설계까지만 이야기하고 끝나는데, 이 paper는 **대규모 학습에 필요한 병렬화 전략**까지 논한다.
- tensor parallelism에서는 input/output projection을 shard하고, 각 SSM head를 한 device 안에 놓는 방식으로 Transformer류와 비슷한 수준의 통신 구조를 맞춘다.
- sequence / context parallelism에서는 chunk 단위로 입력을 나누고 chunk boundary state만 넘기면 된다.
- variable-length batch 처리도 attention처럼 padding을 많이 남기지 않고, 여러 시퀀스를 한 긴 시퀀스로 이어 붙인 뒤 sequence boundary에서 상태 전달을 끊는 방식으로 처리할 수 있다.
- 즉, 이 논문은 "새 operator"를 제안한 게 아니라, **SSM을 foundation model training stack 안으로 끌고 들어오는 방법**까지 같이 제안한다.

# 4. Training / Data / Recipe

## 4-1. Data
- 이 논문의 데이터 측면은 Tulu 3 같은 recipe paper와 다르게 비교적 단순하다. 핵심은 새로운 데이터 큐레이션이 아니라 **공정한 pretraining 비교**다.
- scaling law 실험은 전부 **The Pile** 위에서 진행된다.
- scaling law 섹션에서는 **GPT-2 tokenizer**를 사용하고, fully trained downstream evaluation에서는 **GPT-NeoX tokenizer**를 사용한다.
- synthetic recall 실험으로는 MQAR(Multi-Query Associative Recall)을 사용한다. 저자들은 기존보다 더 어려운 버전을 만들기 위해, filler token을 random token으로 바꾸고 더 긴 시퀀스와 더 많은 key-value pair를 사용한다.
- 즉, 데이터 자체가 논문의 메인이 아니라, **같은 데이터 위에서 어떤 구조와 계산법이 더 잘 스케일하느냐**가 메인 질문이다.

## 4-2. Training strategy
- scaling law 실험은 대략 **125M, 350M, 760M, 1.3B** 규모에서 진행되며, 토큰 수도 각각 **2.5B, 7B, 15B, 26B**로 Chinchilla-style scaling에 맞춘다.
- downstream zero-shot evaluation을 위해서는 **300B tokens on the Pile**로 fully trained Mamba-2 모델을 만든다. 이때 핵심 비교 포인트는 1.3B와 2.7B scale이다.
- optimizer는 AdamW를 사용하고, gradient clip 1.0, weight decay 0.1, dropout 없음, linear warmup + cosine decay를 사용한다.
- training recipe는 GPT-3 기본형을 그대로 쓰지 않고, PaLM / LLaMA 계열의 "improved recipe"를 반영한다. 대표적으로 **RMSNorm, no linear bias, 더 큰 peak learning rate** 같은 요소가 들어간다.
- 즉, 논문의 비교는 "Mamba-2만 특별히 좋은 recipe를 썼다"가 아니라, **SSM과 Transformer 계열이 모두 경쟁력 있는 modern recipe 위에서 비교된 것**에 가깝다.

## 4-3. Engineering notes
- 이 논문의 recipe 섹션에서 진짜 중요한 건 loss보다 **계산 primitive**다.
- SSD는 작은 문제에서는 quadratic dual form을 활용하고, 긴 문맥에서는 chunkwise state passing을 활용한다. 즉 **계산 순서를 바꿔 하드웨어에 맞춘다**는 점이 핵심이다.
- MQAR를 harder setting으로 만든 것도 의미가 있다. 짧고 쉬운 synthetic task로는 finite-state memory 한계가 잘 드러나지 않기 때문이다.
- 또 variable-length sequence를 padding-heavy attention처럼 다루지 않고, sequence boundary에서 recurrence를 끊는 식으로 처리할 수 있다는 점은 실제 finetuning / inference 효율에 중요하다.
- 내가 보기엔 이 논문은 optimizer보다 **"어떤 연산으로 바꾸면 GPU가 좋아하는 계산이 되는가"**를 더 잘 보여주는 paper다.

# 5. Evaluation

## 5-1. Main results
- scaling law 결과에서 Mamba-2는 **Mamba와 strong Transformer++ baseline을 대체로 match or exceed**하고, perplexity / theoretical FLOPs / wall-clock time 관점에서 Pareto dominant하다고 저자들은 정리한다.
- zero-shot evaluation에서는 paper 요약대로 **Mamba-2가 전반적으로 Mamba를 앞서거나 비슷하게 유지하면서, 대체로 2배 정도 큰 Pythia와 맞먹는 그림**이 나온다.
- 예를 들어 780M 규모에서는 average score가 **53.5**로 Mamba 790M의 **53.0**보다 높고, Pythia-1B의 **49.0**보다도 높다.
- 1.3B 규모에서는 Mamba-2-1.3B average가 **56.4**로 Mamba-1.4B의 **56.4**와 거의 동률이고, Pythia-1.4B의 **51.7**보다 높다.
- 2.7B 규모에서는 Mamba-2-2.7B average가 **60.2**로 Mamba-2.8B의 **59.9**, Pythia-2.8B의 **55.7**, Pythia-6.9B의 **58.3**보다 높다.
- speed benchmark에서는 SSD algorithm이 optimized Mamba fused scan보다 **2-8× 빠르고**, FlashAttention-2와 비교했을 때도 **2K sequence length 이후부터 더 빠르며 16K에서 6× 빠르다**고 보고한다.
- synthetic MQAR에서는 Mamba-2가 Mamba-1보다 훨씬 낫고, state expansion `N`을 키울수록 성능이 꾸준히 좋아진다. 이는 "state size를 키우는 게 실제 retrieval-like task에서 의미가 있다"는 점을 보여준다.

## 5-2. What really matters in the experiments
- 내가 가장 중요하게 본 실험은 Table 2 / Table 3의 **hybrid 결과**다.
- 저자들은 pure Mamba-2와 pure Transformer++가 비슷한 수준임을 보이면서, **attention layer를 전체의 약 10% 정도 섞으면 더 좋아진다**는 결과를 제시한다.
- 350M, 48-layer 설정에서는 attention block이 6개일 때 perplexity가 가장 좋고, 2.7B scale에서도 **Mamba-2-Attention**이 pure Mamba-2와 Transformer++보다 평균 점수가 더 높다.
- 이건 굉장히 중요한 메시지다. 이 논문은 "SSM이 attention을 완전히 대체했다"를 보여주지 않는다. 오히려 **SSM은 일반 sequence transformation을 담당하고, attention은 retrieval을 담당한다**는 역할 분담을 보여준다.
- 또 Table 5의 head structure ablation도 의미가 크다. parameter-matched 조건에서도 **MIS / MVA 패턴이 MQA / MKA 패턴보다 뚜렷하게 낫다.** 즉, head sharing 방식은 구현 디테일이 아니라 실제 품질에 영향을 주는 구조적 변수다.
- 반대로 kernel approximation 계열 ablation(Table 6, 7)은 생각보다 큰 개선을 못 준다. 이것도 재밌다. 이 논문이 말하는 핵심은 "softmax 근사"보다 **structured mask + state passing + hardware path** 쪽에 더 있다.
- 마지막으로 speed 실험은 short sequence에서 무조건 Mamba-2가 이긴다고 말하지 않는다. 저자들도 짧은 시퀀스에서는 Transformer의 MLP-heavy block이 더 hardware-efficient할 수 있다고 인정한다. 이 점이 오히려 신뢰를 준다.

# 6. Limitations

1. **제목의 범위를 과하게 일반화하면 안 된다.** 논문 본문도 주석에서 이 연결이 attention의 특정 family에 대한 것임을 분명히 한다. 따라서 "모든 Transformer = 모든 SSM"이라고 읽는 건 과장이다.
2. **SSD는 expressivity와 hardware efficiency 사이의 trade-off를 택한다.** 저자들도 discussion에서 일반 diagonal SSM이 이론적으로는 비슷한 효율을 가질 수 있지만, attention-like quadratic form을 잃어 하드웨어 친화성이 떨어진다고 설명한다.
3. **best empirical story는 pure SSM이 아니라 hybrid다.** attention을 약 10% 섞는 게 더 낫다는 결과는, retrieval/ICL 관점에서 pure recurrent memory만으로는 아직 부족하다는 뜻이기도 하다.
4. **평가 범위가 제한적이다.** Pile 기반 pretraining, LM harness zero-shot, MQAR 같은 실험은 매우 유용하지만, instruction tuning, tool use, multilingual production setting, 실제 서비스 long-context workload를 직접 보여주는 논문은 아니다.
5. **속도 수치는 구현과 하드웨어에 민감하다.** Figure 10의 crossover는 A100 환경과 저자 구현 기준이므로, 다른 kernel stack이나 shorter sequence regime에서는 체감이 달라질 수 있다.

# 7. My Take

## 7-1. Why this matters for my work
- efficient attention / long-context / hybrid architecture를 계속 읽고 있다면, 이 논문은 거의 **교과서 같은 개념 지도**다.
- Kimi Linear, Gated DeltaNet 계열, RetNet류를 읽을 때 자꾸 등장하는 개념들-state size, recurrent memory, sparse/full retrieval, chunkwise kernel-을 하나의 언어로 묶어 준다.
- 특히 "attention을 줄이면 된다"가 아니라, **어떤 mixer가 linear form과 quadratic form을 둘 다 갖는가**를 물어야 한다는 관점은 이후 논문 해석에 바로 도움이 된다.
- 내 기준에서 이 paper의 가장 큰 가치는 Mamba-2 benchmark보다, **sequence mixer를 recurrence / attention / structured matrix 세 축으로 동시에 읽게 해 준다**는 데 있다.

## 7-2. Reuse potential
- 아키텍처 리뷰 관점에서는, 앞으로 어떤 efficient mixer를 보더라도 **state passing path, retrieval path, head sharing pattern, hardware primitive** 네 축으로 분해해서 볼 수 있게 해 준다.
- 실제 모델 설계 관점에서는 "대부분의 layer는 efficient mixer, 일부 layer는 retrieval용 attention"이라는 하이브리드 recipe가 매우 재사용 가능하다.
- 시스템 관점에서는 tensor parallel / sequence parallel / variable-length batching까지 같이 생각해야 한다는 점이 실무적으로 중요하다. asymptotic complexity만 보고는 실제 serving cost를 예측하기 어렵기 때문이다.
- 또 state expansion을 늘릴 때 품질이 왜 개선될 수 있는지, 그리고 그 비용을 어떤 알고리즘으로 감당할 수 있는지까지 같이 보여준다는 점도 좋다.

## 7-3. Follow-up papers
- *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- *Kimi Linear: An Expressive, Efficient Attention Architecture*

# 8. Summary

- 이 논문의 핵심은 "Mamba-2가 빠르다"보다 **attention과 SSM을 같은 structured computation으로 다시 본다**는 데 있다.
- SSM을 semiseparable matrix로 보는 순간, recurrence 알고리즘들이 structured matrix multiplication으로 재해석된다.
- Structured Masked Attention과 SSD는 linear form과 quadratic form이 어떻게 같은 모델의 두 계산 관점이 될 수 있는지 보여준다.
- SSD algorithm은 이 이론을 실제 GPU-friendly matmul path로 옮기며, Mamba보다 더 빠른 구현을 가능하게 한다.
- 하지만 empirical message는 pure SSM의 완전한 승리가 아니라, **SSM과 attention이 보완적이며 hybrid가 종종 더 낫다**는 쪽에 가깝다.
