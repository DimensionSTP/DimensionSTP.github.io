---
layout: single
title: "Memory Caching: RNNs with Growing Memory Review"
categories: Study-concept
tag: [LLM, RNN, LongContext, EfficientAttention, SequenceModeling]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2602.24281)

Memory Caching은 "RNN이 fixed-size hidden state 하나로 과거 전체를 압축해야 한다"는 전제를 다시 여는 논문이다. 요즘 long-context architecture 논의는 보통 attention의 KV cache를 얼마나 줄일 것인가, 혹은 linear attention과 SSM이 Transformer를 얼마나 대체할 수 있는가로 흘러간다. 이 논문은 그 사이를 조금 다르게 본다. RNN이 약한 이유가 recurrence 자체라기보다, 과거를 저장하는 방식이 너무 극단적으로 압축되어 있기 때문이라고 보는 것이다.

이 논문이 흥미로운 이유는 아주 단순한 질문에서 출발한다. Transformer는 모든 past token을 직접 남겨서 memory capacity가 sequence length와 함께 늘어난다. 반면 RNN은 hidden state 크기가 고정되어 있다. 그렇다면 RNN도 매 token을 모두 남기지는 않더라도, segment별 memory checkpoint를 남기면 어떨까. 이게 Memory Caching, 줄여서 MC의 출발점이다.

> 한 줄 요약: Memory Caching은 recurrent model의 hidden state checkpoint를 segment 단위로 cache하고, 현재 token이 online memory와 과거 cached memory를 함께 읽게 만들어 RNN의 effective memory capacity를 sequence length에 따라 늘리는 방법이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- long-context model의 핵심 병목을 compute가 아니라 **memory capacity와 retrieval resolution의 trade-off**로 다시 설명한다.
- MC가 특정 RNN 하나에 붙는 trick이 아니라, linear attention, SWLA, DLA, Titans 같은 memory-bounded module에 붙을 수 있는 일반 framework로 제시된다.
- Transformer와 RNN을 binary choice로 보지 않고, segment size와 cache selection으로 $O(L)$과 $O(L^2)$ 사이의 operating point를 만든다.
- 실험이 language modeling, commonsense reasoning, NIAH, in-context retrieval, LongBench, MQAR까지 이어져서 단순 toy retrieval paper보다 읽을 거리가 많다.

**이 논문의 핵심은 RNN을 Transformer처럼 만들자는 것이 아니라, RNN이 어느 정도의 과거 해상도를 가져야 하는지 조절 가능한 knob을 준다는 데 있다.**

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 recurrent architecture의 **fixed memory bottleneck**이다.

Transformer의 attention은 과거 token의 key-value를 거의 그대로 남긴다. 그래서 retrieval task에서는 매우 강하다. 대신 sequence length가 길어질수록 attention score 계산과 KV cache 비용이 커진다.

반대로 RNN, linear attention, modern recurrent model은 과거를 고정 크기의 state로 압축한다. 그래서 계산량은 줄지만, long-context recall에서는 과거 정보가 한 state 안에서 계속 덮이고 섞인다. 특히 passkey, UUID, associative recall처럼 특정 과거 조각을 다시 꺼내야 하는 문제에서는 이 fixed state가 병목이 된다.

이 논문의 문제 설정은 아래처럼 요약할 수 있다.

| Architecture | Memory shape | Strength | Bottleneck |
| --- | --- | --- | --- |
| Transformer | token-level growing memory | high-resolution retrieval | quadratic attention and KV cache |
| RNN / linear attention | fixed-size online memory | efficient recurrent update | long-range recall degradation |
| Memory Caching | segment-level growing memory | controllable retrieval resolution | cache aggregation and routing cost |

즉 질문은 "RNN이 attention을 완전히 대체할 수 있는가"가 아니다.

질문은 "RNN의 compressed memory를 어느 간격으로 남기면, attention의 retrieval 장점을 일부 가져오면서도 비용을 조절할 수 있는가"에 가깝다.

## 1-2. Why previous approaches are insufficient

기존 efficient sequence model은 대략 세 방향으로 나뉜다.

1. Linear attention 계열
   - softmax attention을 kernelized recurrence로 바꿔 $O(L)$ update를 만든다.
   - 하지만 memory state는 여전히 고정 크기라 long-context recall에서 정보가 사라지기 쉽다.

2. Modern RNN / SSM 계열
   - selective recurrence, gating, state update를 개선해 short-context와 throughput을 끌어올린다.
   - 하지만 recall-intensive task에서는 token-level memory를 가진 Transformer와 격차가 남는다.

3. Hybrid attention 계열
   - 일부 global attention layer를 섞어 long-range retrieval을 보완한다.
   - 효과적이지만, attention block을 다시 넣는 방식이라 RNN 자체의 memory capacity 문제를 직접 다루지는 않는다.

Memory Caching은 이 지점에서 다르게 접근한다. 과거 token을 모두 저장하는 대신, recurrent memory가 sequence를 처리하는 중간 checkpoint를 저장한다. 즉 past token 자체가 아니라, past segment를 압축한 memory module의 state를 남긴다.

이 방식은 다음 중간 지대를 만든다.

$$
RNN: O(L) \quad < \quad MC: O(L^2 / S) \quad < \quad Transformer: O(L^2)
$$

여기서 $S$는 segment size다. $S$가 커질수록 cache 수는 줄고, $S$가 작아질수록 retrieval resolution은 높아진다.

# 2. Core Idea

## 2-1. Main contribution

Memory Caching의 핵심 기여는 크게 4가지다.

1. Recurrent memory checkpointing
   - sequence를 segment로 나누고, 각 segment가 끝난 시점의 hidden state 또는 memory state를 cache한다.
   - 현재 token은 current online memory뿐 아니라 past cached memories도 같이 읽는다.

2. Multiple aggregation strategies
   - Residual Memory, Gated Residual Memory, Memory Soup, Sparse Selective Caching을 제안한다.
   - 단순 sum부터 input-dependent gating, parameter souping, MoE-style sparse routing까지 이어진다.

3. Linear and deep memory module 연결
   - linear attention에서는 일부 variant가 수학적으로 collapse할 수 있지만, deep memory module에서는 새로운 retrieval function을 만든다.
   - 특히 DLA와 Titans처럼 memory 자체가 non-linear module인 경우 Memory Soup의 의미가 커진다.

4. Proof-of-concept experiments
   - SWLA, DLA, Titans에 MC를 붙여 language modeling, NIAH, retrieval, LongBench, MQAR에서 확인한다.
   - 결과적으로 base recurrent model 대비 consistent gain을 보이고, Transformer와의 recall gap도 줄인다.

**중요한 포인트는 MC가 new backbone이라기보다 memory access pattern을 바꾸는 framework라는 점이다.**

## 2-2. Design intuition

MC의 설계 직관은 간단하다.

RNN의 online memory는 sequence를 지나며 계속 업데이트된다. 이 state는 현재까지의 compressed summary다. 문제는 이 summary가 너무 많은 과거를 한 번에 담아야 한다는 점이다. 오래된 segment에서 중요한 정보가 있어도, 이후 token들이 계속 들어오면 그 정보는 쉽게 희석된다.

Memory Caching은 이 압축 과정을 중간중간 저장한다.

$$
m_j = Update(m_{j-1}, x_{(j-1)S+1:jS})
$$

여기서 $m_j$는 $j$번째 segment까지 처리한 memory checkpoint다. current token $x_t$는 online memory $m_{online}$만 읽는 대신, cached memories $m_1, m_2, ..., m_{j-1}$도 함께 읽는다.

개념적으로는 아래처럼 볼 수 있다.

$$
y_t = Aggregate(Read(q_t, m_{online}), Read(q_t, m_1), ..., Read(q_t, m_{j-1}))
$$

이 식의 핵심은 cached memory가 raw token이 아니라 compressed memory state라는 점이다. 따라서 Transformer처럼 모든 token을 직접 보는 것은 아니지만, standard RNN처럼 하나의 state만 보는 것도 아니다.

이 논문에서 가장 중요한 knob은 segment size다. segment size가 1이면 거의 token-level cache에 가까워지고, segment size가 매우 크면 standard RNN에 가까워진다. 결국 MC는 memory resolution을 조절하는 방법이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | RNN의 effective memory capacity를 sequence length와 함께 늘리는 것 |
| Base module | linear attention, SWLA, DLA, Titans 같은 recurrent memory module |
| Key operation | segment별 memory checkpoint cache |
| Retrieval | online memory와 cached memories를 함께 읽음 |
| Main variants | Residual Memory, GRM, Memory Soup, SSC |
| Main trade-off | retrieval accuracy vs cache aggregation cost |

기존 RNN의 출력은 현재 memory state 하나에 의존한다.

$$
y_t = Read(q_t, m_t)
$$

MC에서는 segment별 cache를 추가한다.

$$
C_t = \{m_1, m_2, ..., m_{j-1}, m_{online}\}
$$

그리고 출력은 이 memory set을 aggregation해서 만든다.

$$
y_t = A(q_t, C_t)
$$

여기서 $A$가 어떤 aggregation function인지에 따라 여러 variant가 나온다.

## 3-2. Module breakdown

### 1) Residual Memory

Residual Memory는 가장 단순한 variant다. cached memory들의 readout을 더해서 current output에 residual처럼 더한다.

장점은 구현이 단순하다는 점이다. 단점은 모든 cache를 동일하게 취급하기 때문에, 현재 query와 관련 없는 past segment도 같은 방식으로 들어온다는 점이다.

논문은 linear memory에서는 이런 단순 residual form이 수학적으로 fixed-size memory와 collapse할 수 있다고 설명한다. 하지만 실험상 단순 residual도 recurrent model의 long-past access를 높이는 retention operator처럼 작동할 수 있다.

### 2) Gated Residual Memory

GRM은 Residual Memory에 input-dependent gate를 붙인 형태다. 현재 query와 past segment context의 관계를 보고 segment별 contribution을 조절한다.

$$
y_t = \sum_{i \in C_t} g_i(q_t, c_i) Read(q_t, m_i)
$$

여기서 $g_i$는 현재 token과 $i$번째 segment context의 relevance를 반영한다. 논문은 단순히 position만 보고 gate를 주면 context relevance를 놓칠 수 있으므로, token representation과 segment representation 사이의 similarity를 쓰는 방향을 제안한다.

**GRM은 이 논문에서 가장 실용적인 기본형에 가깝다.**

이유는 명확하다. RNN에 cache를 붙이는 것만으로는 과거가 늘어나지만, 어떤 과거를 얼마나 읽을지 모르면 noise도 같이 늘어난다. GRM은 이 문제를 gating으로 제어한다.

### 3) Memory Soup

Memory Soup은 cached memory state의 output을 ensemble하는 대신, memory parameter 자체를 input-dependent하게 섞는 방식이다. 이름처럼 model soup에서 영감을 받은 variant다.

linear memory에서는 Memory Soup과 GRM이 사실상 같은 형태로 collapse할 수 있다. 하지만 memory module이 MLP처럼 non-linear이면 이야기가 달라진다. parameter를 먼저 섞은 뒤 query를 통과시키는 것과, 각 memory output을 따로 만든 뒤 합치는 것은 같은 연산이 아니다.

따라서 Memory Soup은 deep memory module에서 더 의미가 있다. current token마다 자기에게 맞는 retrieval memory를 즉석에서 구성하는 식으로 해석할 수 있다.

### 4) Sparse Selective Caching

SSC는 ultra-long sequence에서 모든 cache를 다 읽는 비용을 줄이기 위한 variant다. 각 token에 대해 past segment별 relevance score를 계산하고, top-k cache만 선택해서 retrieval한다.

이 구조는 MoE-style router와 비슷하다. 차이는 expert가 layer parameter가 아니라 cached memory state라는 점이다.

SSC의 장점은 두 가지다.

- irrelevant cache를 읽지 않아 memory and compute overhead를 줄일 수 있다.
- segment context 기반으로 필요한 past memory만 활성화하므로 long sequence에서 더 효율적인 operating point를 만들 수 있다.

논문은 efficiency 관점에서 SSC가 특히 중요하다고 본다. 모든 cache를 읽는 GRM은 성능은 강하지만 context가 길어질수록 overhead가 커진다. SSC는 그 비용을 줄이기 위한 sparse retrieval path다.

# 4. Training / Data / Recipe

## 4-1. Data

실험은 크게 두 scale로 진행된다.

| Scale | Training tokens | Main setting |
| --- | --- | --- |
| 760M params | 30B tokens | FineWeb 기반 language modeling and reasoning |
| 1.3B params | 100B tokens | larger scale language modeling and reasoning |

논문은 language modeling과 commonsense reasoning에서 4K context length와 segment length 256을 default로 사용한다. Needle-in-a-haystack, in-context retrieval, LongBench 같은 long-context task에서는 16K context length로 학습해 recurrent model의 long-memory 차이를 더 잘 드러내도록 한다.

사용된 task는 아래처럼 넓다.

- Wikitext, LAMBADA
- PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, SIQA, BoolQ
- Needle-in-a-haystack
- SWDE, SQuAD, FDA, TriviaQA, DROP, NQ
- LongBench
- MQAR

## 4-2. Training strategy

MC 자체는 pretraining recipe라기보다 architecture modification이다. 즉 base recurrent module의 update rule은 유지하고, retrieval 시점에 cached memory를 추가로 읽게 만든다.

논문에서 중요한 설정은 segmenting이다.

- constant-size segmentation
  - segment size를 고정한다.
  - cache resolution이 일정하고, retrieval 성능이 좋다.

- logarithmic segmentation
  - Fenwick tree style의 log-linear 구조와 비슷하게 적은 수의 segment를 만든다.
  - 더 효율적이지만, 오래된 구간의 compression이 너무 거칠어질 수 있다.

실험에서는 constant-size segmentation 기반 MC가 Log-Linear++보다 더 좋은 결과를 자주 보인다. 논문은 이를 larger effective memory size와 더 균일한 compression load 때문으로 해석한다.

## 4-3. Engineering notes

MC를 실제로 붙일 때 중요한 engineering point는 세 가지다.

1. Cache residency
   - 모든 cached memory를 accelerator에 계속 올려둘 필요는 없다.
   - SSC에서는 relevance score와 top-k selection을 먼저 계산한 뒤 필요한 memory만 로드하는 구성이 가능하다.

2. Segment size
   - 작은 segment는 retrieval resolution을 올리지만 cache 수와 read cost를 키운다.
   - 큰 segment는 비용을 줄이지만 segment 내부 정보가 더 강하게 압축된다.

3. Post-training usage
   - 논문은 MC를 pretraining 후 inference에 붙이는 간단한 방식도 논의한다.
   - 각 segment 후 memory state를 cache하고, learnable weight 없이 moving average로 과거 memory를 섞는 방식만으로도 length extrapolation에 도움이 될 수 있다고 보고한다.

**실무적으로는 GRM으로 성능 상한을 보고, SSC로 latency와 memory overhead를 맞추는 식의 탐색이 자연스러워 보인다.**

# 5. Evaluation

## 5-1. Main results

### Language modeling and commonsense reasoning

Table 1에서 MC는 SWLA, DLA, Titans에 붙었을 때 대부분 평균 점수를 올린다.

| Setting | Base | MC variant | Avg |
| --- | --- | --- | --- |
| 760M / SWLA | SWLA | base | 50.05 |
| 760M / SWLA | SWLA + GRM | MC | 51.26 |
| 760M / DLA | DLA | base | 50.48 |
| 760M / DLA | DLA + GRM | MC | 51.41 |
| 760M / Titans | Titans LMM | base | 51.56 |
| 760M / Titans | Titans + GRM | MC | 52.55 |
| 1.3B / DLA | DLA | base | 53.72 |
| 1.3B / DLA | DLA + GRM | MC | 55.96 |
| 1.3B / Titans | Titans LMM | base | 56.82 |
| 1.3B / Titans | Titans + GRM | MC | 58.33 |

내가 중요하게 보는 부분은 단순히 평균이 올랐다는 점보다, MC gain이 SWLA, DLA, Titans 모두에서 반복된다는 점이다. 즉 특정 backbone에만 맞는 trick처럼 보이지 않는다.

### Needle-in-a-haystack

NIAH에서는 fixed-memory bottleneck이 더 잘 드러난다. 예를 들어 16K context에서 DLA의 S-NIAH-3 score는 4.0인데, DLA + GRM은 18.2까지 올라간다. Titans LMM도 S-NIAH-2 16K에서 75.4에서 88.2로, S-NIAH-3 16K에서 21.2에서 32.2로 오른다.

이 결과는 MC의 의도를 잘 보여준다. RNN의 update rule을 더 복잡하게 만들지 않아도, compressed memory checkpoint를 남기고 다시 읽는 것만으로 long-range recall이 개선된다.

### In-context retrieval

Table 3에서는 Transformer가 여전히 가장 강하다. 평균 기준 Transformer는 41.00, Titans MAL은 40.46이다. 하지만 MC를 붙인 recurrent model이 base recurrent model보다 큰 폭으로 올라온다.

| Model | Avg |
| --- | --- |
| Transformer | 41.00 |
| Titans MAL | 40.46 |
| DLA | 30.51 |
| DLA + GRM | 38.03 |
| Titans LMM | 31.75 |
| Titans + GRM | 40.50 |

이 표는 이 논문의 메시지를 가장 잘 압축한다. Transformer를 완전히 넘었다기보다는, recurrent model의 recall gap을 크게 줄였다는 쪽이 더 정확하다.

### LongBench and MQAR

LongBench에서는 DLA와 Titans 모두 MC-enhanced variants가 base RNN보다 좋아진다고 보고한다. MQAR에서도 MC-enhanced variants가 base RNN과 state-of-the-art recurrent model 대비 좋은 성능을 보인다고 설명한다.

다만 HTML 변환본에서는 일부 figure의 수치가 직접 텍스트로 충분히 보이지 않는다. 정확한 MQAR curve와 throughput curve는 PDF figure를 다시 확인하는 편이 좋다.

## 5-2. What really matters in the experiments

실험에서 중요한 해석은 세 가지다.

1. GRM이 강하다
   - 여러 표에서 GRM이 가장 안정적인 성능을 보인다.
   - 단순 cache보다 context-dependent gate가 중요하다는 뜻이다.

2. SSC는 efficiency용 operating point다
   - SSC가 항상 최고 성능은 아니지만, long sequence에서 overhead를 줄이는 방향으로 설계되어 있다.
   - 논문도 SSC가 long context에서 efficiency 측면의 장점을 가진다고 본다.

3. Transformer와의 비교는 신중해야 한다
   - in-context recall에서는 Transformer가 여전히 강하다.
   - MC의 핵심은 Transformer를 즉시 대체했다는 주장이 아니라, recurrent model의 memory capacity를 늘려 gap을 줄였다는 것이다.

Ablation도 이 해석을 지지한다. Titans GRM에서 retrieval accuracy는 40.5인데, context-dependent design을 제거하면 33.0, gating을 제거하면 32.4로 떨어진다. 즉 cache 자체보다 어떤 cache를 어떻게 읽는지가 중요하다.

# 6. Limitations

1. **Exact wall-clock and memory trade-off는 더 세밀한 검증이 필요하다.**
   - 논문은 throughput figure로 MC가 Transformer와 RNN 사이의 중간 지대를 만든다고 보인다.
   - 하지만 deployment 관점에서는 cache layout, top-k routing, accelerator memory movement가 실제 latency를 크게 좌우할 수 있다.

2. **Segment size가 중요한 hyperparameter다.**
   - segment size가 작으면 retrieval은 좋아지지만 cost가 증가한다.
   - segment size가 크면 cost는 줄지만 long-past token이 지나치게 압축될 수 있다.

3. **GRM과 SSC의 선택은 task-dependent일 가능성이 크다.**
   - GRM은 강하지만 모든 cache를 읽는 쪽에 가깝다.
   - SSC는 효율적이지만 routing 품질이 나쁘면 필요한 memory를 놓칠 수 있다.

4. **Transformer의 high-resolution recall을 완전히 대체한 결과는 아니다.**
   - in-context retrieval에서는 Transformer가 여전히 강한 비교군이다.
   - MC는 gap을 줄이는 방법으로 읽는 것이 맞다.

5. **논문은 proof-of-concept 성격이 강하다.**
   - 저자들도 단순한 설계를 유지해 MC idea 자체의 효과를 보이려 했다고 결론에서 설명한다.
   - 더 expressive한 pooling이나 routing은 future work로 남아 있다.

# 7. My Take

## 7-1. Why this matters for my work

Memory Caching은 long-context efficient architecture를 볼 때 꽤 좋은 분석 프레임을 준다.

보통 long-context를 말하면 attention sparsity, KV compression, recurrent replacement를 따로 본다. 그런데 MC 관점에서는 이 셋이 모두 같은 질문으로 묶인다.

"과거를 어떤 resolution으로 남길 것인가"

Transformer는 token resolution을 남긴다. RNN은 full-history를 하나의 state로 남긴다. MC는 segment-level compressed resolution을 남긴다. 이 해석을 쓰면 새로운 architecture를 볼 때, 단순히 complexity만이 아니라 retrieval resolution을 같이 비교할 수 있다.

## 7-2. Reuse potential

실무적으로 재사용 가능성이 있는 지점은 아래와 같다.

- long-context RNN이나 linear attention model의 inference extension
- KV cache가 너무 큰 환경에서 segment-level memory cache 대안 탐색
- retrieval-heavy benchmark에서 recurrent model의 failure mode 분석
- hybrid attention layer를 넣기 전에 cached memory retrieval을 먼저 실험하는 ablation
- agent trajectory나 streaming input처럼 segment boundary가 자연스럽게 존재하는 데이터에 적용

특히 agent나 long video understanding에서는 raw token 전체를 attention으로 남기기 어렵다. 이때 segment별 memory checkpoint를 만들고, 현재 query가 필요한 checkpoint만 읽는 구조는 꽤 자연스럽다.

## 7-3. Follow-up papers

- Titans: Learning to Memorize at Test Time
- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
- Zoology: Measuring and Improving Recall in Efficient Language Models
- Log-Linear Attention / Log-Linear++ 계열 논문
- Infini-attention and compressive memory 계열 long-context model

# 8. Summary

- Memory Caching은 RNN의 hidden state checkpoint를 segment별로 cache해 effective memory capacity를 늘린다.
- 핵심 variant는 Residual Memory, GRM, Memory Soup, SSC다.
- GRM은 context-dependent gate로 cached memory를 가중하고, SSC는 top-k cache만 선택해 overhead를 줄인다.
- SWLA, DLA, Titans에 붙였을 때 language modeling, NIAH, retrieval, LongBench에서 base recurrent model 대비 gain을 보인다.
- 이 논문은 Transformer 대체 주장보다, long-context architecture의 memory resolution knob을 제시했다는 점이 더 중요하다.
