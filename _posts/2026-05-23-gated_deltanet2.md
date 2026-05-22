---
layout: single
title: "Gated DeltaNet-2 Review"
categories: Study-concept
tag: [Linear-Attention, Mamba, Long-Context]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2605.22791)

[Code link](https://github.com/NVlabs/GatedDeltaNet-2)

**한 줄 요약:** Gated DeltaNet-2는 linear attention 계열의 recurrent state update에서 erase와 write를 하나의 scalar gate로 묶던 제약을 풀고, key-side erase gate와 value-side write gate를 channel-wise로 분리한 방법이다.

이 글에서는 이 논문을 "fixed-state memory editing" 문제로 읽는다. Softmax attention은 과거 token 전체에 직접 접근하지만, linear recurrent attention은 과거를 고정 크기의 state에 압축한다. 따라서 긴 context에서 진짜 문제는 단순히 많이 잊는 것이 아니라, 이미 압축된 memory를 어떻게 덜 망가뜨리면서 고치는가에 가깝다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- **Linear attention 계열의 병목을 memory edit 관점에서 정리한다.** Gated DeltaNet, KDA, Mamba-3 같은 흐름이 왜 비슷해 보이면서도 다른지 비교하기 좋다.
- **긴 context retrieval에서 scalar gate의 한계를 직접 겨냥한다.** 특히 multi-key needle retrieval처럼 state interference가 큰 task에서 개선을 보인다.
- **수식 변경이 kernel path까지 연결된다.** 단순 architecture idea가 아니라 chunkwise WY form, gate-aware backward, Triton fused kernel을 함께 고려한다.

Gated DeltaNet-2의 핵심은 linear attention을 attention의 저가형 대체재로 보는 것이 아니다. 오히려 recurrent memory를 작은 database처럼 보고, 어떤 key coordinate를 지울지와 어떤 value coordinate를 쓸지를 분리해 memory edit interface를 다시 설계하는 논문에 가깝다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 **긴 sequence를 효율적으로 처리하면서도 compressed recurrent state의 retrieval 품질을 유지하는 것**이다.

Softmax attention은 각 token이 이전 token 전체를 직접 볼 수 있기 때문에 long-range retrieval에는 강하다. 하지만 sequence length가 길어질수록 attention matrix와 KV cache 부담이 커진다. Linear recurrent attention은 이 문제를 다른 방향으로 푼다. 과거 token을 계속 저장하지 않고, 고정 크기 matrix state $S_t$에 key-value association을 누적한다.

이때 문제는 명확하다.

- state size는 고정되어 있다.
- context length는 계속 늘어난다.
- 많은 key-value association이 같은 state 안에 겹쳐 들어간다.
- 오래된 정보를 지우거나 새 정보를 쓰는 update가 잘못되면 retrieval interference가 생긴다.

따라서 linear attention의 핵심 문제는 단순한 compute reduction이 아니라 **compressed memory control**이다.

## 1-2. Why previous approaches are insufficient

기존 방법들의 한계는 update rule을 보면 더 잘 보인다.

- **Naive linear attention**은 state에 outer product를 더하는 방식이다. 오래된 association이 명시적으로 제거되지 않기 때문에, 나중에는 superposition noise가 커질 수 있다.
- **Mamba-2**는 data-dependent decay를 통해 state horizon을 조절한다. 하지만 decay는 broad forgetting에 가깝고, 특정 key association을 정밀하게 고치는 edit는 아니다.
- **DeltaNet**은 현재 key로 읽은 값을 subtract한 뒤 새 value를 write한다. 이 방식은 targeted overwrite를 가능하게 한다.
- **Gated DeltaNet**은 delta rule에 decay를 추가해 global forgetting과 targeted edit를 결합한다.
- **KDA**는 decay를 channel-wise로 만들지만, active edit 자체는 여전히 하나의 scalar gate에 묶여 있다.

Gated DeltaNet-2가 지적하는 병목은 마지막 항목이다. 기존 delta-rule 계열은 erase와 write를 하나의 scalar strength로 같이 조절한다. 하지만 erase는 key axis에서 old read를 얼마나 제거할지의 문제이고, write는 value axis에서 new content를 얼마나 commit할지의 문제다. 두 decision을 같은 scalar로 묶는 것은 모델링 제약이다.

결국 이 논문은 다음 질문을 던진다.

"고정 크기 recurrent state에서 지우는 정도와 쓰는 정도를 왜 같은 gate로 정해야 하는가?"

# 2. Core Idea

## 2-1. Main contribution

Gated DeltaNet-2의 main contribution은 세 가지로 정리할 수 있다.

1. **Gated Delta Rule-2**
   - scalar delta gate를 key-side erase gate $b_t$와 value-side write gate $w_t$로 분리한다.
   - 두 gate 모두 channel-wise vector다.

2. **Strict generalization**
   - erase gate와 write gate가 같은 scalar로 collapse되면 KDA가 된다.
   - channel-wise decay까지 scalar로 collapse되면 Gated DeltaNet이 된다.
   - 즉 기존 방법을 버리는 것이 아니라, 기존 방법을 tied subspace로 포함한다.

3. **Efficient training path 유지**
   - channel-wise decay를 erase factor 안으로 흡수해 chunkwise WY form을 유지한다.
   - gate-aware backward를 통해 parallel training 효율을 보존한다.

이 아이디어가 중요한 이유는 단순히 gate를 하나 더 붙였기 때문이 아니다. state matrix를 key dimension과 value dimension이 만나는 memory map으로 보면, erase와 write는 서로 다른 축의 operation이다. Gated DeltaNet-2는 이 차이를 architecture에 명시적으로 반영한다.

## 2-2. Design intuition

이 논문에서 가장 중요한 직관은 **memory update를 하나의 scalar confidence로 보지 않는 것**이다.

어떤 token이 들어왔을 때 모델은 세 가지 결정을 해야 한다.

- broad context를 얼마나 decay할 것인가.
- 현재 key와 관련된 old association을 얼마나 erase할 것인가.
- 새 value의 어떤 channel을 state에 write할 것인가.

기존 KDA는 첫 번째 decision인 decay를 channel-wise로 세밀하게 만들었다. 하지만 두 번째와 세 번째 decision은 여전히 하나의 scalar gate에 묶었다. Gated DeltaNet-2는 여기서 한 단계 더 간다. key-side erase는 $b_t$, value-side write는 $w_t$로 분리한다.

이 구조를 retrieval 관점에서 보면 의미가 더 선명하다. Multi-key retrieval에서는 여러 key가 비슷한 state region을 공유할 수 있다. 이때 새 정보를 쓴다는 이유만으로 old read direction 전체를 같은 강도로 지우면 필요한 association까지 손상될 수 있다. 반대로 old association을 약하게만 지우면 새 value가 제대로 반영되지 않을 수 있다.

Gated DeltaNet-2는 이 trade-off를 channel-wise로 풀려고 한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | fixed-size recurrent state에서 long-context memory interference를 줄이는 linear attention update rule |
| Core module | Gated Delta Rule-2 |
| Main gates | key-side erase gate $b_t$, value-side write gate $w_t$, channel-wise decay $D_t$ |
| Prior included | KDA and Gated DeltaNet as special cases |
| Training method | chunkwise WY algorithm with gate-aware backward |
| Model forms | recurrent-only model and hybrid model with SWA |
| Main scale | 1.3B parameters, 100B FineWeb-Edu tokens |

논문에서 사용하는 핵심 update는 다음처럼 이해할 수 있다.

$$
S_t = (I - k_t (b_t \odot k_t)^T) D_t S_{t-1} + k_t (w_t \odot v_t)^T
$$

여기서 $S_t$는 recurrent state, $k_t$는 key, $v_t$는 value, $D_t$는 channel-wise decay, $b_t$는 erase gate, $w_t$는 write gate다.

이 식의 왼쪽 항은 old state를 decay한 뒤, 현재 key에 대응되는 일부 key-side coordinate를 erase한다. 오른쪽 항은 new value 중 write gate가 선택한 channel만 state에 commit한다.

## 3-2. Module breakdown

### 1) Fixed-size recurrent state

Linear recurrent attention은 매 token마다 전체 past token을 다시 보지 않는다. 대신 $S_t$라는 고정 크기 state를 유지한다. 이 state는 key를 query로 읽으면 value를 반환하는 fast-weight memory처럼 동작한다.

이 구조의 장점은 sequence length에 대해 memory가 계속 커지지 않는다는 점이다. decoding에서도 softmax attention처럼 KV cache가 context length에 비례해 커지지 않는다.

하지만 단점도 같은 곳에서 나온다. 고정 크기 state 안에 너무 많은 association을 넣으면 interference가 생긴다. 따라서 update rule이 단순히 write만 잘하는 것이 아니라, stale association을 얼마나 잘 지우는지가 중요해진다.

### 2) Channel-wise decay

Gated DeltaNet-2는 KDA처럼 channel-wise decay를 유지한다. Decay는 broad forgetting 역할을 한다. 특정 token 하나를 지우는 것이 아니라, state 전체의 memory horizon을 조절한다.

논문은 log-decay branch를 두고, 실제 kernel에 들어가기 전 fp32에서 decay activation을 계산한다고 설명한다. 이 부분은 implementation detail이지만 중요하다. 누적 log-decay는 긴 sequence에서 precision issue가 생길 수 있기 때문이다.

### 3) Key-side erase gate

Erase gate $b_t$는 key dimension에 작용한다. 현재 key로 old state를 읽을 때, 어떤 key coordinate를 erase factor에 반영할지 조절한다.

이 점이 Gated DeltaNet-2의 가장 중요한 변화다. 기존 scalar gate는 old read를 지우는 정도와 new value를 쓰는 정도를 같이 움직였다. Gated DeltaNet-2는 old association을 제거하는 decision을 key-side channel별로 독립시킨다.

retrieval benchmark에서 erase gate가 큰 효과를 보인 이유도 여기에 있다. Multi-key task에서는 서로 다른 key association을 분리해야 한다. 이때 key-side coordinate를 선택적으로 보호하거나 수정하는 gate가 value-side write gate보다 더 직접적인 역할을 한다.

### 4) Value-side write gate

Write gate $w_t$는 value dimension에 작용한다. 새 value vector 전체를 무조건 state에 쓰지 않고, 어떤 value channel을 commit할지 선택한다.

이 구조는 new token의 정보를 모두 같은 강도로 state에 넣는 것을 피한다. 어떤 token은 local evidence만 강하고 long-term memory에 남길 필요가 적을 수 있다. 반대로 어떤 token은 value channel 일부만 long-range retrieval에 필요할 수 있다.

즉 write gate는 새 content의 persistence를 value axis에서 조절한다.

### 5) Chunkwise WY training

Gated DeltaNet-2가 practical한 이유는 update rule을 바꿨는데도 efficient training path를 유지한다는 점이다.

논문은 channel-wise decay를 rank-one erase factor 안으로 흡수해 decay-normalized recurrence로 바꾼다. 이렇게 하면 recurrence across token을 그대로 sequential하게 돌리지 않고, chunk 내부를 dense matrix product 형태로 처리할 수 있다.

핵심은 다음과 같다.

- chunk 안에서는 token interaction을 matrix product로 묶는다.
- chunk 사이에서만 recurrent state를 전달한다.
- fixed chunk size를 쓰면 sequence length에 대해 linear complexity를 유지한다.
- gate-aware backward를 통해 추가 gate의 gradient path를 처리한다.

이 부분은 수식보다 implementation이 더 중요하다. Linear attention 계열은 좋은 update rule을 만들더라도 kernel path가 없으면 실제 model training에서 쓰기 어렵다.

### 6) Recurrent-only and hybrid model

논문은 recurrent-only model과 hybrid model을 모두 학습한다.

- recurrent-only model은 Gated DeltaNet-2 token mixer와 MLP를 stack한다.
- hybrid model은 Gated DeltaNet-2, MLP, Sliding-Window Attention, MLP를 반복 cell로 사용한다.

Hybrid setting에서 SWA는 local exact interaction을 담당한다. Gated DeltaNet-2는 long history를 fixed-size state에 압축하고, SWA는 short shift, local comparison, local retrieval 같은 근거리 정보를 더 정확히 처리한다.

이 설계는 현실적인 hybrid operating point에 가깝다. Pure recurrent model의 효율성을 보면서도, local interaction까지 recurrent state 하나로 해결하려 하지 않는다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 모든 비교 모델을 같은 조건에서 학습한다.

- model scale: 1.3B parameters
- training tokens: 100B tokens
- corpus: FineWeb-Edu
- training sequence length: 4K
- hybrid SWA window: 2K

비교 대상은 Mamba-2, Gated DeltaNet, KDA, Mamba-3 SISO, Mamba-3 MIMO, Transformer, 그리고 각 recurrent mixer의 hybrid variant다.

중요한 점은 recurrent state size를 맞춰 비교했다는 것이다. 따라서 결과 차이를 단순히 state capacity가 더 커졌기 때문이라고 보기 어렵다. 논문은 update rule 자체가 더 강해졌다는 해석을 제시한다.

## 4-2. Training strategy

공식 README 기준 default recipe는 다음과 같다.

| Item | Value |
| --- | --- |
| Optimizer | AdamW |
| Peak learning rate | 4e-4 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Schedule | cosine decay |
| Warmup | 1B tokens |
| Global batch size | 0.5M tokens |
| Sequence length | 4K |
| Hybrid SWA size | 2K |
| Heads | 16 |
| Head dimension | $d_k = d_v = 128$ |

논문은 post-training이나 instruction tuning 논문이 아니다. 따라서 recipe의 핵심은 dataset curation보다 architecture comparison의 통제다. 같은 token budget, 같은 scale, 같은 recurrent state size를 두고 update rule만 바꿨을 때 어떤 차이가 생기는지 보는 setup이다.

## 4-3. Engineering notes

구현 관점에서 중요한 포인트는 다음과 같다.

- **q and k path**는 linear projection, short causal convolution, SiLU, L2 normalization을 거친다.
- **value path**는 linear projection, short causal convolution, SiLU를 거친다.
- **decay branch**는 log-decay projection을 만들고 fp32 path를 사용한다.
- **erase gate and write gate**는 각각 별도 linear projection과 sigmoid로 만든다.
- **output path**는 recurrent output을 RMS-normalize하고 SiLU output gate를 곱한 뒤 output projection을 적용한다.
- **kernel path**는 chunkwise WY algorithm과 gate-aware backward를 포함한다.

공식 repository는 PyTorch implementation을 공개하지만, license는 NVIDIA Source Code License-NC다. 연구 재현에는 유용하지만 commercial use나 downstream integration 전에 license 확인이 필요하다.

# 5. Evaluation

## 5-1. Main results

### Language modeling and commonsense reasoning

대표 수치는 다음과 같다. Perplexity는 낮을수록 좋고, accuracy와 average는 높을수록 좋다.

| Setting | Model | Wiki ppl | LMB ppl | LMB acc | Avg acc |
| --- | --- | ---: | ---: | ---: | ---: |
| Recurrent | Mamba-2 | 16.79 | 12.38 | 45.24 | 51.82 |
| Recurrent | Gated DeltaNet | 16.40 | 11.89 | 49.62 | 52.07 |
| Recurrent | KDA | 16.81 | 11.68 | 48.13 | 52.28 |
| Recurrent | Mamba-3 MIMO | 16.45 | 11.66 | 47.82 | 52.39 |
| Recurrent | Gated DeltaNet-2 | 15.90 | 11.41 | 48.09 | 53.11 |
| Hybrid | Transformer | 19.22 | 13.72 | 48.32 | 50.86 |
| Hybrid | KDA | 16.01 | 10.66 | 49.21 | 52.68 |
| Hybrid | Mamba-3 MIMO | 15.81 | 10.92 | 49.82 | 52.72 |
| Hybrid | Gated DeltaNet-2 | 15.62 | 10.43 | 50.90 | 53.97 |

Gated DeltaNet-2는 recurrent-only와 hybrid setting 모두에서 average accuracy가 가장 높다. LAMBADA accuracy만 보면 일부 baseline이 가까운 지점이 있지만, WikiText perplexity, LAMBADA perplexity, commonsense average를 함께 보면 전반적인 profile이 가장 안정적이다.

### RULER needle-in-a-haystack retrieval

논문에서 가장 인상적인 결과는 RULER retrieval이다.

| Setting | Model | S-NIAH-2 at 4K | S-NIAH-3 at 2K | MK-NIAH-1 at 4K |
| --- | --- | ---: | ---: | ---: |
| Recurrent | Gated DeltaNet | 87.2 | 54.2 | 27.8 |
| Recurrent | KDA | 89.0 | 63.2 | 28.0 |
| Recurrent | Mamba-3 MIMO | 64.2 | 72.4 | 18.0 |
| Recurrent | Gated DeltaNet-2 | 93.0 | 89.8 | 37.8 |
| Hybrid | Gated DeltaNet | 57.3 | 91.2 | 44.8 |
| Hybrid | KDA | 56.0 | 93.4 | 40.4 |
| Hybrid | Mamba-3 MIMO | 53.0 | 98.4 | 46.6 |
| Hybrid | Gated DeltaNet-2 | 57.9 | 99.0 | 48.0 |

특히 recurrent-only setting의 S-NIAH-3 at 2K에서 89.8, MK-NIAH-1 at 4K에서 37.8을 기록한다. 이 결과는 논문의 claim과 잘 맞는다. Fixed-size state에서 key association이 충돌할수록, erase와 write를 분리하는 update가 유리해진다.

### Real-world retrieval

Real-world retrieval에서는 SWDE, SQuAD, FDA, TriviaQA, NQ, DROP을 본다. Input은 2K tokens로 truncate된다.

| Setting | Gated DeltaNet | KDA | Mamba-3 MIMO | Gated DeltaNet-2 |
| --- | ---: | ---: | ---: | ---: |
| Recurrent avg | 28.09 | 28.67 | 28.35 | 29.88 |
| Hybrid avg | 39.11 | 40.14 | 40.11 | 42.28 |

Gated DeltaNet-2는 recurrent와 hybrid 모두에서 평균이 가장 높다. 다만 DROP 같은 항목에서는 recurrent-only가 항상 강한 것은 아니다. 논문도 local evidence aggregation이 필요한 task에서는 SWA가 보완 역할을 한다고 해석한다.

### Gate ablation

Ablation은 이 논문에서 특히 중요하다. 단순히 gate를 늘려 parameter가 늘었기 때문에 좋아진 것이 아니라, channel-wise structure가 실제로 쓰이는지 확인하기 때문이다.

| Variant | Wiki ppl | LMB ppl | Common avg | S-NIAH-2 at 4K | S-NIAH-3 at 2K | MK-NIAH-1 at 4K | Recall avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w-only, scalar erase and channel write | 16.55 | 11.62 | 52.45 | 90.6 | 71.4 | 30.6 | 28.92 |
| b-only, channel erase and scalar write | 16.12 | 11.50 | 52.79 | 92.1 | 84.6 | 35.2 | 29.51 |
| Gated DeltaNet-2 | 15.90 | 11.41 | 53.11 | 93.0 | 89.8 | 37.8 | 29.88 |
| Expanded erase range | 15.95 | 11.44 | 53.04 | 93.1 | 89.4 | 37.6 | 29.81 |

결과를 보면 erase gate 쪽 channel structure가 더 큰 역할을 한다. b-only variant가 w-only보다 full model에 더 가깝다. 이는 key-side erase가 compressed state의 association interference를 직접 조절한다는 설계 직관과 맞다.

## 5-2. What really matters in the experiments

### 1) Score보다 update rule의 위치가 중요하다

Gated DeltaNet-2는 새 tokenizer나 새 training corpus를 제안하는 논문이 아니다. 핵심은 recurrent state update의 residual edit를 바꾸는 것이다. 따라서 성능 개선은 data recipe보다 memory update rule의 inductive bias를 보여주는 evidence로 읽는 편이 맞다.

### 2) Long-context retrieval에서 가장 강한 메시지가 나온다

Language modeling 결과도 좋지만, 논문의 설계가 가장 직접적으로 검증되는 지점은 RULER와 real-world retrieval이다. Fixed-size state가 여러 association을 동시에 담아야 하는 상황에서 erase와 write decoupling이 의미를 가진다.

### 3) Hybrid result를 pure recurrent의 승리로 오독하면 안 된다

Hybrid model은 SWA를 포함한다. 따라서 local exact interaction은 SWA가 담당하고, Gated DeltaNet-2는 long-range compressed memory를 담당한다. 좋은 결과를 "linear attention alone beats attention"으로 읽으면 과장이다. 더 정확한 해석은 recurrent memory와 bounded local attention의 역할 분담이 잘 작동한다는 것이다.

### 4) Throughput overhead는 작지만 free는 아니다

논문은 single H100 training throughput에서 near-flat scaling을 유지한다고 보고한다. Hybrid 1.3B setting에서 sequence length가 늘어날 때 38.0 Kt/s에서 36.1 Kt/s 정도로만 감소한다. 다만 KDA 대비 작은 constant overhead가 있다. Gate를 늘려도 효율성이 유지된다는 점은 강점이지만, production inference에서는 kernel, batching, cache layout까지 별도로 검증해야 한다.

# 6. Limitations

1. **Scale extrapolation은 아직 제한적이다.** 논문은 1.3B parameters와 100B training tokens에서 비교한다. 10B 이상 scale이나 trillion-token pretraining에서 같은 trade-off가 유지되는지는 추가 검증이 필요하다.

2. **Million-token context claim과는 다른 종류의 evidence다.** 이 논문은 long-context retrieval과 fixed-state memory를 다루지만, 평가가 곧바로 million-token production context를 보장하지는 않는다.

3. **Hybrid result에는 SWA의 역할이 포함된다.** Gated DeltaNet-2 자체의 update rule도 좋지만, 최종적으로 강한 hybrid result는 local attention과 함께 해석해야 한다.

4. **Kernel complexity가 증가한다.** Channel-wise erase와 write gate는 modeling flexibility를 주지만, kernel implementation, backward pass, numerical stability 관리가 더 중요해진다.

5. **Instruction following이나 post-training 능력을 직접 평가한 논문은 아니다.** 1.3B pretrained model benchmark 중심이므로 assistant quality나 alignment quality로 바로 확장하면 안 된다.

6. **Code license 확인이 필요하다.** 공식 implementation은 공개되어 있지만 NVIDIA Source Code License-NC 기반이다. 연구용과 상업적 사용 가능성은 분리해서 봐야 한다.

7. **Ablation은 gate 구조 중심이다.** 더 다양한 data mixture, larger context, decoding latency, serving engine integration까지는 후속 검증이 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

Gated DeltaNet-2는 linear attention 계열을 이해할 때 좋은 기준점을 준다. 그동안 efficient attention 논의는 자주 "attention cost를 줄인다" 수준에서 끝났다. 하지만 실제 long-context model에서 중요한 것은 state를 얼마나 싸게 유지하느냐만이 아니라, state를 어떻게 안정적으로 edit하느냐다.

이 논문은 그 관점을 명확하게 만든다.

- **forgetting**은 broad memory horizon control이다.
- **erasing**은 key-side stale association removal이다.
- **writing**은 value-side new content commit이다.

이 세 operation을 분리해서 보면, Mamba-2, DeltaNet, Gated DeltaNet, KDA, Gated DeltaNet-2가 하나의 lineage로 정리된다. 연구 관점에서는 새로운 token mixer를 볼 때 어떤 축의 memory control을 추가했는지 묻는 기준이 생긴다.

## 7-2. Reuse potential

재사용하고 싶은 포인트는 네 가지다.

1. **Axis-aware gating**
   - 어떤 gate가 어떤 tensor axis에 작용하는지 명시적으로 설계한다.
   - scalar gate 하나로 여러 decision을 묶지 않는다.

2. **State edit decomposition**
   - decay, erase, write를 하나의 update 안에서 분리한다.
   - RNN, memory model, retrieval cache 설계에도 비슷한 관점을 적용할 수 있다.

3. **Hybrid operating point**
   - long-range compressed memory와 local exact attention을 역할별로 나눈다.
   - 모든 문제를 하나의 token mixer로 해결하려 하지 않는다.

4. **Ablation style**
   - gate를 추가했으면 channel degree of freedom이 실제로 쓰이는지 확인한다.
   - b-only, w-only처럼 기능별 scalarization ablation을 두는 방식이 깔끔하다.

## 7-3. Follow-up papers

함께 읽으면 좋은 논문은 다음과 같다.

- DeltaNet: Parallelizing Linear Transformers with the Delta Rule over Sequence Length
- Gated Delta Networks: Improving Mamba2 with Delta Rule
- Kimi Linear: An Expressive, Efficient Attention Architecture
- Mamba-2: Transformers are SSMs
- Mamba-3: An Efficient State Space Model with Selection and Compression
- FG2-GDN: Enhancing Long-Context Gated Delta Networks with Doubly Fine-Grained Control
- Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences

# 8. Summary

- Gated DeltaNet-2는 erase와 write를 하나의 scalar gate로 묶던 KDA and Gated DeltaNet의 제약을 푼다.
- 핵심은 key-side erase gate $b_t$와 value-side write gate $w_t$를 channel-wise로 분리하는 것이다.
- 기존 KDA와 Gated DeltaNet을 special case로 포함하므로, 새 update rule은 strict generalization이다.
- 1.3B model을 100B FineWeb-Edu tokens로 학습한 비교에서 language modeling, commonsense reasoning, synthetic retrieval, real-world retrieval 평균이 개선된다.
- 가장 중요한 결과는 RULER multi-key retrieval과 gate ablation이며, erase gate가 compressed memory interference를 줄이는 데 특히 중요해 보인다.
