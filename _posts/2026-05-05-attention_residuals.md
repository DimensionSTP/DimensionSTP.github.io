---
layout: single
title: "Attention Residuals Review"
categories: Study-concept
tag: [Architecture, ResidualConnection, AttentionResiduals]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2603.15031)

[Code link](https://github.com/MoonshotAI/Attention-Residuals)

Attention Residuals는 "residual connection에 attention을 붙인 논문" 정도로 읽으면 핵심을 놓치기 쉽다. 이 논문이 흥미로운 이유는 attention mechanism을 token sequence 방향이 아니라 **network depth 방향**으로 적용한다는 점이다. 즉 각 layer가 바로 직전 hidden state만 받는 것이 아니라, 앞선 layer output들을 source로 보고 그중 무엇을 가져올지 softmax attention으로 고르게 만든다.

현대 LLM에서 PreNorm residual connection은 거의 표준처럼 쓰인다. 일반적인 residual update는 단순하고 안정적이다. 하지만 unroll해서 보면 모든 이전 layer output을 동일한 weight 1로 계속 더하는 구조다. depth가 깊어질수록 hidden-state magnitude가 커지고, 각 layer가 새로 만든 representation의 상대적 기여도는 희석될 수 있다. 논문은 이 현상을 **PreNorm dilution** 문제로 보고, residual stream 자체를 "고정 합산"이 아니라 "선택적 depth-wise retrieval" 문제로 다시 정의한다.

이 관점이 mHC와도 잘 이어진다. mHC가 multi-stream residual topology를 안정적으로 섞기 위한 constraint를 제안했다면, Attention Residuals는 residual accumulation 자체를 attention matrix로 재해석한다. 둘 다 block 내부 attention/FFN을 바꾸기보다, **layer 사이에서 정보가 흐르는 방식을 다음 architecture scaling axis로 본다**는 점에서 같은 흐름에 있다.

> 한 줄 요약: Attention Residuals는 PreNorm residual의 fixed unit-weight accumulation을 depth-wise softmax attention으로 바꾸고, 대규모 학습에서의 memory/communication 문제를 Block AttnRes와 pipeline/cache 최적화로 완화한 residual topology 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM architecture에서 residual connection이 단순한 skip path가 아니라 **depth 방향 information routing mechanism**이라는 점을 명확하게 보여준다.
- Full AttnRes와 Block AttnRes를 나눠, 아이디어의 원형과 large-scale training에서의 실용적 변형을 함께 제시한다.
- 48B total / 3B activated Kimi Linear 계열 모델의 1.4T token pretraining까지 넣어, small-scale ablation에 머물지 않고 실제 scaling setting에서 residual topology 변경을 검증한다.
- mHC, DenseFormer, Highway, DeepNorm 같은 기존 residual generalization을 depth mixing matrix 관점으로 정리해 주기 때문에, 후속 residual architecture를 읽는 기준점으로 쓰기 좋다.

이 논문의 핵심 메시지는 명료하다. **Residual connection은 모든 이전 layer를 무차별적으로 더하는 누적기가 아니라, depth 방향 memory를 읽는 retrieval interface로 볼 수 있다.** 그리고 그 retrieval을 fixed linear accumulation에서 softmax attention으로 바꾸면, layer별 representation의 기여도를 더 명시적으로 제어할 수 있다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 PreNorm Transformer/LLM에서 residual stream이 depth 방향으로 정보를 어떻게 누적하는가이다.

일반적인 residual update는 다음처럼 쓸 수 있다.

$$
h_l = h_{l-1} + f_{l-1}(h_{l-1})
$$

이를 펼치면 layer $l$의 입력은 embedding과 모든 이전 layer output의 합이 된다.

$$
h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)
$$

이 구조는 gradient highway로 매우 강력하다. 어떤 layer든 identity path를 통해 loss까지 직접 연결될 수 있고, 이 때문에 깊은 network를 안정적으로 학습할 수 있다. 하지만 동시에 이 구조는 모든 이전 layer output에 동일한 coefficient를 부여한다. 즉 layer 1의 output, layer 20의 output, layer 80의 output이 모두 weight 1로 누적된다.

논문은 이 지점을 문제로 본다. sequence modeling에서는 token마다 어떤 과거 token을 볼지 attention으로 고른다. expert routing에서는 input에 따라 어떤 expert를 쓸지 고른다. 하지만 depth 방향 residual aggregation은 여전히 고정된 합산 규칙에 묶여 있다.

따라서 문제는 "residual connection이 필요한가?"가 아니다. 문제는 **residual connection이 너무 고정된 방식으로 이전 representation을 합산하고 있는가?** 이다.

## 1-2. Why previous approaches are insufficient

기존 residual connection은 단순하고 강하지만, 다음 한계를 갖는다.

1. **No selective access**  
   각 layer는 개별 이전 layer output을 직접 선택할 수 없다. 바로 직전 hidden state $h_{l-1}$ 안에 모든 과거 정보가 이미 섞여 있으므로, 특정 early-layer feature를 다시 꺼내 쓰는 것이 어렵다.

2. **Irreversible aggregation**  
   한 번 합쳐진 residual state는 어떤 layer가 어떤 정보를 만들었는지 분리하기 어렵다. 이후 layer가 "이전 attention layer output은 강하게 쓰고, 특정 MLP output은 약하게 쓰자" 같은 선택을 하려면 구조적으로 source가 분해되어 있어야 한다.

3. **PreNorm dilution**  
   PreNorm에서는 각 sublayer가 normalized input을 받지만, residual stream 자체는 계속 누적된다. depth가 깊어질수록 hidden-state magnitude가 커지고, 새로 추가되는 layer output의 상대적 영향력은 작아질 수 있다. 논문은 이로 인해 깊은 layer가 영향력을 유지하려면 더 큰 output을 학습해야 하고, gradient 분포도 불균형해질 수 있다고 본다.

4. **기존 확장법의 trade-off**  
   ReZero, LayerScale, DeepNorm, Highway 계열은 residual path의 scale이나 gate를 조정하지만, 대부분은 여전히 immediate previous state 중심이다. DenseFormer처럼 모든 이전 output에 접근하는 방법은 있지만, fixed/static coefficient에 가깝고 input-dependent depth-wise selection은 제한적이다. HC/mHC는 multi-stream recurrence로 input-dependent mixing을 넣지만, 여러 stream을 유지하는 비용과 구현 복잡도가 있다.

Attention Residuals는 이 문제를 sequence modeling의 역사와 연결한다. RNN이 time dimension에서 모든 과거 정보를 single state에 압축하다가 Transformer attention으로 넘어간 것처럼, residual connection도 depth dimension에서 single accumulated state만 넘기지 말고, 이전 layer output들에 직접 attention하자는 것이다.

# 2. Core Idea

## 2-1. Main contribution

Attention Residuals의 핵심은 residual accumulation을 다음처럼 바꾸는 것이다.

기존 residual은

$$
h_l = \sum_{i=0}^{l-1} v_i
$$

처럼 모든 source를 동일하게 더한다. 여기서 $v_0 = h_1$, $v_i = f_i(h_i)$로 볼 수 있다.

AttnRes는 이를 다음처럼 바꾼다.

$$
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i
$$

여기서 $\alpha_{i \to l}$는 softmax attention weight다. 즉 layer $l$은 이전 source $v_i$들을 모두 보면서, 어떤 source를 더 강하게 가져올지 학습한다.

논문의 주요 기여는 세 가지다.

1. **Full Attention Residuals**  
   모든 layer가 모든 이전 layer output에 softmax attention을 수행한다. depth가 보통 sequence length보다 훨씬 작기 때문에, 개념적으로는 $O(L^2d)$ depth attention도 가능하다는 판단이다.

2. **Block Attention Residuals**  
   Full AttnRes는 아이디어는 깔끔하지만, large-scale training에서는 activation recomputation과 pipeline parallelism 때문에 모든 layer output을 유지/전달하는 비용이 커진다. 이를 해결하기 위해 layer를 block으로 묶고, block 내부는 standard residual sum으로 누적하며, block 사이에서만 attention을 수행한다.

3. **Infrastructure design**  
   Block AttnRes가 실제 pretraining/inference에서 동작하도록 cross-stage caching, two-phase computation, online softmax merge, sequence-sharded prefilling을 제안한다. 논문은 이를 통해 pipeline parallelism 환경에서 training overhead를 작게 유지하고, typical inference workload에서 latency overhead도 제한적이라고 보고한다.

## 2-2. Design intuition

설계 직관은 sequence attention과 거의 동일하다.

RNN은 time direction에서 과거 정보를 hidden state 하나에 압축한다. Transformer는 이를 바꿔, 각 token이 모든 과거 token representation을 직접 보고 softmax attention으로 선택하게 만들었다.

Attention Residuals는 이 아이디어를 depth direction에 옮긴다.

- Standard residual = depth-wise recurrence
- AttnRes = depth-wise attention
- 이전 layer output = depth direction의 memory slot
- 현재 layer input = 이전 source를 attention으로 읽은 결과

특히 중요한 점은 AttnRes가 거대한 새 module을 추가하지 않는다는 것이다. 각 layer는 learned pseudo-query $w_l \in \mathbb{R}^d$ 하나를 갖고, 이전 layer output을 key/value로 쓴다. attention weight는 다음처럼 계산된다.

$$
\alpha_{i \to l} = \frac{\exp(w_l^\top \mathrm{RMSNorm}(v_i))}{\sum_{j=0}^{l-1} \exp(w_l^\top \mathrm{RMSNorm}(v_j))}
$$

여기서 query $w_l$은 layer-specific learned vector이고, key/value는 source representation $v_i$다. query 자체는 input-dependent하지 않지만, key/value가 token별 hidden representation이므로 최종 weight는 input/token에 따라 달라진다.

RMSNorm을 key 쪽에 넣는 것도 중요한 설계다. 이를 넣지 않으면 magnitude가 큰 layer output이 softmax에서 과도하게 우세해질 수 있다. 논문은 RMSNorm 제거 시 Full AttnRes와 Block AttnRes 모두 validation loss가 나빠지는 ablation을 제시한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | PreNorm residual의 fixed accumulation을 content-dependent depth-wise aggregation으로 바꾸는 것 |
| Key module | Learned pseudo-query 기반 softmax attention over previous layer outputs |
| Full variant | 각 layer가 모든 이전 layer output에 attention |
| Scalable variant | Block 단위로 source를 압축하고 block-level representation에 attention |
| Main system issue | Activation recomputation / pipeline parallelism에서 이전 layer output 저장 및 전송 비용 증가 |
| Main system solution | Block AttnRes + cross-stage caching + two-phase computation + online softmax merge |
| Experimental backbone | Kimi Linear 계열 MoE Transformer, KDA/MLA hybrid attention, MoE FFN |
| Difference from prior residual variants | Single previous state가 아니라 개별 source 또는 block source에 input-dependent softmax selection 적용 |

## 3-2. Module breakdown

### 1) Full AttnRes: 모든 이전 layer output을 source로 둔다

Full AttnRes는 가장 직접적인 형태다. 각 layer $l$의 입력은 embedding $h_1$과 앞선 sublayer output $f_i(h_i)$들의 weighted sum이다.

$$
h_l = \alpha_{0 \to l} h_1 + \sum_{i=1}^{l-1} \alpha_{i \to l} f_i(h_i)
$$

이때 attention은 sequence attention이 아니라 depth attention이다. token sequence의 각 위치마다, layer depth 축을 따라 source를 고른다고 보면 된다.

Full AttnRes의 장점은 가장 표현력이 높다는 점이다. 각 layer는 먼 과거 layer output까지 직접 볼 수 있다. 논문이 visualized attention weights에서 보여주는 것처럼, 대부분은 diagonal dominance, 즉 바로 이전 source를 많이 보지만, 특정 layer에서는 embedding이나 멀리 떨어진 source에 non-trivial weight를 주는 learned skip path가 나타난다.

단점은 large-scale setting에서 저장/통신 비용이다. vanilla training에서는 backpropagation을 위해 activation을 이미 저장하므로 추가 memory overhead가 작다고 볼 수 있다. 하지만 실제 대규모 LLM 학습은 activation recomputation과 pipeline parallelism을 사용한다. 이 경우 모든 이전 layer output을 계속 보존하고 pipeline stage를 넘어 전달해야 하므로 비용이 $O(Ld)$로 커진다.

### 2) Block AttnRes: layer source를 block source로 압축한다

Block AttnRes는 Full AttnRes의 practical variant다. 전체 $L$개 layer를 $N$개 block으로 나누고, block 내부 output은 standard residual처럼 합산해 하나의 block representation으로 만든다.

block $n$의 representation은 개념적으로 다음과 같다.

$$
b_n = \sum_{j \in B_n} f_j(h_j)
$$

layer $l$이 block $n$ 안에 있을 때는 다음 source들을 본다.

- token embedding $b_0 = h_1$
- 이전 block representation들 $b_1, \ldots, b_{n-1}$
- 같은 block 안에서 지금까지 누적된 partial block representation $b_n^i$

즉 Full AttnRes가 모든 과거 sublayer output을 source로 둔다면, Block AttnRes는 과거를 block-level summary로 압축한다. 이로써 memory/communication이 $O(Ld)$에서 $O(Nd)$로 줄어든다.

논문은 $N \approx 8$ blocks 정도가 Full AttnRes의 이득을 상당 부분 회복한다고 보고한다. 가장 큰 모델에서는 54 layers를 6 layers per block으로 묶어 9 blocks를 만들고, embedding까지 포함해 총 10개의 depth-wise source를 사용한다.

### 3) Learned pseudo-query: 작지만 중요한 parameterization

AttnRes에서 각 layer는 learned pseudo-query $w_l$ 하나를 갖는다. 이 선택은 단순해 보이지만 중요한 trade-off를 갖는다.

input-dependent query를 쓰면 현재 hidden state에서 query를 projection해 더 expressive한 attention을 만들 수 있다. 실제 ablation에서도 input-dependent query는 validation loss를 더 낮춘다. 하지만 그 대가로 layer마다 $d \times d$ projection이 필요하고, decoding 시 memory access가 sequential해진다. 그래서 논문은 기본 설계로 learned static pseudo-query를 택한다.

이 선택 덕분에 block 안의 여러 query를 한 번에 batch 처리할 수 있다. query가 현재 forward computation에 의존하지 않기 때문에, block 내부 $S$개 layer의 inter-block attention을 한 번의 matrix multiplication으로 계산할 수 있다. 이 점이 two-phase computation의 핵심이다.

또 하나 중요한 구현 detail은 initialization이다. pseudo-query vector는 모두 zero로 초기화된다. 그러면 초기 softmax weight가 source들 사이에서 uniform해지고, 학습 초기에 특정 source가 과도하게 선택되는 volatility를 줄일 수 있다. 이는 기존 residual의 unnormalized sum과는 다르지만, AttnRes가 처음부터 depth-wise magnitude를 제어하려는 설계라고 볼 수 있다.

### 4) Two-phase computation: inter-block은 병렬, intra-block은 순차

Block AttnRes의 inference/training 효율은 two-phase computation에서 나온다.

- **Phase 1: Parallel inter-block attention**  
  block $n$ 안의 모든 layer query $w_l$를 모아, 이전 block representations $[b_0, \ldots, b_{n-1}]$에 한 번에 attention한다. 이 단계는 block source를 한 번만 읽고 여러 layer query를 batch 처리한다.

- **Phase 2: Sequential intra-block attention + online softmax merge**  
  block 안에서는 partial sum $b_n^i$가 layer를 지나며 계속 바뀐다. 따라서 intra-block part는 순차적으로 계산해야 한다. 논문은 Phase 1의 softmax statistics와 Phase 2의 partial result를 online softmax 방식으로 정확히 merge한다.

이 구조의 좋은 점은 inter-block source에 대한 반복 read를 줄인다는 점이다. naïve하게는 layer마다 모든 block representation을 읽어야 하지만, two-phase design에서는 block 단위로 amortize할 수 있다. 논문은 typical setting에서 Block AttnRes의 residual mechanism I/O를 standard residual보다 조금 큰 수준으로 낮추고, mHC-style multi-stream residual보다 훨씬 작게 만든다고 분석한다.

### 5) Cross-stage caching과 prefilling 최적화

대규모 pretraining에서 pipeline parallelism은 핵심 변수다. standard residual은 stage 사이에 hidden state 하나만 넘기면 된다. 반면 Block AttnRes는 각 stage가 누적된 block representation history를 알아야 한다.

naïve하게 매 stage transition마다 전체 block history를 다시 보내면 redundant communication이 커진다. 논문은 각 physical rank가 이전 virtual stage에서 받은 block들을 cache하도록 만들어, 다음 transition에서는 새로 생긴 incremental block만 보내게 한다. 이 방식으로 peak per-transition communication cost를 낮추고 1F1B steady-state에서 computation과 overlap할 수 있다고 설명한다.

long-context prefilling에서도 block representation cache가 부담이 된다. 논문은 128K context, 8 blocks 예시에서 block representation 저장이 15GB 수준까지 갈 수 있다고 보고하고, 이를 sequence dimension으로 tensor-parallel sharding해 device당 memory를 줄인다. 여기에 chunked prefill을 결합하면 per-device overhead를 더 낮출 수 있다고 설명한다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 final model 학습에서 Kimi Linear 1.4T-token run과 동일한 data/training recipe를 따른다고 설명한다. 다만 Attention Residuals technical report 자체에는 pretraining corpus의 세부 mixture가 충분히 풀려 있지는 않다. 따라서 데이터 구성은 Kimi Linear 논문 또는 관련 technical report에서 추가 확인이 필요하다.

본문에서 명시적으로 확인되는 training scale은 다음과 같다.

- scaling law 실험: 5개 model size, 각 size에서 Baseline / Full AttnRes / Block AttnRes 비교
- scaling law context length: 8192 tokens
- final main run: Kimi Linear 48B total / 3B activated parameter configuration
- final pretraining: 총 1.4T tokens
  - 1T tokens WSD pre-training phase
  - 약 400B high-quality tokens mid-training phase
- mid-training 이후 32K sequence length로 추가 long-context training 진행

여기서 중요한 점은 Attention Residuals가 별도의 supervised tuning 논문이 아니라는 것이다. AttnRes의 효과는 기본적으로 pretraining validation loss, training dynamics, downstream benchmark 성능으로 검증된다.

## 4-2. Training strategy

논문 기준 architecture는 Kimi Linear 계열 MoE Transformer와 동일하다. Kimi Delta Attention(KDA)과 Multi-Head Latent Attention(MLA)을 3:1 비율로 interleave하고, 각 attention layer 뒤에 MoE feed-forward layer를 둔다. AttnRes를 넣는 것 외에 model depth, hidden dimension, expert routing, MLP 구조는 유지한다.

final 48B / 3B activated model 설정은 다음과 같다.

| Item | Value |
| --- | --- |
| Total parameters | 48B |
| Activated parameters | 3B |
| Transformer blocks | 27 |
| Sublayers counted by AttnRes | 54 layers |
| Experts | 8 routed experts out of 256 + 1 shared expert |
| Block AttnRes grouping | 6 layers per block |
| Depth-wise sources | 9 blocks + token embedding = 10 sources |
| Context window during pretraining | 4096 tokens |
| Optimizer | Muon |
| LR schedule | WSD, Warmup-Stable-Decay |
| Global batch size | 8M tokens |

AttnRes 자체는 별도의 auxiliary objective를 추가하지 않는다. 핵심은 residual aggregation path를 바꾸는 것이다. 따라서 loss 설계보다 중요한 것은 다음 두 가지다.

1. **동일 recipe 비교**  
   baseline과 AttnRes 모델은 같은 pretraining recipe로 비교된다. scaling law 실험에서는 각 size group의 hyperparameter를 baseline 기준으로 선택해 비교가 보수적으로 baseline에 유리하도록 했다고 설명한다.

2. **zero initialization**  
   pseudo-query $w_l$는 zero initialization을 사용한다. 초기 attention weight가 uniform하게 시작하므로, 학습 초기에 특정 source가 임의로 dominate하는 현상을 줄인다. 논문은 이 초기화가 training volatility를 막는 데 중요하다고 본다.

## 4-3. Engineering notes

이 논문의 engineering message는 꽤 분명하다. AttnRes의 이론적 overhead는 depth $L$에 대한 attention이므로 sequence attention에 비해 작아 보일 수 있다. 하지만 실제 LLM training에서는 memory layout, activation checkpointing, pipeline stage communication이 병목이 된다.

중요한 engineering point는 다음과 같다.

- Full AttnRes는 vanilla training에서는 activation 저장과 겹쳐 추가 memory가 작지만, activation recomputation과 pipeline parallelism을 쓰면 모든 이전 layer output을 유지/전송해야 한다.
- Block AttnRes는 stored representation 수를 $L$개 layer output에서 $N$개 block representation으로 줄인다.
- cross-stage caching은 interleaved pipeline schedule에서 block history를 매번 재전송하지 않고 incremental block만 전달하게 한다.
- two-phase computation은 pseudo-query가 forward hidden state와 decoupled되어 있다는 점을 이용해 block 안의 inter-block attention을 batch 처리한다.
- online softmax merge는 inter-block result와 intra-block partial result를 정확히 합치기 위한 장치다.
- long-context prefilling에서는 block representation을 sequence dimension으로 sharding해 per-device memory overhead를 낮춘다.

논문에서 보고하는 system 수치는 다음과 같다.

- pipeline parallelism이 없을 때 Block AttnRes training overhead는 negligible하다고 설명한다.
- pipeline parallelism 하에서는 measured end-to-end training overhead가 4% 미만이라고 보고한다.
- typical inference workload에서는 latency overhead가 2% 미만이라고 보고한다.
- 128K context, 8 blocks 예시에서 block representation memory는 전체 15GB 수준이지만, sequence sharding으로 device당 약 1.9GB, chunked prefill과 결합하면 0.3GB 미만으로 줄일 수 있다고 설명한다.

다만 이 수치는 Kimi/Moonshot 쪽 distributed training/inference stack과 최적화 구현에 강하게 묶여 있다. 다른 infra에서 그대로 재현된다고 보면 안 되고, 구현 난이도까지 포함해 해석해야 한다.

# 5. Evaluation

## 5-1. Main results

평가는 크게 네 축으로 볼 수 있다.

1. scaling law 실험
2. 48B / 3B activated model pretraining 및 downstream benchmark
3. ablation study
4. training dynamics / learned attention pattern analysis

### Scaling law

논문은 5개 model size에서 Baseline, Block AttnRes, Full AttnRes, mHC(-lite)를 비교한다. 모든 모델은 8192 context length로 학습된다.

| Activated Params | Tokens | Baseline | Block AttnRes | Full AttnRes | mHC(-lite) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 194M | 38.7B | 1.931 | 1.909 | 1.899 | 1.906 |
| 241M | 45.4B | 1.895 | 1.875 | 1.874 | 1.869 |
| 296M | 62.1B | 1.829 | 1.809 | 1.804 | 1.807 |
| 436M | 87.9B | 1.766 | 1.746 | 1.737 | 1.747 |
| 528M | 119.0B | 1.719 | 1.693 | 1.692 | 1.694 |

읽을 때 중요한 점은 세 가지다.

- Full AttnRes가 대체로 가장 낮은 validation loss를 보인다.
- Block AttnRes는 Full AttnRes보다 조금 낮은 표현력을 갖지만, largest scale에서는 차이가 0.001까지 좁혀진다.
- 논문은 fitted scaling curve 기준으로 Block AttnRes가 baseline 대비 1.25x compute advantage에 해당한다고 해석한다.

mHC와의 비교도 흥미롭다. Table 2 기준으로 mHC(-lite)는 강한 baseline이지만, Full AttnRes는 대부분의 size에서 mHC와 비슷하거나 더 낮은 loss를 보인다. Block AttnRes는 mHC와 유사한 loss를 내면서, 논문 기준 residual mechanism I/O cost가 더 낮다고 주장한다. 이 부분은 mHC와 Attention Residuals를 이어서 읽을 때 중요한 포인트다. 둘 다 residual topology를 다루지만, mHC는 multi-stream recurrence이고 AttnRes는 cross-layer source selection이다.

### Final model downstream benchmark

48B total / 3B activated configuration에서 baseline과 AttnRes를 같은 pretraining recipe로 비교한 결과는 다음처럼 요약할 수 있다.

| Benchmark | Baseline | AttnRes | Delta |
| --- | ---: | ---: | ---: |
| MMLU | 73.5 | 74.6 | +1.1 |
| MMLU-Pro | 52.2 | 52.2 | 0.0 |
| GPQA-Diamond | 36.9 | 44.4 | +7.5 |
| BBH | 76.3 | 78.0 | +1.7 |
| ARC-Challenge | 64.6 | 65.7 | +1.1 |
| HellaSwag | 83.2 | 83.4 | +0.2 |
| TriviaQA | 69.9 | 71.8 | +1.9 |
| GSM8K | 81.7 | 82.4 | +0.7 |
| MATH | 53.5 | 57.1 | +3.6 |
| HumanEval | 59.1 | 62.2 | +3.1 |
| MBPP | 72.0 | 73.9 | +1.9 |
| CMMLU | 82.0 | 82.9 | +0.9 |
| C-Eval | 79.6 | 82.5 | +2.9 |

논문은 downstream performance가 전반적으로 개선된다고 해석한다. 표만 놓고 보면 MMLU-Pro는 동일하고 나머지는 개선이다. 그래서 블로그 초안에서는 "모든 task에서 개선"보다는 **대부분의 benchmark에서 개선, MMLU-Pro는 동일**이라고 보수적으로 적는 편이 맞아 보인다.

가장 눈에 띄는 것은 GPQA-Diamond, MATH, HumanEval 쪽이다. 논문은 이를 depth-wise information flow 개선이 compositional reasoning, multi-step reasoning, code generation에 특히 도움이 될 수 있다는 가설과 연결한다.

## 5-2. What really matters in the experiments

이 논문의 실험에서 진짜 봐야 할 것은 최종 score보다 ablation과 dynamics다.

### 1) input-dependent depth-wise selection이 중요하다

16-layer model ablation은 AttnRes의 어떤 설계가 중요한지 꽤 잘 보여준다.

| Variant | Validation Loss | 해석 |
| --- | ---: | --- |
| Baseline PreNorm | 1.766 | fixed residual accumulation |
| DenseFormer | 1.767 | 모든 이전 output 접근은 있지만 static coefficient에 가까움 |
| mHC | 1.747 | multi-stream dynamic mixing |
| AttnRes Full | 1.737 | full depth-wise softmax attention |
| Input-dependent query | 1.731 | 더 좋지만 비용/decoding trade-off 큼 |
| Input-independent mixing | 1.749 | source 접근만으로는 부족, input-dependent weighting 필요 |
| Sigmoid instead of softmax | 1.741 | softmax의 competitive normalization이 이득 |
| Full w/o RMSNorm | 1.743 | magnitude bias 방지에 RMSNorm 필요 |
| Block S=4 | 1.746 | Full보다 약하지만 효율적 trade-off |
| Block multihead H=16 | 1.752 | channel별 depth mixture가 항상 이득은 아님 |
| Block w/o RMSNorm | 1.750 | block source magnitude bias가 더 커질 수 있음 |

내가 보기엔 핵심은 DenseFormer와 input-independent mixing 결과다. 단순히 모든 layer output에 접근할 수 있게 하는 것만으로는 충분하지 않다. **어떤 input/token에서 어떤 depth source가 필요한지**를 동적으로 정해야 한다.

### 2) softmax는 단순 normalization이 아니라 competition이다

sigmoid variant가 Full AttnRes보다 나쁜 점도 중요하다. sigmoid는 각 source를 독립적으로 켜고 끌 수 있지만, source들 사이의 probability mass competition을 강하게 만들지는 않는다. softmax는 source들끼리 같은 mass를 두고 경쟁하게 만든다. 이게 depth-wise selection을 더 날카롭게 만든다는 해석이 가능하다.

즉 AttnRes의 본질은 "가중합을 학습했다"가 아니라, **이전 layer source들 사이에 경쟁을 도입했다**는 데 있다.

### 3) 멀리 있는 layer를 볼 수 있어야 한다

sliding-window aggregation은 최근 $W=8$ layer output과 embedding만 유지하는 방식이다. 이 variant는 baseline보다는 약간 낫지만 Full/Block AttnRes보다 훨씬 낮다. 논문은 이를 selective access to distant layers가 중요하다는 근거로 본다.

이 점은 residual stream 해석에 중요한 힌트를 준다. 깊은 layer가 항상 바로 직전 layer만 필요로 하는 것이 아니라, 특정 상황에서는 embedding 또는 훨씬 앞선 layer의 representation을 다시 읽을 수 있어야 한다.

### 4) Block size는 performance/system trade-off다

Block size sweep에서는 $S=1$이 Full AttnRes이고, $S$가 커질수록 더 coarse한 block compression이 된다. validation loss는 block size가 커질수록 점진적으로 나빠지지만, $S=2,4,8$은 비슷한 수준을 유지하고 $S=16,32$ 쪽에서 baseline에 가까워진다.

이 결과는 Block AttnRes를 단순 approximation으로만 볼 필요가 없다는 뜻이다. 적당한 block granularity에서는 Full AttnRes의 핵심 구조, 즉 locality + occasional long skip + embedding persistence가 유지된다. 논문은 실용성을 위해 약 8 blocks를 기본 선택으로 둔다.

### 5) Training dynamics가 논문의 설득력을 만든다

Figure 5의 training dynamics가 이 논문의 중요한 증거다.

- baseline은 depth가 깊어질수록 output magnitude가 증가하는 경향을 보인다.
- Block AttnRes는 block boundary에서 selective aggregation이 reset 역할을 하며, output magnitude가 bounded periodic pattern을 보인다.
- baseline은 earliest layer gradient가 disproportionately large해지는 반면, Block AttnRes는 gradient magnitude가 더 균일하게 분포한다.

이 부분이 단순 benchmark improvement보다 더 중요하다. Attention Residuals가 성능만 올린 것이 아니라, 논문이 제기한 PreNorm dilution 문제를 실제 training dynamics에서 완화하는 방향으로 작동한다는 증거이기 때문이다.

### 6) learned pattern은 "전부 멀리 본다"가 아니다

Figure 8의 learned attention weight를 보면 대부분의 layer는 여전히 바로 이전 source를 강하게 본다. 즉 AttnRes가 residual locality를 파괴하는 것은 아니다. 오히려 standard residual의 local path를 유지하면서, 필요할 때 embedding이나 특정 distant source로 jump하는 learned skip path를 만든다.

이 해석이 중요하다. 좋은 residual architecture는 모든 layer를 fully connected하게 만든다고 좋은 것이 아니라, 기본 locality는 유지하되 필요한 cross-depth retrieval을 열어주는 쪽이 더 자연스럽다.

# 6. Limitations

1. **Kimi Linear / Moonshot stack에 강하게 묶인 검증**  
   실험 backbone은 Kimi Linear 계열 MoE Transformer다. KDA/MLA hybrid attention, MoE routing, Moonlight/DeepSeek-V3 스타일 design과 함께 검증된다. Dense Transformer, 다른 MoE routing, 다른 normalization recipe에서도 같은 크기의 gain이 나는지는 추가 검증이 필요하다.

2. **Full AttnRes는 아직 system constraint가 크다**  
   Full AttnRes가 가장 깨끗한 아이디어지만, large-scale setting에서는 모든 이전 layer output을 저장/전달해야 한다. 논문도 현재 hardware/distributed training constraint에서는 Block AttnRes를 practical default로 둔다. Full AttnRes의 잠재력은 future interconnect나 memory system 개선과 함께 다시 봐야 한다.

3. **Block AttnRes는 compression trade-off를 갖는다**  
   block representation은 여러 layer output을 하나로 합친다. 적당한 block size에서는 성능 손실이 작지만, source granularity가 coarse해질수록 Full AttnRes와의 차이가 생긴다. block boundary를 어떻게 잡는가도 architecture/hardware에 따라 달라질 수 있다.

4. **input-dependent query가 더 좋지만 채택되지 않았다**  
   ablation에서는 input-dependent query가 Full AttnRes보다 더 낮은 loss를 보인다. 하지만 decoding cost와 projection overhead 때문에 default로 쓰지 않는다. 즉 현재 AttnRes는 성능 최상 설계라기보다 efficiency-aware compromise다.

5. **downstream benchmark 개선의 causal attribution은 조심해야 한다**  
   baseline과 recipe를 맞췄다고 해도, downstream score는 pretraining dynamics, data recipe, architecture interaction의 결과다. AttnRes가 어떤 task에 왜 특히 도움되는지에 대해서는 더 세밀한 mechanistic analysis가 필요하다.

6. **논문 본문만으로 data recipe 재현은 어렵다**  
   final run은 Kimi Linear 1.4T-token recipe를 따른다고 되어 있지만, Attention Residuals paper 자체가 corpus mixture를 자세히 설명하는 논문은 아니다. 재현 관점에서는 Kimi Linear/Moonlight 쪽 recipe를 함께 봐야 한다.

7. **깊은 모델 선호가 곧 deployment recommendation은 아니다**  
   architecture sweep에서는 AttnRes가 더 deeper/narrower한 allocation을 잘 활용할 수 있음을 시사한다. 하지만 깊은 모델은 sequential depth 때문에 inference latency가 커질 수 있다. 논문도 이 결과를 deployment recommendation이 아니라 diagnostic으로 해석한다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 residual connection을 다시 보게 만든다. 보통 우리는 Transformer block 안의 attention type, FFN variant, MoE routing, positional encoding에는 많은 관심을 두지만, block들을 연결하는 residual stream은 거의 고정된 배관처럼 취급한다. Attention Residuals는 이 배관 자체가 중요한 modeling interface일 수 있음을 보여준다.

내가 특히 흥미롭게 본 지점은 residual connection을 **depth-wise retrieval**로 재해석한 부분이다. RAG에서 query가 document chunk를 고르듯, layer도 이전 layer representation을 고른다. 이 관점으로 보면 deep network의 forward pass는 단순한 sequential transform이 아니라, depth memory를 계속 쌓고 필요한 시점에 다시 읽는 과정이다.

mHC와 함께 보면 더 명확하다.

- mHC: residual stream을 여러 개로 넓히고, stream mixing을 안정적으로 제한한다.
- Attention Residuals: residual source를 layer/block memory로 보고, 어떤 source를 읽을지 attention으로 선택한다.

둘 다 residual topology가 LLM architecture의 다음 설계 축이 될 수 있다는 신호다. 앞으로는 attention kernel이나 MoE routing만이 아니라, **representation이 layer 사이에서 어떻게 저장되고, 압축되고, 다시 읽히는가**가 더 중요해질 수 있다.

## 7-2. Reuse potential

실무나 연구에서 바로 가져갈 수 있는 포인트는 다음 네 가지다.

1. **Residual path를 metric으로 모니터링하기**  
   단순 validation loss만 보지 말고 layer별 output magnitude, gradient norm, residual contribution ratio를 같이 봐야 한다. AttnRes 논문은 구조 변경의 설득력을 dynamics 분석으로 만든다.

2. **Block-level cross-layer aggregation**  
   Full cross-layer access는 비용이 크지만, block-level summary를 두고 필요할 때 attention하는 방식은 다른 architecture에도 적용 가능해 보인다. 예를 들어 long-context model, MoE routing, memory-augmented layer stack에서 block summary를 source로 쓰는 방향을 생각해볼 수 있다.

3. **pseudo-query parameterization**  
   input-dependent query가 더 강하지만 비싸고, static learned query는 약간 단순하지만 system-friendly하다. 이 trade-off는 production model에서 매우 현실적이다. 좋은 architecture는 항상 가장 expressive한 구조가 아니라, training/inference path에서 살아남는 구조다.

4. **softmax competition over depth**  
   residual scaling이나 gating을 독립적으로 학습하는 대신, source 간 경쟁을 도입하는 방식은 다른 residual variant에도 붙일 수 있다. 이 경쟁 구조가 gradient 분포를 더 균일하게 만드는지 보는 것은 후속 실험으로 가치가 있다.

내가 직접 실험한다면 처음부터 48B급 pretraining을 할 수는 없으니, 작은 dense Transformer에서 다음 순서로 볼 것 같다.

- PreNorm baseline vs Block AttnRes
- block count $N$ sweep
- learned query vs input-dependent query
- RMSNorm key on/off
- layer별 output magnitude / gradient norm / source attention heatmap
- reasoning-heavy validation subset에서 gain이 더 큰지 확인

## 7-3. Follow-up papers

- **mHC: Manifold-Constrained Hyper-Connections**  
  residual topology를 multi-stream 관점에서 확장하고, stability constraint를 Birkhoff polytope로 넣는 논문이다. Attention Residuals와 함께 읽으면 residual connection의 두 방향, 즉 stream expansion과 source retrieval을 비교하기 좋다.

- **Kimi Linear Technical Report**  
  Attention Residuals의 main experiment backbone이 되는 Kimi Linear architecture를 이해하기 위해 필요하다. KDA/MLA hybrid, MoE configuration, 1.4T-token recipe를 같이 봐야 final result 해석이 정확해진다.

- **DeepSeek-V3 Technical Report / Moonlight 관련 보고서**  
  논문이 언급하는 MoE Transformer recipe와 WSD/Muon-style training stack의 배경을 이해하는 데 도움이 된다.

- **DenseFormer / DenseNet 계열 cross-layer access 논문**  
  모든 이전 layer output에 접근하는 아이디어가 AttnRes 이전에 어떻게 다뤄졌는지 볼 수 있다. AttnRes의 차이는 단순 access가 아니라 input-dependent softmax selection이라는 점이다.

- **DeepNet / DeepNorm / ReZero / LayerScale**  
  residual scaling과 deep network stability를 다룬 고전적 비교축이다. Attention Residuals의 Table 5를 제대로 이해하려면 이 계열과 함께 보는 것이 좋다.

# 8. Summary

- Attention Residuals는 residual connection을 fixed accumulation이 아니라 depth-wise attention problem으로 재정의한다.
- Full AttnRes는 모든 이전 layer output을 source로 보고 softmax attention을 수행하며, Block AttnRes는 이를 block-level summary로 압축해 large-scale training 비용을 줄인다.
- 핵심 설계는 learned pseudo-query, RMSNorm-normalized source keys, softmax competition, zero initialization이다.
- 실험에서는 scaling law, 48B/3B activated pretraining, downstream benchmark, ablation, training dynamics를 통해 PreNorm dilution 완화를 주장한다.
- 내가 보기엔 이 논문은 residual connection을 LLM의 depth memory / retrieval interface로 보게 만든다는 점에서, mHC와 함께 residual topology 연구의 중요한 흐름으로 읽을 만하다.
