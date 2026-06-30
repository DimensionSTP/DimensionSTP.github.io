---
layout: single
title: "Log-Linear Attention Review"
categories: Study-concept
tag: [EfficientAttention, LinearAttention, Mamba2, GatedDeltaNet, LongContext]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2506.04761)

[Code link](https://github.com/HanGuo97/log-linear-attention)

Log-Linear Attention은 efficient attention 논문을 읽을 때 늘 보게 되는 trade-off를 정면으로 다룬다. Transformer softmax attention은 prefix를 고해상도로 보지만 compute가 $O(T^2)$다. Linear attention and SSM은 $O(T)$에 가깝게 학습하고 decoding memory도 $O(1)$로 줄이지만, 결국 fixed-size hidden state 하나에 과거 전체를 압축한다.

이 논문은 그 사이에 새로운 operating point를 만든다. 과거 전체를 token별 KV cache로 들고 있지는 않되, hidden state를 하나만 두지도 않는다. 대신 prefix를 Fenwick tree 방식으로 여러 bucket으로 나누고, 최근 token은 fine-grained bucket으로, 먼 token은 coarse bucket으로 요약한다. 그 결과 hidden state 수는 sequence length에 대해 logarithmic하게 증가한다.

> 한 줄 요약: Log-Linear Attention은 linear attention/SSM의 fixed-size memory 한계를 완화하기 위해 Fenwick-tree 기반 hierarchical memory를 도입하고, hidden state 수를 $O(\log T)$로 늘리면서 training compute $O(T \log T)$, decoding memory $O(\log T)$의 middle ground를 만드는 efficient attention framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Transformer와 linear attention 사이를 binary choice가 아니라 memory resolution trade-off로 다시 본다.
- Fenwick tree partitioning으로 recent tokens는 high resolution, distant tokens는 low resolution으로 다룬다.
- Mamba-2 and Gated DeltaNet 위에 log-linear variant를 실제로 구성한다.
- Custom Triton kernel and chunk-scan을 통해 hardware-friendly parallel form을 제시한다.
- MQAR, 50B-token language modeling, NIAH, per-position loss에서 fixed-state linear models 대비 gain을 보인다.
- 한계도 솔직하다. Transformer와의 gap은 여전히 있고, engineering complexity가 높다.

이 글에서는 Log-Linear Attention을 "linear attention을 더 복잡하게 만든 논문"보다, **sequence length가 길어질수록 memory capacity를 어떻게 증가시킬 것인가에 대한 architecture design paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Sequence modeling에서 attention/memory mechanism은 크게 세 operating point를 가진다.

| Method | Training compute | Decoding memory | Context representation |
| --- | ---: | ---: | --- |
| Softmax attention | $O(T^2)$ | $O(T)$ | Token-level KV cache |
| Linear attention / SSM | $O(T)$ | $O(1)$ | Fixed-size recurrent state |
| Log-linear attention | $O(T \log T)$ | $O(\log T)$ | Logarithmically growing hidden states |

Softmax attention은 가장 expressive한 편이지만 long sequence에서 quadratic cost가 병목이다. Linear attention and SSM은 hidden state recurrence로 효율을 얻지만, 과거 전체를 fixed-size state에 압축한다. 이 fixed memory는 associative recall, needle retrieval, long-range dependency에서 한계가 될 수 있다.

Log-Linear Attention은 질문을 이렇게 바꾼다.

> Hidden state를 하나만 쓰지 않고, sequence length에 따라 조금씩 늘리면 어떤 trade-off가 가능한가?

## 1-2. Why previous approaches are insufficient

### 1) Full attention is too expensive

Transformer attention은 training에서 $O(T^2)$ compute를 요구하고, decoding에서는 $O(T)$ KV cache를 유지한다. FlashAttention류 kernel optimization은 상수를 줄이지만 asymptotic complexity 자체를 바꾸지는 않는다.

### 2) Linear attention compresses too much

Linear attention은 recurrent form으로 쓸 수 있다.

$$
S_t
=
S_{t-1}
+
\mathbf{v}_t
\mathbf{k}_t^\top
$$

$$
\mathbf{o}_t
=
S_t
\mathbf{q}_t
$$

이 구조는 fixed-size $S_t$ 하나로 prefix 전체를 표현한다. Efficient하지만, 과거 token을 서로 다른 granularity로 보관하는 능력이 부족하다.

### 3) Convolution and FFT models are another trade-off

Hyena류 model은 $O(T \log T)$ compute를 갖지만 decoding memory는 linear하게 남을 수 있다. Log-linear attention은 $O(T \log T)$ compute를 허용하되 decoding memory를 $O(\log T)$로 유지하는 point를 목표로 한다.

### 4) Sparse attention만으로는 recurrent memory problem을 직접 해결하지 않는다

Sparse attention은 어떤 token pair를 볼지 제한한다. Log-linear attention은 prefix를 hierarchical state로 요약한다. 즉 KV cache sparsification보다 recurrent memory capacity를 늘리는 접근에 가깝다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 contribution은 세 가지다.

1. **Logarithmically growing memory**
   - Fixed hidden state 대신 $O(\log T)$개의 hidden state를 유지한다.
   - Fenwick tree partitioning으로 prefix를 exponentially growing buckets로 나눈다.

2. **Parallel form with hierarchical matrix**
   - Recurrent form만 제시하지 않고, training을 위한 matmul-rich parallel form을 만든다.
   - Causal lower-triangular mask를 hierarchical matrix로 바꿔 $O(T \log T)$ compute를 얻는다.

3. **General lifting framework**
   - Log-linear attention을 vanilla linear attention뿐 아니라 Mamba-2 and Gated DeltaNet에 적용한다.
   - Existing linear-attention/SSM architecture 위에 temporal mask를 compose하는 방식으로 확장한다.

## 2-2. Design intuition

Log-linear attention의 직관은 recent vs distant memory resolution이다.

- 최근 token은 정확히 기억해야 한다.
- 멀리 있는 token은 개별 token 단위보다 summary로 충분할 수 있다.
- 하지만 summary 하나로 전체 prefix를 압축하는 것은 너무 거칠다.
- 따라서 prefix를 여러 scale의 bucket으로 나눈다.

Fenwick tree는 prefix $[0,t)$를 power-of-two segment로 분해한다. 예를 들어 어떤 time $t$의 prefix는 크기 1, 2, 4, 8 같은 bucket들의 disjoint union으로 표현될 수 있다. 이때 bucket 수는 at most $O(\log T)$다.

이를 memory 관점으로 쓰면 다음과 같다.

$$
\mathrm{Memory}(t)
=
\left\{
S_t^{(0)},
S_t^{(1)},
\ldots,
S_t^{(L-1)}
\right\}
$$

$$
L
=
O(\log T)
$$

각 $S_t^{(\ell)}$는 level $\ell$ bucket에 해당하는 recurrent state다. Output은 여러 bucket contribution을 가중합으로 만든다.

$$
\mathbf{o}_t
=
\sum_{\ell=0}^{L-1}
\lambda_t^{(\ell)}
\mathbf{q}_t^\top
S_t^{(\ell)}
$$

여기서 $\lambda_t^{(\ell)}$는 current input에서 나온 nonnegative coefficient다. 이 coefficient가 어떤 temporal scale을 더 강조할지 조절한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Linear attention의 fixed-state bottleneck 완화 |
| Core mechanism | Fenwick tree prefix partitioning |
| Memory size | $O(\log T)$ hidden states |
| Training compute | $O(T \log T)$ |
| Decoding memory | $O(\log T)$ |
| Parallel form | Hierarchical matrix and chunk-scan |
| Instantiations | Log-Linear Mamba-2, Log-Linear Gated DeltaNet |
| Implementation | Triton chunkwise parallel scan |

## 3-2. Fenwick tree partitioning

Attention을 prefix partitioning 문제로 보자. Token $t$가 과거 $[0,t)$를 볼 때 세 가지 extreme이 있다.

1. Full attention
   - Every past token is one bucket.
   - Number of buckets is $t$.

2. Linear attention
   - Entire prefix is one bucket.
   - Number of buckets is 1.

3. Log-linear attention
   - Prefix is decomposed into power-of-two buckets.
   - Number of buckets is $O(\log T)$.

Fenwick tree partitioning은 최근 token을 작은 bucket으로, 먼 token을 큰 bucket으로 둔다. 이는 natural inductive bias다.

| Region | Resolution |
| --- | --- |
| Recent past | Fine-grained |
| Mid-range past | Medium bucket |
| Distant past | Coarse summary |

이 design은 long-context memory를 "얼마나 많은 token을 그대로 저장할 것인가"가 아니라 "어떤 scale로 요약할 것인가"로 바꾼다.

## 3-3. Recurrent form

Vanilla linear attention에서 recurrent state는 하나다.

$$
S_t
=
\sum_{s < t}
\mathbf{v}_s
\mathbf{k}_s^\top
$$

Log-linear attention에서는 level별 state를 둔다.

$$
S_t^{(\ell)}
\in
\mathbb{R}^{d \times d}
$$

Output은 bucket별 state의 weighted sum이다.

$$
\mathbf{o}_t
=
\sum_{\ell=0}^{L-1}
\lambda_t^{(\ell)}
\mathbf{q}_t^\top
S_t^{(\ell)}
$$

모든 $\lambda_t^{(\ell)}$가 같으면 구조가 linear attention처럼 collapse될 수 있다. 따라서 level마다 다른 $\lambda$를 두는 것이 multi-scale temporal structure를 잡는 데 중요하다.

## 3-4. Parallel form

Recurrent form은 decoding 설명에는 좋지만, training에서는 GPU/TPU가 좋아하는 matmul-rich parallelism이 필요하다. 논문은 log-linear attention을 hierarchical matrix로 표현한다.

$$
\mathbf{O}
=
\left(
\mathbf{Q}
\mathbf{K}^{\top}
\odot
M^{H}
\right)
\mathbf{V}
$$

여기서 $M^{H}$는 Fenwick-tree partitioning이 만드는 hierarchical matrix다. Ordinary causal mask를 hierarchical structured mask로 바꾸는 셈이다.

이 matrix는 lower-triangular hierarchical structure를 가지며, 논문은 이를 chunkwise scan으로 계산한다. Training compute는 $O(T \log T)$다.

## 3-5. Mamba-2 and Gated DeltaNet variants

논문은 log-linear framework를 Mamba-2 and Gated DeltaNet에 적용한다. 두 model은 모두 gating mechanism으로 semiseparable temporal structure를 갖는다.

Log-linear extension은 original temporal mask $M^S$에 hierarchical mask $M^H$를 compose한다.

$$
M
=
M^S
\odot
M^H
$$

이 방식으로 기존 architecture의 transition matrix parameterization은 유지하면서, temporal memory structure만 multi-scale로 확장한다.

### Log-Linear Mamba-2

Mamba-2의 semiseparable structure에 hierarchical mask를 붙인다. 논문은 custom Triton kernel을 구현했고, 특정 setup에서 sequence length 8K 이후 forward plus backward가 FlashAttention-2보다 빠르다고 보고한다.

### Log-Linear Gated DeltaNet

Gated DeltaNet에도 같은 lifting을 적용한다. Empirical result는 Gated DeltaNet 쪽에서 더 강한 gain이 보고된다. WikiText and LAMBADA perplexity, LMEval average에서 linear Gated DeltaNet보다 log-linear variant가 개선된다.

## 3-6. Memory-efficient decoding

Decoding에서는 Fenwick tree state가 update된다. State 수는 binary representation의 significant bits와 관련되어 $O(\log T)$ 수준으로 유지된다.

Full attention의 KV cache는 $O(T)$다. Linear attention은 $O(1)$이다. Log-linear attention은 그 사이에 있다.

$$
\mathrm{Memory}_{\mathrm{decode}}
=
O(\log T)
$$

이 design은 fixed-state bottleneck을 줄이지만, full token cache만큼 정보가 보존되는 것은 아니다. Distant token은 bucket summary로 압축된다.

# 4. Training / Data / Recipe

## 4-1. Synthetic benchmark

논문은 MQAR, Multi-Query Associative Recall, task로 recall capability를 본다. Sequence length는 256이고, 4-64 key-value pairs를 포함한다. Dimension 16, 32, 64를 사용하며, five seeds로 평균과 standard deviation을 보고한다.

| Model | Dim 16 | Dim 32 | Dim 64 |
| --- | ---: | ---: | ---: |
| Transformer | >=99 | >=99 | >=99 |
| Mamba-2 | 46.9 | 75.1 | 89.6 |
| Log-Linear Mamba-2 | 55.9 | 76.5 | 92.9 |
| Gated DeltaNet | 38.4 | 79.0 | >=99 |
| Log-Linear Gated DeltaNet | 40.0 | 84.4 | >=99 |

Transformer가 여전히 강하지만, log-linear variant는 linear baseline을 개선한다. 이 결과는 fixed memory를 늘리는 것이 associative recall에 도움이 될 수 있음을 보여준다.

## 4-2. Language modeling

Academic-scale pretraining은 Long-Data-Collections dataset에서 50B tokens로 수행한다. Sequence length는 16K다.

Model configuration은 다음과 같다.

| Model | Parameters |
| --- | ---: |
| Transformer | 693M |
| Transformer, 24 layers | 778M |
| Mamba-2 | 802M |
| Log-Linear Mamba-2 | 825M |
| Gated DeltaNet | 793M |
| Log-Linear Gated DeltaNet | 796M |

Log-linear variant의 extra parameter는 Mamba-2에서 less than 3%, Gated DeltaNet에서 less than 0.4%라고 보고된다.

Main result는 다음과 같다.

| Model | WikiText ppl | LAMBADA ppl | LMEval avg |
| --- | ---: | ---: | ---: |
| Transformer | 21.56 | 22.14 | 44.0 |
| Transformer, 24 layers | 21.13 | 21.17 | 45.6 |
| Hyena | 29.50 | / | / |
| Mamba-2 | 22.44 | 24.14 | 44.8 |
| Log-Linear Mamba-2 | 22.11 | 21.86 | 44.9 |
| Gated DeltaNet | 21.73 | 19.71 | 45.0 |
| Log-Linear Gated DeltaNet | 21.45 | 18.09 | 45.5 |

Log-Linear Mamba-2는 perplexity를 개선하고 LMEval average는 소폭 올라간다. Log-Linear Gated DeltaNet은 더 큰 perplexity improvement를 보인다.

다만 Transformer와의 gap은 여전히 남는다. 특히 parameter-matched transformer와 비교하면 모든 metric에서 이긴다고 보기는 어렵다.

## 4-3. Per-position loss

Long-context utilization을 보기 위해 Book3 39M tokens에서 position별 loss를 본다. Running average window size는 501이다.

논문은 Mamba-2 and Gated DeltaNet을 log-linear variant로 확장했을 때 여러 position에서 smoothed loss가 낮아진다고 보고한다. 이는 fixed-state linear model보다 long-range context를 더 잘 활용할 가능성을 보여준다.

## 4-4. NIAH

RULER의 Needle-In-A-Haystack task에서도 log-linear variant는 linear baseline을 개선한다.

논문 summary에 따르면 다음과 같다.

- Log-Linear Mamba-2는 simpler single-needle task에서 8/9 metrics 개선
- Log-Linear Gated DeltaNet은 일부 single-needle task에서 이미 perfect인 cell을 유지하고, 몇몇 metric에서 개선
- More challenging multi-needle task에서 Log-Linear Mamba-2는 8/9 metrics 개선
- Log-Linear Gated DeltaNet은 multi-needle metrics 전반에서 개선

다만 table을 보면 Transformer baseline이 여전히 여러 retrieval task에서 강하다. Log-linear attention은 linear/SSM family의 memory capacity를 늘리는 방향이지, full attention을 완전히 대체했다고 읽으면 안 된다.

## 4-5. Implementation notes

논문은 custom Triton chunkwise parallel scan kernel을 구현한다. Log-linear Mamba-2 kernel은 특정 setup에서 sequence length 8K 이후 FlashAttention-2보다 forward plus backward가 빠르다고 보고한다.

하지만 전체 training throughput은 architecture, chunk size, heads, hidden dimension, convolution/MLP overhead에 따라 달라진다. 논문은 131K에서 gradient checkpointing 때문에 throughput drop이 있다고도 언급한다.

실제로 재사용하려면 다음을 확인해야 한다.

- Chunk size
- Memory level count
- Lambda parameterization
- Triton kernel support
- Backward pass implementation
- Integration with existing Mamba/DeltaNet kernels
- Inference KV/state cache layout

# 5. Evaluation

## 5-1. Main results

Log-linear attention은 linear baseline 대비 여러 결과에서 개선된다.

| Evaluation | Main observation |
| --- | --- |
| MQAR | Mamba-2와 Gated DeltaNet 변형이 linear baseline보다 개선된다 |
| Language modeling | WikiText와 LAMBADA perplexity가 전반적으로 개선된다 |
| Per-position loss | Log-linear variant가 sequence position 전반의 smoothed loss를 낮춘다 |
| NIAH | 여러 single/multi-needle metric에서 개선된다 |
| Transformer comparison | 여러 benchmark에서 gap이 남아 있다 |

가장 중요한 result는 "log-linear가 transformer를 압도한다"가 아니다. 이 논문은 fixed-state linear model에 $O(\log T)$ memory를 주면 recall과 long-context utilization이 개선될 수 있음을 보인다.

## 5-2. What really matters in the experiments

### 1) Memory capacity is a knob

Linear attention의 $O(1)$ hidden state는 매력적이지만 너무 강한 compression일 수 있다. Log-linear attention은 memory capacity를 $O(\log T)$로 늘려 중간 지점을 만든다.

### 2) Transformer gap remains

논문은 모든 benchmark에서 Transformer와의 gap이 남는다고 명시한다. 따라서 log-linear attention은 softmax attention을 곧바로 대체하는 상위 방법이라기보다, efficient model family의 memory capacity를 확장하는 방법으로 읽는 편이 안전하다.

### 3) Engineering cost is part of the method

Theoretical complexity가 좋아도 실제 성능은 kernel에 크게 의존한다. Log-linear attention은 inter-chunk와 intra-chunk implementation이 복잡하다. Backward pass도 lambda term gradient 때문에 더 어렵다.

### 4) Fenwick bias is not universally optimal

최근 token은 fine-grained, 먼 token은 coarse-grained라는 bias는 자연스럽지만 모든 task에 맞지는 않을 수 있다. 어떤 task는 먼 위치의 token을 정확히 recall해야 한다. 이런 경우 full attention이나 retrieval mechanism이 여전히 필요하다.

### 5) Lambda parameterization matters

$\lambda_t^{(\ell)}$가 어떤 temporal scale을 강조할지 결정한다. 논문은 compute limit 때문에 lambda parameterization과 hyperparameter sweep을 충분히 하지 못했다고 말한다. 후속 연구 여지가 크다.

# 6. Limitations

1. **Transformer와의 gap이 남아 있다**
   - Log-linear variant는 linear baseline보다 나아지지만, Transformer와의 gap을 모든 benchmark에서 일관되게 닫지는 못한다.

2. **모든 task에서 개선되는 것은 아니다**
   - 논문도 log-linear attention이 linear baseline보다 개선되지 않은 task가 적지 않다고 명시한다.

3. **Compute limit이 있다**
   - 700M-800M model을 한 번씩만 run할 수 있었다고 밝힌다.
   - Hyperparameter와 lambda parameterization search가 제한적이다.

4. **Engineering complexity가 높다**
   - Intra-chunk mechanism과 backward pass에는 custom implementation이 필요하다.
   - Lambda term gradient도 구현 복잡도를 높인다.

5. **Fenwick tree inductive bias가 모든 task에 맞지는 않는다**
   - 최근 context는 높은 해상도로, 먼 context는 낮은 해상도로 보관하는 memory bias가 모든 application에 최적은 아닐 수 있다.

6. **Academic scale을 넘는 scaling은 별도 검증이 필요하다**
   - 50B token, 700M-800M scale result는 유망하지만 multi-billion production scale은 별도 검증이 필요하다.

7. **Retrieval-heavy task에서는 Transformer가 여전히 강하다**
   - MQAR와 NIAH에서 log-linear gain은 확인되지만, 여러 cell에서는 Transformer가 여전히 강하다.

8. **Architecture별 gain pattern이 다르다**
   - Mamba-2와 Gated DeltaNet에 적용했을 때 gain pattern이 다르다.
   - 다른 linear attention variant에서도 같은 result가 나올지는 확인이 필요하다.

9. **Hardware dependency가 크다**
   - Kernel performance와 throughput claim은 GPU, head dimension, state dimension, batch size, chunk size에 의존한다.

10. **Memory cost가 더 이상 $O(1)$이 아니다**
    - Linear attention의 strongest efficiency point를 포기하고 $O(\log T)$ memory를 선택한다.
    - Deployment에서는 이 memory trade-off가 허용 가능한지 따로 봐야 한다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 핵심은 "linear attention을 더 강하게 만들었다"보다, **long-context model에서 memory capacity를 asymptotic knob으로 설계했다는 점**이다.

최근 efficient sequence model 논의는 자주 다음 둘 중 하나로 흐른다.

- Full attention을 어떻게 더 싸게 계산할 것인가
- Fixed-state recurrent model을 어떻게 더 강하게 만들 것인가

Log-linear attention은 세 번째 선택지를 준다.

```text
fixed state < log-growing state < full KV cache
```

이 axis는 practical design에서 중요하다. 모든 past token을 그대로 저장할 필요는 없지만, 하나의 state에 모두 압축하는 것도 너무 거칠 수 있기 때문이다.

## 7-2. Reuse potential

### Long-context LM

Long document에서는 recent context와 distant summary를 함께 쓰는 구조가 필요하다. Log-linear attention은 learned hierarchical memory의 architecture template으로 볼 수 있다.

### Document and code modeling

Codebase나 document에는 local syntax와 long-range structure가 함께 있다. Recent token은 exact하게, distant region은 function/class/document-level summary로 다루는 bias가 잘 맞을 수 있다.

### Hybrid architecture

Mamba-2, DeltaNet, Gated DeltaNet처럼 linear/SSM family를 transformer와 비교할 때는 memory size 자체를 늘리는 variant를 넣어야 더 공정한 comparison이 될 수 있다.

### Inference serving

$O(\log T)$ memory는 $O(T)$ KV cache보다 훨씬 작지만 $O(1)$보다 크다. Long-context serving에서 memory budget에 따라 middle-point를 선택할 수 있다.

### Follow-up design

다음 아이디어가 바로 떠오른다.

- Input-dependent adaptive bucket size
- Learned memory decay per level
- Hybrid of log-linear memory and retrieval cache
- Exact recent KV plus log-linear distant memory
- Domain-specific Fenwick schedule
- Sparse recovery for distant exact token recall

## 7-3. Follow-up papers

- Linear Transformers
- RetNet
- Mamba-2
- Gated DeltaNet
- Multi-Hyena
- DeltaNet
- RULER
- H3 and Hyena
- Memory Caching: RNNs with Growing Memory
- Adaptive Memory Decay for Log-Linear Attention

# 8. Summary

- Log-linear attention은 하나의 fixed hidden state를 sequence length에 따라 logarithmic하게 늘어나는 hidden state 묶음으로 대체한다.
- Fenwick tree partitioning은 recent context를 high-resolution으로, distant context를 coarse하게 유지한다.
- Complexity는 training compute $O(T \log T)$와 decoding memory $O(\log T)$가 된다.
- Mamba-2와 Gated DeltaNet의 log-linear variant는 여러 task에서 linear baseline보다 개선된다.
- 이 방법은 유망하지만 아직 transformer replacement는 아니며, engineering complexity가 높다.
