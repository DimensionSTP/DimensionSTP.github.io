---
layout: single
title: "SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales Review"
categories: Study-concept
tag: [LLM-Training, Optimizer, Distributed-Training]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2607.20548)

[Code link](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)

> 한 줄 요약: 이 논문은 AdamW, Muon, SOAP을 update RMS 기준으로 공정하게 비교하고, 대규모 batch와 fine-grained MoE에서 나타나는 optimizer 안정성 문제를 분석하며, SOAP의 stale preconditioner 문제와 matrix optimizer의 distributed systems 병목을 함께 해결한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Muon과 SOAP을 small-scale benchmark가 아니라 8B dense, 30B MoE, 72B MoE와 trillion-token pretraining regime에서 비교한다.
- Optimizer 비교에서 learning rate를 그대로 옮기는 대신 update RMS를 맞춰 scale 차이를 통제한다.
- SOAP의 loss spike를 단순 numerical accident가 아니라 current gradient와 stale eigenbasis 사이의 시차 문제로 진단한다.
- Fine-grained MoE에서 global batch와 per-expert effective batch가 다르게 움직인다는 점을 optimizer 관점에서 해설한다.
- Matrix-aware optimizer를 Megatron-LM 규모에서 실행하기 위한 layer-wise distributed optimizer까지 제안한다.

LLM pretraining에서 optimizer는 흔히 algorithm hyperparameter 중 하나로 취급된다. 하지만 model scale이 커지면 optimizer는 loss curve뿐 아니라 memory layout, tensor sharding, communication pattern, maximum stable batch size까지 결정한다.

AdamW는 거의 모든 대규모 학습 stack에서 기본값이다. Parameter coordinate마다 first moment와 second moment를 유지하므로 구현과 sharding이 단순하다. 반면 linear layer의 weight가 matrix라는 사실은 거의 이용하지 않는다. Row, column, head, expert 사이의 correlation을 보지 않고 각 scalar coordinate를 독립적으로 rescale한다.

Muon과 SOAP은 이 한계를 다른 방식으로 다룬다.

- Muon은 momentum matrix의 singular direction을 정리해 update를 polar factor에 가깝게 만든다.
- SOAP은 Shampoo-style covariance와 eigenbasis를 사용해 gradient를 회전한 뒤, 그 basis에서 Adam-like adaptive update를 수행한다.
- 두 방법 모두 weight matrix의 operator structure를 이용하지만, full matrix operation과 optimizer state 때문에 distributed training이 어려워진다.

이 논문의 핵심은 "어느 optimizer가 loss를 더 빨리 낮추는가"만 비교하는 데 있지 않다. 실제 frontier-scale training에서 matrix optimizer가 살아남으려면 다음 네 문제가 동시에 풀려야 한다.

1. AdamW와 공정하게 learning rate를 비교할 수 있어야 한다.
2. Large batch에서 update가 안정적이어야 한다.
3. Tensor parallelism과 data parallelism 아래에서 full matrix semantics를 보존해야 한다.
4. Extra optimizer compute와 communication을 training step 안에 숨겨야 한다.

따라서 이 논문은 optimization paper이면서 systems paper다. Algorithm, numerical stability, batch scaling, distributed execution을 하나의 pretraining recipe로 묶는다.

# 1. Problem Setting

## 1-1. Problem definition

LLM pretraining의 한 step을 단순화하면 다음과 같다.

1. Global batch에서 gradient를 계산한다.
2. Distributed worker 사이에서 gradient를 aggregate한다.
3. Optimizer가 update direction과 scale을 만든다.
4. Parameter를 갱신하고 다음 step으로 넘어간다.

AdamW는 flattened gradient $g_t$에 대해 first moment와 second moment를 coordinate-wise로 유지한다.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t
$$

단순화한 update는 다음과 같다.

$$
u_t = \frac{m_t}{\sqrt{v_t}+\epsilon}
$$

이 방식의 강점은 명확하다.

- 모든 state가 parameter와 같은 element-wise layout을 가진다.
- ZeRO나 distributed optimizer로 쉽게 shard할 수 있다.
- Matrix shape가 달라도 같은 kernel을 적용하기 쉽다.
- Numerical behavior가 오랜 기간 검증되어 있다.

그러나 matrix $G_t \in \mathbb{R}^{m \times n}$를 flatten하면 row와 column correlation이 사라진다. Linear layer는 input direction과 output direction을 연결하는 operator인데, AdamW는 이 구조를 직접 이용하지 않는다.

Shampoo 계열은 row covariance와 column covariance를 따로 추정한다.

$$
L_t = \beta_2 L_{t-1} + (1-\beta_2) G_t G_t^\top
$$

$$
R_t = \beta_2 R_{t-1} + (1-\beta_2) G_t^\top G_t
$$

이후 Kronecker-factored preconditioner를 적용한다.

$$
u_t =
L_t^{-1/4}
G_t
R_t^{-1/4}
$$

이론적으로는 gradient geometry를 더 잘 반영할 수 있다. 하지만 practical cost가 커진다.

- $L_t$와 $R_t$를 full precision으로 유지해야 한다.
- Matrix inverse root나 eigendecomposition이 필요하다.
- Weight가 tensor-parallel shard로 나뉘면 full matrix statistics를 바로 계산할 수 없다.
- Eigenbasis를 자주 갱신하면 compute가 늘고, 드물게 갱신하면 stale basis가 된다.
- Fine-grained MoE는 expert 수가 많아 layer shape와 load balance가 더 복잡하다.

논문이 겨냥하는 문제는 다음과 같이 정리할 수 있다.

| Question | Meaning |
| --- | --- |
| Algorithm | AdamW보다 matrix structure를 잘 이용하는가 |
| Fairness | Optimizer마다 update scale이 다른데 learning rate를 어떻게 맞출 것인가 |
| Stability | Large batch에서 SOAP preconditioner가 왜 loss spike를 만드는가 |
| MoE scaling | Global batch가 커질 때 expert와 dense parameter가 같은 regime을 보는가 |
| Systems | Full matrix update를 TP와 DP 아래에서 어떻게 계산할 것인가 |
| Throughput | Extra matrix compute와 communication을 실제 step에서 숨길 수 있는가 |

## 1-2. Why previous approaches are insufficient

### 1) Small-scale optimizer comparison은 frontier regime을 대표하지 못한다

Optimizer는 model size, token budget, batch size, architecture에 따라 behavior가 달라진다. 수억 parameter와 짧은 training run에서 보인 advantage가 30B 또는 72B MoE의 trillion-token training에서도 유지된다고 볼 수 없다.

특히 large batch는 중요한 stress test다. Synchronous training에서 accelerator 수를 늘릴수록 global batch가 커지기 쉽다. Batch를 계속 키워도 token efficiency와 stability가 유지되어야 cluster scale을 활용할 수 있다.

### 2) Learning rate를 그대로 복사하면 공정한 비교가 아니다

AdamW, Muon, SOAP은 같은 learning rate를 사용해도 parameter update RMS가 다르다. Muon은 momentum matrix를 orthogonalize하고, SOAP은 eigenbasis에서 adaptive scaling을 수행한다. Raw learning rate만 같다고 effective step size가 같지는 않다.

Optimizer A가 더 좋은 loss를 보였더라도 실제로는 update가 더 컸을 수 있다. 반대로 conservative learning rate 때문에 성능이 과소평가될 수도 있다.

### 3) SOAP의 eigenbasis freshness 문제는 large batch에서 커진다

SOAP은 gradient covariance의 eigenbasis로 momentum을 회전한다. Standard implementation은 eigenbasis와 covariance statistics를 일정 주기로 갱신한다. 이때 current gradient와 basis가 같은 시점의 geometry를 반영하지 않으면 preconditioner가 stale해진다.

Small batch에서는 gradient noise가 어느 정도 이를 가릴 수 있다. Large batch에서는 noise가 줄어들고 update가 더 deterministic해지므로 basis lag가 갑작스러운 overshoot나 loss spike로 나타날 수 있다.

### 4) MoE의 global batch는 expert batch가 아니다

Top-$k$ routing을 사용하는 MoE에서 global token batch를 $B_{\mathrm{global}}$, expert 수를 $N$, token당 active expert 수를 $k$라고 하자. Ideal load balance 아래에서 한 expert가 보는 effective batch는 대략 다음과 같다.

$$
B_{\mathrm{eff}}^{\mathrm{expert}}
=
B_{\mathrm{global}}
\frac{k}{N}
$$

Fine-grained MoE에서는 $k \ll N$이다. 따라서 global batch가 매우 커져도 개별 expert는 그 일부만 본다. 반대로 attention, embedding, shared expert 같은 dense parameter는 전체 global batch의 large-batch stress를 그대로 받는다.

즉 optimizer stability는 model 전체에서 균일한 문제가 아니다. Dense component가 먼저 critical batch를 넘을 수 있다.

### 5) Element-wise sharding은 matrix optimizer의 semantics를 깨뜨릴 수 있다

ZeRO-style sharding은 parameter와 optimizer state를 element 단위로 균등하게 나눈다. AdamW에는 잘 맞지만 Muon이나 SOAP은 full weight matrix가 필요하다.

각 rank가 자기 shard만 orthogonalize하면 full matrix의 polar factor와 달라진다. 작은 block으로 쪼개는 approximation은 systems cost를 낮추지만 optimizer 자체를 바꾼다. 이 논문은 convergence benefit을 유지하려면 matrix atomicity를 보존해야 한다고 본다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 네 축으로 볼 수 있다.

### 1) Update RMS matching을 통한 optimizer 비교

Optimizer별 update RMS를 맞춰 learning rate를 transfer한다. Weight matrix $W$의 update를 $\Delta W$라고 하면 비교 기준은 다음과 같다.

$$
\mathrm{RMS}(\Delta W)
=
\frac{\|\Delta W\|_F}{\sqrt{|W|}}
$$

SOAP은 orthogonal rotation 전후의 Frobenius norm이 같기 때문에 AdamW와 update scale을 연결하기 상대적으로 쉽다. Muon은 momentum correction과 matrix shape에 따라 scale이 달라지므로 별도 normalization이 필요하다.

Momentum coefficient가 $\beta_1$일 때 stationary momentum variance에 대한 correction은 다음 factor와 연결된다.

$$
c_{\mathrm{mom}}
=
\sqrt{
\frac{1-\beta_1}
{1+\beta_1}
}
$$

$\beta_1=0.9$이면 약 $0.2$다. 이 factor와 matrix dimension normalization을 사용해 Muon의 learning rate를 AdamW와 comparable한 update RMS로 맞춘다.

이 접근의 장점은 optimizer마다 대규모 learning rate sweep을 반복하는 비용을 줄이면서도, raw LR equality보다 공정한 비교를 제공한다는 점이다.

### 2) Muon, Shampoo, SOAP의 연결을 SVD 관점에서 정리

Muon은 momentum matrix $M_t$의 polar factor를 근사한다. SVD가 다음과 같다고 하자.

$$
M_t = U \Sigma V^\top
$$

Polar factor는 다음과 같다.

$$
\mathrm{Polar}(M_t) = U V^\top
$$

Muon은 Newton-Schulz iteration으로 이를 근사한다. 큰 singular value가 update를 지배하지 않도록 singular spectrum을 평평하게 만든다.

Shampoo에서 EMA를 끄고 current gradient만 사용하면 다음 관계가 성립한다.

$$
(GG^\top)^{-1/4}
G
(G^\top G)^{-1/4}
=
UV^\top
$$

즉 특정 limit에서 Shampoo update가 Muon의 polar update로 이어진다. SOAP도 covariance eigenbasis 안에서 Adam-like update를 수행하므로 세 optimizer는 완전히 별개의 heuristic이 아니라 matrix geometry를 다르게 근사하는 family로 볼 수 있다.

### 3) SOAP의 slingshot instability 진단과 수정

논문은 SOAP loss spike를 다음 조합으로 설명한다.

1. Covariance factor와 eigenbasis가 이전 step statistics에 머문다.
2. Current gradient가 preconditioner update에 제때 포함되지 않는다.
3. Momentum이 stale basis에서 회전되고 adaptive scaling된다.
4. Large batch에서 잘못 정렬된 update가 큰 loss jump를 만든다.

해결은 단순히 QR을 더 자주 수행하는 것만이 아니다.

- Current gradient를 covariance estimate에 포함한다.
- Eigenbasis를 매 step 갱신하거나 real-time에 가깝게 유지한다.
- Per-step QR orthogonalization으로 basis quality를 보존한다.
- KL-divergence 관점의 covariance estimator를 사용해 condition number를 낮춘다.

논문은 이 조합을 통해 large-batch SOAP의 spike를 제거하고 stability를 높인다.

### 4) Layer-wise distributed optimizer

Matrix optimizer의 full-layer semantics를 유지하기 위해 layer 단위로 optimizer ownership을 DP rank에 배정한다.

- 각 parameter matrix는 한 optimizer rank에 온전히 존재한다.
- Rank마다 맡는 layer의 총 optimizer cost가 비슷하도록 load balance한다.
- Forward 직전에 필요한 updated parameter를 all-gather한다.
- Communication을 computation과 overlap한다.
- Tensor parallel matrix는 duplicated mode 또는 distributed mode로 full-layer statistics를 복원한다.

이 설계는 element shard보다 coarse하지만 matrix operation을 approximation 없이 수행할 수 있다.

## 2-2. Design intuition

논문의 설계 직관은 다음과 같다.

### 1) Optimizer quality와 update scale을 분리해야 한다

Optimizer comparison에서 중요한 것은 동일 LR이 아니라 동일한 effective movement다. Update RMS를 먼저 맞추면 이후 loss 차이를 geometry와 conditioning의 차이로 해석하기 쉬워진다.

### 2) Large batch는 optimizer의 numerical geometry를 드러낸다

Batch가 커질수록 gradient noise가 줄어든다. 이때 stale preconditioner, singular direction imbalance, inadequate epsilon 같은 문제가 더 직접적으로 loss curve에 나타난다.

Muon과 SOAP이 large batch에서 강하다는 주장은 단순히 더 큰 step을 쓴다는 뜻이 아니다. Matrix structure를 이용해 update direction을 정리하므로 batch noise 감소 이후에도 안정적인 movement를 유지한다는 해석이다.

### 3) MoE에서는 dense parameter가 large-batch bottleneck이 된다

Expert는 sparse routing 때문에 effective batch가 작다. Global batch 확대의 압력은 attention과 shared layer에 더 크게 걸린다. 따라서 optimizer를 expert와 dense parameter에 동일하게 적용하더라도 실제 benefit은 component별로 다를 수 있다.

### 4) Algorithm을 살리려면 systems layout도 바뀌어야 한다

Muon과 SOAP의 핵심은 matrix-wide operation이다. Parameter를 element-wise로 분할한 뒤 local approximation을 쓰면 algorithmic benefit 자체가 약해질 수 있다.

그래서 논문은 optimizer state sharding을 matrix structure에 맞춰 재설계한다. Optimization algorithm과 distributed system을 별도로 최적화하지 않고 co-design하는 접근이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Muon과 SOAP을 multi-billion-parameter, trillion-token pretraining으로 확장 |
| Baseline | AdamW |
| Matrix optimizers | Muon, SOAP, KL-SOAP, related variants |
| Fair comparison | Update RMS matching |
| Main stress test | Large global batch up to 100M tokens |
| Architectures | 8B dense GPT, 30B total / 3B active MoE, 72B total / 8B active hybrid MoE |
| Stability fix | Current-gradient-aware basis update, per-step QR, improved covariance |
| Distributed system | Megatron-LM-compatible layer-wise distributed optimizer |
| Open artifact | Emerging-Optimizers codebase |

## 3-2. Module breakdown

### 1) AdamW: coordinate-wise adaptation

AdamW update는 각 coordinate의 variance로 first moment를 normalize한다. Implementation은 단순하지만 matrix correlation을 사용하지 않는다.

Matrix $W$가 attention projection인지, MLP projection인지, expert FFN인지와 관계없이 같은 element-wise rule을 적용한다. 이 universality가 강점이지만, structured layer에서 available한 geometry를 버린다.

### 2) Muon: spectral update through polar factor

Muon은 momentum matrix를 만든 뒤 Newton-Schulz iteration으로 polar factor를 근사한다.

개념적으로는 singular value를 모두 비슷한 scale로 만들어 update가 특정 direction에 과도하게 집중되는 것을 막는다. Matrix가 rectangular해도 closest semi-orthogonal factor를 사용한다.

논문 recipe는 다음 요소를 사용한다.

- Momentum coefficient $\beta_1=0.9$
- Weight decay $0.1$
- PolarExpress 계열 Newton-Schulz coefficients
- 16 Newton-Schulz iterations
- Numerical epsilon $10^{-7}$
- Fused QKV tensor는 필요한 경우 logical matrix 단위로 분리한 뒤 orthogonalization

Iteration 수가 적으면 approximation error가 남고, 많으면 optimizer compute가 커진다. 논문은 orthogonalization quality도 별도로 측정해 large-scale conclusion이 지나치게 거친 approximation에 의존하지 않는지 확인한다.

### 3) SOAP: covariance eigenbasis plus Adam

SOAP은 row and column covariance의 eigenvector를 $Q_L$, $Q_R$라고 할 때 momentum을 해당 basis로 회전한다.

$$
\tilde{M}_t
=
Q_L^\top M_t Q_R
$$

그 basis에서 element-wise Adam update를 수행한 뒤 다시 원래 space로 돌린다.

$$
U_t
=
Q_L
\mathrm{Adam}(\tilde{M}_t)
Q_R^\top
$$

회전이 orthogonal하면 Frobenius norm은 보존된다. 하지만 covariance와 eigenbasis를 유지해야 하므로 memory와 numerical cost가 크다.

### 4) Real-time eigenbasis update

Standard SOAP의 computational shortcut은 eigenbasis를 여러 step 동안 재사용하는 것이다. 논문은 large batch에서 이 staleness가 위험하다고 본다.

수정된 path는 다음과 같다.

1. Current gradient로 covariance statistics를 먼저 업데이트한다.
2. Updated factor에 대해 basis를 refresh한다.
3. QR orthogonalization으로 basis drift를 교정한다.
4. 같은 step의 momentum을 current basis에서 precondition한다.

핵심은 preconditioner와 gradient의 temporal alignment다.

### 5) KL-Shampoo covariance estimator

Sample covariance를 직접 누적하면 condition number가 커지고 inverse root가 불안정해질 수 있다. KL-Shampoo은 Gaussian approximation 사이의 KL objective에서 covariance estimator를 유도해 inversion과 decomposition에 더 유리한 conditioning을 얻는다.

논문에서 KL-SOAP은 stability뿐 아니라 controlled comparison에서도 Muon보다 약간 나은 cross-entropy loss를 보인다. 다만 epsilon, architecture, budget에 따라 ordering이 달라질 수 있어 절대적 승자로 해석하면 안 된다.

### 6) Tensor-parallel full-matrix mode

Tensor parallelism이 weight matrix를 나누면 Muon과 SOAP은 full matrix operation을 수행하기 어렵다. 논문은 두 mode를 제공한다.

| Mode | Operation | Suitable regime |
| --- | --- | --- |
| Duplicated | Full weight를 TP rank마다 all-gather한 뒤 각 rank가 같은 matrix operation 수행 | Smaller layer, communication-limited |
| Distributed | Newton-Schulz 중 intermediate matrix multiplication 결과를 collective communication으로 합침 | Larger layer, compute-limited |

두 mode 모두 full-layer normalization statistics를 사용해 non-TP result와 mathematically equivalent한 update를 목표로 한다.

### 7) Layer-wise optimizer ownership

Data parallel rank별로 layer ownership을 나눈다.

예를 들어 DP rank가 8개라면 parameter element를 8등분하는 대신 layer 또는 matrix 단위 job을 8개 rank에 배치한다. 각 rank는 맡은 matrix의 full optimizer state를 가진다.

Load balancing은 parameter count만 보면 부족하다. Muon의 Newton-Schulz cost와 SOAP의 factor size는 matrix shape에 따라 달라진다. 따라서 layer-wise scheduler는 memory와 compute를 함께 고려해야 한다.

# 4. Training / Data / Recipe

## 4-1. Model and data

논문은 다음 scale에서 optimizer를 평가한다.

| Model | Total / active scale | Architecture |
| --- | --- | --- |
| 8B Dense GPT | 8B active | Dense Transformer |
| Nemotron-3-Nano-30B-A3B | 30B total / 3B active | Transformer MoE |
| Nemotron-3-72B-A8B | 72B total / 8B active | Hybrid Mamba-Transformer MoE |
| Qwen3-30B-A3B | 30B total / 3B active | Controlled optimizer comparison |

Pretraining data는 Nemotron-3 dataset의 1T-token 및 3T-token subset을 사용한다.

MoE setting은 model마다 expert 수와 active expert 수가 다르다. 예를 들어 72B hybrid MoE는 512 total experts와 top-6 routing을 사용한다. 이런 fine-grained routing은 expert당 effective batch를 크게 낮춘다.

## 4-2. Baseline training recipe

공통 baseline은 다음과 같다.

- Global batch: 3072 sequences
- Sequence length: 8192
- Token batch: 약 25M tokens
- Micro-batch size: 1
- Weight decay: 0.1
- Schedule: Warmup-Stable-Decay
- Final decay: inverse-square-root style
- MoE load balance: sigmoid score와 sequence-level auxiliary loss
- Auxiliary loss coefficient: $10^{-4}$

Batch scaling experiment에서는 global token batch를 최대 100M tokens까지 확장한다. Learning rate는 square-root scaling rule을 기반으로 조정한다.

$$
\eta'
=
\eta
\sqrt{
\frac{B'}{B}
}
$$

이 rule은 batch가 커질 때 update noise variance를 비슷하게 유지하려는 heuristic이다. Linear scaling보다 large jump에서 conservative하다.

## 4-3. Optimizer comparison protocol

공정한 비교를 위해 다음 절차가 중요하다.

1. AdamW baseline의 stable learning rate를 잡는다.
2. 각 optimizer의 raw update를 같은 checkpoint 또는 warmup regime에서 측정한다.
3. Parameter update RMS가 비슷하도록 LR multiplier를 구한다.
4. Momentum correction과 matrix-shape normalization을 반영한다.
5. 같은 data, model, schedule, weight decay 아래에서 long run을 수행한다.

이 방식은 모든 optimizer에 동일한 hyperparameter search budget을 쓰는 완전한 grid search는 아니다. 대신 frontier-scale에서 practical하게 transfer 가능한 comparison protocol이다.

## 4-4. Engineering notes

### 1) Epsilon은 작은 구현 세부가 아니다

SOAP과 Muon의 numerical floor는 inversion, normalization, Newton-Schulz convergence에 영향을 준다. 논문도 epsilon tuning을 conclusion의 limitation으로 언급한다.

같은 algorithm 이름이라도 epsilon, precision, factor update frequency가 다르면 결과가 달라질 수 있다.

### 2) Matrix definition을 명확히 해야 한다

Fused QKV, grouped linear layer, convolution filter, expert stack을 어떤 2D matrix로 볼 것인지가 Muon update를 결정한다.

예를 들어 fused QKV tensor 전체를 하나로 orthogonalize하면 query, key, value projection 사이를 하나의 operator로 묶는다. 논문은 필요한 경우 이를 분리해 optimization semantics를 보존한다.

### 3) Optimizer compute는 training step과 overlap해야 한다

Layer-wise optimizer가 이론적으로 memory를 분산하더라도 all-gather가 critical path에 그대로 노출되면 throughput advantage가 사라진다.

실제 system에서는 다음이 필요하다.

- Layer execution order를 아는 prefetch
- Forward 직전 updated weight all-gather
- Optimizer compute와 backward 또는 communication overlap
- Variable matrix size에 대한 load balancing
- Contiguous buffer와 aligned memory layout

# 5. Evaluation

## 5-1. Main results

### 1) Muon과 SOAP은 large batch에서 AdamW보다 안정적이다

논문은 8B dense와 30B 및 72B MoE에서 global batch를 늘려 비교한다. AdamW는 critical batch를 넘으면 token-efficient convergence가 악화되거나 instability가 나타난다.

Muon과 SOAP은 tested scale에서 최대 100M-token batch까지 quality와 stability를 유지한다. 중요한 점은 accelerator utilization만 좋아진 것이 아니라 같은 token budget에서 loss degradation이 상대적으로 작다는 점이다.

### 2) SOAP의 standard implementation은 large-batch spike를 보인다

Standard SOAP은 특정 point에서 loss가 갑자기 튀는 slingshot behavior를 보인다. Per-step QR만 적용하면 basis orthogonality는 좋아지지만 문제를 완전히 제거하지 못한다.

Current gradient를 covariance update에 포함하고 basis를 real-time에 가깝게 refresh해야 spike가 사라진다. KL-style covariance estimator는 conditioning을 추가로 개선한다.

### 3) KL-SOAP과 Muon의 차이는 작지만 일관된 경향이 있다

Controlled Qwen3-30B-A3B experiment에서 KL-SOAP은 cross-entropy loss에서 Muon보다 약간 앞서는 경향을 보인다. 반면 exact-polar MOP 계열은 Muon보다 약간 나은 result를 보이는 setting도 있다.

따라서 conclusion은 "SOAP이 항상 Muon보다 우월하다"가 아니다.

- Muon은 state와 implementation이 더 단순하다.
- SOAP은 더 expressive한 adaptive preconditioning을 제공한다.
- SOAP은 factor memory, basis freshness, epsilon sensitivity가 더 크다.
- 실제 choice는 convergence gain과 systems cost를 함께 봐야 한다.

### 4) Downstream gain은 task마다 균일하지 않다

Pretraining loss improvement가 모든 downstream benchmark로 같은 비율로 전이되지는 않는다. Coding과 commonsense 쪽에서 강한 gain이 보이는 반면, 일부 math evaluation은 regression 또는 작은 차이를 보인다.

Optimizer 평가를 validation loss 하나로 끝내면 안 되는 이유다. 같은 loss라도 learned representation과 task transfer가 다를 수 있다.

### 5) Layer-wise distributed implementation이 full semantics를 보존한다

System contribution은 matrix를 작은 block으로 근사하지 않고 full-layer update를 유지한다는 데 있다. Memory를 DP rank 사이에 분배하고 communication을 숨겨 multi-billion model에서도 optimizer를 실행한다.

다만 정확한 wall-clock benefit은 layer shape, TP degree, accelerator, network topology에 따라 달라진다. 공개 code와 target cluster에서 profiling이 필요하다.

## 5-2. What really matters in the experiments

이 논문에서 가장 중요한 실험 해석은 다섯 가지다.

### 1) Large batch가 목적이 아니라 scaling flexibility가 목적이다

100M-token batch가 항상 최적이라는 주장이 아니다. 중요한 것은 optimizer가 critical batch를 더 늦게 만나도록 만들어 cluster scaling option을 넓힌다는 점이다.

### 2) Update RMS matching은 comparison의 출발점이지 완전한 tuning이 아니다

RMS가 같아도 direction distribution, layer-wise scale, weight decay interaction은 다르다. 이 protocol은 unfair scale advantage를 줄이지만 각 optimizer의 global optimum을 보장하지는 않는다.

### 3) SOAP stability gain은 algorithm과 schedule을 함께 바꾼 결과다

Per-step basis refresh, current gradient inclusion, QR, KL covariance가 결합된다. 어느 component가 어느 scale에서 필수인지 ablation을 구분해 읽어야 한다.

### 4) MoE 전체 평균만 보면 component별 bottleneck을 놓친다

Expert parameter는 smaller effective batch를 보고 dense component는 full batch를 본다. Layer-wise update RMS와 instability location을 함께 분석해야 실제 mechanism을 이해할 수 있다.

### 5) System overhead를 포함한 time-to-quality가 최종 지표다

Tokenizer throughput이나 loss-per-token만으로 optimizer를 고르면 안 된다. Matrix optimizer는 step당 compute가 더 크다. 최종적으로는 target validation quality까지 걸리는 wall-clock, energy, memory headroom을 봐야 한다.

# 6. Limitations

1. Optimizer별 exhaustive hyperparameter search가 아니다.
   - Update RMS matching은 fair transfer에 유용하지만 각 method의 best learning rate, epsilon, momentum, factor schedule을 완전히 찾지는 않는다.
   - 작은 performance gap은 tuning budget에 따라 뒤집힐 수 있다.

2. Tested architecture와 data distribution에 conclusion이 묶인다.
   - Dense Transformer, Transformer MoE, hybrid Mamba-Transformer MoE를 포함하지만 모든 architecture를 대표하지는 않는다.
   - Multimodal pretraining, encoder-only model, diffusion model에서 같은 ordering이 유지되는지는 별도 검증이 필요하다.

3. SOAP의 memory와 numerical complexity가 여전히 크다.
   - Real-time eigenbasis update와 full covariance factor는 stability를 높이지만 compute와 memory를 늘린다.
   - Large expert count에서 factor 관리가 operationally 복잡하다.

4. Muon의 matrix partition rule이 model design과 결합된다.
   - Fused QKV, MLA projection, convolution, low-rank adapter를 어떤 matrix로 취급하는지에 따라 update가 달라진다.
   - Full-rank 2D weight에 가장 자연스럽고, LoRA처럼 low-rank parameterization에서는 별도 설계가 필요하다.

5. Large-batch scaling law를 완전히 도출하지 않는다.
   - Square-root LR rule과 empirical batch ramp를 사용하지만 optimizer-specific critical batch를 예측하는 closed-form law는 제시하지 않는다.

6. Downstream effect가 일관되지 않다.
   - Pretraining loss가 좋아도 일부 math task처럼 gain이 작거나 regression이 있는 영역이 남는다.
   - 최종 model selection에는 broad evaluation이 필요하다.

7. System result는 hardware topology에 민감하다.
   - Layer-wise ownership, duplicated TP mode, distributed TP mode의 break-even point는 GPU memory, interconnect, matrix shape에 따라 달라진다.
   - 다른 cluster에서 같은 throughput을 기대하려면 profiling이 필요하다.

8. Precision floor와 orthogonalization quality가 scale에 따라 달라질 수 있다.
   - BF16, FP32 accumulator, Newton-Schulz iteration 수가 polar approximation과 stability에 영향을 준다.
   - 더 큰 model과 더 긴 run에서는 numerical drift를 계속 관찰해야 한다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 큰 가치는 Muon과 SOAP의 ranking보다 optimizer를 production-scale co-design 문제로 바꿨다는 데 있다.

최근 optimizer 논의는 종종 "AdamW보다 loss가 몇 퍼센트 낮다"는 curve 비교로 끝난다. 실제 large model training에서는 다음 질문이 더 중요하다.

- Update matrix를 어떤 logical operator로 나눌 것인가.
- TP shard 상태에서 full-matrix semantics를 어떻게 복원할 것인가.
- Optimizer state와 compute를 어디에 배치할 것인가.
- Batch를 키울 때 dense layer와 expert layer가 각각 어떤 regime으로 이동하는가.
- Convergence gain이 extra optimizer time을 상쇄하는가.

이 논문은 algorithmic gain을 주장하면서도 그 gain을 유지하기 위한 memory layout과 communication path를 함께 제시한다. LLM training framework를 설계하는 입장에서는 이 연결이 특히 실용적이다.

## 7-2. Reuse potential

### 1) Small-scale optimizer bake-off

바로 72B experiment를 재현할 필요는 없다. 동일 model checkpoint와 fixed token budget에서 다음 protocol을 만들 수 있다.

1. AdamW baseline을 충분히 튜닝한다.
2. Muon, SOAP의 update RMS를 layer별로 측정한다.
3. Global RMS뿐 아니라 attention, MLP, embedding, expert group별 scale을 비교한다.
4. Same wall-clock 또는 same token budget으로 validation loss를 본다.
5. Downstream task를 최소한 math, code, commonsense로 나눈다.

### 2) Large-batch stability sweep

Single GPU 또는 small cluster에서도 gradient accumulation을 사용해 effective batch를 키울 수 있다.

- Batch doubling
- Square-root LR scaling
- Fixed token budget
- Loss spike, gradient norm, update RMS, layer-wise singular value 기록

이를 통해 optimizer가 어느 batch에서 degrade하는지 local scaling map을 만들 수 있다.

### 3) MoE component-aware optimization

Dense component와 expert component에 같은 optimizer를 강제할 필요는 없다.

- Attention and shared layer: Muon 또는 SOAP
- Sparse expert: AdamW 또는 lower-cost variant
- Embedding and norm: AdamW

이런 hybrid optimizer는 algorithmic benefit과 systems overhead를 절충할 수 있다. 다만 weight decay와 global update scale을 group별로 다시 맞춰야 한다.

### 4) Optimizer observability

Training log에 다음 항목을 추가하면 numerical issue를 조기에 찾을 수 있다.

- Layer-wise update RMS
- Momentum singular value distribution
- Newton-Schulz residual
- SOAP factor condition number
- Basis update age
- Current-gradient inclusion 여부
- Optimizer step wall-clock과 communication overlap

### 5) Layer-wise ownership scheduler

PyTorch FSDP나 custom distributed trainer에서도 parameter element가 아니라 logical matrix job을 rank에 배치하는 scheduler를 실험할 수 있다. Matrix size뿐 아니라 expected orthogonalization FLOPs와 state memory를 cost로 사용해야 한다.

## 7-3. Follow-up papers

- Muon: An Optimizer for Hidden Layers in Neural Networks
- SOAP: Improving and Stabilizing Shampoo using Adam
- Shampoo: Preconditioned Stochastic Tensor Optimization
- Understanding and Improving Shampoo's Preconditioner via KL Divergence
- Scion: Scaling Deep Learning through Spectral Descent
- veScale-FSDP: Flexible Sharding for Structure-Aware Optimizers
- Canzona: Asynchronous Systems for Matrix Optimizers
- Nemotron-3 Technical Report

# 8. Summary

- 이 논문은 Muon과 SOAP을 multi-billion-parameter, trillion-token pretraining과 최대 100M-token batch에서 검증한다.
- Update RMS matching으로 optimizer마다 다른 effective step scale을 통제하고 AdamW와 비교한다.
- SOAP의 large-batch loss spike를 stale eigenbasis와 current-gradient lag 문제로 진단하고 real-time basis update, QR, KL covariance로 안정화한다.
- Fine-grained MoE에서는 expert effective batch와 dense global batch가 다르므로 optimizer bottleneck도 component별로 달라진다.
- Layer-wise distributed optimizer는 full matrix semantics를 유지하면서 Megatron-LM scale에서 matrix optimizer를 실행한다.
- Practical choice는 validation loss뿐 아니라 epsilon sensitivity, optimizer memory, communication, time-to-quality를 함께 봐야 한다.
