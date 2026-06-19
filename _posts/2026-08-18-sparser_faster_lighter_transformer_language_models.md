---
layout: single
title: "Sparser, Faster, Lighter Transformer Language Models Review"
categories: Study-concept
tag: [LLM, ModelSparsity, CUDA, EfficientTraining, EfficientInference]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2603.23198)

[Code link](https://github.com/SakanaAI/sparser-faster-llms)

[Official blog](https://pub.sakana.ai/sparser-faster-llms/)

[Checkpoints](https://huggingface.co/collections/SakanaAI/sparser-faster-lighter-transformers)

Sparser, Faster, Lighter Transformer Language Models는 "activation을 99% 이상 sparse하게 만들었다"는 숫자만 보고 읽으면 핵심을 놓치기 쉬운 논문이다. Sparse activation이 많다고 GPU가 자동으로 빨라지지는 않는다. 오히려 index를 만들고, row를 정렬하고, irregular memory access를 처리하는 비용 때문에 theoretical FLOPs가 줄어도 wall-clock latency는 더 나빠질 수 있다.

이 논문이 실제로 겨냥하는 문제는 sparsity 자체가 아니다. 더 정확히는, **unstructured activation sparsity를 modern GPU의 tiled execution, memory hierarchy, kernel fusion과 맞물리게 설계해 실제 throughput, energy, memory benefit으로 바꿀 수 있는가**다.

저자들은 이 문제를 세 층에서 동시에 푼다. Model 쪽에서는 gated FFN의 gate activation에 ReLU와 mild L1 regularization을 적용해 높은 sparsity를 유도한다. Data format 쪽에서는 기존 ELLPACK의 row-wide packing을 tile-local packing으로 바꾼 TwELL을 제안한다. Kernel 쪽에서는 TwELL construction을 gate projection의 epilogue에 넣고, inference에서는 up projection과 down projection을 fuse하며, training에서는 non-uniform sparsity를 견디는 hybrid sparse format을 사용한다.

이 세 요소를 분리해서 보면 평범해 보일 수 있다. ReLU와 L1은 오래된 도구고, sparse matrix format도 오래된 주제며, fused CUDA kernel도 새로운 발상만은 아니다. 하지만 이 논문의 기여는 이들을 하나의 execution path로 묶어, sparsity를 "model statistic"이 아니라 **end-to-end system property**로 만든 데 있다.

> 한 줄 요약: 이 논문은 ReLU plus L1로 gated FFN activation을 99% 이상 sparse하게 만들고, TwELL과 hybrid sparse format, custom CUDA kernel을 함께 설계해 H100에서 최대 20.5% forward throughput gain과 21.9% training throughput gain을 실측한 hardware-aware sparse LLM 연구다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Sparse FLOPs와 실제 GPU speedup 사이의 간극을 data layout과 kernel pipeline 관점에서 정면으로 다룬다.
- Model objective, sparse representation, inference fusion, backward storage를 하나의 co-design 문제로 본다.
- 0.5B에서 2B까지 같은 recipe를 확장하면서 scale이 커질수록 sparsity와 speedup이 어떻게 변하는지 보여준다.
- Throughput뿐 아니라 energy per token과 peak memory를 함께 측정해 deployment와 training 관점을 동시에 제공한다.
- MoE나 structured pruning과 달리, token마다 달라지는 unstructured FFN activation을 직접 활용하는 다른 conditional computation 경로를 보여준다.

이 논문은 sparse model paper이면서 동시에 GPU systems paper다. 따라서 downstream accuracy만 보거나 sparsity ratio만 보면 부족하다. TwELL이 왜 필요한지, training에서 왜 inference format을 그대로 쓰지 않는지, 어떤 workload에서 speedup이 커지는지를 함께 읽어야 한다.

# 1. Problem Setting

## 1-1. Problem definition

Modern Transformer의 gated feed-forward block은 보통 gate, up, down projection으로 구성된다. Input batch를 $x$, model dimension을 $K$, expanded FFN dimension을 $N$이라고 하면 계산은 다음과 같다.

$$
h_u = xW_u, \quad h_g = \operatorname{ReLU}(xW_g), \quad h = h_u \odot h_g, \quad y = hW_d
$$

여기서 $W_g$와 $W_u$는 $K$에서 $N$으로 확장하고, $W_d$는 다시 $N$에서 $K$로 줄인다. 일반적으로 $N$은 $K$보다 훨씬 크다. 논문은 larger model에서 FFN이 전체 parameter의 3분의 2 이상, execution FLOPs의 80% 이상을 차지할 수 있다고 설명한다.

ReLU gate를 쓰면 $h_g$의 많은 원소가 정확히 0이 된다. 이때 같은 위치의 $h$도 0이므로, 이론적으로는 해당 hidden dimension에 대응하는 up projection column과 down projection row를 계산할 필요가 없다. Token마다 활성화된 neuron index만 처리할 수 있다면 FFN compute와 intermediate activation storage를 크게 줄일 수 있다.

문제는 이 sparsity가 **unstructured하고 input-dependent하다**는 점이다.

- 어떤 token은 수십 개 neuron만 활성화한다.
- 어떤 token은 평균보다 10배 이상 많은 neuron을 활성화한다.
- Layer마다 평균 non-zero count가 다르다.
- Sequence 초반과 후반의 activation pattern도 다르다.
- 같은 batch 안에서도 row별 sparse workload가 크게 달라진다.

즉 sparse pattern은 static mask가 아니며, token마다 runtime에 결정된다. Dense GEMM에 최적화된 GPU에서 이런 irregular workload를 효율적으로 처리하려면 단순히 zero를 skip하는 것보다 훨씬 많은 문제가 생긴다.

## 1-2. Why previous approaches are insufficient

기존 sparse computation이 실제 LLM pipeline에서 잘 확장되지 않는 이유는 크게 네 가지다.

첫째, **packing overhead**다. Sparse matrix multiplication을 하려면 non-zero value와 index를 먼저 compact representation으로 만들어야 한다. Dense gate output을 DRAM에 쓴 뒤 별도 kernel로 scan, count, pack하면, sparse compute에서 절약한 시간보다 conversion overhead가 커질 수 있다.

둘째, **row-wide alignment와 tiled matmul의 mismatch**다. ELLPACK은 각 row의 non-zero를 row 앞쪽에 모아 저장한다. 하지만 modern dense matmul kernel은 output을 2D tile로 나눠 여러 CTA가 독립적으로 계산한다. 한 row의 전체 non-zero를 정렬하려면 서로 다른 CTA 사이의 synchronization이나 별도 pass가 필요하다.

셋째, **non-uniform sparsity**다. ELL 계열 format은 흔히 row당 maximum non-zero count에 맞춰 padding한다. 평균은 매우 작아도 일부 token이 많은 neuron을 켜면 storage capacity를 크게 잡아야 한다. 그러면 padding waste가 커지고, capacity를 작게 잡으면 overflow가 발생한다.

넷째, **training은 inference보다 훨씬 긴 lifecycle을 본다**. Inference에서는 forward path를 fuse하는 것이 중요하다. 반면 training에서는 intermediate activation을 저장하고, backward에서 weight와 input gradient를 계산하며, optimizer state까지 유지해야 한다. Inference용 sparse layout을 그대로 쓰면 backward access pattern과 memory traffic이 오히려 비효율적일 수 있다.

그래서 이 논문의 질문은 다음처럼 정리할 수 있다.

> Sparse activation을 만드는 것뿐 아니라, 그 activation이 생성되는 순간부터 forward, storage, backward까지 sparse representation을 유지해 dense pipeline보다 실제로 빠르게 만들 수 있는가?

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 세 가지 축으로 압축할 수 있다.

1. **Sparse model recipe**
   - Gated FFN의 gate activation에 ReLU를 사용한다.
   - Cross-entropy loss에 activation L1 penalty를 추가한다.
   - Mild regularization으로 downstream accuracy를 거의 유지하면서 average active neuron 수를 크게 줄인다.

2. **TwELL for inference**
   - ELLPACK을 row-wide가 아니라 horizontal tile 단위로 local packing한다.
   - Gate projection을 계산하는 dense matmul의 epilogue에서 TwELL을 바로 만든다.
   - 별도 packing kernel과 extra DRAM round trip을 제거한다.
   - Sparse gate를 따라 필요한 up column과 down row만 읽어 두 projection을 하나의 kernel에서 fuse한다.

3. **Hybrid sparse format for training**
   - 대부분의 sparse row는 compact ELL-like storage에 넣는다.
   - Non-zero count가 capacity를 넘는 소수 row는 dense backup에 넣는다.
   - Hybrid-to-dense와 dense-to-hybrid kernel을 사용해 forward와 backward를 sparse path로 유지한다.
   - Intermediate activation storage와 backward compute를 함께 줄인다.

이 논문의 중요한 설계 선택은 inference와 training에 같은 format을 강요하지 않는다는 점이다. TwELL은 tiled gate projection과 fusion에 맞고, hybrid format은 training 전체 step의 memory와 backward access에 맞는다.

## 2-2. Design intuition

이 논문의 직관은 "zero를 저장하지 않는다"보다 더 구체적이다.

Dense pipeline은 대략 다음처럼 볼 수 있다.

$$
\text{dense gate} \to \text{dense up} \to \text{dense hidden} \to \text{dense down}
$$

Naive sparse pipeline은 다음처럼 되기 쉽다.

$$
\text{dense gate} \to \text{write} \to \text{scan and pack} \to \text{sparse matmul} \to \text{unpack}
$$

이 경우 arithmetic은 줄어도 kernel launch, synchronization, index traffic, global memory access가 추가된다. Modern GPU에서는 이 overhead가 매우 비싸다.

논문이 원하는 pipeline은 아래에 가깝다.

$$
\text{gate matmul plus TwELL epilogue} \to \text{fused sparse up and down}
$$

Training에서는 목적이 달라진다.

$$
\text{TwELL gate} \to \text{hybrid storage} \to \text{sparse forward} \to \text{sparse backward}
$$

즉 sparsity를 활용하는 핵심은 sparse matrix를 만든 뒤 빠르게 곱하는 것이 아니다. **Dense activation이 만들어지는 경계, sparse storage로 바뀌는 경계, 다음 operator가 소비하는 경계를 하나의 pipeline으로 설계하는 것**이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Gated FFN의 unstructured activation sparsity를 실제 GPU efficiency로 변환 |
| Sparse induction | ReLU gate plus activation L1 regularization |
| Inference format | TwELL, tile-wise ELLPACK |
| Training format | Compact sparse rows plus dense backup의 hybrid representation |
| Inference kernel | Gate projection과 TwELL construction fuse, sparse up and down projection fuse |
| Training kernel | Hybrid-to-dense, dense-to-hybrid, transpose, L1 gradient injection |
| Main hardware | NVIDIA H100 GPU |
| Main evaluation | 0.5B, 1B, 1.5B, 2B Transformer++ models |
| Main benefit | Throughput, energy per token, activation memory 개선 |
| Key distinction | Sparsity ratio가 아니라 end-to-end sparse execution path를 설계 |

## 3-2. Sparse training objective

논문은 standard cross-entropy loss에 layer-wise activation L1 penalty를 더한다.

$$
\mathcal{L}
= \mathcal{L}_{CE}
+ \lambda \frac{1}{L}
\sum_{l=1}^{L}
\frac{1}{MN}
\sum_{m=1}^{M}
\sum_{n=1}^{N}
\left| h^{l}_{m,n} \right|
$$

여기서 $L$은 FFN layer 수, $M$은 token을 합친 effective batch dimension, $N$은 expanded hidden dimension이다. $\lambda$는 sparsity strength를 조절한다.

ReLU는 negative pre-activation을 exact zero로 만든다. L1 penalty는 remaining positive activation의 magnitude도 줄여 더 많은 값이 0으로 밀리게 한다. 중요한 점은 별도 top-k router나 thresholding module을 넣지 않는다는 것이다. Sparse pattern은 gate activation에서 자연스럽게 결정된다.

논문은 1.5B model에서 unregularized ReLU model도 평균 911개의 non-zero를 보인다고 보고한다. FFN hidden dimension이 5632이므로 이미 상당한 natural sparsity가 있다. Mild L1을 적용하면 average non-zero가 수십 개 수준으로 떨어진다.

다만 ReLU 선택에는 trade-off가 있다. Appendix ablation에서 dense SiLU baseline은 dense ReLU baseline보다 mean task accuracy와 cross-entropy가 약간 좋다. 즉 sparse path의 efficiency는 activation function choice와 분리할 수 없다. ReLU가 sparse kernel에 유리하다고 해서 SiLU의 modeling advantage가 자동으로 사라지는 것은 아니다.

## 3-3. Why ELL is not enough

ELL은 sparse row마다 non-zero value와 original column index를 row 앞쪽에 모아 저장한다. Row별 capacity는 전체 row 중 maximum non-zero count에 맞춘다.

이를 단순화하면 다음과 같다.

| Component | Role |
| --- | --- |
| Value matrix | Row별 non-zero value 저장 |
| Index matrix | 각 value의 original column index 저장 |
| Count vector | Row별 실제 non-zero 개수 저장 |
| Padding | Maximum row length에 맞추기 위해 남는 slot 채움 |

ELL은 sparse row를 소비하는 kernel에는 편리하다. 하지만 gate projection output을 바로 ELL로 만들기는 어렵다. Dense matmul은 output tile마다 독립적으로 계산되는데, ELL은 row 전체에서 non-zero를 global하게 정렬해야 하기 때문이다.

별도 packing kernel을 두면 구현은 쉬워진다. 그러나 gate output을 DRAM에 한번 쓰고, 다시 읽어 scan하고, packed output을 다시 쓰는 memory traffic이 생긴다. 이 overhead가 sparse arithmetic saving을 상쇄할 수 있다.

## 3-4. TwELL: Tile-wise ELLPACK

TwELL은 ELL의 alignment 범위를 row 전체에서 horizontal tile로 줄인다. Column dimension을 size $T$의 tile로 나누고, 각 tile 안에서만 non-zero value와 index를 local하게 pack한다.

Dense gate output $h_g \in \mathbb{R}^{M \times N}$를 TwELL로 바꾸면 대략 세 component를 저장한다.

| Tensor | Description |
| --- | --- |
| $h_v$ | Tile-local packed non-zero values |
| $h_I$ | Original hidden index |
| $h_{nz}$ | Token-row와 tile별 non-zero count |

TwELL의 핵심은 tile size를 matmul output tile의 column size와 맞출 수 있다는 점이다. 각 CTA는 자신이 계산한 gate tile 안에서 ReLU를 적용하고, local count를 사용해 non-zero를 바로 pack한다. 다른 CTA가 계산한 같은 row의 결과를 기다릴 필요가 없다.

따라서 gate projection과 sparse materialization을 같은 kernel에 넣을 수 있다.

1. Dense input tile과 gate weight tile을 읽는다.
2. Tensor Core matmul로 gate pre-activation tile을 계산한다.
3. ReLU를 적용한다.
4. Warp-local count로 non-zero value와 index를 pack한다.
5. Dense gate 대신 TwELL output을 DRAM에 쓴다.

이 설계가 중요한 이유는 sparse conversion을 별도 preprocessing step이 아니라 **matmul epilogue**로 바꿨기 때문이다.

## 3-5. Fused sparse inference kernel

Gate activation이 TwELL로 만들어지면, 다음 kernel은 active hidden index만 순회한다. 각 active index $n$에 대해 up projection의 $n$번째 column과 down projection의 $n$번째 row만 읽는다.

Token row $m$의 output은 아래처럼 쓸 수 있다.

$$
y_m
= \sum_{n \in \mathcal{A}_m}
(h_g)_{m,n}
\left(x_m W_{u,:,n}\right)
W_{d,n,:}
$$

여기서 $\mathcal{A}_m$은 token $m$에서 활성화된 gate index set이다.

Dense implementation은 $N$개 hidden dimension을 모두 계산하고 $h_u$와 $h$를 materialize한다. Sparse fused kernel은 active index만 처리하고, scalar up activation을 kernel 내부에서 계산한 뒤 바로 down row에 곱해 output accumulator에 더한다.

이렇게 하면 다음 비용을 줄일 수 있다.

- Inactive up projection column compute
- Inactive down projection row compute
- Dense $h_u$ materialization
- Dense hidden activation materialization
- Extra kernel launch와 global memory traffic

논문은 single-warp CTA가 각 token row를 담당하도록 설계해 concurrency와 cache locality를 높인다. Token sequence 안에서 active index가 상관되는 경향이 있어 nearby CTA가 같은 weight region을 재사용할 가능성도 있다.

## 3-6. Hybrid format for training

Inference에서 TwELL은 fusion에 유리하지만, training activation을 장시간 저장하기에는 row별 non-zero unevenness가 문제다. Average non-zero count가 매우 작아도 일부 token은 수백 개 neuron을 켤 수 있다. TwELL capacity를 worst case에 맞추면 storage가 커지고, 작게 잡으면 overflow가 난다.

논문은 각 row를 두 경로로 나눈다.

1. **Sparse path**
   - Non-zero count가 threshold 안에 들어오는 대부분의 row를 compact ELL-like matrix에 저장한다.
   - Value와 index를 tight capacity 안에 넣는다.

2. **Dense backup path**
   - Sparse capacity를 넘는 소수 row를 dense matrix에 저장한다.
   - Outlier token이 전체 sparse storage capacity를 키우지 않게 한다.

Binary routing vector는 원래 row가 sparse storage와 dense backup 중 어디에 있는지 기록한다. Hybrid matmul kernel은 sparse row를 CUDA Core path에서, dense backup row를 tiled Tensor Core path에서 처리한다.

이 design은 average case를 aggressively compress하면서 worst case를 안전하게 처리한다. 중요한 것은 outlier를 제거하거나 clipping하지 않고 dense fallback으로 보존한다는 점이다.

## 3-7. Sparse backward pipeline

Training의 핵심 이득은 forward speedup만이 아니다. Activation을 hybrid form으로 저장하면 backward에서 dense hidden state를 다시 만들지 않고 gradient를 계산할 수 있다.

논문은 다음 operation을 위한 custom kernel을 구성한다.

- Hybrid-to-dense matmul
- Dense-to-hybrid matmul
- Hybrid sparse transpose
- L1 gradient injection
- Sparse activation pattern을 사용한 weight gradient 계산

Forward에서는 up와 down projection을 inference처럼 완전히 fuse하지 않는다. Training에서는 intermediate state가 backward에 필요하므로, separate step으로 계산하되 hybrid form을 유지한다. 이 선택은 operator-level fusion보다 full training step의 memory traffic과 recomputation을 줄이는 데 초점을 둔다.

이 부분이 논문의 systems contribution에서 중요하다. Inference는 kernel launch와 materialization을 줄이는 것이 핵심이고, training은 saved activation과 backward access를 줄이는 것이 핵심이다. 같은 sparsity를 쓰더라도 optimization target이 다르다.

# 4. Training / Data / Recipe

## 4-1. Model and data

논문은 Transformer++ 계열 architecture를 사용한다. Main scaling experiment의 설정은 다음과 같다.

| Model | Layers | Model dimension | FFN hidden | Training tokens | Steps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 8 | 2048 | 5632 | 10.49B | 10K |
| 1B | 18 | 2048 | 5632 | 20.97B | 20K |
| 1.5B | 28 | 2048 | 5632 | 31.46B | 30K |
| 2B | 38 | 2048 | 5632 | 41.94B | 40K |

공통 설정은 다음과 같다.

- Dataset: FineWeb
- Context length: 2048
- Global batch: 1,048,576 tokens per step
- Optimizer: AdamW
- Weight decay: 0.1
- Learning-rate schedule: cosine
- Warmup: 600 steps
- Precision: bf16
- Attention heads: 32
- KV heads: 32
- Vocabulary size: 49,152
- Tokenizer: GPT-2 tokenizer
- Hardware: 8 NVIDIA H100 PCIe GPUs

Sparse model은 dense baseline과 동일한 main optimization hyperparameter를 사용하고, activation L1 term만 추가한다. 이 점은 method가 별도 distillation, router balancing, pruning schedule 없이 기존 pretraining recipe에 비교적 작게 개입한다는 의미다.

## 4-2. Choosing the L1 coefficient

1.5B model에서는 $\lambda$를 0에서 $10^{-4}$까지 변화시키며 sparsity, cross-entropy, downstream accuracy, throughput, energy, memory를 비교한다.

핵심 관찰은 세 구간으로 나눌 수 있다.

1. **Natural sparsity regime**
   - ReLU만 사용해도 average non-zero가 911개다.
   - Dense SiLU처럼 모든 hidden element를 쓰는 model과 달리 exact zero가 자연스럽게 생긴다.

2. **Performance-preserving sparse regime**
   - Mild L1이 non-zero count를 수십 개 수준으로 줄인다.
   - 논문은 $\lambda = 2 \times 10^{-5}$를 conservative scaling setting으로 선택한다.
   - $3 \times 10^{-5}$까지는 task accuracy drop이 거의 보이지 않고 final cross-entropy increase도 unregularized baseline 대비 2% 안쪽이라고 설명한다.

3. **Over-regularized regime**
   - Active fraction이 0.5% 아래로 내려가면 performance degradation이 나타나기 시작한다.
   - 더 강한 sparsity는 dead neuron과 optimization instability 위험을 높인다.

Main scaling table은 $\lambda = 2 \times 10^{-5}$를 사용한다. 이 값은 maximum speedup을 노린 aggressive setting이 아니라 accuracy degradation을 피하려는 conservative point다.

## 4-3. Static capacity and overflow handling

Public appendix 기준 hybrid storage는 static capacity를 사용한다.

- Sparse row의 maximum stored non-zero: 128
- Dense backup capacity: token batch의 최대 1/8
- Capacity overflow 발생 시 CPU-side flag를 확인하고 더 큰 storage로 retry

이 설정은 benchmark model과 sparsity distribution에 맞춰 선택되었다. 다른 hidden size, context, domain, regularization strength에서는 capacity를 다시 조정해야 한다.

Static allocation은 dynamic memory management를 피한다는 장점이 있다. 반면 sparsity distribution이 바뀌면 overflow rate가 늘거나 memory를 과할당할 수 있다. 따라서 production 적용에서는 average sparsity보다 tail distribution을 monitoring해야 한다.

## 4-4. Public release status

Official repository는 다음 artifact를 공개한다.

- Sparse training code
- TwELL inference kernels
- 0.5B, 1B, 1.5B, 2B checkpoints
- Inference benchmark script
- Energy measurement helper
- Hydra training configurations

다만 repository roadmap 기준으로 **efficient TwELL training kernels는 아직 공개 완료 상태가 아니다**. Paper의 training speedup을 그대로 재현하려면 공개 repository의 current scope와 추후 release를 구분해서 봐야 한다.

Repository는 H100 GPU와 CUDA 12.8 이상 환경을 전제로 한다. 따라서 이 방법은 plain PyTorch model을 어떤 GPU에서나 바로 빨라지게 하는 drop-in optimization이 아니다.

# 5. Evaluation

## 5-1. Main scaling results

Main Table 1은 dense ReLU baseline과 $\lambda = 2 \times 10^{-5}$ sparse model을 비교한다.

| Scale | Mean accuracy dense | Mean accuracy sparse | Forward gain | Energy per token change | Training gain | Peak memory change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 40.4% | 40.4% | +17.0% | -11.8% | -1.5% | -19.2% |
| 1B | 44.6% | 44.7% | +18.1% | -14.6% | +7.1% | -25.5% |
| 1.5B | 46.4% | 46.2% | +18.8% | -15.0% | +11.6% | -28.1% |
| 2B | 49.1% | 48.8% | +20.5% | -17.0% | +21.9% | +22.3% |

이 표는 몇 가지 주의해서 읽어야 한다.

첫째, forward gain은 scale이 커질수록 17.0%에서 20.5%로 증가한다. 논문은 model scale이 커질수록 average non-zero count가 39에서 24로 줄어드는 관찰과 연결한다.

둘째, training gain은 0.5B에서 -1.5%로 오히려 느리다. Sparse pipeline에는 packing, index, hybrid routing 같은 fixed overhead가 있다. Model이 작을 때는 saved compute가 overhead를 이기지 못한다. 1B 이상에서 gain이 나타나고 2B에서 21.9%까지 커진다.

셋째, 2B의 peak memory가 +22.3%로 표시된 것은 sparse representation이 더 많은 activation memory를 썼다는 뜻으로 읽으면 안 된다. 논문은 memory saving으로 micro-batch size를 두 배로 키워 throughput을 높였기 때문에 absolute peak memory가 dense baseline보다 커졌다고 설명한다. 같은 micro-batch 조건의 direct memory reduction과 throughput-optimized run을 구분해야 한다.

넷째, mean task accuracy 차이는 작지만 완전히 0은 아니다. 1.5B와 2B sparse model은 각각 0.2 point와 0.3 point 낮다. 논문은 random deviation 범위로 해석하지만, 더 큰 model과 더 긴 training budget에서도 같은 trade-off가 유지되는지는 별도 검증이 필요하다.

## 5-2. ReLU, SiLU, and sparsity trade-off

Appendix ablation은 activation function과 sparse kernel의 관계를 보여준다.

| Variant | Mean accuracy | Cross-entropy | Average non-zero | Forward throughput | Energy per token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense ReLU | 46.4% | 2.255 | 911 | 117.1 tok/ms | 5.77 mJ |
| Dense SiLU | 47.1% | 2.240 | 5632 | 116.5 tok/ms | 5.82 mJ |
| Sparse ReLU | 46.2% | 2.297 | 29 | 138.0 tok/ms | 5.07 mJ |

Dense SiLU는 accuracy와 cross-entropy에서 가장 좋다. Sparse ReLU는 dense ReLU 대비 accuracy가 0.2 point 낮지만 forward throughput은 약 17.9% 높고 energy per token은 약 12.1% 낮다.

이 결과는 method를 "free speedup"으로 보면 안 된다는 점을 보여준다. 실제 comparison은 sparse kernel 대 dense kernel만이 아니라, ReLU sparse architecture 대 modern SiLU dense architecture의 trade-off까지 포함한다.

## 5-3. Sparsity is uneven but predictable

논문은 1.5B model의 $2^{20}$ input token에서 activation을 수집해 layer와 token별 pattern을 분석한다.

핵심 관찰은 다음과 같다.

- Layer별 average non-zero count는 크게 다르다.
- Row별 maximum은 average보다 10배 이상 클 수 있다.
- Layer average non-zero와 speedup은 매우 강한 negative correlation을 보인다.
- URL fragment, contraction 같은 low-context token은 적은 neuron을 쓰는 경향이 있다.
- Context-dependent word와 sequence 초반 token은 더 많은 neuron을 쓰는 경향이 있다.
- Sequence가 진행되면서 active neuron 수가 감소하는 pattern도 관찰된다.

이 분석은 sparsity가 무작위 noise가 아니라 token difficulty와 context state에 따라 capacity를 배분하는 dynamic compute처럼 동작할 가능성을 보여준다.

다만 이 해석은 correlation이다. More active neuron이 더 어려운 reasoning을 의미한다는 causal evidence는 아니다. Tokenization artifact, frequency, positional effect, domain distribution이 섞여 있을 수 있다.

## 5-4. What really matters in the experiments

### 1) Sparse ratio보다 packing boundary가 중요하다

같은 non-zero count라도 sparse format을 별도 kernel로 만들면 speedup이 사라질 수 있다. TwELL의 핵심은 representation 자체보다 gate matmul epilogue에서 바로 materialize할 수 있다는 점이다.

### 2) Training과 inference를 다른 optimization problem으로 본다

Inference는 operator fusion과 dense hidden materialization 제거가 중요하다. Training은 activation storage, backward compute, tail sparsity handling이 중요하다. 논문은 두 lifecycle에 같은 sparse format을 강요하지 않는다.

### 3) Scale이 fixed overhead를 amortize한다

0.5B training slowdown과 2B training speedup의 대비는 중요한 결과다. Sparse kernel은 언제나 빠른 것이 아니라, skipped compute가 packing과 control overhead를 충분히 넘어서는 regime에서만 유리하다.

### 4) Tail distribution이 average보다 중요하다

Average active neuron은 20개에서 40개 수준이어도 일부 token은 훨씬 많은 neuron을 사용한다. Training storage가 dense backup을 둔 이유다. Production tuning에서도 mean sparsity 하나만 보고 capacity를 정하면 위험하다.

### 5) Memory saving은 throughput operating point를 바꾼다

2B result는 같은 micro-batch에서 memory를 적게 쓰는 것보다, saved memory로 micro-batch를 두 배 키워 throughput을 높이는 방식으로 사용한다. System benefit은 단일 metric이 아니라 feasible batch size와 scheduler setting까지 바꾼다.

# 6. Limitations

1. **Hardware specificity가 크다**
   - Main kernel은 H100 GPU execution model에 맞춰 설계되었다.
   - Public repository도 H100과 CUDA 12.8 이상을 전제로 한다.
   - A100, B200, consumer GPU, AMD GPU에서 같은 speedup이 나오는지는 별도 kernel engineering이 필요하다.

2. **Model scale가 아직 제한적이다**
   - Main experiment는 0.5B에서 2B model이다.
   - 7B, 70B, MoE, very long context에서 같은 sparsity distribution과 kernel bottleneck이 유지되는지는 확인되지 않았다.

3. **Context length가 2048이다**
   - Long-context serving에서는 attention과 KV cache cost가 더 커져 FFN optimization의 end-to-end 비중이 달라질 수 있다.
   - 이 논문은 attention cost를 줄이지 않는다.

4. **From-scratch architecture choice에 가깝다**
   - ReLU plus L1 recipe는 pretraining 시점에 넣는다.
   - 대부분의 existing open LLM은 SiLU 계열이다.
   - Existing dense SiLU checkpoint를 low-cost로 sparse ReLU model로 바꾸는 path는 main result가 아니다.

5. **Accuracy trade-off가 완전히 사라진 것은 아니다**
   - Dense SiLU ablation은 dense ReLU보다 약간 좋은 quality를 보인다.
   - Main sparse model도 일부 scale에서 0.2에서 0.3 point 낮다.
   - 더 어려운 reasoning, coding, multilingual benchmark에서는 작은 gap이 커질 수 있다.

6. **Workload가 batched forward와 pretraining 중심이다**
   - Single-token decode, low-batch interactive serving, tensor parallel, pipeline parallel 환경의 end-to-end latency는 별도 분석이 필요하다.
   - Kernel-level input tokens/ms가 실제 request latency와 동일하지 않다.

7. **Hybrid capacity tuning이 필요하다**
   - Static sparse capacity와 dense backup size는 sparsity tail에 민감하다.
   - Domain shift나 regularization change로 overflow pattern이 바뀌면 retry overhead와 memory waste가 생길 수 있다.

8. **Dead neuron 문제가 남는다**
   - Recommended L1에서도 layer average로 30% 이상 neuron이 permanently inactive해질 수 있다고 appendix가 분석한다.
   - Targeted reinitialization은 preliminary result이며 broad validation이 부족하다.

9. **Public artifact와 paper result 사이에 release gap이 있다**
   - Inference kernel과 standard sparse training code는 공개되어 있다.
   - Repository roadmap 기준 efficient TwELL training kernel은 아직 공개되지 않았다.
   - Paper의 full training speedup reproduction은 current public artifact만으로 바로 가능하다고 단정하면 안 된다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 큰 가치는 99%라는 sparsity 숫자보다 **sparsity를 wall-clock property로 만드는 조건을 구체화했다는 점**에 있다.

Model research에서는 activation sparsity를 distribution statistic으로 측정하기 쉽다. 하지만 engineering 관점에서 중요한 질문은 다르다.

- Zero가 언제 생기는가.
- Zero pattern을 누가 기록하는가.
- Index를 어디에 저장하는가.
- 다음 operator가 sparse data를 어떤 access pattern으로 읽는가.
- Backward에서 dense state를 다시 만들어야 하는가.
- Outlier row가 전체 layout을 망치지 않는가.
- Saved memory를 smaller footprint로 쓸지 larger batch로 쓸지 어떻게 결정하는가.

이 논문은 이 질문을 모두 architecture와 kernel 사이의 interface로 가져온다. 그래서 sparse LLM을 algorithm paper가 아니라 model-system co-design으로 읽게 만든다.

특히 TwELL은 좋은 systems lesson을 준다. 최적화 대상 operation만 빨리 만드는 것보다, **producer와 consumer 사이의 representation boundary를 바꾸는 것**이 더 큰 이득을 줄 수 있다. Gate matmul output을 dense tensor로 확정한 뒤 sparse conversion을 붙이는 대신, epilogue에서 sparse format을 바로 만든다. 이 원칙은 sparsity 외에도 quantization, routing, KV compression, fused sampling 같은 영역에 그대로 적용할 수 있다.

## 7-2. Reuse potential

재사용 가치가 큰 포인트는 다음과 같다.

1. **Epilogue-time representation conversion**
   - Dense result를 쓴 뒤 다시 읽어 변환하지 않는다.
   - Producer kernel의 epilogue에서 consumer-friendly format을 바로 만든다.

2. **Fast path plus safe fallback**
   - Average case는 compact sparse storage로 처리한다.
   - Tail case는 dense backup으로 보존한다.
   - Outlier 때문에 전체 format을 worst case에 맞추지 않는다.

3. **Lifecycle-specific layout**
   - Inference와 training에 같은 layout을 강요하지 않는다.
   - 각 lifecycle의 bottleneck에 맞게 format과 fusion boundary를 다르게 둔다.

4. **Scale-aware benchmark**
   - 작은 model에서 overhead가 드러나는지 확인한다.
   - 큰 model에서 theoretical saving이 실제 speedup으로 커지는지 본다.

5. **Quality, throughput, energy, memory joint evaluation**
   - Sparsity ratio만 보고 성공을 선언하지 않는다.
   - Model quality와 system metric을 같은 table에서 비교한다.

실제 적용 순서는 아래처럼 가져가는 편이 현실적이다.

1. Target model의 layer별, token별 non-zero distribution을 측정한다.
2. Average뿐 아니라 p95, p99, maximum row activity를 본다.
3. Dense baseline의 FFN wall-clock share를 profiler로 확인한다.
4. Packing cost와 sparse compute saving을 따로 측정한다.
5. Batch size와 decode mode별 break-even point를 찾는다.
6. Accuracy regression을 SiLU dense baseline과도 비교한다.
7. Hardware generation별 kernel portability를 검증한다.

## 7-3. Follow-up papers

- [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://arxiv.org/abs/2310.17157)
- [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516)
- [Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters](https://arxiv.org/abs/2406.05955)
- [TEAL: Training-Free Activation Sparsity in Large Language Models](https://arxiv.org/abs/2408.14690)
- [CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models](https://arxiv.org/abs/2404.08763)

# 8. Summary

- ReLU plus mild L1 regularization은 gated FFN activation을 99% 이상 sparse하게 만들 수 있다.
- TwELL은 row-wide ELL packing을 tile-local packing으로 바꿔 gate matmul epilogue에서 sparse format을 바로 만든다.
- Inference kernel은 active hidden index만 따라 up와 down projection을 fuse하고, training kernel은 hybrid sparse format으로 activation storage와 backward compute를 줄인다.
- H100의 0.5B에서 2B experiment에서 scale이 커질수록 benefit이 증가하며, 2B에서 forward +20.5%, training +21.9%, energy per token -17.0%를 보고한다.
- 핵심 lesson은 높은 sparsity ratio가 아니라 model objective, data layout, kernel fusion, workload operating point를 함께 설계해야 실제 speedup이 나온다는 점이다.
