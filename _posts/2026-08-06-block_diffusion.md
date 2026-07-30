---
layout: single
title: "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models Review"
categories: Study-concept
tag: [DiffusionLM, LanguageModeling, SemiAutoregressive, KVCache, GenerativeModel]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2503.09573)

[Project page](https://m-arriola.com/bd3lms)

[Code link](https://github.com/kuleshov-group/bd3lms)

> 한 줄 요약: Block Diffusion은 sequence를 block 단위로 autoregressive하게 생성하되, 각 block 내부 token은 discrete denoising diffusion으로 병렬 복원하는 BD3-LM을 제안하고, variable-length generation, previous-block KV cache, parallel token sampling, gradient-variance-aware noise schedule을 하나의 training and inference recipe로 묶는다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Autoregressive LM과 diffusion LM을 서로 배타적인 두 architecture가 아니라 block size로 이어지는 연속적인 design space로 본다.
- Diffusion LM의 fixed-length generation과 KV cache 부재를 block-wise factorization으로 해결하려 한다.
- Block size가 작아질수록 likelihood와 length flexibility가 좋아지고, block size가 커질수록 parallel generation과 controllability 여지가 커지는 trade-off를 명시한다.
- 단순히 architecture를 제안하는 데 그치지 않고, block diffusion objective의 높은 gradient variance를 측정하고 줄이는 noise schedule을 설계한다.
- Vectorized training mask와 FlexAttention implementation까지 포함해 이론적 factorization을 실제 Transformer training recipe로 연결한다.

Text generation에서 autoregressive model은 매우 강한 baseline이다. 앞 token을 확정하고 다음 token 하나를 예측하는 causal factorization은 variable-length generation, streaming, KV caching에 자연스럽다. 하지만 token을 한 개씩 생성하므로 decoding parallelism이 제한된다.

Discrete diffusion language model은 반대쪽 장점을 가진다. 여러 masked token을 동시에 갱신할 수 있고, 양방향 context를 사용하며, 특정 token을 반복적으로 수정하는 generation path를 만들 수 있다. 반면 최신 diffusion LM은 다음 문제를 가진다.

- Training context에 맞춘 fixed-length vector를 생성하는 경우가 많다.
- 전체 sequence가 계속 바뀌므로 이전 step의 KV를 쉽게 재사용하기 어렵다.
- Likelihood와 sample quality가 강한 autoregressive baseline에 뒤처진다.
- 많은 denoising step이 필요하면 theoretical parallelism이 wall-clock speed로 이어지지 않는다.

Block Diffusion의 출발점은 이 둘 사이에 구조적인 중간 지점이 있다는 것이다.

- Sequence 전체는 left-to-right block 순서로 생성한다.
- 이미 생성된 block은 고정된 clean context가 된다.
- 현재 block 내부에서는 여러 token을 diffusion으로 함께 복원한다.
- 다음 block으로 넘어갈 때 이전 block의 KV를 cache한다.

Block size가 1이면 token-level autoregression에 가까워진다. Block size가 전체 context length이면 ordinary full-sequence diffusion에 가까워진다. 따라서 block size는 단순한 implementation parameter가 아니라 quality, parallelism, cacheability, controllability를 조절하는 핵심 axis다.

# 1. Problem Setting

## 1-1. Problem definition

길이 $L$인 token sequence를 다음처럼 둔다.

$$
\mathbf{x} = (x_1, x_2, \ldots, x_L)
$$

Autoregressive LM은 sequence probability를 token 단위로 factorize한다.

$$
\log p_\theta(\mathbf{x})
=
\sum_{i=1}^{L}
\log p_\theta(x_i \mid \mathbf{x}_{<i})
$$

이 방식의 장점은 명확하다.

- EOS가 나올 때까지 variable-length generation이 가능하다.
- 이전 token은 다시 바뀌지 않으므로 KV cache를 쓸 수 있다.
- Standard next-token objective가 안정적이다.
- Likelihood evaluation이 직접적이다.

대신 한 번에 생성하는 unit이 token 하나이므로 $L$개 token에 대략 $L$번의 sequential decision이 필요하다.

Full-sequence discrete diffusion은 clean sequence를 점차 mask하고, reverse process에서 masked token을 병렬 복원한다. 각 step에서 여러 token을 업데이트할 수 있지만, output length를 미리 정해야 하는 경우가 많고 sequence 전체 representation이 바뀌기 때문에 cache reuse가 어렵다.

Block Diffusion은 sequence를 $B$개의 block으로 나눈다.

$$
\mathbf{x}
=
(\mathbf{x}^{1}, \mathbf{x}^{2}, \ldots, \mathbf{x}^{B})
$$

각 block의 길이를 $L'$라고 하면, block-level probability는 다음처럼 factorize된다.

$$
\log p_\theta(\mathbf{x})
=
\sum_{b=1}^{B}
\log p_\theta(\mathbf{x}^{b} \mid \mathbf{x}^{<b})
$$

여기서 핵심은 각 conditional distribution

$$
p_\theta(\mathbf{x}^{b} \mid \mathbf{x}^{<b})
$$

을 autoregressive token decoder가 아니라 discrete denoising diffusion model로 구현한다는 점이다.

즉 dependency는 두 층으로 나뉜다.

- Inter-block dependency: autoregressive
- Intra-block dependency: diffusion and bidirectional

이 구조가 해결하려는 문제는 세 가지다.

1. Diffusion LM이 EOS 기반 variable-length generation을 할 수 있는가.
2. 이미 완료된 prefix block에 KV caching을 적용할 수 있는가.
3. Block 내부 parallelism을 유지하면서 likelihood gap을 줄일 수 있는가.

## 1-2. Why previous approaches are insufficient

### 1) Full diffusion의 fixed-length output

많은 diffusion LM은 length $L$의 masked vector에서 시작해 같은 length의 clean vector를 복원한다. 실제 chat response나 document generation에서는 output length를 미리 알기 어렵다.

Length predictor를 별도로 둘 수 있지만, length error가 generation constraint가 된다. Block Diffusion은 block을 하나씩 생성하다가 현재 block에서 EOS가 나오면 멈추므로 autoregressive model과 비슷한 variable-length interface를 만든다.

### 2) Full diffusion의 KV cache 부재

Autoregressive inference에서는 이전 prefix token이 변하지 않는다. 따라서 각 layer의 key와 value를 cache하고 새 token의 computation만 추가한다.

Full diffusion에서는 denoising step마다 많은 token representation이 바뀐다. 이전 step의 KV를 그대로 재사용하기 어렵다. Block Diffusion에서는 완료된 이전 block은 더 이상 바뀌지 않으므로 그 부분의 KV를 cache할 수 있다. Current block만 반복 갱신한다.

### 3) Pure AR의 낮은 token-level parallelism

Autoregressive model은 다음 token이 이전 token에 의존한다. Speculative decoding이나 multi-token prediction을 쓰지 않으면 한 step에 한 token을 확정한다.

Block Diffusion은 현재 block 안의 여러 masked position을 같은 denoising step에서 업데이트할 수 있다. Sequential unit을 token에서 block으로 키운다.

### 4) Semi-autoregressive model의 quality gap

Block 단위 병렬 생성 자체는 새로운 아이디어가 아니다. 문제는 block 안 token dependency를 얼마나 잘 모델링하느냐다. 한 번의 independent prediction으로 block 전체를 생성하면 intra-block dependency가 약해질 수 있다.

BD3-LM은 block 내부에 iterative denoising을 사용한다. 같은 block의 token들이 여러 reverse step 동안 서로를 condition으로 사용한다.

### 5) Diffusion objective의 gradient variance

논문이 강조하는 가장 중요한 optimization 문제다. Block size가 1이면 distribution factorization은 autoregressive objective와 연결되지만, diffusion-style Monte Carlo estimator를 그대로 쓰면 gradient variance가 커질 수 있다.

실제로 block size 1 setting에서도 naive diffusion training은 autoregressive baseline보다 PPL이 나빠진다. 이는 representation capacity 차이가 아니라 estimator variance와 noise schedule 문제일 수 있다.

따라서 좋은 block diffusion model에는 architecture뿐 아니라 다음 recipe가 필요하다.

- Efficient block-aware training algorithm
- Gradient variance estimator
- Block-size-specific noise schedule
- Prefix cache와 current-block diffusion을 지원하는 inference loop

# 2. Core Idea

## 2-1. Main contribution

Block Diffusion의 핵심 기여는 네 가지로 정리할 수 있다.

### 1) BD3-LM factorization

Sequence를 block 단위로 autoregressive하게 factorize하고, 각 block conditional을 discrete denoising diffusion으로 모델링한다.

| Block size | 가까워지는 endpoint | 특징 |
| --- | --- | --- |
| $L'=1$ | Autoregressive LM | Strong likelihood, minimal intra-block parallelism |
| Small $L'$ | Semi-autoregressive | Quality와 parallelism의 절충 |
| Large $L'$ | Diffusion LM | More parallel token updates, harder modeling |
| $L'=L$ | Full-sequence diffusion | Fixed canvas에 가까운 generation |

이 design space는 AR와 diffusion을 architecture label로 나누기보다 generation granularity로 연결한다.

### 2) Variable-length and cacheable diffusion generation

Block을 순차 생성하므로 다음이 가능하다.

- Current block에서 EOS가 나오면 generation 종료
- Training context보다 긴 sequence를 여러 block으로 이어 생성
- Completed block의 KV cache 재사용
- Current block token의 parallel denoising

### 3) Vectorized block training

Block conditional loss를 naive하게 계산하면 clean prefix encoding과 noisy block prediction을 반복해야 한다. 논문은 clean sequence와 noisy sequence를 하나의 길이 $2L$ input으로 붙이고, specialized attention mask를 적용해 여러 block loss를 한 번의 vectorized forward로 계산한다.

이 방식은 standard diffusion training보다 무겁지만, naive multiple-pass implementation보다 효율적이다.

### 4) Gradient-variance-aware noise schedule

Masking probability를 전체 $[0,1]$ 범위에서 uniform하게 sampling하지 않고, block size에 따라 low-noise와 high-noise extreme을 clip한다.

$$
1 - \alpha_t \sim \mathcal{U}[\beta, \omega]
$$

여기서 $1-\alpha_t$는 masking rate다. 논문은 validation에서 NELBO estimator variance가 작아지는 $[\beta, \omega]$를 grid search한다.

작은 block과 큰 block은 optimal interval이 다르다. Noise schedule을 universal hyperparameter가 아니라 block-size-dependent optimization parameter로 본다.

## 2-2. Design intuition

### 1) Block는 dependency boundary다

완료된 이전 block은 immutable context다. Current block만 uncertain state다. 이 boundary 덕분에 bidirectional denoising과 causal caching을 한 model 안에 함께 넣을 수 있다.

### 2) Parallelism은 무료가 아니다

Block이 커질수록 한 denoising step에서 더 많은 token을 갱신할 수 있다. 하지만 conditional distribution도 더 복잡해진다.

- Small block: 쉬운 prediction, 더 많은 block step
- Large block: 어려운 prediction, 더 적은 block step

따라서 최적 block size는 model scale, task, target latency, control requirement에 따라 달라질 수 있다.

### 3) Noise schedule은 learning problem의 난도를 정한다

Masking rate가 너무 낮으면 대부분 token이 보이는 쉬운 example만 학습하고, objective weight 때문에 variance가 커질 수 있다. Masking rate가 너무 높으면 current block에 정보가 거의 없어 prediction이 어려워진다.

Block size에 따라 useful noise regime이 다르므로, fixed linear schedule이 항상 적절하지 않다.

### 4) Model architecture보다 estimator가 bottleneck일 수 있다

Block size 1에서 AR와 같은 conditional structure를 표현할 수 있어도 training estimator가 noisy하면 optimization 결과는 달라진다. 논문은 theoretical equivalence와 practical trainability를 구분한다.

이 논문의 가장 중요한 기여는 block factorization 자체보다 이 지점이다. Diffusion LM의 quality gap을 architecture capacity만으로 설명하지 않고, objective estimator variance와 sampling distribution까지 내려가 분석한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Model name | Block Discrete Denoising Diffusion LM, BD3-LM |
| Global generation | Block-autoregressive |
| Local generation | Discrete denoising diffusion within a block |
| Transformer | One shared Transformer |
| Attention | Block-aware specialized mask |
| Prefix reuse | Completed blocks use KV cache |
| Current block | Multiple tokens updated in parallel across denoising steps |
| Length | EOS-based variable-length generation |
| Training | Vectorized clean and noisy sequence forward |
| Optimization | Gradient variance estimator and clipped noise schedule |
| Main hyperparameter | Block size $L'$ |

## 3-2. Block-autoregressive factorization

Sequence length $L$과 block size $L'$가 주어지면 block 수는 대략 다음과 같다.

$$
B = \left\lceil \frac{L}{L'} \right\rceil
$$

Generation은 block index $b=1$부터 순서대로 진행한다.

1. Previous clean blocks $\mathbf{x}^{<b}$를 condition으로 둔다.
2. Current block $\mathbf{x}^{b}$를 all-mask state에서 시작한다.
3. Reverse diffusion step을 반복해 current block token을 복원한다.
4. Current block을 확정하고 KV cache에 추가한다.
5. EOS가 없으면 다음 block으로 이동한다.

Block 안에서는 token order를 causal하게 고정하지 않는다. Current block token은 서로의 partially denoised state를 볼 수 있다.

## 3-3. Forward masking process

Absorbing-state discrete diffusion에서는 clean token이 시간 $t$에 따라 mask token으로 바뀐다. Simplified form은 다음처럼 볼 수 있다.

$$
q(\mathbf{x}^{b}_{t} \mid \mathbf{x}^{b}_{0})
=
\mathrm{Cat}
\left(
\alpha_t \mathbf{x}^{b}_{0}
+
(1-\alpha_t)\mathbf{m}
\right)
$$

여기서

- $\mathbf{x}^{b}_{0}$: clean current block
- $\mathbf{x}^{b}_{t}$: noisy or masked current block
- $\mathbf{m}$: absorbing mask state
- $\alpha_t$: token이 clean하게 남을 probability
- $1-\alpha_t$: masking rate

Reverse model은 previous clean block과 noisy current block을 보고 clean token distribution을 예측한다.

$$
p_\theta
\left(
\mathbf{x}^{b}_{0}
\mid
\mathbf{x}^{<b},
\mathbf{x}^{b}_{t},
 t
\right)
$$

논문 model은 explicit timestep conditioning을 쓰지 않는다. Mask pattern 자체가 noise level 정보를 제공할 수 있다는 prior diffusion LM recipe를 따른다.

## 3-4. Specialized attention structure

Block Diffusion의 attention은 ordinary causal mask도, full bidirectional mask도 아니다.

Current noisy token이 볼 수 있는 범위는 다음과 같다.

- 같은 noisy block의 token
- 이전 clean block의 token

볼 수 없는 범위는 다음과 같다.

- 미래 block의 clean token
- 다른 future noisy block

Training에서 clean representation과 noisy representation을 함께 처리하기 위해 논문은 세 mask component를 조합한다.

### 1) Block diagonal mask

같은 noisy block 안에서 bidirectional self-attention을 허용한다.

### 2) Offset block-causal mask

Noisy current block이 그보다 앞선 clean block을 condition으로 볼 수 있게 한다.

### 3) Block-causal mask

Clean sequence side에서는 현재 block과 이전 block의 clean token을 보게 한다.

이 mask를 사용하면 각 block conditional에 필요한 dependency를 하나의 sparse attention pattern으로 표현할 수 있다.

## 3-5. Vectorized training algorithm

Naive implementation은 block마다 다음을 반복할 수 있다.

1. Previous clean block을 encode한다.
2. Current noisy block을 입력한다.
3. Loss를 계산한다.
4. 다음 block으로 이동한다.

이렇게 하면 한 sequence에서 여러 forward pass가 필요하다.

논문은 noisy sequence와 clean sequence를 concatenate한다.

$$
\mathbf{z}
=
[\mathbf{x}_{t};\mathbf{x}_{0}]
$$

Input length는 $2L$이 되지만, specialized mask를 통해 각 position이 필요한 context만 본다. 모든 block의 conditional loss를 vectorized하게 계산할 수 있다.

중요한 해석은 `ordinary diffusion과 같은 비용`이 아니라 `naive block-by-block 반복보다 효율적인 single vectorized computation`이다. Token이 clean side와 noisy side에 나타나므로 effective compute는 standard $L$-length forward보다 크다.

논문은 vectorized algorithm이 separate forward-pass implementation보다 약 20%에서 25% 빠르다고 보고하고, standard diffusion training 대비 2배 미만의 범위로 비용을 제한한다고 설명한다.

## 3-6. KV caching at inference

Block $1$부터 $b-1$까지는 이미 clean하게 확정되었다. Reverse diffusion step이 바뀌어도 이 prefix는 바뀌지 않는다.

따라서 각 Transformer layer에서 previous block의 key와 value를 cache할 수 있다.

Current block의 denoising step에서는 다음만 다시 계산한다.

- Current block query
- Current block key and value
- Cached prefix에 대한 attention
- Current block 내부 attention

Full diffusion처럼 전체 sequence KV를 매 step 다시 만들 필요가 없다.

다만 current block 내부 KV는 token state가 바뀔 수 있으므로 denoising step마다 갱신해야 한다.

## 3-7. Sampling procedure

Simplified inference는 다음과 같다.

```text
prefix = []
cache = empty

while not stopped:
    block = [MASK] * block_size

    for step in reverse_schedule:
        logits = model(block, prefix_cache=cache)
        block = denoise(block, logits, step)

    append block to prefix
    update cache with completed block

    if EOS appears in block:
        stop
```

Block 안의 여러 token을 동시에 갱신하므로 token-level AR보다 parallelism이 있다. 하지만 block 자체는 순차 생성하므로 완전한 one-shot generation은 아니다.

## 3-8. Block size as interpolation parameter

### $L'=1$

각 block에 token 하나만 있다. Block conditional은 token conditional이 된다. Intra-block parallelism은 없지만 AR에 가까운 likelihood를 기대할 수 있다.

다만 diffusion objective estimator를 쓰므로 training dynamics가 standard cross-entropy AR와 완전히 같지는 않다. 논문은 noise schedule을 조정해야 PPL gap이 닫힌다고 보여준다.

### Small $L'$

$L'=4$나 $8$은 몇 token을 병렬 복원하면서 previous block context를 자주 갱신한다. Experiments에서는 작은 block이 diffusion baseline보다 좋은 likelihood와 sample quality를 보인다.

### Large $L'$

$L'=16$ 이상에서는 block 수가 줄고 더 많은 token을 동시에 처리한다. 하지만 current block conditional이 어려워져 PPL이 나빠질 수 있다.

### $L'=L$

Sequence 전체가 하나의 block이 된다. Full diffusion에 가까워지고 previous-block KV cache의 장점은 사라진다.

## 3-9. Gradient variance estimator

Diffusion objective는 random time 또는 masking rate를 sampling해 loss를 추정한다. 같은 sequence에서도 sampled mask pattern과 noise level에 따라 gradient가 크게 달라질 수 있다.

논문은 validation batch에서 NELBO estimator variance를 측정하고, block size와 schedule에 따른 variance를 비교한다.

핵심 관찰은 다음과 같다.

- High variance setting은 더 나쁜 PPL과 연결된다.
- Block size마다 variance를 줄이는 masking interval이 다르다.
- Full $[0,1]$ linear schedule이 항상 최적이 아니다.
- Schedule tuning으로 block size 1에서 AR PPL을 회복할 수 있다.

## 3-10. Clipped noise schedule

Masking rate를 다음 interval에서 sampling한다.

$$
r = 1-\alpha_t
$$

$$
r \sim \mathcal{U}[\beta, \omega]
$$

$\beta > 0$이면 지나치게 clean한 example을 줄인다. $\omega < 1$이면 거의 모든 token이 masked된 extreme example을 줄인다.

논문은 block size별로 candidate interval을 평가하고, NELBO variance가 작은 interval을 선택한다. 이 search는 validation epoch마다 수행되며 약 5,000 update 간격으로 설명된다.

결과적으로

- $L'=4$는 상대적으로 heavy masking을 포함한 $[0.45,0.95]$가 좋다.
- $L'=16$은 더 낮은 masking range인 $[0.3,0.8]$가 좋다.

작은 block은 previous context가 많고 current target이 작으므로 더 강한 masking을 견딜 수 있다. 큰 block은 current block 자체의 uncertainty가 크므로 너무 heavy한 masking이 어렵다는 해석이 가능하다.

## 3-11. FlexAttention implementation

Specialized attention mask는 structured sparse pattern을 가진다. Dense attention matrix를 그대로 materialize하면 $2L$ input의 비용이 커진다.

논문은 PyTorch FlexAttention으로 block mask를 compile한다.

- Invalid block을 compile time에 제거
- Sparse block만 계산
- Full attention matrix materialization 회피
- `torch.compile`과 Triton kernel fusion 사용

A5000, $L=1024$, batch size 16 microbenchmark에서 naive PyTorch scaled dot product attention 대비 최대 약 5배 attention speedup을 보고한다. End-to-end model forward에서는 FlashAttention 기반 custom mask를 FlexAttention으로 교체했을 때 약 15% speedup을 보고한다.

이 수치는 architecture-level guarantee가 아니라 특정 hardware, shape, kernel setting의 implementation result다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 두 language modeling corpus를 사용한다.

| Dataset | Tokenizer | Context length | Training tokens |
| --- | --- | ---: | ---: |
| LM1B | bert-base-uncased | 128 | 65B |
| OpenWebText | GPT-2 tokenizer | 1024 | 524B |

LM1B example은 대부분 짧은 sentence이므로 sequence를 concatenate하고 128 token으로 wrap한다. OWT도 document를 pad or truncate하는 대신 concatenate and wrap하여 1024 token training sequence를 만든다.

OWT에는 official validation split이 없어 마지막 100,000 document를 validation으로 사용한다.

## 4-2. Architecture

Main model setting은 다음과 같다.

- 12 Transformer layers
- Hidden size 768
- 12 attention heads
- 약 110M parameters
- RoPE positional encoding
- Explicit timestep conditioning 없음
- Context length 128 or 1024

AR, SEDD, MDLM, BD3-LM comparison은 가능한 한 같은 model scale과 training token 조건을 맞춘다.

## 4-3. Optimization

논문은 prior MDLM recipe를 가깝게 따른다.

- Optimizer: AdamW
- Global batch size: 512
- Learning rate warmup: 0에서 $3 \times 10^{-4}$
- Warmup updates: 2,500
- Base pretraining: maximum block size $L'=L$
- Base pretraining updates: 850,000
- Block-size-specific fine-tuning: 150,000 updates

전체 sequence diffusion setting으로 먼저 pretrain한 뒤, 원하는 block size로 fine-tune하는 recipe다. 처음부터 모든 block size를 별도 pretrain하지 않아 compute를 줄인다.

## 4-4. Recommended recipe from the paper

논문 결과를 실제 recipe로 정리하면 다음 순서다.

### Step 1. Standard diffusion base를 학습한다

Maximum block size로 large-scale pretraining을 수행한다. Shared representation을 먼저 확보한다.

### Step 2. Target block size를 정한다

목표가 likelihood인지, parallel generation인지, control인지에 따라 $L'$를 고른다.

- Quality 우선: small block
- Parallelism 우선: larger block
- AR-like behavior: $L'=1$

### Step 3. Vectorized mask로 fine-tune한다

Clean and noisy sequence를 concatenate하고 block-aware mask로 all-block loss를 계산한다.

### Step 4. Gradient variance를 측정한다

Validation batch에서 candidate noise schedule별 NELBO estimator variance를 계산한다.

### Step 5. Clipped schedule을 선택한다

Block size별로 $[\beta,\omega]$를 search하고 variance가 작은 schedule로 fine-tune한다.

### Step 6. Inference에서는 prefix KV를 cache한다

Completed block은 고정하고 current block만 denoise한다.

### Step 7. NFE와 wall-clock을 함께 측정한다

같은 NFE라도 block size, mask ratio, kernel, cache hit에 따라 latency가 달라질 수 있다.

## 4-5. Engineering notes

### 1) EOS 처리

EOS가 current block 중간에 나올 수 있다. EOS 뒤 token을 output에서 제거하고 cache update 범위를 명확히 해야 한다.

### 2) Long generation stop rule

논문은 EOS 또는 sample entropy가 크게 악화될 때 sampling을 멈춘다. Production에서는 entropy threshold, max block, repetition detection을 함께 둬야 한다.

### 3) Block size와 batch shape

Block size가 바뀌면 sparse mask pattern과 kernel efficiency가 달라진다. PPL만 보고 block size를 고르면 실제 throughput이 예상과 다를 수 있다.

### 4) Schedule search cost

Validation마다 schedule candidate를 평가하면 추가 compute가 든다. 작은 pilot run에서 block size별 interval을 먼저 좁히는 방법이 실용적이다.

### 5) Cache memory

Long sequence에서는 completed block KV가 계속 쌓인다. Arbitrary-length generation을 하려면 KV memory, sliding window, quantized cache 같은 별도 strategy가 필요할 수 있다.

### 6) Training mask correctness

Vectorized attention mask가 future clean token을 누출하면 likelihood가 잘못 좋아질 수 있다. Block index별 visibility unit test가 필수적이다.

### 7) Objective normalization

Block size와 masking rate에 따라 loss token 수가 달라진다. Schedule comparison에서는 같은 normalization과 estimator definition을 유지해야 한다.

# 5. Evaluation

## 5-1. Likelihood on LM1B

LM1B test PPL은 다음과 같다.

| Model | PPL |
| --- | ---: |
| Autoregressive Transformer | 22.83 |
| D3PM absorbing | <= 82.34 |
| SEDD | <= 32.68 |
| MDLM | <= 31.78 |
| BD3-LM, $L'=16$ | <= 30.60 |
| BD3-LM, $L'=8$ | <= 29.83 |
| BD3-LM, $L'=4$ | <= 28.23 |

BD3-LM은 비교한 diffusion baseline보다 낮은 PPL을 보인다. 작은 block이 더 좋은 likelihood를 보이며 AR 22.83과의 gap은 여전히 남는다.

`<=` 표시는 diffusion model의 NELBO-based likelihood bound라는 점을 기억해야 한다. AR의 exact autoregressive PPL과 숫자를 같은 의미로 완전히 동일시하면 안 된다.

## 5-2. Likelihood on OpenWebText

| Model | PPL |
| --- | ---: |
| Autoregressive | 17.54 |
| SEDD | <= 24.10 |
| MDLM | <= 22.98 |
| BD3-LM, $L'=16$ | <= 22.27 |
| BD3-LM, $L'=8$ | <= 21.68 |
| BD3-LM, $L'=4$ | <= 20.73 |

$L'=4$ BD3-LM은 MDLM 22.98 대비 20.73으로 개선된다. 논문은 LM1B에서 MDLM 대비 최대 13% PPL improvement를 보고한다.

하지만 AR 17.54에는 아직 미치지 못한다. Block Diffusion은 diffusion LM 안에서 strong result이지, autoregressive likelihood를 넘어선 결과는 아니다.

## 5-3. Zero-shot likelihood transfer

OWT에서 학습한 model을 PTB, Wikitext, LM1B, Lambada, AG News, Pubmed, Arxiv에 zero-shot 평가한다.

BD3-LM은 Wikitext, LM1B, AG News에서 비교한 diffusion model 중 가장 좋은 결과를 보이고, Pubmed에서는 표 전체에서 강한 결과를 보인다. 그러나 모든 dataset과 모든 block size에서 일관되게 최고는 아니다.

이 실험의 의미는 block factorization이 training corpus likelihood만 개선한 것이 아니라 일부 domain transfer에서도 유지되는지를 보는 데 있다.

## 5-4. Variable-length generation

OWT로 학습한 model에서 500개 document를 sampling한 length statistics는 다음과 같다.

| Source or model | Median tokens | Max tokens |
| --- | ---: | ---: |
| OWT train set | 717 | 131K |
| Autoregressive | 4,008 | 131K |
| SEDD | 1,021 | 1,024 |
| BD3-LM, $L'=16$ | 798 | 9,982 |

SEDD는 training context 1,024에 사실상 묶인다. BD3-LM은 9,982 token까지 생성해 약 10배 긴 maximum을 보인다.

이 결과는 `10K token에서도 품질이 완전히 유지된다`는 뜻은 아니다. 논문 sampling은 EOS가 나오거나 sample entropy가 크게 나빠질 때 멈춘다. 길이 capability와 long-range coherence는 별도로 평가해야 한다.

또한 AR이 131K까지 간다는 결과는 variable length가 항상 좋은 calibration을 의미하지 않는다. Median 4,008은 OWT train median 717보다 훨씬 길어 stop behavior가 과도할 수 있다.

## 5-5. Generative perplexity and NFE

GPT-2 Large를 evaluator로 사용한 generative PPL은 다음과 같다.

### Generation length 1024

| Model | Generative PPL | NFE |
| --- | ---: | ---: |
| AR | 14.1 | 1K |
| SEDD | 52.0 | 1K |
| MDLM | 46.8 | 1K |
| SSD-LM, standard | 37.2 | 40K |
| SSD-LM, matched NFE | 281.3 | 1K |
| BD3-LM, $L'=16$ | 33.4 | 1K |
| BD3-LM, $L'=8$ | 30.4 | 1K |
| BD3-LM, $L'=4$ | 25.7 | 1K |

### Generation length 2048

| Model | Generative PPL | NFE |
| --- | ---: | ---: |
| AR | 13.2 | 2K |
| MDLM | 41.3 | 2K |
| SSD-LM, standard | 35.3 | 80K |
| SSD-LM, matched NFE | 281.9 | 2K |
| BD3-LM, $L'=16$ | 31.5 | 2K |
| BD3-LM, $L'=8$ | 28.2 | 2K |
| BD3-LM, $L'=4$ | 23.6 | 2K |

BD3-LM은 비교 diffusion model보다 낮은 generative PPL을 같은 NFE budget에서 보인다. Small block이 가장 좋은 sample quality를 보인다.

다만 NFE는 wall-clock latency와 같지 않다.

- AR NFE는 token별 sequential step이다.
- Diffusion NFE는 block 내 parallel update를 포함한다.
- Kernel efficiency와 batch size가 다르다.
- Current block length에 따라 한 NFE의 FLOPs가 달라진다.

따라서 실제 serving speed claim에는 end-to-end latency, throughput, memory measurement가 추가로 필요하다.

## 5-6. Noise schedule ablation

### $L'=4$

| Noise schedule | PPL | NELBO variance |
| --- | ---: | ---: |
| Clipped U[0.45, 0.95] | 29.21 | 6.24 |
| Clipped U[0.3, 0.8] | 29.38 | 10.33 |
| Linear U[0, 1] | 30.18 | 23.45 |
| Logarithmic | 30.36 | 23.53 |
| Square root | 31.41 | 26.43 |

### $L'=16$

| Noise schedule | PPL | NELBO variance |
| --- | ---: | ---: |
| Clipped U[0.45, 0.95] | 31.42 | 3.60 |
| Clipped U[0.3, 0.8] | 31.12 | 3.58 |
| Linear U[0, 1] | 31.72 | 7.62 |
| Square | 31.43 | 13.03 |
| Cosine | 31.41 | 13.00 |

두 block size 모두 clipped schedule이 linear schedule보다 낮은 variance와 더 좋은 PPL을 보인다. 하지만 optimal interval은 다르다.

이 결과는 다음을 지지한다.

- Gradient variance는 단순 diagnostic이 아니라 schedule selection signal로 쓸 수 있다.
- Diffusion LM schedule은 image diffusion recipe를 그대로 가져오기보다 token dependency와 block size에 맞춰야 한다.

## 5-7. Block size 1 and AR equivalence

논문은 block size 1에서 objective가 expectation상 AR과 연결되지만 naive sampling에서는 PPL gap이 생긴다고 분석한다.

Reported ablation의 핵심 흐름은 다음과 같다.

- Standard AR: PPL 22.88
- Randomized diffusion-style estimator: PPL 24.37
- Naive BD3 with $L'=1$: PPL bound가 더 나쁨
- Tuned noise schedule: PPL 22.88 회복

또한 일부 setting에서 NELBO variance가 1.52에서 full masking에 가까운 schedule의 0.11로 줄어든다고 보고한다.

정확한 의미는 `BD3가 AR보다 좋다`가 아니다. Same conditional structure라도 estimator가 noisy하면 training result가 나빠지고, variance-aware schedule로 theoretical endpoint에 더 가까워질 수 있다는 것이다.

## 5-8. Training and kernel efficiency

논문은 두 종류의 speed result를 제시한다.

1. Vectorized training
   - Separate clean and noisy forward보다 약 20%에서 25% speedup
   - Standard diffusion training 대비 2배 미만의 training speed 범위

2. FlexAttention
   - Attention microbenchmark에서 naive PyTorch implementation 대비 최대 약 5배
   - A5000, $L=1024$, batch 16의 end-to-end model forward에서 약 15% speedup

여기서 microbenchmark와 full model speed를 구분해야 한다. Sparse attention kernel이 5배 빨라도 embedding, MLP, normalization, memory movement를 포함한 model 전체는 15% 정도 개선될 수 있다.

## 5-9. 무엇을 진짜 봐야 하는가

### 1) Diffusion SOTA와 AR parity는 다르다

BD3-LM은 diffusion baseline을 크게 개선하지만 AR likelihood에는 여전히 gap이 있다.

### 2) Small block이 quality에는 유리하다

$L'=4$가 주요 table에서 가장 좋은 PPL과 generative PPL을 보인다. 그러나 small block은 more sequential block step을 요구한다.

### 3) Variable length는 architecture capability다

Training context보다 긴 generation이 가능하다는 점은 중요한 구조적 개선이다. 하지만 very long coherence와 safe stop은 별도 문제다.

### 4) Gradient variance가 central result다

Architecture diagram만 보면 block factorization이 전부처럼 보이지만, 실제 PPL improvement의 중요한 부분은 schedule tuning과 estimator variance reduction에서 나온다.

### 5) Serving claim은 hardware measurement가 더 필요하다

KV cache와 block parallelism은 promising하지만, NFE, attention microbenchmark, end-to-end latency는 서로 다른 지표다.

# 6. Limitations

1. Autoregressive likelihood gap이 남아 있다.
   - OWT에서 best BD3-LM은 20.73, AR은 17.54다.
   - Diffusion model 중 strong result이지만 next-token LM을 quality 측면에서 대체했다고 보기는 어렵다.

2. Training cost가 standard diffusion보다 크다.
   - Clean and noisy representation을 함께 처리한다.
   - Vectorization으로 2배 미만에 제한하지만 ordinary $L$-length forward보다 무겁다.

3. Block는 여전히 sequential하다.
   - Current block token은 parallel하게 갱신하지만 next block은 previous block 완료를 기다린다.
   - Small block에서는 AR과 비슷한 sequential bottleneck이 다시 커질 수 있다.

4. Optimal block size가 task specific이다.
   - Likelihood에는 small block이 좋을 수 있다.
   - Editability나 control에는 larger block이 유리할 수 있다.
   - 하나의 pretrained checkpoint와 block size가 모든 workload에 최적이라는 근거는 없다.

5. Experiment scale이 modern instruction LLM보다 작다.
   - Main model은 약 110M parameter의 unconditional language model이다.
   - Multi-billion parameter, instruction tuning, chat alignment, tool use에서 같은 trend가 유지되는지 확인이 필요하다.

6. Long generation quality 평가가 제한적이다.
   - 9,982 token maximum은 length capability를 보여준다.
   - Long-range factual consistency, topic drift, repetition, discourse structure를 정량 평가하지는 않는다.

7. Generative PPL은 proxy evaluator에 의존한다.
   - GPT-2 Large의 likelihood를 sample quality proxy로 사용한다.
   - Evaluator bias와 training corpus overlap의 영향을 받을 수 있다.

8. NFE는 latency가 아니다.
   - Block 내부 parallelism, block 수, cache, sparse kernel, batch size에 따라 wall-clock이 달라진다.
   - Online serving benchmark가 더 필요하다.

9. Clipped schedule search가 추가 hyperparameter를 만든다.
   - Block size마다 $\beta$와 $\omega$를 찾아야 한다.
   - Model scale과 data가 바뀌면 optimal interval도 달라질 수 있다.

10. Arbitrary-length라는 표현에는 practical bound가 있다.
    - Architecture상 block을 계속 이어 생성할 수 있다.
    - 실제로는 KV memory, positional extrapolation, entropy degradation, maximum block limit에 제약을 받는다.

11. Cache가 long-context cost를 제거하지는 않는다.
    - Previous KV를 재사용해도 current token이 긴 prefix에 attention하면 attention cost와 cache memory가 계속 증가한다.

12. General generative risk가 남는다.
    - Hallucination, harmful output, copyright, controllability 문제는 architecture만으로 해결되지 않는다.

# 7. My Take

## 7-1. Why this matters for my work

Block Diffusion은 diffusion LM을 볼 때 유용한 기준점을 제공한다. 핵심 질문을 `AR인가 diffusion인가`에서 다음으로 바꾼다.

- 어떤 dependency를 sequential하게 유지할 것인가.
- 어떤 token group을 동시에 수정할 것인가.
- Cacheable boundary를 어디에 둘 것인가.
- Parallelism을 위해 얼마의 likelihood gap을 허용할 것인가.

이 질문은 text generation 외에도 적용된다.

### Code generation

Function이나 statement block을 순차적으로 생성하고, block 내부 token은 반복 수정할 수 있다. Fill-in-the-middle이나 local constraint와 결합할 여지가 있다.

### Structured document generation

Section을 autoregressive하게 만들고, table row나 field group을 block diffusion으로 채울 수 있다. EOS 대신 schema completion condition을 사용할 수도 있다.

### Long-form planning

High-level plan block은 순차적으로 확정하고, 각 block의 wording은 병렬 refinement할 수 있다.

### Multimodal token generation

Image or audio token을 시간 block으로 나누고, previous block cache와 current block denoising을 결합할 수 있다. 다만 modality마다 block boundary semantics가 다르다.

### Agent trajectory

Action chunk를 block으로 생성하고, chunk 내부 계획을 iterative refinement하는 semi-autoregressive policy와 연결할 수 있다. Environment feedback이 block 사이에만 들어간다는 제약은 별도 연구 문제다.

## 7-2. Reuse potential

### 1) Block-size sweep을 architecture ablation으로 본다

$L'=1,4,8,16$을 단순 latency hyperparameter가 아니라 dependency structure ablation으로 평가한다.

측정할 지표는 다음과 같다.

- NLL or PPL
- End-to-end latency
- Tokens per second
- KV memory
- Repetition and stop error
- Editability
- Constraint satisfaction

### 2) Variance를 training dashboard에 넣는다

Loss 평균만 보지 않고 noise level별 gradient norm과 estimator variance를 기록한다. Schedule failure를 early diagnosis할 수 있다.

### 3) Data-driven schedule search

Image diffusion에서 가져온 cosine schedule을 default로 고정하지 않는다. Block size와 token domain별로 useful masking interval을 찾는다.

### 4) Prefix-cache-aware serving

Completed block KV와 current block KV를 분리한다. Current block만 invalidation하고 prefix cache는 유지한다.

### 5) Sparse mask unit test

각 query position이 볼 수 있는 key set을 block index별로 검증한다. Future clean leakage를 막는 것이 가장 중요하다.

### 6) Quality-latency Pareto curve

NFE만 보고 결론을 내리지 않는다. Block size, denoising step, cache, batch를 함께 sweep해 actual wall-clock Pareto를 만든다.

### 7) Dynamic block size

Easy span은 larger block, uncertain span은 smaller block을 사용하는 adaptive policy를 생각할 수 있다. 다만 boundary decision과 cache layout이 복잡해진다.

### 8) Retrieval or constraint boundary와 block를 맞춘다

Sentence, code statement, schema field group처럼 semantic boundary에 block을 맞추면 fixed token block보다 dependency가 자연스러울 수 있다.

## 7-3. Follow-up papers

- D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces
- SEDD: Score Entropy Discrete Diffusion
- MDLM: Simple and Effective Masked Diffusion Language Models
- SSD-LM: Semi-Autoregressive Simplex-Based Diffusion Language Model
- LLaDA: Large Language Diffusion Models
- Diffusion Forcing and block-wise generative modeling
- FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention
- Speculative decoding and multi-token prediction literature

# 8. Summary

- Block Diffusion은 sequence를 block 단위로 autoregressive하게 생성하고, current block 내부 token은 discrete diffusion으로 병렬 복원한다.
- Block size 1은 AR endpoint에 가깝고, full context block은 ordinary diffusion endpoint에 가까워 quality와 parallelism을 연속적으로 조절한다.
- Completed block은 고정되므로 KV caching이 가능하고, EOS가 나올 때까지 block을 이어 variable-length sequence를 생성할 수 있다.
- Vectorized clean-noisy input과 specialized sparse attention mask로 all-block training loss를 효율적으로 계산한다.
- Clipped noise schedule은 NELBO estimator variance를 줄이고, block-size-specific optimization이 PPL 개선에 중요하다는 점을 보여준다.
- BD3-LM은 diffusion baseline보다 좋은 likelihood와 generative PPL을 보이지만, autoregressive baseline과의 quality gap과 block-level sequential bottleneck은 남는다.
