---
layout: single
title: "DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation Review"
categories: Study-concept
tag: [TrainingEfficiency, DiffusionModels, BlockWiseTraining, MemoryEfficiency, Transformer]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2506.14202)

[Code link](https://github.com/SakanaAI/DiffusionBlocks)

[OpenReview link](https://openreview.net/forum?id=pwVSmK71cS)

DiffusionBlocks는 end-to-end backpropagation의 activation memory bottleneck을 block-wise training으로 줄이려는 논문이다. 그런데 단순히 layer를 잘라서 local loss를 붙이는 방식은 아니다. 이 논문은 residual connection을 diffusion process의 Euler step으로 해석하고, 각 block을 특정 noise range를 담당하는 denoiser로 바꾸는 recipe를 제안한다.

한 줄 framing은 "residual network를 diffusion training problem으로 다시 쓰는 block-wise recipe"라고 할 수 있다.

> 한 줄 요약: DiffusionBlocks는 transformer 기반 residual network를 noise-conditioned denoising blocks로 변환해, 매 step에서 한 block만 gradient를 계산하도록 만들고, equi-probability noise partition으로 block별 학습 난도를 맞추는 block-wise training framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- activation memory를 줄이는 접근을 checkpointing이나 offloading이 아니라 **objective conversion** 관점에서 다룬다.
- residual connection과 diffusion ODE 사이의 연결을 이용해 **block-wise local objective**를 비교적 principled하게 만든다.
- vision classification, image diffusion, masked diffusion LM, autoregressive LM, recurrent-depth model까지 넓게 적용한다.
- 단순 memory saving만이 아니라 diffusion inference와 recurrent-depth training에서 compute pattern 자체가 달라지는 지점을 보여준다.
- ICLR 2026 paper로 공개되었고, official code도 최소 ViT classification setting 기준으로 열려 있다.

이 논문은 **대규모 학습을 싸게 만든다**보다 **global backprop 없이 block별 objective를 어떻게 설계할 것인가**에 더 가깝다. 메모리 절감 claim은 중요하지만, 더 흥미로운 지점은 residual stack을 denoising trajectory로 재해석하면 block마다 독립적인 supervised signal을 줄 수 있다는 점이다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 end-to-end backpropagation의 memory bottleneck이다.

일반적인 L-layer residual network는 아래처럼 layer output을 순차적으로 갱신한다.

$$
z_{l+1} = z_l + f_{theta_l}(z_l)
$$

End-to-end training에서는 backward pass를 위해 각 layer의 activation을 저장해야 한다. 그래서 model depth가 커질수록 activation memory가 거의 layer 수에 비례해서 증가한다. activation checkpointing으로 일부 줄일 수는 있지만, parameter, gradient, optimizer state는 그대로 남고 recomputation cost가 붙는다.

Block-wise training은 이 문제를 오래 전부터 다뤄왔다. Network를 여러 block으로 쪼개고 block마다 local objective를 붙이면, 한 번에 전체 layer activation을 들고 있을 필요가 없다. 이상적으로는 B개 block으로 나누었을 때 active block만 학습하므로, 필요한 activation과 optimizer state memory를 줄일 수 있다.

하지만 기존 block-wise training의 어려움은 local objective다.

1. 각 block이 무엇을 예측해야 하는지 정의하기 어렵다.
2. Local objective가 global task performance와 잘 맞는다는 보장이 약하다.
3. Classification 중심 custom architecture에서는 돌아가도 generative model이나 transformer stack에는 일반화하기 어렵다.
4. Block 사이의 coordination을 heuristic으로 처리하면 end-to-end baseline 대비 성능이 쉽게 무너진다.

즉 문제는 "layer를 나눌 수 있는가"가 아니라 "나눈 block을 어떤 objective로 독립 학습할 것인가"다.

## 1-2. Why previous approaches are insufficient

기존 접근을 간단히 나누면 아래와 같다.

| Approach | Main idea | Limitation |
| --- | --- | --- |
| Activation checkpointing | activation을 저장하지 않고 backward 때 recompute | activation memory만 줄이고 recomputation cost가 붙음 |
| Offloading | activation 또는 optimizer state를 CPU나 storage로 이동 | bandwidth와 latency bottleneck이 생김 |
| Greedy layer-wise training | layer 또는 block별 local loss 사용 | global objective와 local objective alignment가 약함 |
| Forward-Forward | positive와 negative contrastive objective 사용 | generation task와 transformer conversion에는 직접 쓰기 어려움 |
| NoProp | diffusion-like no-backprop training | custom CNN architecture 중심이라 범용 transformer recipe가 약함 |

DiffusionBlocks의 출발점은 여기다. Block-wise training이 실패한 이유를 **memory saving idea가 틀려서**가 아니라 **local objective가 ad-hoc이어서**라고 본다. 그래서 이 논문은 local objective를 새로 발명하기보다, diffusion model의 score matching objective가 이미 noise level별로 독립 학습 가능하다는 성질을 끌어온다.

핵심 질문은 다음과 같다.

- Residual network의 layer update를 diffusion denoising step처럼 볼 수 있는가.
- 그렇다면 layer block마다 담당 noise interval을 부여할 수 있는가.
- 각 block이 자기 interval만 denoise하도록 학습해도 전체 network가 end-to-end model처럼 작동할 수 있는가.

# 2. Core Idea

## 2-1. Main contribution

DiffusionBlocks의 핵심 기여는 transformer-style residual network를 diffusion denoising process로 변환하는 3-step recipe다.

1. L개 layer를 B개 block으로 나눈다.
2. Noise range $[sigma_min, sigma_max]$를 B개 interval로 나눈다.
3. 각 block에 noise conditioning을 붙이고, 해당 interval에서 target을 denoise하도록 독립 학습한다.

Block b의 objective는 개념적으로 아래처럼 쓸 수 있다.

$$
L_b(theta_b) = E_{x,y,sigma,eps} [ w(sigma) Loss(f_{theta_b,sigma}(x, y + sigma eps), y) ]
$$

여기서 $sigma$는 block b가 담당하는 noise interval에서 sample된다. 중요한 점은 block b가 이전 block의 output을 기다리지 않는다는 것이다. Training 시에는 target $y$에 noise를 더한 $y + sigma eps$를 block input으로 주고, block은 clean target $y$를 복원한다.

이렇게 바꾸면 각 block은 자기 noise range에 대한 denoiser가 된다. 전체 network는 inference 때 high noise에서 low noise로 내려가며 block을 순차 적용한다.

이 논문의 주장 중 가장 중요한 것은 **각 block이 independent trainable unit이 된다**는 점이다. End-to-end backprop에서는 모든 layer를 지나 loss를 계산하고 전체 activation graph를 유지해야 한다. DiffusionBlocks에서는 한 step에 하나의 block만 활성화해서 그 block의 denoising loss만 계산한다.

## 2-2. Design intuition

DiffusionBlocks의 design intuition은 세 가지로 볼 수 있다.

첫째, residual update는 dynamical system의 discretized update와 닮아 있다.

$$
z_{l+1} = z_l + f_{theta_l}(z_l)
$$

Diffusion probability flow ODE도 discretization하면 noisy state를 조금씩 clean state 쪽으로 이동시키는 update가 된다. 논문은 이 structural similarity를 이용해 residual stack을 denoising trajectory로 해석한다.

둘째, diffusion training은 noise level별 objective가 자연스럽게 분해된다.

Score-based diffusion에서는 다양한 noise level에서 denoiser를 학습한다. 특정 noise level에서 denoising objective를 최적화하는 것은 다른 noise level의 sample을 꼭 지나가야만 가능한 것이 아니다. DiffusionBlocks는 이 independence를 block-wise training의 theoretical handle로 사용한다.

셋째, block을 noise range로 나누면 specialization이 생긴다.

초기 high-noise block은 거친 구조를 만들고, intermediate block은 어려운 denoising region을 다루며, low-noise block은 refinement에 가까운 역할을 맡는다. 논문이 제안하는 equi-probability partitioning은 바로 이 난도 배분을 위한 장치다.

이 논문은 "backprop을 안 한다"는 주장보다 "backprop이 필요한 범위를 block 내부로 제한한다"는 주장으로 읽는 편이 정확하다. 각 block 안에서는 여전히 gradient descent를 사용한다. 다만 gradient가 전체 depth를 관통하지 않는다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Transformer-style residual network를 independently trainable blocks로 변환 |
| Core interpretation | Residual update를 diffusion reverse process의 Euler step으로 해석 |
| Training unit | One block at a time |
| Key objective | Noise-conditioned denoising objective |
| Partition strategy | Equi-probability partition over noise distribution |
| Main memory claim | B blocks이면 active training memory를 roughly 1/B로 줄임 |
| Main evaluation domains | ViT classification, DiT image generation, masked diffusion LM, AR LM, recurrent-depth LM |
| Public code scope | GitHub는 ViT image classification implementation 중심 |

전체 절차는 아래처럼 요약할 수 있다.

```text
Standard residual network:
  input -> block 1 -> block 2 -> ... -> block B -> output

DiffusionBlocks training:
  sample block b
  sample sigma from block b noise interval
  corrupt target y into y + sigma eps
  train only block b to denoise y

DiffusionBlocks inference:
  start from high-noise target state
  apply block 1, block 2, ..., block B along noise schedule
  output final denoised state
```

## 3-2. Module breakdown

### 1) Block partitioning

먼저 L개의 layer를 B개의 block으로 나눈다. 각 block $F_b$는 연속된 layer subset을 포함한다. Block b의 composed function을 $f_{theta_b}$라고 쓰면, 기존 network는 여러 block composition으로 표현된다.

$$
F = F_B circ F_{B-1} circ ... circ F_1
$$

DiffusionBlocks에서 중요한 것은 block이 단순 sub-network가 아니라 denoiser role을 갖는다는 점이다. Block b는 전체 target space에서 특정 noise interval을 담당한다.

### 2) Noise range assignment

Noise range $[sigma_min, sigma_max]$를 B개 구간으로 나눈다. 여기서 naive uniform partition은 좋은 선택이 아니다. Diffusion model의 학습 난도는 noise scale에 균등하게 분포하지 않기 때문이다.

논문은 EDM 스타일의 log-normal noise distribution을 사용하고, cumulative probability mass가 block마다 같도록 boundary를 둔다.

$$
\int_{sigma_b}^{sigma_{b-1}} p_{noise}(sigma) d sigma = 1 / B
$$

이것이 equi-probability partitioning이다. Uniform spacing이 아니라 training distribution의 probability mass를 균등하게 나눈다. 그래서 denoising이 어려운 intermediate noise region에는 더 좁은 interval이 배정되고, 상대적으로 쉬운 high-noise 또는 low-noise region에는 더 넓은 interval이 배정된다.

이 부분은 이 논문의 practical detail 중 가장 중요하다. Block-wise training에서 block마다 학습 난도가 크게 달라지면 어떤 block은 underfit되고 어떤 block은 낭비된다. Equi-probability partition은 이 imbalance를 줄이려는 장치다.

### 3) Noise conditioning

각 block은 noise-conditioned denoiser가 된다. 원래 block input이 $x$였다면, DiffusionBlocks에서는 task input $x$와 noisy target state $z$를 함께 받는다.

$$
z = y + sigma eps
$$

Block은 아래 target을 학습한다.

$$
f_{theta_b,sigma}(x, z) -> y
$$

Transformer 계열에서는 noise conditioning을 AdaLN 같은 normalization path로 넣을 수 있다. Discrete output을 다루는 경우, 예를 들어 class label이나 token id는 continuous embedding space에서 noise를 더한다.

### 4) Block-independent objective

Block b의 objective는 아래처럼 독립적으로 정의된다.

$$
L_b(theta_b) = E_{(x,y),sigma,eps} [ w(sigma) Loss(f_{theta_b,sigma}(x, y + sigma eps), y) ]
$$

전체 objective는 block objective의 합으로 볼 수 있다.

$$
L(theta) = sum_{b=1}^{B} L_b(theta_b)
$$

하지만 학습 step에서는 모든 block을 동시에 backprop하지 않는다. 논문은 training iteration마다 block을 random sample하고, 해당 block만 gradient update한다. 따라서 training graph는 active block 내부로 제한된다.

### 5) Architecture-specific conversion

이 논문이 흥미로운 이유는 단일 architecture trick으로 끝나지 않는다는 점이다.

| Architecture | DiffusionBlocks adaptation |
| --- | --- |
| ViT classification | class label embedding에 noise를 더하고, patch embedding을 condition으로 사용 |
| DiT image generation | 기존 diffusion denoising process를 noise interval별 block으로 partition |
| Masked diffusion LM | continuous noise 대신 masking schedule을 partition |
| AR language model | token embedding space에서 noisy future token embedding을 denoise |
| Recurrent-depth model | K-step recurrence 자체를 diffusion trajectory로 보고 single-pass denoiser로 학습 |

특히 AR LM 적용은 그냥 diffusion model처럼 바꾸면 causal leakage가 생길 수 있다. 논문은 clean past token과 noisy future token을 concat하고 modified causal mask를 사용해, future denoising이 clean past에만 condition되도록 만든다. 이 선택은 single forward pass efficiency를 유지하지만 sequence memory를 doubling할 수 있다는 trade-off가 있다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 하나의 dataset만 보는 것이 아니라 architecture별로 다른 task를 사용한다.

| Task | Model family | Dataset |
| --- | --- | --- |
| Image classification | ViT | CIFAR-100, Tiny-ImageNet |
| Image generation | DiT | CIFAR-10, ImageNet-256 |
| Masked diffusion text generation | MD4-style model | text8 |
| Autoregressive text generation | Llama-2-style transformer | LM1B, OpenWebText |
| Recurrent-depth text generation | Huginn-style model | LM1B |

이 구성은 논문의 메시지와 맞다. 저자들은 DiffusionBlocks가 task-specific local objective가 아니라 residual/diffusion interpretation에서 나오는 general recipe라고 주장한다. 그래서 classification, continuous diffusion, discrete diffusion, AR, recurrent-depth를 모두 걸쳐 보여준다.

## 4-2. Training strategy

공통 recipe는 아래와 같다.

1. Original network를 B개 block으로 나눈다.
2. EDM-style noise distribution을 정의한다.
3. Noise range를 equi-probability partitioning으로 나눈다.
4. Training step마다 block을 uniform random sample한다.
5. 해당 block의 noise interval에서 $sigma$를 sample한다.
6. Target side에 noise를 넣고 block denoising loss를 계산한다.
7. Active block만 update한다.

논문에서 기본적으로 쓰는 noise setting은 아래와 같다.

| Item | Value |
| --- | --- |
| Noise distribution | log-normal |
| Pmean | -1.2 |
| Pstd | 1.2 |
| sigma range | [0.002, 80] |
| Sampling | Euler sampling |
| Default block overlap | 0.05 |
| Text generation overlap | 0.1 |

Block overlap은 block boundary를 조금 부드럽게 만들기 위한 장치다. Noise interval이 딱 끊기면 boundary에서 denoising behavior가 불안정할 수 있다. 논문은 log-sigma space에서 interval을 살짝 확장하는 방식으로 overlap을 둔다.

## 4-3. Engineering notes

가장 중요한 engineering note는 DiffusionBlocks가 **총 compute를 항상 줄이는 방법은 아니라는 점**이다.

Training 관점에서 standard training이 K iterations 동안 L개 layer를 평가한다면, DiffusionBlocks는 B개 block 각각을 K iterations 학습할 때 총 layer evaluation 수가 대략 같아질 수 있다.

$$
K * L = K * B * (L / B)
$$

즉 memory는 줄지만 total training compute가 자동으로 줄어드는 것은 아니다. 다만 active block만 graph를 유지하므로 peak memory가 줄고, block들을 여러 device에서 communication 없이 병렬 학습할 여지가 생긴다.

논문 appendix는 memory accounting을 activation checkpointing과 비교한다. Standard training의 memory를 단순화하면 아래처럼 볼 수 있다.

$$
M_{standard} = (4P + A) * L
$$

DiffusionBlocks는 B개 independent block으로 나누므로 active block 기준 memory는 아래처럼 줄어든다.

$$
M_{dblock} = (4P + A) * (L / B)
$$

여기서 P는 layer parameter size, A는 activation size다. Adam optimizer를 가정하면 parameter, gradient, momentum, variance까지 포함해 4P가 된다. Activation checkpointing은 activation A를 줄이는 반면, DiffusionBlocks는 block 단위로 parameter, gradient, optimizer state, activation을 함께 줄인다는 것이 논문의 주장이다.

또 하나의 note는 public code scope다. GitHub README는 official implementation이라고 설명하지만, 현재 공개 repository는 image classification using Vision Transformers 중심이다. 논문 전체의 DiT, AR LM, recurrent-depth 실험을 그대로 재현하려면 추가 코드나 구현 해석이 필요할 수 있다.

# 5. Evaluation

## 5-1. Main results

### ViT classification

CIFAR-100에서 12-layer ViT를 B=3 block으로 나누고, 한 번에 4 layers만 training한다.

| Method | Accuracy |
| --- | --- |
| ViT | 60.25 |
| Forward-Forward | 7.85 |
| DiffusionBlocks | 59.30 |

결과만 보면 DiffusionBlocks는 end-to-end ViT보다 약간 낮지만 거의 비슷한 수준이다. 반면 Forward-Forward adaptation은 크게 무너진다. 이 표는 DiffusionBlocks가 단순 local objective보다 denoising objective를 쓰는 이유를 잘 보여준다.

Tiny-ImageNet 추가 실험에서는 ViT 35.32, DiffusionBlocks 36.16으로 보고된다. 이 결과는 적어도 작은 classification task 하나에만 overfit된 결과는 아니라는 보조 근거다.

### DiT image generation

DiT 실험은 DiffusionBlocks가 가장 자연스럽게 들어맞는 setting이다. 이미 diffusion model은 noise level별 denoising을 하므로, block을 noise interval별 denoiser로 나누는 해석이 자연스럽다.

| Dataset | Method | FID train / test |
| --- | --- | --- |
| CIFAR-10 | DiT | 32.84 / 39.83 |
| CIFAR-10 | DiffusionBlocks | 30.59 / 37.20 |
| ImageNet-256 | DiT | 9.01 / 12.09 |
| ImageNet-256 | DiffusionBlocks | 9.00 / 10.63 |

논문은 이 setting에서 3x memory reduction과 inference cost reduction을 함께 주장한다. 이유는 inference에서도 denoising step마다 full network가 아니라 relevant block만 호출하기 때문이다.

### Masked diffusion LM

Text8 masked diffusion experiment에서는 MD4-style model을 B=3 block으로 나눈다.

| Method | BPC |
| --- | --- |
| MD4 | 1.56 |
| DiffusionBlocks | 1.45 |

낮은 BPC가 더 좋다. 여기서는 continuous sigma가 아니라 masking schedule을 나눈다는 점이 중요하다. Appendix derivation에 따르면 masked diffusion에서는 time t 자체가 아니라 masking probability schedule의 mass를 나누는 식으로 block objective가 분해된다.

### Autoregressive LM

AR transformer 실험은 가장 논쟁적인 적용이다. AR model은 본질적으로 next-token prediction model이지 diffusion denoiser가 아니다. 논문은 token embedding space에 noise를 더하고, clean past token에 condition해서 noisy future token embedding을 denoise하도록 바꾼다.

| Dataset | Method | MAUVE | PPL Llama-2 | PPL GPT2-XL |
| --- | --- | --- | --- | --- |
| LM1B | AR | 0.50 | 14.58 | 38.87 |
| LM1B | DiffusionBlocks | 0.71 | 12.32 | 30.99 |
| OWT | AR | 0.85 | 15.05 | 25.24 |
| OWT | DiffusionBlocks | 0.82 | 14.99 | 26.33 |

LM1B에서는 DiffusionBlocks가 더 좋아 보이고, OWT에서는 MAUVE와 GPT2-XL PPL에서 약간 떨어진다. 그래서 이 결과는 "AR도 무조건 개선"이 아니라, diffusion-style block conversion이 AR setting에서도 비교 가능한 품질을 유지한다는 정도로 보는 편이 맞다.

### Recurrent-depth model

Huginn-style recurrent-depth model에서는 같은 block을 평균 32 iterations 반복 적용한다. DiffusionBlocks는 recurrence를 diffusion trajectory로 보고 single-pass denoiser training으로 바꾼다.

| Method | MAUVE | PPL Llama-2 | PPL GPT2-XL |
| --- | --- | --- | --- |
| Huginn | 0.49 | 17.04 | 46.73 |
| DiffusionBlocks | 0.70 | 16.08 | 42.43 |

이 부분은 꽤 흥미롭다. 일반 block-wise training보다 더 큰 의미가 있다. Recurrent-depth model의 training-time recurrence를 denoising objective로 치환하면, BPTT-style iterative training을 줄일 수 있기 때문이다.

## 5-2. What really matters in the experiments

실험에서 중요한 포인트는 성능 숫자보다 아래 네 가지다.

첫째, DiffusionBlocks는 **memory reduction을 performance collapse 없이 보여준다**. CIFAR-100, CIFAR-10, ImageNet, text8, LM1B에서 baseline과 비슷하거나 일부 더 좋은 수치가 나온다.

둘째, equi-probability partitioning이 실제로 중요하다. CIFAR-10 ablation에서 uniform partition보다 equi-probability partition이 대체로 더 좋은 FID를 낸다.

| Partitioning | Layer distribution | FID |
| --- | --- | --- |
| Uniform | [4,4,4] | 43.53 |
| Uniform | [2,4,6] | 42.37 |
| Equi-Probability | [4,4,4] | 38.03 |
| Equi-Probability | [2,4,6] | 40.40 |

셋째, block count B는 무조건 크게 할수록 좋은 것이 아니다. ImageNet에서는 B=2와 B=3이 end-to-end B=1보다 FID가 좋지만, B=6에서는 FID가 악화된다.

| B | FID | Layers per block | Relative speed |
| --- | --- | --- | --- |
| 1 | 12.09 | 24 | 1.0x |
| 2 | 9.90 | 12 | 2.0x |
| 3 | 11.11 | 8 | 3.0x |
| 4 | 11.90 | 6 | 4.0x |
| 6 | 14.43 | 4 | 6.0x |

즉 DiffusionBlocks에는 quality-efficiency trade-off가 있다. Block을 많이 나누면 memory와 per-step compute는 줄지만, block당 capacity가 줄어서 quality가 나빠질 수 있다.

넷째, wall-time claim은 조심해서 봐야 한다. 12-layer ViT 기준으로 standard ViT는 0.0507 sec/iter, DiffusionBlocks per-block은 0.0181 sec/iter이고, B=3 aggregate는 0.0543 sec/iter다. 즉 total aggregated training time은 비슷하다. 이 논문의 강한 claim은 wall-clock speedup보다 peak memory reduction에 있다.

# 6. Limitations

1. Input-output dimension matching assumption이 있다. 논문도 U-Net 같은 architecture에는 바로 적용하기 어렵다고 명시한다.

2. 대부분의 실험은 from-scratch training이다. 이미 학습된 large model을 DiffusionBlocks로 convert하고 fine-tuning하는 setting은 future work로 남아 있다.

3. Optimal block count B가 task마다 다르다. Image generation은 B=2 또는 B=3이 좋고, LM1B autoregressive experiment는 B=4를 선택한다. 자동 partition search는 아직 명확하지 않다.

4. AR language model conversion은 causal consistency를 위해 special attention mask가 필요하고, 논문 구현에서는 sequence memory가 doubling될 수 있다.

5. Public code는 현재 ViT image classification implementation 중심이다. 논문의 모든 architecture 실험을 end-to-end reproduce하려면 추가 구현이 필요할 수 있다.

6. Memory reduction과 total compute reduction을 혼동하면 안 된다. General training에서는 total layer evaluation이 비슷할 수 있고, speedup은 diffusion inference나 recurrent-depth training처럼 구조가 맞는 setting에서 더 선명하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 long-context inference나 efficient attention 논문과는 다른 축의 efficiency paper다. Attention cost를 줄이는 것이 아니라, training graph를 어떻게 쪼갤 것인가를 다룬다.

중요한 포인트는 **block-wise training의 local objective를 diffusion으로 정당화한다**는 점이다. 대규모 model을 학습할 때 memory 병목은 계속 문제이고, checkpointing/offloading/ZeRO 계열은 system-level 해결책에 가깝다. DiffusionBlocks는 objective 자체를 바꿔 active graph를 작게 만드는 쪽이다.

물론 이 방식이 바로 trillion-scale LLM pretraining에 들어갈 수 있다는 뜻은 아니다. 하지만 residual stack이 점점 깊어지고 recurrent-depth, latent diffusion, masked diffusion, hybrid LM이 늘어나는 상황에서는 이런 objective-level decomposition이 중요해질 수 있다.

## 7-2. Reuse potential

바로 재사용해볼 수 있는 포인트는 세 가지다.

1. Equi-probability partitioning

Noise schedule을 block별로 나눌 때 uniform boundary를 쓰지 않고 probability mass를 균등하게 나누는 아이디어는 다른 diffusion architecture에도 쉽게 옮길 수 있다.

2. Block overlap

Noise interval boundary에서 behavior가 끊기지 않도록 log-sigma space에서 overlap을 주는 detail은 practical하다. Boundary artifact가 생기는 modular model training에도 비슷한 intuition을 쓸 수 있다.

3. Recurrent-depth training conversion

Recurrent-depth model을 K-step unroll로 학습하지 않고 denoising objective로 학습하는 아이디어는 후속 연구 가치가 크다. 특히 adaptive computation time, recurrent transformer, iterative reasoning model training과 연결될 수 있다.

## 7-3. Follow-up papers

- Score-based Generative Modeling through Stochastic Differential Equations
- Elucidating the Design Space of Diffusion-Based Generative Models
- Deep Networks with Stochastic Depth
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- Scalable Diffusion Models with Transformers
- Simple and Effective Masked Diffusion Language Models
- Block Diffusion: Interpolating between Autoregressive and Diffusion Language Models
- NoProp: Training Neural Networks without Back-propagation or Forward-propagation

# 8. Summary

- DiffusionBlocks는 residual network를 diffusion denoising trajectory로 해석해 block-wise training objective를 만든다.
- 핵심은 layer를 나누는 것이 아니라, block마다 특정 noise range를 담당하는 independent denoiser role을 부여하는 것이다.
- Equi-probability partitioning은 block별 denoising 난도를 맞추기 위한 중요한 design choice다.
- 실험은 ViT, DiT, masked diffusion LM, AR LM, recurrent-depth model에서 baseline과 비교 가능한 성능을 보여준다.
- 가장 강한 claim은 total training compute 절감보다 peak memory reduction이며, diffusion inference와 recurrent-depth training에서는 compute pattern도 더 유리해질 수 있다.
