---
layout: single
title: "ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models Review"
categories: Study-concept
tag: [Diffusion, GenerativeModels, ImageGeneration, 3DGeneration]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2405.13729)

[Project page](https://ruixu.me/html/ComboStoc/index.html)

[Code link](https://github.com/Xrvitd/ComboStoc)

ComboStoc은 "diffusion model을 더 크게 만들었다"거나 "새 sampler를 만들었다"는 논문이 아니다. 이 논문이 진짜로 건드리는 지점은 diffusion training에서 너무 당연하게 쓰던 **하나의 scalar timestep** 자체다.

일반적인 diffusion 또는 flow matching 계열 학습에서는 한 data sample 전체가 같은 시간 $t$에 놓인다. 이미지라면 모든 patch와 모든 latent channel이 같은 noise level에 있고, 구조화된 3D shape라면 모든 part와 모든 attribute가 같은 progression stage에 있다고 가정한다. ComboStoc은 이 가정이 고차원 데이터의 조합 구조를 충분히 보지 못하게 만든다고 본다.

핵심은 단순하다. 하나의 sample 안에서도 patch, feature dimension, part, attribute마다 서로 다른 timestep을 주면, 모델은 source와 target을 잇는 하나의 diagonal path만 보는 것이 아니라, 그 주변의 off-diagonal 조합 공간까지 학습하게 된다. 논문은 이 아이디어를 **combinatorial stochasticity**라고 부르고, 이미지 생성과 structured 3D shape generation에서 training convergence와 controllability를 동시에 개선한다고 주장한다.

> 한 줄 요약: ComboStoc은 scalar timestep을 tensorized asynchronous timestep으로 바꿔 diffusion model이 patch, feature, part, attribute 조합 공간을 더 넓게 보게 만들고, 이를 통해 ImageNet generation convergence와 structured 3D shape generation, graded control을 개선하는 training scheme이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- diffusion 개선을 architecture나 sampler가 아니라 **training path coverage** 문제로 다시 본다.
- 이미지 latent patch와 3D semantic part를 같은 관점, 즉 dimension과 attribute의 조합 공간으로 묶어 설명한다.
- scalar $t$를 tensor $T$로 바꾸는 작은 수정만으로, training acceleration과 test-time graded control이라는 두 효과를 동시에 만든다.
- structured 3D generation처럼 part, bounding box, shape code가 얽힌 문제에서 diffusion training이 왜 더 어려운지 직관적으로 보여준다.
- 최근 multimodal generation, controllable generation, 3D asset generation 쪽에서 asynchronous schedule이 하나의 재사용 가능한 설계 패턴이 될 수 있음을 보여준다.

ComboStoc의 가장 중요한 메시지는 이것이다. **Diffusion model이 보는 trajectory는 model architecture만큼 중요하다.** 모델이 같은 data point를 보더라도, 모든 dimension을 같은 시간에 움직이게 할지, 아니면 서로 다른 진행 상태의 조합을 보게 할지에 따라 학습 난이도와 제어 가능성이 달라진다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 diffusion generative model에서 **combinatorial complexity가 training stage에서 충분히 샘플링되지 않는다**는 점이다.

일반적인 stochastic interpolant 또는 flow matching setting에서는 source sample $z$와 target data sample $x_1$ 사이를 아래처럼 잇는다.

$$
x_t = (1 - t) z + t x_1
$$

여기서 $t$는 scalar다. 즉 한 sample 안의 모든 coordinate가 같은 interpolation stage에 놓인다.

이미지 latent를 생각하면, $x_1$은 $C x H x W$ 형태의 feature grid다. Transformer 기반 diffusion model은 이를 patch token으로 보고 병렬 생성한다. 그런데 각 patch는 위치 정보를 갖고 있고, 각 patch 안에는 여러 channel이 있으며, 이 patch와 channel 사이의 조합이 이미지의 구조를 만든다.

3D structured shape에서는 이 문제가 더 강해진다. 하나의 object는 여러 semantic part로 구성되고, 각 part에는 다음과 같은 attribute가 있다.

- part existence
- bounding box center
- bounding box size
- part shape code

즉 모델은 단순히 하나의 continuous vector를 생성하는 것이 아니라, part의 존재 여부, 위치, 크기, 세부 shape code가 서로 맞는 조합을 만들어야 한다.

문제는 standard training이 이런 조합 공간을 충분히 보지 못한다는 것이다. 모든 dimension이 같은 $t$를 가지면, 모델은 source에서 target으로 가는 synchronized diagonal path를 주로 본다. 하지만 test-time generation이나 editing에서는 일부 region은 거의 보존되고, 일부 region은 거의 noise이며, 일부 attribute만 고정되는 상황이 자주 생긴다.

따라서 논문의 질문은 다음과 같다.

- diffusion model이 dimension과 attribute의 조합 공간을 더 넓게 보게 만들 수 있는가?
- 그 과정에서 backbone을 크게 바꾸지 않고 training scheme만 수정할 수 있는가?
- 이렇게 학습한 model을 test time에 graded control 또는 part-level control로 사용할 수 있는가?

## 1-2. Why previous approaches are insufficient

기존 diffusion 개선은 보통 세 축에 집중한다.

| Approach | Main idea | Limitation from ComboStoc view |
| --- | --- | --- |
| Better backbone | DiT, SiT처럼 transformer architecture를 개선 | model capacity는 늘지만 training path coverage 문제를 직접 다루지는 않는다 |
| Better sampler | ODE/SDE solver, distillation, consistency model | inference cost와 sample quality를 다루지만 training distribution 자체는 그대로일 수 있다 |
| Better conditioning | mask, ControlNet, task-specific condition | 특정 task에는 강하지만, partial observation의 degree를 continuous하게 다루기 어렵다 |

ComboStoc은 이들과 다른 축을 본다. 문제는 network가 약하다는 것만이 아니라, network가 학습 중에 보는 noisy sample의 조합이 너무 제한적이라는 것이다.

예를 들어 image inpainting을 생각해보면, binary mask는 특정 region을 보존하고 나머지를 생성한다. 하지만 실제 편집에서는 보존 강도가 binary가 아닐 수 있다. 얼굴은 강하게 보존하고, 주변 물체는 반쯤 유지하고, 배경은 자유롭게 생성하는 식의 graded control이 필요하다. Standard scalar timestep training은 이런 continuous preservation map을 자연스럽게 다루기 어렵다.

3D structured shape도 비슷하다. 어떤 part의 shape는 고정하고, 다른 part의 bounding box만 바꾸고, 나머지 part existence는 새로 생성하고 싶을 수 있다. 이런 경우 part와 attribute마다 다른 generation progress가 필요하다.

ComboStoc의 관점에서 기존 방식의 부족함은 다음처럼 정리된다.

- scalar timestep은 sample 내부 structure를 지나치게 동기화한다.
- patch, feature, part, attribute의 조합을 training 중 충분히 보지 않는다.
- task-specific editing module을 따로 붙이지 않으면 graded partial control이 어렵다.
- structured domain에서는 part permutation, part existence, attribute coupling까지 있어서 diagonal path만으로는 학습 신호가 약할 수 있다.

# 2. Core Idea

## 2-1. Main contribution

ComboStoc의 핵심 기여는 scalar timestep $t$를 tensorized timestep $T$로 일반화하는 것이다.

기존 interpolant는 아래와 같다.

$$
x_t = (1 - t) z + t x_1
$$

ComboStoc은 이를 다음처럼 바꾼다.

$$
x_T = (1 - T) \odot z + T \odot x_1
$$

여기서 $T$는 data sample과 compatible한 shape를 갖는 timestep tensor다. 이미지라면 patch 또는 feature dimension별로 다른 value를 가질 수 있고, 3D shape라면 part 또는 attribute dimension별로 다른 value를 가질 수 있다.

이 변화가 만드는 효과는 다음과 같다.

1. **Off-diagonal training samples**
   - 모든 coordinate가 같은 time에 있는 sample만 보지 않는다.
   - 어떤 patch는 target에 가깝고, 어떤 patch는 source noise에 가깝고, 어떤 channel은 중간 단계에 있을 수 있다.

2. **Combinatorial path coverage**
   - source-target pair 하나에서 더 많은 interpolation combination을 만들 수 있다.
   - 특히 dimension과 attribute가 많은 structured data에서 training signal이 풍부해진다.

3. **Asynchronous test-time generation**
   - inference에서도 서로 다른 patch, part, attribute에 다른 initial time을 줄 수 있다.
   - 따라서 binary mask보다 더 유연한 graded control이 가능해진다.

4. **Backbone-light modification**
   - 핵심은 model을 새로 크게 설계하는 것이 아니다.
   - timestep embedding과 velocity target을 바꾸는 쪽에 가깝다.

## 2-2. Design intuition

ComboStoc의 설계 직관은 diagonal path와 off-diagonal space의 차이로 이해하면 쉽다.

Scalar timestep에서는 한 source-target pair가 만들어내는 training point가 아래와 같은 선 위에 있다.

```text
z ---- t ---- x_1
```

하지만 data sample이 실제로는 patch, channel, part, attribute의 집합이라면, 각 coordinate가 반드시 같은 속도로 target으로 갈 필요는 없다. 어떤 coordinate는 target에 가까워지고, 어떤 coordinate는 아직 source에 가까운 상태도 학습해야 한다.

그래서 ComboStoc은 sample 전체를 하나의 time으로 묶지 않는다. 모델에게 다음과 같은 상태를 보여준다.

```text
patch 1: t = 0.9
patch 2: t = 0.2
patch 3: t = 0.5
channel 1: t = 0.7
channel 2: t = 0.1
```

이런 sample은 standard diffusion training의 관점에서는 이상한 off-diagonal state다. 하지만 editing과 structured generation 관점에서는 매우 자연스러운 상태다. 사용자가 어떤 region은 보존하고, 어떤 region은 새로 만들고, 어떤 attribute만 condition으로 주는 상황과 비슷하기 때문이다.

즉 ComboStoc은 training data augmentation이라기보다 **trajectory augmentation**에 가깝다. 원본 data sample을 늘리는 것이 아니라, source와 target 사이를 오가는 path의 sampling density를 늘린다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Diffusion training에서 dimension / attribute combination space를 더 넓게 샘플링 |
| Core change | scalar timestep $t$를 tensorized asynchronous timestep $T$로 대체 |
| Base framework | stochastic interpolants, SiT-style image generation, structured 3D shape generation |
| Image axis | patch axis, feature vector axis |
| 3D shape axis | part axis, attribute axis, feature vector axis |
| Key technical issue | asynchronous $T$에서 naive velocity target을 쓰면 off-diagonal drift가 생길 수 있음 |
| Test-time feature | region / part / attribute마다 다른 generation progress를 주는 graded control |
| Difference from prior work | backbone이나 sampler보다 training path coverage를 직접 바꾼다 |

## 3-2. Module breakdown

### 1) Tensorized timestep schedule

ComboStoc의 가장 중요한 module은 timestep schedule 자체다. 논문은 image domain과 structured 3D shape domain에서 서로 다른 granularity를 둔다.

이미지에서는 latent image $x_1$이 $C x H x W$ 형태다. 따라서 timestep $T$도 이 구조에 맞춰 구성할 수 있다.

대표적인 setting은 다음과 같다.

| Setting | Meaning |
| --- | --- |
| unsync_none | 모든 patch와 feature dimension이 같은 timestep을 사용 |
| unsync_patch | patch별로 다른 timestep을 사용 |
| unsync_vec | feature vector dimension별로 다른 timestep을 사용 |
| unsync_all | patch와 feature dimension을 모두 비동기화 |

3D structured shape에서는 part와 attribute가 있다. 논문은 3 x 2 조합으로 6가지 setting을 둔다.

| Setting | Meaning |
| --- | --- |
| unsync_none | 모든 part와 attribute가 같은 timestep |
| unsync_part | part별 timestep만 다르게 설정 |
| unsync_att | attribute-level timestep 사용 |
| unsync_att_part | attribute와 part를 함께 비동기화 |
| unsync_vec | feature vector dimension level 비동기화 |
| unsync_all | part, attribute, feature dimension을 가장 세밀하게 비동기화 |

여기서 중요한 것은 granularity다. 단순히 random timestep을 여러 개 뽑는 것이 아니라, data structure에 맞춰 어떤 축을 비동기화할지 정해야 한다.

### 2) Timestep embedding adaptation

기존 SiT는 scalar timestep을 embedding한 뒤 class conditioning과 함께 modulation에 사용한다. ComboStoc에서는 $T$가 tensor이므로 timestep embedding도 그에 맞게 바뀌어야 한다.

논문은 $T$를 data shape에 맞게 입력하고, 이를 MLP와 patch embedding 계열 module을 통해 network conditioning으로 넣는다. 이 부분은 작아 보이지만 중요하다. $T$가 scalar일 때는 하나의 global noise level만 알려주면 되지만, ComboStoc에서는 patch별, channel별 noise level을 network가 알아야 한다.

즉 model input은 단순히 noisy sample $x_T$가 아니라, 그 sample의 각 coordinate가 어느 stage에 있는지 알려주는 tensorized schedule 정보까지 포함한다.

### 3) Velocity compensation

ComboStoc에서 가장 주의해야 할 부분은 velocity target이다.

Scalar interpolant에서는 velocity가 단순하다.

$$
u = x_1 - z
$$

하지만 asynchronous $T$에서는 sample이 diagonal path 밖에 있다. 이때 그대로 $x_1 - z$를 예측하게 만들면, test-time integration 중 target data point에서 벗어나는 drift가 생길 수 있다.

논문은 이 문제를 완화하기 위해 compensation drift를 도입한다.

$$
u_{combo} =  u + u_{cmpn}
$$

여기서 $u_{cmpn}$은 off-diagonal state에서 trajectory가 target 쪽으로 안정적으로 돌아오도록 보정하는 항이다. 논문은 두 가지 possible approach를 제시하지만, 실험에서는 제한된 compute 때문에 첫 번째 접근, 즉 off-diagonal drift를 gradient descent로 줄이는 방법만 사용했다고 설명한다.

이 부분은 ComboStoc을 단순한 timestep noise augmentation과 구분하는 핵심이다. 비동기 timestep을 주는 것만으로 끝나지 않고, 그 off-diagonal state에서 어떤 vector field를 학습해야 하는지 다시 생각해야 한다.

### 4) Image generation implementation

이미지 실험은 SiT를 baseline으로 한다. ImageNet-scale generation에서 이미 강한 transformer diffusion baseline 위에 ComboStoc을 얹는다.

구성은 다음과 같다.

- image는 VAE encoder를 통해 latent image로 바뀐다.
- latent image shape는 $C x H x W$다.
- model은 noisy latent $x_T$, class label $c$, tensorized timestep $T$를 조건으로 velocity를 예측한다.
- 학습 중에는 unsync_none, unsync_patch, unsync_vec, unsync_all 같은 granularity를 비교한다.

중요한 점은 ComboStoc이 image generation backbone 자체를 크게 바꾸는 논문이 아니라는 것이다. Backbone은 SiT-style transformer를 유지하고, training schedule과 timestep embedding을 바꾸는 쪽에 가깝다.

### 5) Structured 3D shape generation implementation

3D shape 실험은 논문에서 매우 중요한 축이다. 이유는 image보다 combinatorial complexity가 더 노골적으로 드러나기 때문이다.

논문은 structured 3D object를 part collection으로 표현한다.

$$
x = \{p_i\}, i = 1, ..., L
$$

여기서 $L = 256$으로 설정해 dataset의 최대 part 수를 cover한다. 각 part는 다음처럼 표현된다.

$$
p = (s, b, e)
$$

- $s$: part existence
- $b$: bounding box center와 size
- $e$: normalized coordinate에서 part shape를 encoding한 latent shape code

논문 설명 기준으로 $b$는 center $(x, y, z)$와 length, width, height를 포함하고, $e$는 512-dimensional shape code다.

여기서 diffusion model은 velocity보다 target data sample $x_1$을 직접 예측하는 방식으로 단순화된다.

$$
f_\theta(x_T, c, T) = x_1
$$

3D shape에서는 $T$가 part와 attribute, feature dimension마다 다른 timestep을 가질 수 있다. 이게 ComboStoc의 장점이 가장 선명하게 보이는 부분이다. part existence만 target에 가깝고 shape code는 아직 noise에 가깝거나, 어떤 part의 bounding box만 고정된 상태 같은 조합을 training 중에 볼 수 있기 때문이다.

### 6) Test-time asynchronous control

ComboStoc이 흥미로운 이유는 training acceleration만이 아니다. 같은 idea가 inference-time control로 이어진다.

사용자는 test time에 region, channel, part, attribute마다 다른 initial time을 줄 수 있다. 예를 들어 image에서는 다음과 같은 설정이 가능하다.

- 얼굴 region은 높은 preservation weight로 유지한다.
- 얼굴 주변은 중간 정도만 유지한다.
- 배경은 거의 새로 생성한다.

3D shape에서는 다음이 가능하다.

- 일부 part shape를 고정한다.
- 나머지 part는 모델이 새로 생성하게 한다.
- part assembly 또는 shape completion에 활용한다.

이건 binary mask 기반 inpainting보다 더 부드러운 control interface다. Mask가 0 또는 1만 표현하는 것이 아니라, $T$ map이 preservation degree를 continuous하게 표현할 수 있다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 두 domain에서 ComboStoc을 평가한다.

| Domain | Dataset / setting | Purpose |
| --- | --- | --- |
| Image generation | ImageNet-scale latent image generation | SiT baseline 대비 convergence와 FID 개선 확인 |
| Structured 3D shape generation | PartNet, 약 18K shapes, mainly chair and table | part / attribute / feature 조합이 강한 domain에서 ComboStoc 필요성 확인 |

ImageNet에서는 image를 VAE latent로 바꾼 뒤 SiT-style transformer를 학습한다. Structured 3D shape에서는 semantic part structure를 leaf-level part collection으로 표현하고, 각 part의 existence, bounding box, shape code를 함께 생성한다.

PartNet 실험에서 중요한 점은 dataset이 비교적 작다는 것이다. 논문은 약 18K shape라고 설명하고, 대부분 chair와 table category에 집중되어 있다. 따라서 3D 실험은 broad 3D asset generation 전체를 해결했다기보다, structured shape representation에서 combinatorial stochasticity의 효과를 확인하는 controlled setting에 가깝다.

## 4-2. Training strategy

Image domain에서는 baseline SiT와 비교한다. 논문은 unsync_none, unsync_patch, unsync_vec, unsync_all을 비교하고, 더 강한 combinatorial stochasticity를 사용할수록 더 빠르게 구조적인 image가 나타난다고 설명한다.

Structured 3D shape에서는 6가지 configuration을 비교한다.

- unsync_none
- unsync_part
- unsync_att
- unsync_att_part
- unsync_vec
- unsync_all

논문은 PartNet dataset이 상대적으로 작기 때문에 ImageNet과 달리 batch mixing scheme을 복잡하게 쓰지 않고, 각 batch의 sample 전체에 대응하는 asynchronous timestep setting을 적용했다고 설명한다. 결과는 1.5K epochs 기준으로 보고된다. 이유는 더 이른 checkpoint에서는 unsync_none 같은 setting이 valid manifold shape로 decoding되지 않아 metric 평가가 어렵기 때문이다.

## 4-3. Engineering notes

실무적으로 중요한 포인트는 다음과 같다.

1. **Schedule granularity가 핵심 hyperparameter다**
   - image에서는 patch와 vector dimension을 나눌 수 있다.
   - 3D shape에서는 part, attribute, feature vector dimension을 나눌 수 있다.
   - 어떤 축을 비동기화할지는 data representation에 강하게 의존한다.

2. **Timestep embedding을 대충 바꾸면 baseline보다 나빠질 수 있다**
   - 논문은 unsync_none이 baseline SiT보다 약간 나쁜 이유로 timestep embedding module 차이를 언급한다.
   - 따라서 ComboStoc의 gain은 단순 module 변경이 아니라 asynchronous schedule 자체를 제대로 쓰는 setting에서 봐야 한다.

3. **Velocity compensation은 필수적이다**
   - asynchronous $T$는 off-diagonal state를 만든다.
   - 이 상태에서 original velocity만 예측하면 integration drift가 생길 수 있다.
   - 실제 구현에서는 compensation drift 또는 target prediction choice를 신중하게 설계해야 한다.

4. **Structured data에서는 permutation issue가 있다**
   - 3D part representation에서는 part index permutation이 같은 shape를 나타낼 수 있다.
   - image grid처럼 fixed order가 있는 data보다 더 까다롭다.
   - 이 때문에 part-level diffusion은 representation design과 schedule design이 함께 중요하다.

5. **ComboStoc은 sampler trick이 아니라 training distribution change다**
   - inference만 바꿔서 되는 접근이 아니다.
   - model이 asynchronous state를 training 중에 충분히 봐야 test-time graded control이 자연스럽다.

# 5. Evaluation

## 5-1. Main results

### Image generation

논문 Table 1의 핵심 수치는 ImageNet FID-50K 기준으로 정리할 수 있다.

| Model | Params | Training steps | FID |
| --- | ---: | ---: | ---: |
| DiT-XL | 675M | 400K | 19.5 |
| SiT-XL | 675M | 400K | 17.2 |
| ComboStoc | 673M | 400K | 15.69 |
| DiT-XL | 675M | 800K | 14.3 |
| SiT-XL | 675M | 800K | 12.6 |
| ComboStoc | 673M | 800K | 11.41 |

이 표의 메시지는 명확하다. 같은 400K, 800K training step에서 ComboStoc은 SiT-XL보다 낮은 FID를 보고한다. 즉 논문의 claim은 단순 final SOTA가 아니라 **같은 iteration budget에서 더 빠른 convergence**에 가깝다.

다만 CFG를 켠 long training comparison은 조심해서 읽어야 한다.

| Model | Params | Training steps | FID with cfg=1.5 |
| --- | ---: | ---: | ---: |
| DiT-XL | 675M | 7M | 2.27 |
| SiT-XL | 675M | 7M | 2.06 |
| ComboStoc | 673M | 800K | 2.85 |

여기서 ComboStoc은 800K step으로도 strong sample quality를 보이지만, 7M step까지 학습한 DiT-XL, SiT-XL을 넘는다는 claim은 아니다. 이 부분은 논문을 hype 없이 읽을 때 중요하다. ComboStoc의 strength는 **training step efficiency**와 **control interface**다.

### Structured 3D shape generation

3D shape에서는 결과가 더 흥미롭다. 논문은 unsync_none이 거의 의미 있는 shape를 만들지 못한다고 설명한다. 반대로 unsync_part, unsync_att, unsync_vec, unsync_all처럼 더 많은 combinatorial axis를 쓰는 setting일수록 더 그럴듯한 semantic part structure가 나온다.

논문 appendix의 comparison에서는 ComboStoc 결과가 StructureNet, StructRe처럼 part hierarchy를 직접 활용하는 baseline과 비교된다. 중요한 차이는 이렇다.

- StructureNet과 StructRe는 part hierarchy를 progressive generation constraint로 사용한다.
- ComboStoc은 leaf-level part를 직접 생성하고, hierarchy constraint를 직접 쓰지 않는다.
- 그럼에도 ComboStoc은 일부 metric에서 comparable한 결과를 보고하고, visual diversity 측면에서 장점을 보인다고 설명한다.

따라서 3D result는 "ComboStoc이 모든 3D structured generation baseline을 압도했다"가 아니다. 더 정확히는 **hierarchical prior 없이도 diffusion model이 structured part distribution을 학습할 수 있게 만드는 training scheme으로 작동했다**는 것이다.

### Test-time control

ComboStoc은 asynchronous timestep을 inference에도 사용할 수 있다. 논문은 다음 application을 보여준다.

- graded image inpainting
- soft preservation map 기반 image variation
- structured 3D shape completion
- part-level shape specification
- random part assembly

여기서 핵심은 binary condition이 아니라 degree control이다. 예를 들어 어떤 image region을 완전히 고정하거나 완전히 지우는 것이 아니라, region마다 다른 $T$ value를 주어 보존 강도를 조절할 수 있다.

## 5-2. What really matters in the experiments

### 1) 이 논문은 architecture novelty보다 training coverage 논문이다

ComboStoc을 읽을 때 가장 먼저 구분해야 할 것은 이것이다. 논문이 보여주는 gain은 새로운 transformer block에서 나온 것이 아니다. 같은 backbone 계열에서 training sample이 놓이는 timestep geometry를 바꾼 효과다.

그래서 이 논문은 diffusion architecture paper라기보다 **diffusion training distribution paper**에 가깝다.

### 2) unsync_none은 좋은 control이다

논문에서 unsync_none은 중요한 control baseline이다. 같은 modified timestep embedding module을 쓰되, 실제 asynchronous schedule을 쓰지 않는 setting에 가깝다. 이 setting이 baseline SiT보다 약간 나빠질 수 있다는 점은 중요하다.

즉 ComboStoc의 gain은 단순히 module을 바꿔서 생긴 것이 아니라, 실제로 combinatorial stochasticity를 더 많이 사용할 때 나타난다는 해석을 지지한다.

### 3) 3D shape가 image보다 더 좋은 stress test다

ImageNet FID 개선도 의미 있지만, 개인적으로는 structured 3D shape 결과가 논문의 메시지를 더 잘 보여준다고 본다.

이미지는 patch grid가 고정되어 있고, channel structure도 비교적 일정하다. 반면 3D part-based shape에서는 part existence, position, box size, shape code, category-level regularity가 모두 섞인다. 이 경우 scalar timestep은 훨씬 더 강한 병목이 된다.

그래서 3D 실험은 ComboStoc이 단순 image trick이 아니라 structured generation 일반의 training recipe가 될 수 있음을 보여준다.

### 4) graded control은 downstream application 관점에서 더 중요할 수 있다

FID improvement는 다른 training trick이나 더 긴 학습으로도 어느 정도 따라잡을 수 있다. 하지만 asynchronous timestep을 그대로 control interface로 쓰는 점은 더 독특하다.

특히 image editing과 3D editing에서는 binary mask만으로 충분하지 않은 경우가 많다.

- region마다 preservation strength가 다르다.
- shape part마다 condition confidence가 다르다.
- geometry는 보존하고 texture는 바꾸고 싶다.
- 일부 attribute만 고정하고 나머지는 생성하고 싶다.

ComboStoc은 이런 요구를 하나의 $T$ map으로 표현할 수 있는 가능성을 보여준다.

### 5) final SOTA보다 sample efficiency로 읽어야 한다

Table 1을 잘못 읽으면 ComboStoc이 DiT/SiT final model을 압도한다고 생각할 수 있다. 하지만 cfg=1.5 row를 보면 7M step DiT/SiT가 여전히 더 낮은 FID를 갖는다. ComboStoc의 장점은 800K step에서 strong quality를 얻는 convergence speed와 asynchronous control이다.

따라서 이 논문은 "new best image generator"보다 **same backbone family에서 training path를 더 잘 쓰는 방법**으로 읽는 편이 맞다.

# 6. Limitations

1. **이론적 설명은 아직 완결적이지 않다**
   - 논문은 combinatorial complexity와 insufficient sampling을 강하게 주장하지만, 어떤 data distribution에서 scalar timestep이 얼마나 부족한지에 대한 일반 이론은 제한적이다.
   - 결과는 주로 empirical evidence로 설득한다.

2. **Schedule granularity 선택이 domain-specific하다**
   - image에서는 patch와 feature vector가 자연스러운 축이다.
   - 3D shape에서는 part, attribute, feature dimension이 자연스러운 축이다.
   - video, audio, text-token diffusion, multimodal generation에서는 어떤 축을 비동기화할지 별도 설계가 필요하다.

3. **Timestep embedding mismatch가 baseline 해석을 어렵게 만든다**
   - unsync_none이 baseline SiT보다 약간 불리해지는 이유로 embedding module 차이가 언급된다.
   - 따라서 ablation은 architecture adaptation과 schedule effect를 완전히 분리해서 보기 어렵다.

4. **Velocity compensation은 더 깊은 검증이 필요하다**
   - asynchronous $T$에서는 naive velocity target이 drift를 만들 수 있다.
   - 논문은 두 가지 compensation approach를 제시하지만, 실험은 제한된 compute 때문에 첫 번째 접근 위주로 진행된다.
   - 더 다양한 solver와 sampler에서 compensation design이 어떻게 작동하는지는 추가 확인이 필요하다.

5. **3D shape 실험은 dataset scale이 작다**
   - PartNet 약 18K shapes, mainly chair/table setting이다.
   - 최근 3D asset generation처럼 훨씬 더 큰 text-conditioned 3D dataset에서 같은 효과가 유지되는지는 별도 검증이 필요하다.

6. **Large-scale text-to-image setting은 아직 열려 있다**
   - ImageNet class-conditioned generation에서는 효과가 보인다.
   - 하지만 prompt-conditioned text-to-image, video diffusion, world model generation에서는 conditioning interface가 훨씬 복잡하다.
   - 이 경우 asynchronous timestep이 condition alignment를 돕는지, 아니면 modality mismatch를 키우는지 확인해야 한다.

7. **Control quality는 user-facing metric으로 더 봐야 한다**
   - graded control은 매우 흥미롭지만, 실제 editing product에서는 identity preservation, boundary smoothness, semantic consistency, user controllability를 따로 평가해야 한다.
   - FID만으로 이 기능의 가치를 평가하기 어렵다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 흥미로운 이유는 diffusion model의 개선 축을 조금 다른 곳으로 옮기기 때문이다. 보통 generation paper를 읽을 때는 backbone, loss, sampler, dataset을 먼저 본다. ComboStoc은 그 사이에 있는 **training trajectory design**을 전면에 올린다.

나에게 가장 중요한 인사이트는 다음이다.

> Generation model은 data sample만 학습하는 것이 아니라, source에서 target으로 가는 path distribution도 학습한다.

이 관점은 diffusion뿐 아니라 multimodal generation에도 중요하다. 예를 들어 unified model이 RGB, depth, normal, mask를 동시에 생성하거나, video frame과 camera trajectory를 함께 생성하거나, 3D part와 geometry를 함께 생성할 때, 모든 modality와 attribute가 같은 denoising stage에 있어야 한다는 가정은 너무 강할 수 있다.

ComboStoc은 그 가정을 깔끔하게 깨는 방법을 보여준다.

## 7-2. Reuse potential

재사용하고 싶은 포인트는 다음과 같다.

1. **Timestep as structured control map**
   - timestep을 scalar schedule이 아니라 structured control map으로 본다.
   - region, part, modality, attribute마다 다른 generation progress를 줄 수 있다.

2. **Trajectory augmentation**
   - data augmentation이 아니라 path augmentation을 설계한다.
   - 같은 source-target pair에서도 더 다양한 intermediate state를 학습하게 한다.

3. **Structured generation에 적합한 schedule granularity**
   - 3D shape, document layout, UI generation, scene graph generation처럼 part/attribute가 있는 task에 특히 잘 맞을 수 있다.

4. **Binary mask를 continuous control로 확장**
   - inpainting과 editing에서 mask를 0/1로만 보지 않는다.
   - preservation degree를 continuous $T$ map으로 표현한다.

5. **Conditioned generation과 joint generation의 연결**
   - 어떤 attribute는 target에 가깝게 두고, 어떤 attribute는 noise에서 시작하게 하면, 같은 model로 generation과 conditional completion을 함께 다룰 수 있다.

개인적으로는 document AI나 multimodal UI generation에도 이 관점이 유용해 보인다. 예를 들어 document page에서 layout은 강하게 보존하고 text style은 중간 정도로 유지하며 image region은 새로 생성하는 식의 partial generation이 필요할 수 있다. 이런 경우 ComboStoc식 timestep map이 하나의 interface가 될 수 있다.

## 7-3. Follow-up papers

- Scalable Interpolant Transformer
- DiT: Scalable Diffusion Models with Transformers
- Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
- StructureNet: Hierarchical Graph Neural Networks for 3D Shape Generation
- StructRe: Reconstructing Structured 3D Shapes
- ControlNet
- REPA: Representation Alignment for Diffusion Transformers
- MMGen: Unified Multi-modal Image Generation and Understanding in One Go

# 8. Summary

- ComboStoc은 diffusion model의 scalar timestep을 tensorized asynchronous timestep으로 바꿔 dimension과 attribute 조합 공간을 더 넓게 샘플링한다.
- 핵심 수식은 $x_t = (1 - t) z + t x_1$에서 $x_T = (1 - T) \odot z + T \odot x_1$로 바뀌는 것이다.
- ImageNet generation에서는 같은 400K, 800K training step에서 SiT-XL보다 낮은 FID를 보고하며, convergence speed 측면의 장점이 크다.
- Structured 3D shape generation에서는 part, attribute, feature dimension의 비동기화가 거의 필수적인 training signal로 작동한다.
- 가장 재사용성 높은 아이디어는 timestep을 단순 noise level이 아니라 region, part, attribute별 control map으로 보는 것이다.
