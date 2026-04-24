---
layout: single
title: "MultiWorld: Scalable Multi-Agent Multi-View Video World Models Review"
categories: Study-concept
tag: [World-Model, Video-Generation, Multi-Agent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.18564)

MultiWorld는 video world model을 단일 agent의 action-conditioned video prediction에서 **multi-agent, multi-view shared environment simulation**으로 확장하려는 논문이다. 기존 interactive video world model은 보통 하나의 agent가 움직이고, 하나의 camera view에서 다음 frame을 예측하는 문제로 설정된다. 하지만 실제 game, robotics, embodied AI 환경에서는 여러 agent가 동시에 행동하고, 각 agent 또는 camera가 같은 world를 서로 다른 view에서 관찰한다.

이 논문이 흥미로운 이유는 단순히 여러 view의 video를 한꺼번에 생성하는 것이 아니라, **agent별 action controllability**와 **view 간 3D consistency**를 동시에 architectural problem으로 잡았다는 점이다. MultiWorld는 Multi-Agent Condition Module, Global State Encoder, parallel view generation을 결합해 multi-player game과 multi-robot manipulation을 하나의 framework 안에서 다룬다.

> 한 줄 요약: MultiWorld는 여러 agent의 action을 구분하고 상호작용을 모델링하는 MACM과, multi-view observation을 3D-aware global state로 압축하는 GSE를 통해 scalable multi-agent multi-view video world model을 구현하려는 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Video world model이 단순 video generation에서 **interactive simulator** 방향으로 이동하고 있다.
- Single-agent world model만으로는 game, robotics, embodied AI에서 필요한 multi-agent interaction을 설명하기 어렵다.
- Multi-view consistency는 visual quality보다 더 system-level 문제다. 각 view가 따로 그럴듯해도 shared environment가 어긋나면 simulator로 쓰기 어렵다.
- MACM과 GSE는 LLM/VLM의 attention 설계와도 연결되는 흥미로운 abstraction이다. Agent dimension과 view dimension을 어떻게 factorize할 것인지 보여준다.

내가 보기엔 MultiWorld의 핵심은 더 좋은 video generator가 아니라, **shared world state를 중심으로 여러 agent와 여러 view를 묶는 conditioning contract**다. 즉 world model을 하나의 video predictor로 보는 대신, action source와 observation sink가 여러 개인 simulation system으로 재정의한다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 문제는 **multi-agent multi-view world modeling**이다.
- 입력은 여러 camera view의 initial observation과 각 agent의 per-frame action이다.
- 출력은 각 camera view에서의 future video sequence다.
- 중요한 조건은 세 가지다.
  - 각 agent의 action이 해당 agent에 정확히 연결되어야 한다.
  - agent들 사이의 interaction과 shared environment 변화가 반영되어야 한다.
  - 서로 다른 view에서 생성된 video가 같은 world state를 바라보는 것처럼 일관되어야 한다.

이를 간단히 쓰면 다음과 같은 조건부 generation 문제로 볼 수 있다.

$$
p(x_{1:K}^{t+1:t+T} | x_{1:K}^{1:t}, a_{1:N}^{t+1:t+T})
$$

여기서 $N$은 agent 수, $K$는 camera view 수다. 기존 single-agent setting은 사실상 $N = 1$, $K = 1$에 가깝다. MultiWorld가 다루는 setting은 $N$과 $K$가 모두 커질 수 있고, 심지어 configuration이 고정되어 있지 않을 수 있다.

## 1-2. Why previous approaches are insufficient

기존 접근은 크게 세 가지로 볼 수 있다.

1. **Standard image-action-to-video model**
   - 각 view를 독립적인 image-action-conditioned video generation 문제로 처리한다.
   - 구현은 단순하지만 view 간 shared environment를 보장하지 못한다.
   - agent가 사라지거나, 한 view에서는 발생한 interaction이 다른 view에서는 사라지는 문제가 생길 수 있다.

2. **Concat-View model**
   - 여러 view를 하나의 video input처럼 붙여서 처리한다.
   - fixed number of views에서는 어느 정도 cross-view signal을 줄 수 있다.
   - 하지만 view 수가 늘어나면 memory와 compute가 커지고, variable view setting에 약하다.
   - 논문에서도 robotics setting의 Concat-View는 두 camera view만 학습되어 다른 방법과 직접 비교하기 어렵다고 표시한다.

3. **COMBO-style compositional multi-agent model**
   - 여러 single-agent model을 조합해 multi-agent behavior를 만들려는 방식이다.
   - agent별 dynamics는 다룰 수 있지만, agent 간 interaction을 architecture 수준에서 충분히 모델링하지 못한다.
   - shared object를 함께 밀거나, 한 agent의 행동이 다른 agent의 시야와 환경 상태에 영향을 주는 경우에 약할 수 있다.

결국 이 논문이 보는 병목은 video generator 자체보다 **conditioning structure**다. Multi-agent world model에서는 action token을 그냥 stack하면 identity ambiguity가 생기고, view token을 그냥 concat하면 scalability 문제가 생긴다. 따라서 agent dimension과 view dimension을 분리해서 설계해야 한다.

# 2. Core Idea

## 2-1. Main contribution

MultiWorld의 핵심 기여는 다음과 같이 정리할 수 있다.

- **Multi-Agent Condition Module**
  - Agent Identity Embedding으로 action token에 agent identity를 부여한다.
  - agent token 사이 self-attention으로 inter-agent interaction을 모델링한다.
  - Adaptive Action Weighting으로 active agent의 action을 더 강하게 반영한다.

- **Global State Encoder**
  - frozen VGGT backbone을 사용해 multi-view observation에서 3D-aware latent state를 추출한다.
  - 여러 partial observation을 compact global representation으로 압축한다.
  - 이 global state를 DiT backbone에 cross-attention으로 주입해 view 간 consistency를 높인다.

- **Scalable multi-view generation**
  - multi-view generation을 shared global state를 조건으로 한 여러 single-view generation task로 분해한다.
  - view별 video를 병렬 생성할 수 있어 view 수가 늘어도 sequential rendering처럼 latency가 선형으로 늘지 않도록 설계한다.

- **Multi-domain evaluation**
  - It Takes Two 기반 multi-player video game dataset과 RoboFactory 기반 multi-robot manipulation dataset에서 평가한다.
  - visual quality, action following, multi-view reprojection error를 함께 본다.

## 2-2. Design intuition

이 논문의 설계 직관은 단순하다. Multi-agent multi-view world model은 두 가지 symmetry를 깨야 한다.

첫째, agent symmetry다. 여러 agent의 action이 같은 encoder를 통과하면, 모델은 어떤 action이 어떤 agent에 속하는지 혼동할 수 있다. 예를 들어 Agent 1이 왼쪽으로 움직이고 Agent 2가 오른쪽으로 움직이는 경우와, 그 반대의 경우는 action set만 보면 비슷해 보일 수 있다. 그래서 **Agent Identity Embedding**이 필요하다.

둘째, view independence다. 각 view를 독립적으로 생성하면 frame 하나하나는 그럴듯할 수 있지만, shared environment가 view마다 다르게 변할 수 있다. 같은 물체가 한 view에서는 움직였는데 다른 view에서는 정지해 있거나, shadow와 footprint가 view마다 불일치할 수 있다. 그래서 **3D-aware global state**가 필요하다.

MultiWorld의 핵심은 agent와 view를 하나의 거대한 sequence로 밀어 넣지 않는다는 점이다. Agent dimension은 MACM으로 정리하고, view dimension은 GSE로 정리한다. 이 factorization이 scalability의 핵심이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Multi-agent action을 따르면서 multi-view-consistent future video를 생성 |
| Backbone | Flow Matching 기반 Transformer video generation model, Wan2.2-5B 사용 |
| Key modules | MACM, GSE, causal action cross-attention, autoregressive chunk generation |
| Agent scalability | Agent Identity Embedding과 per-agent action weighting으로 agent 수 확장 |
| View scalability | Multi-view observation을 compact global state로 압축하고 view별 generation을 병렬화 |
| Main evaluation domains | It Takes Two multi-player game, RoboFactory multi-robot manipulation |

## 3-2. Module breakdown

### 1) Flow Matching video backbone

MultiWorld는 Flow Matching 기반 video generation backbone 위에 올라간다. 논문은 Wan2.2-5B를 사용하고, 각 camera view에 대해 noisy future video token에서 target velocity를 예측하는 방식으로 학습한다.

개념적으로는 다음 목적함수로 볼 수 있다.

$$
L = E[||v_theta(z_t, t, A, E) - u_t||_2^2]
$$

여기서 $A$는 multi-agent action condition, $E$는 environment observation 또는 global state condition이다. 중요한 점은 video token이 action cross-attention을 할 때 **frame-wise causal mask**를 사용한다는 것이다. 현재 frame의 video token이 미래 action을 보지 않게 막아 long-horizon autoregressive simulation에서 leakage를 줄인다.

### 2) Multi-Agent Condition Module

MACM은 multi-agent controllability를 담당한다. 구체적으로는 다음 흐름을 따른다.

1. 각 agent의 action을 latent action token으로 embed한다.
2. Agent Identity Embedding을 action token에 주입한다.
3. agent token 사이 self-attention을 적용해 inter-agent interaction을 모델링한다.
4. Adaptive Action Weighting으로 agent별 action importance를 추정한다.
5. weighted action token을 frame별 unified action token으로 aggregate한다.
6. 이 token을 DiT backbone에 causal cross-attention으로 주입한다.

**Agent Identity Embedding**은 RoPE를 agent dimension에 적용하는 방식이다. LLM에서 position index를 구분하듯이, 여기서는 agent index를 구분한다. 이 설계의 목적은 agent order와 identity를 action token에 명시적으로 부여하는 것이다.

**Adaptive Action Weighting**은 active agent와 static agent를 구분한다. 어떤 시점에는 한 agent가 움직이고 다른 agent는 멈춰 있을 수 있다. 모든 agent action을 같은 weight로 합치면 실제 environment change를 만드는 action이 희석될 수 있다. AAW는 MLP로 action token별 weight를 예측하고, 움직임에 더 큰 영향을 주는 agent를 더 강하게 반영한다.

이 부분은 LLM의 mixture routing과 비슷하게 볼 수도 있다. 모든 agent를 같은 강도로 보는 것이 아니라, 현재 frame의 environment dynamics를 설명하는 agent에 더 큰 weight를 주는 구조다.

### 3) Global State Encoder

GSE는 multi-view consistency를 담당한다. 논문은 pretrained VGGT를 frozen backbone으로 사용해 여러 view의 observation에서 3D-aware latent feature를 추출한다. 그 다음 MLP로 DiT backbone dimension에 맞춘 compact global representation을 만들고, 이를 cross-attention condition으로 주입한다.

중요한 점은 MultiWorld가 explicit point cloud를 매번 reconstruct하지 않는다는 것이다. 대신 VGGT latent가 가진 3D spatial information을 global state로 사용한다. 따라서 GSE는 다음 역할을 한다.

- 여러 partial view를 하나의 shared environment representation으로 묶는다.
- camera 수가 달라져도 variable-length multi-view input을 처리한다.
- 각 view의 generation이 같은 global state에 anchored되도록 만든다.

내가 보기엔 GSE는 이 논문의 가장 중요한 system abstraction이다. Multi-view generation에서 view 간 attention을 직접 모든 token에 걸면 memory가 커진다. 반대로 view를 완전히 독립적으로 만들면 consistency가 깨진다. GSE는 두 극단 사이에서 **compact shared world memory** 역할을 한다.

### 4) Scalable view generation

MultiWorld는 multi-view video를 하나의 거대한 video로 concat하지 않는다. 대신 각 camera view를 별도의 image-action-conditioned video generation task로 두고, 모두 같은 global state를 조건으로 사용한다.

이 구조의 장점은 다음과 같다.

- view별 generation을 병렬로 실행할 수 있다.
- view 수가 늘어나도 architecture를 바꿀 필요가 작다.
- multi-view interaction은 token-level concat이 아니라 global state를 통해 전달된다.
- project page 기준으로 double-view simulation에서 parallel generation이 sequential generation 대비 1.5x speedup을 보고한다.

이 설계는 practical deployment 관점에서 중요하다. 실제 simulator는 하나의 hero camera만 필요한 경우도 있지만, robotics나 multi-agent game에서는 여러 actor의 observation을 동시에 만들어야 한다. 이때 sequential하게 view를 하나씩 생성하면 serving latency가 커진다.

### 5) Autoregressive long-horizon simulation

MultiWorld는 긴 horizon을 위해 chunk 단위 autoregressive generation을 사용한다. 먼저 모든 view의 첫 chunk를 생성하고, 각 view의 마지막 frame을 다시 GSE에 넣어 global state를 update한다. 그 다음 다음 chunk를 생성한다.

논문은 이 방식으로 training context window보다 2x 긴 sequence를 큰 품질 저하 없이 생성하고, 4x 긴 sequence까지도 작은 품질 저하로 확장할 수 있다고 설명한다. 다만 이 결과는 qualitative figure 중심이기 때문에, 실제 ultra-long simulation에서는 drift와 memory accumulation을 별도로 검증해야 한다.

# 4. Training / Data / Recipe

## 4-1. Data

MultiWorld는 두 가지 dataset을 사용한다.

| Dataset | Domain | Main property |
| --- | --- | --- |
| It Takes Two | Multi-player video game | 두 player의 synchronized view와 keyboard, mouse, gamepad action |
| RoboFactory | Multi-robot manipulation | 2 to 4 agents, multiple camera views, success and failure trajectories |

Video game dataset은 It Takes Two에서 real-player gameplay를 수집한다. 논문 본문 기준으로는 500 hours를 기록하고, preprocessing 후 clear actions와 stable camera motion을 가진 100 hours를 남긴다. 이 데이터는 60 fps, original resolution 2560 x 1440, over 21 million frames로 설명된다.

Robotics dataset은 RoboFactory 기반이다. 논문은 striking, two-robot stacking, three-robot stacking, four-robot passing을 포함하는 4개 multi-robot manipulation task를 사용한다. 각 task마다 1,000 successful episodes와 2,000 failure episodes를 수집해 success-only bias를 줄인다. Failure episode는 완전히 random action이 아니라 성공 trajectory에 controlled perturbation을 넣어 nearly successful failure가 되도록 만든다.

이 데이터 설계는 꽤 중요하다. World model은 성공 trajectory만 보면 실패 상황을 제대로 시뮬레이션하기 어렵다. 특히 robotics에서는 failure trajectory를 실제 robot에서 많이 모으기 어렵고 위험할 수 있다. MultiWorld는 failure trajectory simulation을 qualitative result로도 강조한다.

## 4-2. Training strategy

논문에서 공개한 주요 training recipe는 다음과 같다.

| Item | Setting |
| --- | --- |
| Base model | Wan2.2-5B |
| Frame length | 81 frames |
| Game resolution | 320 x 320 per view |
| Robot resolution | 320 x 256 per view |
| Iterations | 40,000 |
| Learning rate | 5e-5 |
| Scheduler | Cosine learning rate scheduler |
| Global batch size | 64 |
| Hardware | 8 NVIDIA A800 GPUs |
| Training time | Approximately 4 days |

학습은 action-conditioned video generation으로 볼 수 있지만, 단순히 action만 넣는 것이 아니라 MACM과 GSE를 통해 action condition과 environment condition을 구조화한다. Flow Matching objective는 video generation backbone을 학습하고, MACM/GSE는 각각 action controllability와 view consistency를 높이는 conditioning path로 작동한다.

## 4-3. Engineering notes

실무적으로 재사용할 만한 포인트는 다음과 같다.

1. **Agent identity는 action encoder 안에서 해결해야 한다**
   - 여러 agent action을 concat한 뒤 모델이 알아서 구분하길 기대하면 identity ambiguity가 생긴다.
   - RoPE 기반 AIE처럼 agent dimension에 identity를 직접 넣는 편이 안정적이다.

2. **View consistency는 shared latent state로 관리한다**
   - view token을 모두 self-attention에 넣는 방식은 view 수가 늘수록 비싸다.
   - GSE처럼 multi-view observation을 compact global state로 압축하면 scalability가 좋아진다.

3. **Static action을 그대로 평균내면 움직임이 희석된다**
   - multi-agent environment에서는 모든 agent가 항상 중요한 것은 아니다.
   - AAW는 active agent를 강조해 frame-level dynamics를 더 잘 반영하게 만든다.

4. **Failure data는 random failure보다 near-success failure가 유용하다**
   - robotics에서 완전 random action은 학습 신호가 약할 수 있다.
   - 성공 trajectory에 perturbation을 넣은 failure는 realistic failure mode를 만들기 좋다.

5. **Long-horizon generation은 memory update policy가 핵심이다**
   - chunk를 이어 붙이는 것만으로는 충분하지 않다.
   - 마지막 frame들을 다시 global state encoder에 넣어 shared world state를 갱신하는 루프가 중요하다.

# 5. Evaluation

## 5-1. Main results

논문은 visual quality, action following, multi-view consistency를 함께 평가한다. 주요 metric은 FVD, LPIPS, SSIM, PSNR, Action, RPE다. FVD, LPIPS, RPE는 낮을수록 좋고, SSIM, PSNR, Action은 높을수록 좋다.

### Multi-Player Video Game

| Method | FVD | LPIPS | SSIM | PSNR | Action | RPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 245 | 0.36 | 0.50 | 17.48 | 88.4 | 0.75 |
| Concat-View | 215 | 0.36 | 0.49 | 17.54 | 89.1 | 0.74 |
| Combo | 207 | 0.34 | 0.51 | 17.82 | 89.3 | 0.72 |
| MultiWorld | 179 | 0.35 | 0.51 | 17.72 | 89.8 | 0.67 |

Game setting에서 MultiWorld는 FVD, Action, RPE에서 가장 좋다. PSNR과 LPIPS는 Combo가 더 좋은 항목이 있으므로, 이 결과를 모든 metric에서 압도한다고 읽으면 안 된다. 더 정확한 해석은 MultiWorld가 **visual fidelity, action controllability, multi-view geometry consistency의 균형**을 가장 잘 맞춘다는 것이다.

### Multi-Robot Manipulation

| Method | FVD | LPIPS | SSIM | PSNR | Action | RPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 100 | 0.07 | 0.90 | 26.39 | 88.2 | 1.60 |
| Concat-View* | 106 | 0.06 | 0.90 | 27.44 | 92.0 | 0.82 |
| Combo | 99 | 0.08 | 0.90 | 26.49 | 88.5 | 1.54 |
| MultiWorld | 96 | 0.07 | 0.90 | 26.60 | 88.7 | 1.52 |

Robotics setting에서는 MultiWorld가 FVD 기준으로 가장 좋고, Standard/Combo 대비 Action과 RPE도 개선된다. 다만 Concat-View*는 두 camera view만 학습되어 full setting과 직접 비교하기 어렵다. 이 표는 MultiWorld가 variable agent/view setting에서 더 general framework라는 점을 보여주지만, 일부 metric만 보면 Concat-View*가 높게 보이는 항목도 있으므로 주석을 반드시 같이 읽어야 한다.

## 5-2. What really matters in the experiments

가장 중요한 실험은 ablation이다. MultiWorld의 gain이 어디서 오는지 보여주기 때문이다.

### Main component ablation

| Config | FVD | LPIPS | SSIM | PSNR | Action | RPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 245 | 0.36 | 0.50 | 17.48 | 88.4 | 0.75 |
| MACM | 228 | 0.36 | 0.51 | 17.56 | 89.7 | 0.76 |
| MACM + GSE | 179 | 0.35 | 0.51 | 17.72 | 89.8 | 0.67 |

이 표는 역할 분담이 명확하다. MACM은 action following을 크게 올린다. GSE까지 넣으면 FVD와 RPE가 크게 개선된다. 즉 **action controllability는 MACM에서, multi-view consistency는 GSE에서 온다**고 읽을 수 있다.

### MACM design ablation

Agent Identity Embedding의 base frequency에서는 base=20이 base=10k보다 좋다.

| Config | FVD | PSNR | Action |
| --- | ---: | ---: | ---: |
| base=10k | 234 | 17.53 | 89.2 |
| base=20 | 228 | 17.56 | 89.7 |

논문 해석은 LLM에서 쓰는 default base가 agent identity setting에는 너무 완만할 수 있다는 것이다. Agent 수는 token position처럼 수천 개가 아니라 훨씬 작은 dimension이므로, 인접 agent identity를 충분히 구분하려면 더 작은 base가 맞을 수 있다.

Adaptive Action Weighting도 작은 폭이지만 도움이 된다.

| Config | FVD | PSNR | Action |
| --- | ---: | ---: | ---: |
| w/o AAW | 245 | 17.48 | 88.4 |
| w/ AAW | 236 | 17.52 | 88.6 |

AAW의 수치 gain은 크지 않지만 설계 의도는 납득된다. Multi-agent video에서 모든 agent가 매 frame 같은 중요도를 갖지 않으므로, active action에 더 큰 weight를 주는 것이 자연스럽다.

### GSE backbone ablation

| Global State Encoder | FVD | LPIPS | SSIM | PSNR | RPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| w/o Global State | 228 | 0.36 | 0.51 | 17.56 | 0.75 |
| Wan VAE | 256 | 0.36 | 0.50 | 17.38 | 0.71 |
| DINOv2 | 232 | 0.36 | 0.50 | 17.48 | 0.72 |
| VGGT | 179 | 0.35 | 0.51 | 17.72 | 0.67 |

여기서 가장 큰 메시지는 **3D reconstruction-oriented feature가 multi-view world state에 더 잘 맞는다**는 점이다. Wan VAE나 DINOv2 feature는 image representation으로는 강할 수 있지만, view 간 geometry relation을 담는 global state로는 VGGT가 더 적합했다.

Qualitative result도 이 해석과 맞다. 논문은 competing methods의 실패 유형으로 inaccurate action following, agent disappearance, multi-view inconsistency를 제시한다. MultiWorld는 MACM과 GSE를 통해 이 세 가지를 줄이는 방향으로 설계되어 있다.

# 6. Limitations

1. **Scale이 아직 작다**
   - 논문은 current scale이 limited이며, computational constraints 때문에 large-scale training은 아직 탐색하지 못했다고 명시한다.
   - 따라서 이 결과를 frontier-scale world model로 일반화하기에는 이르다.

2. **Full game dataset release에 제약이 있다**
   - 논문 appendix는 It Takes Two dataset 중 한 chapter subset만 reproduction을 위해 공개할 계획이며, full game dataset은 external constraints 때문에 공개할 수 없다고 설명한다.
   - 재현성 측면에서는 robotics dataset이 더 중요해질 가능성이 있다.

3. **Video world model은 physics simulator가 아니다**
   - MultiWorld는 physical consistency를 일부 보여주지만, explicit simulator처럼 causal physical state를 보장하는 구조는 아니다.
   - RPE와 qualitative consistency가 좋아도 long-horizon에서 compounding error가 누적될 수 있다.

4. **Small or distant agent ambiguity가 남는다**
   - Appendix failure analysis는 agent가 view에서 작은 영역을 차지할 때 ambiguous shape가 생긴다고 설명한다.
   - Multi-view world model에서는 distant agent가 중요한 상호작용을 만들 수 있으므로, 이 문제는 실사용에서 작지 않을 수 있다.

5. **Real-time generation과 ultra-long memory가 아직 future work다**
   - 논문은 real-time multi-agent generation과 ultra-long multi-agent simulation memory를 future work로 둔다.
   - 실제 robotics closed-loop control이나 game serving에 쓰려면 latency와 memory mechanism이 더 중요해진다.

6. **Evaluation metric이 simulator usefulness를 완전히 대변하지는 않는다**
   - FVD, PSNR, SSIM, LPIPS, RPE, IDM action score는 필요한 metric이지만, downstream policy learning이나 planning 성능을 직접 측정하지는 않는다.
   - World model을 simulator로 쓸 때는 generated video quality보다 policy improvement와 failure prediction usefulness가 더 중요할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

- MultiWorld는 video generation 논문이라기보다 **multi-agent simulation interface** 논문으로 읽는 편이 좋다.
- LLM agent나 VLA agent에서 중요한 문제는 하나의 actor가 세계를 보는 것이 아니라, 여러 actor가 같은 environment를 공유하며 서로 영향을 주는 것이다.
- 이 논문은 그 문제를 video world model 관점에서 풀기 위해 agent conditioning과 view consistency를 분리한다.
- 특히 GSE는 multimodal memory나 spatial context encoder를 설계할 때 참고할 만하다. 모든 observation을 self-attention으로 묶지 않고 compact global state로 압축하는 방식은 긴 context를 다룰 때도 유용하다.

## 7-2. Reuse potential

재사용하고 싶은 포인트는 네 가지다.

1. **RoPE-style agent identity embedding**
   - Multi-agent action token에 agent identity를 넣는 간단한 방법이다.
   - LLM의 position embedding을 agent index embedding으로 다시 해석하는 점이 흥미롭다.

2. **Adaptive action aggregation**
   - 모든 agent action을 같은 weight로 합치지 않는다.
   - frame별로 environment change를 만드는 active agent를 더 강하게 반영한다.

3. **3D-aware global state as shared memory**
   - multi-view observation을 VGGT latent로 압축해 shared environment state로 쓴다.
   - view별 generation은 독립적으로 실행하되, 같은 global state에 anchor된다.

4. **Failure trajectory augmentation**
   - 성공 trajectory에 controlled perturbation을 넣어 realistic failure를 만든다.
   - Robotics dataset 구축에서 매우 실용적인 recipe다.

## 7-3. Follow-up papers

- Genie: Generative Interactive Environments
- Genie 2: A Large-Scale Foundation World Model
- Oasis: A Universe in a Transformer
- WHAM: World and Human Action Model
- Geometry Forcing for Video World Models
- VGGT: Visual Geometry Grounded Transformer
- RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints
- COMBO: Compositional World Models for Multi-Agent Simulation
- ShareVerse: Multi-Agent Consistent Video Generation

# 8. Summary

- MultiWorld는 single-agent video world model을 multi-agent, multi-view shared environment simulation으로 확장한다.
- MACM은 Agent Identity Embedding과 Adaptive Action Weighting으로 agent별 action controllability를 높인다.
- GSE는 frozen VGGT 기반 3D-aware global state를 사용해 multi-view consistency를 개선한다.
- It Takes Two와 RoboFactory evaluation에서 visual quality, action following, RPE 기준으로 Standard/Combo 대비 개선을 보인다.
- 다만 scale, dataset release, real-time inference, ultra-long memory, small agent ambiguity는 실무 적용 전에 반드시 확인해야 한다.
