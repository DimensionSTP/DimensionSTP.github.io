---
layout: single
title: "PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation Review"
categories: Study-concept
tag: [PhysisForcing, WorldModel, Robotics, VideoGeneration, EmbodiedAI]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.28128)

[Project page](https://dagroup-pku.github.io/PhysisForcing.github.io/)

[Code link](https://github.com/DAGroup-PKU/PhysisForcing)

PhysisForcing은 "robot video generation benchmark score를 올린 논문"으로만 읽으면 핵심을 놓치기 쉽다. 이 논문이 실제로 겨냥하는 문제는 더 구체적이다. **video generation model을 robot manipulation world simulator로 쓰려면, visual realism보다 contact와 interaction physics가 먼저 안정되어야 한다**는 것이다.

최근 robot foundation model 흐름에서는 video generation model을 world model처럼 쓰려는 시도가 많다. Image와 prompt가 주어지면 action outcome video를 예측하고, 그 video representation을 action planning이나 policy learning에 활용한다. 하지만 일반 video generator는 물체 deformation, discontinuous trajectory, robot-object contact inconsistency를 쉽게 만든다. 결과 image는 그럴듯해도, manipulation simulator로는 신뢰하기 어렵다.

PhysisForcing은 이 문제를 pixel-level과 semantic-level physics alignment로 푼다. Pixel level에서는 CoTracker3 reference point trajectory를 사용해 DiT feature가 implied하는 local motion trajectory를 맞춘다. Semantic level에서는 frozen video understanding encoder가 보는 inter-region relation structure를 DiT token relation과 맞춘다. 그리고 이 supervision을 전체 pixel에 균일하게 주지 않고, robot, object, contact처럼 physics-informative region에 집중한다.

> 한 줄 요약: PhysisForcing은 embodied video generator의 physical instability가 object deformation과 implausible spatio-temporal relation에서 온다고 보고, depth-aware region mask, pixel-level trajectory alignment, semantic-level relational alignment를 DiT feature에 적용해 manipulation video generator를 더 reliable한 world simulator로 만드는 training-time framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Video generation model을 "예쁜 video generator"가 아니라 robotic world simulator로 평가한다.
- Physical plausibility를 단일 reward가 아니라 local trajectory와 inter-region relation alignment로 분해한다.
- Auxiliary tracker와 video encoder를 inference-time이 아니라 training-time supervision으로만 사용해 deployment overhead를 늘리지 않는다.
- R-Bench, PAI-Bench, EZS-Bench뿐 아니라 WorldArena action-planner와 RoboTwin policy learning까지 연결한다.
- World Action Model 흐름에서 "future video quality가 action success로 이어지는가"를 확인하는 concrete case다.

이 글에서는 PhysisForcing을 "robot video generation model"보다, **physical alignment signal이 video world model의 downstream control utility를 어떻게 바꾸는지 보여주는 training framework**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Robot manipulation world simulator는 current observation $o_t$, instruction 또는 action condition $c$, 필요하다면 history $h_t$를 보고 future visual state를 예측한다.

$$
\hat{o}_{t+1:T}
=
G_{\theta}(o_t,c,h_t)
$$

이 future는 단순 visual output이 아니라 action planning이나 policy learning에 쓰인다.

$$
a
=
\pi(\hat{o}_{t+1:T},o_t,c)
$$

문제는 video generator의 likelihood나 perceptual quality가 manipulation physics를 보장하지 않는다는 점이다. Robot object contact에서는 작은 physical inconsistency가 매우 큰 policy error로 이어질 수 있다.

논문은 physical instability의 주요 원인을 두 가지로 요약한다.

1. Moving object deformation
   - 움직이는 object shape이나 structure가 video에서 무너진다.
   - 같은 object가 frame 사이에서 늘어나거나 녹거나 부자연스럽게 변한다.

2. Implausible spatio-temporal correlation
   - interacting entities의 relation이 contact 순간에 깨진다.
   - Gripper가 object를 잡았는데 object가 같이 움직이지 않거나, push action 후 object response가 inconsistent하다.

따라서 world simulator training은 pixel fidelity만이 아니라 motion trajectory와 relation consistency를 직접 supervision해야 한다.

## 1-2. Why previous approaches are insufficient

### 1) General-domain video generators

General video generator는 visual realism에는 강하다. 하지만 manipulation에는 action-conditioned physical contact가 필요하다. 일반 video generation training data와 objective는 robot-object interaction에 최적화되어 있지 않은 경우가 많다.

### 2) Robot-specific finetuning

Robot video data로 vanilla finetuning을 하면 domain match는 좋아진다. 그러나 논문은 vanilla finetuning 이후에도 unstable grasp, object drift, broken contact가 남는다고 보고한다. Robot data를 더 넣는 것만으로는 어떤 region이 physics-informative한지 명시적으로 가르치기 어렵다.

### 3) Pixel reconstruction 또는 diffusion loss

Standard denoising objective는 모든 region과 error를 비슷하게 취급한다. 하지만 manipulation video에서는 contact region과 moving object trajectory가 훨씬 큰 physical meaning을 가진다. Background reconstruction이 loss를 지배하는 동안 contact inconsistency가 남을 수 있다.

### 4) Reward-only alignment

Scalar realism reward나 preference reward는 global quality를 개선할 수 있지만, point trajectory나 inter-object relation structure에 supervision을 정확히 localize하지는 못할 수 있다. PhysisForcing은 explicit pixel-level 및 semantic-level alignment signal을 사용한다.

# 2. Core Idea

## 2-1. Main contribution

PhysisForcing의 core component는 세 가지다.

1. **Depth-aware physics-informative region mask**
   - Robot, manipulated object, contact-related region을 localize한다.
   - Static하거나 irrelevant한 background에 supervision을 낭비하지 않는다.

2. **Pixel-level trajectory alignment**
   - Point tracking reference trajectory를 사용한다.
   - DiT feature에서 유도된 motion을 masked trajectory MSE로 supervision한다.

3. **Semantic-level relational alignment**
   - Frozen video understanding encoder로 inter-region relation을 추출한다.
   - DiT token-to-token relation matrix를 reference semantic relation에 맞춘다.

핵심은 local motion continuity와 global interaction relation을 함께 맞추는 hierarchical alignment다.

## 2-2. Design intuition

Design intuition은 physical plausibility에 적어도 두 layer가 있다는 것이다.

| Layer | What can fail | PhysisForcing signal |
| --- | --- | --- |
| Pixel-level motion | Object deformation, discontinuous trajectory | Trajectory alignment |
| Semantic relation | Broken grasp, inconsistent push, wrong coupling | Relational alignment |

Pixel-level trajectory만 강제하면 point motion은 smooth해질 수 있지만 higher-level coupling을 놓칠 수 있다. 반대로 semantic relation만 강제하면 object motion이 locally unstable한 채로 남을 수 있다. 두 loss는 상호 보완적이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Robot manipulation video world model의 physical consistency 개선 |
| Base models | Wan2.2-I2V-A14B, Cosmos3-Nano, Wan2.2-5B for action planner setting |
| Core supervision | Pixel-level trajectory와 semantic-level relation |
| Region focus | Physics-informative region을 위한 depth-aware motion mask |
| Auxiliary models | CoTracker3와 frozen video understanding encoder |
| Inference overhead | 없음, auxiliary model은 training 중에만 사용 |
| Evaluation | R-Bench, PAI-Bench, EZS-Bench, RoboTwin 2.0, WorldArena |

## 3-2. Module breakdown

### 1) Physics-informative region mining

Method는 먼저 robot-object interaction이 future physical plausibility를 결정할 가능성이 높은 region을 찾는다. Project page는 이를 depth-aware motion mask localization으로 설명한다.

이 region mask는 다음에 더 큰 weight를 준다.

- manipulator
- manipulated object
- contact region
- moving entities

Static background에는 더 낮은 weight를 준다.

### 2) Pixel-level trajectory alignment

Point tracker는 reference point trajectory를 만든다.

$$
p_i^{ref}(t)
$$

Video generator의 DiT feature는 predicted trajectory-like correspondence를 암시한다.

$$
p_i^{pred}(t)
$$

여기에 masked trajectory alignment loss를 적용한다.

$$
\mathcal{L}_{traj}
=
\sum_{i,t}
m_i
\left\|
p_i^{pred}(t)
-
p_i^{ref}(t)
\right\|_2^2
$$

여기서 $m_i$는 physics-informative region weight를 의미한다. 목적은 local motion을 continuous하고 contact-compatible하게 만드는 것이다.

### 3) Semantic-level relational alignment

Frozen video understanding encoder는 physics-informative region 사이의 relational structure를 추출한다. DiT feature token도 token-to-token similarity matrix를 정의한다.

다음과 같이 두 matrix를 두자.

$$
R^{ref}
$$

는 reference relation matrix이고,

$$
R^{dit}
$$

는 DiT-side relation matrix다. Semantic alignment loss는 이 두 matrix를 맞추는 것으로 볼 수 있다.

$$
\mathcal{L}_{rel}
=
\left\|
R^{dit}
-
R^{ref}
\right\|_F^2
$$

이 loss는 globally consistent interaction을 유도한다. Grasp된 object는 gripper와 함께 움직이고, push된 object는 밀려나며, coupled entity는 frame 사이에서도 coupling을 유지해야 한다.

### 4) Total objective

Training objective는 video model training objective에 physics alignment term을 추가한다.

$$
\mathcal{L}
=
\mathcal{L}_{video}
+
\lambda_{traj}
\mathcal{L}_{traj}
+
\lambda_{rel}
\mathcal{L}_{rel}
$$

정확한 weight는 paper에서 다시 확인해야 한다. 핵심은 모든 additional supervision이 training-time에만 사용된다는 점이다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 PhysisForcing을 robot manipulation video generation과 world-action modeling benchmark에서 평가한다.

| Benchmark | 측정 대상 |
| --- | --- |
| R-Bench | Task와 embodiment를 가로지르는 robot video generation |
| PAI-Bench | Robot video quality와 domain plausibility |
| EZS-Bench | Embodied zero-shot 또는 broader generalization setting |
| RoboTwin 2.0 | Generated video backbone을 사용한 policy learning |
| WorldArena | Action-planner closed-loop success |

정확한 training data mixture와 finetuning recipe는 reproduction 전에 paper와 code release에서 확인해야 한다.

## 4-2. Training strategy

Training은 standard diffusion video backbone을 사용하고 physics alignment supervision을 추가한다.

High-level recipe는 다음과 같다.

1. Video generation backbone에서 시작한다.
2. Robot manipulation video로 finetune한다.
3. Physics-informative region mask를 mining한다.
4. CoTracker3로 point trajectory를 얻는다.
5. Frozen video understanding encoder로 relation reference를 얻는다.
6. DiT feature에 trajectory와 relation alignment를 적용한다.
7. Inference에서는 auxiliary model을 버린다.

## 4-3. Engineering notes

1. **Contact region에 loss를 집중한다**
   - Uniform supervision은 physical bottleneck을 충분히 크게 반영하지 못할 수 있다.

2. **Local signal과 global signal을 함께 사용한다**
   - Trajectory smoothness와 relation consistency는 서로 다른 failure mode를 잡는다.

3. **Auxiliary model은 training-only로 유지한다**
   - 이렇게 하면 inference cost가 늘어나지 않는다.

4. **Downstream action을 평가한다**
   - World simulator quality는 video metric만이 아니라 planner나 policy setting에서도 평가해야 한다.

5. **Negative transfer를 확인한다**
   - Average가 좋아져도 일부 RoboTwin task는 나빠질 수 있다.

# 5. Evaluation

## 5-1. Video generation results

Project page는 다음 selected result를 보고한다.

| Benchmark | Baseline | PhysisForcing | Difference |
| --- | ---: | ---: | ---: |
| R-Bench, Wan2.2-A14B ft avg | 57.9 | 62.0 | +4.1 |
| R-Bench, Cosmos3-Nano ft avg | 61.5 | 63.8 | +2.3 |
| PAI-Bench, Wan2.2-A14B ft avg | 79.90 | 81.73 | +1.83 |
| PAI-Bench, Cosmos3-Nano ft avg | 84.03 | 85.17 | +1.14 |
| EZS-Bench, Wan2.2-A14B ft avg | 79.04 | 80.54 | +1.50 |
| EZS-Bench, Cosmos3-Nano ft avg | 80.29 | 81.08 | +0.79 |

ArXiv abstract는 R-Bench relative improvement를 base model 대비 22.3%와 9.2%, vanilla finetuning 대비 7.1%와 3.7%로 요약한다. 대상 model은 Wan2.2-I2V-A14B와 Cosmos3-Nano다.

## 5-2. Policy and world-action results

Project page는 RoboTwin 2.0 Fast-WAM average success rate 증가도 보고한다.

| Setting | Success |
| --- | ---: |
| Fast-WAM | 68.2% |
| Fast-WAM + PhysisForcing | 72.8% |

WorldArena Action Planner IDM에서는 다음과 같다.

| Model | Avg success |
| --- | ---: |
| Wan2.2-5B base | 16.0% |
| PF-Wan5B | 24.0% |

이 부분이 가장 중요한 evidence다. Physically aligned video generation이 downstream action planning과 policy learning으로 이어진다는 점을 보여주기 때문이다.

## 5-3. What really matters in the experiments

### 1) Downstream action 평가가 video-only evaluation보다 중요하다

Robot world simulator라면 action success를 개선해야 한다. 따라서 WorldArena와 RoboTwin result는 video benchmark average만 보는 것보다 더 의미가 크다.

### 2) Physical alignment가 항상 양의 효과만 내지는 않는다

RoboTwin task table에는 negative task delta도 일부 포함된다. Average는 개선되지만 task-level trade-off는 남아 있다.

### 3) Training-only alignment는 deployable하다

Auxiliary tracker와 encoder는 비싸지만 training 후에는 버려진다. 이 점은 deployment 관점에서 실용적이다.

### 4) Physical metric은 아직 발전 중이다

R-Bench, PAI-Bench, EZS-Bench는 유용하지만 physical plausibility를 측정하는 일은 어렵다. Human evaluation과 closed-loop evaluation은 여전히 중요하다.

# 6. Limitations

1. **Benchmark dependence가 있다**
   - Physical plausibility metric은 아직 성숙하지 않았고 benchmark-specific하다.

2. **Auxiliary model dependence가 있다**
   - CoTracker3와 frozen video understanding encoder의 품질이 supervision에 영향을 준다.

3. **Task-level regression이 남아 있다**
   - Average가 개선되어도 일부 RoboTwin task는 성능이 떨어진다.

4. **Training cost가 증가한다**
   - 추가 tracking과 relational alignment는 training pipeline complexity를 높인다.

5. **Physics simulator는 아니다**
   - PhysisForcing은 learned video dynamics를 개선하지만 physical law satisfaction을 보장하지는 않는다.

6. **Action condition detail을 확인해야 한다**
   - Action, prompt, image, embodiment condition이 어떻게 표현되는지는 paper에서 다시 확인해야 한다.

7. **Generalization boundary가 있다**
   - 결과는 여러 benchmark를 포함하지만 arbitrary robot이나 object generalization을 증명하지는 않는다.

8. **WorldArena task는 제한적이다**
   - Closed-loop success는 promising하지만 real robot deployment에 대한 작은 proxy일 뿐이다.

9. **Visual generator backbone dependence가 있다**
   - Gain은 Wan/Cosmos architecture와 finetuning setup에 의존할 수 있다.

10. **Safety와 physical execution 문제는 남아 있다**
   - 그럴듯한 generated video가 safe real-world control을 보장하지는 않는다.

# 7. My Take

## 7-1. Why this matters for my work

PhysisForcing의 핵심은 "video generation benchmark를 올렸다"가 아니라, **world simulator training에서 어디를 supervision해야 action utility가 올라가는지 보여준 것**이다.

Robot video world model에서는 background를 예쁘게 그리는 것보다 contact와 coupling을 맞추는 것이 중요하다. PhysisForcing은 그 bottleneck을 region-focused loss로 잡는다.

## 7-2. Reuse potential

### Robot world model training

Manipulation video generator를 world model로 쓰려면 pixel loss보다 contact-aware trajectory와 relation loss가 필요하다.

### Video-to-policy representation

Physically aligned video model feature는 더 좋은 policy representation이 될 수 있다. RoboTwin과 WorldArena result는 이 방향을 뒷받침한다.

### General video model alignment

같은 아이디어는 robotics 바깥으로도 확장될 수 있다. Sports, human-object interaction, driving, tool use는 모두 physically plausible contact와 trajectory에 의존한다.

### Evaluation design

Target application이 control이라면 video generation evaluation에는 closed-loop policy success가 포함되어야 한다.

## 7-3. Production considerations

- Generated video는 uncertainty check와 함께 planning에 사용한다.
- Average만 보지 말고 per-task로 평가한다.
- Predicted future와 real outcome의 mismatch를 logging한다.
- Real-world safety controller는 generated-video planner와 분리한다.
- Physical alignment는 guarantee가 아니라 training signal로 사용한다.

## 7-4. Follow-up papers

- World Action Models: A Survey
- ABot-PhysWorld
- PhysWorld
- Cosmos Predict and Cosmos world models
- Wan video generation series
- Fast-WAM
- WorldArena
- RoboTwin

# 8. Summary

- PhysisForcing은 robot manipulation video generator의 physical instability를 겨냥한다.
- Supervision을 physics-informative region에 집중한다.
- Pixel-level trajectory alignment는 local motion continuity를 개선한다.
- Semantic-level relational alignment는 robot-object interaction consistency를 개선한다.
- 가장 강한 evidence는 WorldArena와 RoboTwin policy setting에서의 downstream improvement다.
