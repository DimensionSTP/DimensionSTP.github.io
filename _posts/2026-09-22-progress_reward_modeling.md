---
layout: single
title: "Progress Reward Modeling for Robotic Learning: A Comprehensive Survey Review"
categories: Study-concept
tag: [Robotics, Reward-Modeling, Reinforcement-Learning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2607.21655)

[Project page](https://github.com/sterzhang/Awesome-Progress-Models)

> 한 줄 요약: 이 survey는 robotic learning의 progress reward를 progress-model interface, reward construction mechanism, data and benchmark evidence의 세 connected step으로 정리하고, terminal success만으로는 보이지 않는 advance, stagnation, regression, recovery를 비교할 공통 framework를 제안한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Robot task의 sparse terminal reward와 dense progress signal을 명확히 구분한다.
- Progress model이 무엇을 입력받고 어떤 형태의 output을 내는지 interface 관점에서 통일한다.
- Frozen foundation model, temporal supervision, instruction tuning, reward program generation을 하나의 taxonomy로 묶는다.
- Reward fidelity와 downstream policy improvement가 같은 평가가 아니라는 점을 강조한다.
- Long-horizon manipulation, planning, recovery에서 필요한 memory와 multimodal signal의 한계를 정리한다.

Robotic learning에서 terminal success는 task가 끝났는지는 알려주지만, 현재 behavior가 목표에 가까워지고 있는지는 알려주지 않는다. Cup을 옮기는 task에서 gripper가 object에 접근했는지, 잡았는지, 떨어뜨렸는지, 다시 회복 중인지가 모두 같은 terminal failure로 보일 수 있다.

Progress reward는 trajectory 중간의 상태가 목표 달성에 얼마나 기여하는지 표현하려 한다. 잘 설계된 progress signal은 RL reward뿐 아니라 trajectory filtering, rollout ranking, planning heuristic, failure detection, recovery trigger로도 사용할 수 있다.

하지만 기존 연구는 observation, goal representation, output scale, supervision source, benchmark가 서로 달라 직접 비교하기 어렵다. 이 survey의 핵심 가치는 새로운 reward model 하나보다 comparison coordinate를 제공하는 데 있다.

# 1. Problem Setting

## 1-1. Problem definition

Task goal을 $g$, time $t$까지의 history를 $h_t$, latent task progress를 $z_t$라고 하자. Progress model은 보통 다음 mapping을 근사한다.

$$
\hat{z}_t = f(h_t,g)
$$

그러나 $z_t$는 직접 관찰되지 않는 경우가 많다. 또한 progress는 단순히 time에 따라 증가하지 않는다.

- Advance: Goal에 가까워진다.
- Stagnation: 상태가 거의 변하지 않는다.
- Regression: 이미 달성한 subgoal을 되돌린다.
- Recovery: 실패 뒤 다시 유효한 path로 돌아온다.
- Detour: 단기적으로 멀어 보이지만 장기적으로 필요한 행동을 한다.

따라서 progress는 state-only scalar가 아니라 history-dependent, task-conditioned latent quantity에 가깝다.

Survey는 progress reward modeling을 세 질문으로 나눈다.

1. Interface: 무엇을 입력받고 무엇을 출력하는가?
2. Construction: Signal을 어떤 mechanism으로 만드는가?
3. Evidence: 어떤 data와 benchmark로 quality를 검증하는가?

## 1-2. Why previous approaches are insufficient

### 1) Terminal reward는 credit assignment가 sparse하다

긴 manipulation trajectory에서 마지막 success or failure만 주면 어느 action이 유용했는지 알기 어렵다. Exploration cost가 크고, partial success를 구분하지 못한다.

### 2) Hand-designed dense reward는 task-specific하다

Distance, angle, contact, object height 같은 simulator state로 reward를 만들 수 있지만, task마다 formula가 다르고 real-world sensor에서는 privileged state가 없을 수 있다.

### 3) Temporal order가 항상 progress는 아니다

Demonstration의 later frame이 earlier frame보다 목표에 가깝다는 가정은 monotonic trajectory에서는 유용하다. 하지만 pause, regression, backtracking, recovery가 있는 behavior에서는 noisy label이 된다.

### 4) Foundation model score는 semantic alignment와 physical feasibility를 혼동할 수 있다

CLIP이나 VLM은 image-goal similarity를 평가할 수 있지만 contact, force, stability, hidden object state를 직접 알기 어렵다.

### 5) Policy gain만으로 reward fidelity를 증명할 수 없다

Reward를 사용한 policy가 좋아져도 reward 자체가 calibrated progress를 측정한다는 뜻은 아니다. Reward shaping이 exploration을 우연히 돕거나 특정 planner bias와 맞았을 수 있다.

# 2. Core Idea

## 2-1. Main contribution

Survey의 taxonomy는 세 connected step으로 정리할 수 있다.

1. Progress model interface
   - State를 어떤 관측 범위로 표현하는가
   - Goal을 language, image, demonstration, predicate 중 무엇으로 주는가
   - Scalar, delta, preference, program 중 어떤 output을 내는가

2. Reward construction mechanism
   - Frozen foundation model을 직접 scoring에 사용하는가
   - Temporal 또는 relative supervision으로 학습하는가
   - Instruction-tuned progress prediction을 사용하는가
   - Programmatic reward를 생성하는가

3. Data and benchmark evidence
   - Data construction은 Human-driven, Human-in-the-loop, Fully automated 방식으로 나눈다
   - Evaluation evidence는 Progress fidelity, Robustness and generalization, Downstream utility를 본다

아래 overview table에서는 실무 비교를 위해 세 번째 step을 supervision과 evaluation 두 row로 펼쳐 표시한다.

이 구조의 장점은 서로 다른 paper를 model architecture가 아니라 problem contract로 비교할 수 있다는 점이다.

## 2-2. Design intuition

Progress model을 평가하려면 먼저 어떤 형태의 progress를 원하는지 정해야 한다.

| Use case | 필요한 output |
| --- | --- |
| Online RL | Dense scalar reward with stable scale |
| Offline relabeling | State or transition score |
| Rollout selection | Pairwise ranking or preference |
| Planning | Smooth, query-efficient value proxy |
| Monitoring | Progress plus uncertainty and regression flag |
| Recovery | Completed, pending, invalidated subgoal state |

한 model이 모든 use case에 최적인 것은 아니다. Pairwise ranking은 absolute calibration 없이도 rollout selection에 유용할 수 있지만, online RL에서 reward magnitude가 불안정하면 optimization이 어려워진다.

Survey가 반복해서 강조하는 점은 progress signal의 meaning을 downstream use에서 역으로 정의해야 한다는 것이다.

# 3. Architecture / Method

## 3-1. Overview

| Axis | Main categories |
| --- | --- |
| State input | Single observation, temporal window, full history, state pair, rollout set, simulator state |
| Goal input | Language instruction, goal image, demonstration, structured predicate, program |
| Output | Scalar progress, progress delta, pairwise preference, ranking, reward code |
| Construction | Frozen model, temporal supervision, instruction tuning, program synthesis |
| Supervision | Human, hybrid, automated |
| Evaluation | Fidelity, robustness, downstream utility |

## 3-2. Module breakdown

### 1) State representation

Progress model이 받는 state representation은 다섯 형태로 나뉜다.

#### Single observation

현재 RGB image나 proprioceptive state만 본다. Latency는 낮지만 history가 필요한 task에서 regression and recovery를 구분하기 어렵다.

#### Temporal prefix or window

최근 frame sequence나 trajectory prefix를 본다. Motion direction과 completed subgoal을 추론할 수 있지만 context cost가 늘어난다.

#### Full trajectory

Long-horizon history 전체를 입력한다. Rich signal을 제공하지만 real-time use에서 expensive하고, long-context forgetting이 생길 수 있다.

#### Comparison input

Before-after, initial-current, current-goal, candidate rollout pair를 비교한다. Absolute score보다 relative ordering을 배우기 쉬울 수 있다.

#### Direct state or API access

Simulator object pose, contact, task predicate를 입력한다. 정확하지만 privileged information에 의존하며 real-world transfer가 어렵다.

### 2) Goal specification

#### Language goal

"Place the red block in the left drawer"처럼 natural-language instruction을 사용한다. Flexible하지만 ambiguity와 object grounding 문제가 있다.

#### Visual goal or demonstration

Goal image, target video, expert demonstration을 사용한다. Visual alignment에는 강하지만 viewpoint and embodiment variation에 민감하다.

#### Structured goal

Predicate, symbolic state, programmatic condition을 사용한다. Verification이 명확하지만 task authoring cost가 크다.

### 3) Output signal

#### State-wise scalar

현재 progress level을 하나의 값으로 출력한다. RL reward로 사용하기 쉽지만 calibration이 필요하다.

#### Progress delta

$\Delta_t = \hat{z}_{t+1} - \hat{z}_t$처럼 transition improvement를 출력한다. Advance and regression을 직접 표현할 수 있다.

#### Pairwise preference or ranking

두 state or trajectory 중 어느 쪽이 goal에 가까운지 판단한다. Annotation이 쉽고 reranking에 적합하지만 absolute stopping threshold를 만들기 어렵다.

#### Reward program

LLM이 simulator API를 사용한 executable reward code를 만든다. Physical state를 직접 활용할 수 있지만 generated code validation이 필요하다.

### 4) Frozen foundation model scoring

CLIP, VLM, video-language model의 embedding similarity나 token probability를 progress proxy로 사용한다.

장점은 task-specific training이 적고 open-vocabulary goal을 처리한다는 것이다. 단점은 semantic similarity가 physical progress와 같지 않다는 점이다. Example image와 비슷해 보여도 grasp stability나 force condition이 실패했을 수 있다.

### 5) Temporal and relative supervision

Demonstration에서 later state를 earlier state보다 높은 progress로 label한다. Pairwise ranking loss로 학습하기 쉽고 manual scalar annotation이 필요 없다.

하지만 다음 assumption이 들어간다.

- Demonstration이 대체로 monotonic하다.
- Time index가 goal proximity와 correlation한다.
- Expert trajectory의 detour가 크지 않다.

Non-monotonic trajectory에서는 hard negative, plateau, regression annotation이 필요하다.

### 6) Instruction-tuned progress prediction

VLM or video-LM을 instruction and trajectory에 맞춰 fine-tune해 progress score, success probability, delta, reasoning을 출력한다.

이 방식은 open-vocabulary task와 explicit explanation을 지원하지만 training label quality와 inference latency가 bottleneck이다.

### 7) Programmatic reward construction

Text2Reward, Eureka 계열은 task description과 simulator API를 바탕으로 reward code를 생성하고, policy performance를 feedback으로 program을 수정한다.

이 방식은 state feature를 명시적으로 사용할 수 있지만 simulator-specific API와 execution sandbox가 필요하다. Generated reward가 unintended shortcut을 만들지 audit해야 한다.

# 4. Training / Data / Recipe

## 4-1. Data

이 paper는 survey이므로 새로운 training dataset 하나를 제안하지 않는다. 대신 progress supervision source를 세 범주로 정리한다.

### Human-driven data

- Expert scalar annotation
- Pairwise preference
- Success stage label
- Subgoal boundary
- Failure and recovery tag

High quality지만 expensive하고, dense frame annotation은 consistency가 어렵다.

### Human-in-the-loop data

Model이 candidate label or reward program을 만들고 human이 수정하거나 rank한다. Cost를 줄이면서 semantic control을 유지할 수 있지만 annotation interface와 active sampling이 중요하다.

### Fully automated data

- Demonstration의 temporal order
- Simulator state와 predicate
- Foundation model이 만든 pseudo-label
- Synthetic success 및 failure trajectory
- Policy rollout과 automatic verifier의 결합

Scale은 크지만 assumption error가 label에 들어간다.

## 4-2. Training strategy

Progress model recipe는 output form에 따라 달라진다.

### Scalar regression

Human or simulator progress target에 MSE or ordinal loss를 적용한다. Absolute reward가 필요할 때 적합하지만 annotator scale alignment가 필요하다.

### Pairwise ranking

State pair $(s_i,s_j)$에서 어느 쪽이 더 진행되었는지 학습한다.

$$
L_{rank} = -\log \sigma(f(s_j,g)-f(s_i,g))
$$

Later-is-better label을 사용할 수 있지만 non-monotonic example을 반드시 포함해야 한다.

### Contrastive goal alignment

Current observation과 goal representation의 similarity를 높인다. Open-vocabulary generalization에는 유리하지만 physical state feature를 놓칠 수 있다.

### Instruction tuning

Trajectory, goal, task description을 입력하고 progress score와 rationale를 생성한다. Numeric calibration loss와 text generation loss를 함께 쓰는 hybrid design이 가능하다.

### Reward program search

Generated code로 policy를 학습하고 downstream return을 feedback으로 program을 반복 개선한다. Code validity, reward exploit, simulator overfitting을 별도 검사해야 한다.

## 4-3. Engineering notes

### 1) Reward scale를 normalize해야 한다

Task마다 progress range가 다르면 multi-task RL에서 gradient scale이 흔들린다. Per-task normalization, rank-based reward, bounded output을 고려해야 한다.

### 2) Plateau and regression sample을 의도적으로 넣어야 한다

Success demonstration만 사용하면 model이 time index를 progress로 외울 수 있다. Pause, failed grasp, dropped object, wrong drawer, recovery trajectory를 포함해야 한다.

### 3) Uncertainty output이 필요하다

Occlusion, unseen embodiment, ambiguous instruction에서는 confident score보다 abstention이 안전하다. Ensemble, calibration, conformal interval을 사용할 수 있다.

### 4) Reward model query cost를 기록해야 한다

VLM을 every control step에 호출하면 latency가 너무 크다. Streaming inference, feature cache, distilled small model, hierarchical query policy가 필요하다.

### 5) Sensor fusion을 분리해야 한다

Visual progress와 physical progress가 다를 수 있다. RGB, depth, proprioception, force, tactile signal을 task에 따라 결합해야 한다.

### 6) Reward hacking test를 둬야 한다

High progress score를 얻지만 task를 수행하지 않는 behavior를 adversarial rollout로 찾는다. Reward model and policy를 같은 distribution에서만 평가하면 exploit을 놓칠 수 있다.

# 5. Evaluation

## 5-1. Main results

이 survey는 새로운 model의 single leaderboard result를 제시하지 않는다. 대신 progress reward evaluation을 세 level로 구분한다.

### 1) Progress fidelity

Progress model 자체가 intended quantity를 측정하는지 본다.

| Property | Example metric |
| --- | --- |
| Absolute calibration | Predicted progress vs annotated stage |
| Temporal consistency | Valid demonstration에서 non-decreasing rate |
| Relative ordering | Pairwise ranking accuracy |
| Task grounding | Goal change에 따른 score sensitivity |
| Regression detection | Backward transition identification |
| Uncertainty | Calibration error, selective risk |

### 2) Robustness and generalization

- 학습에 없던 task
- 학습에 없던 object와 scene
- Viewpoint 변화
- 새로운 robot embodiment
- Distractor와 occlusion
- Non-monotonic trajectory
- Plateau, reversal, recovery 구간
- Long-horizon context

Progress model이 training demonstration의 timestamp shortcut만 배웠다면 이 평가에서 무너질 수 있다.

### 3) Downstream utility

- Online RL의 sample efficiency
- Offline data relabeling 품질
- Trajectory filtering과 retrieval 성능
- Candidate rollout ranking 품질
- Planning과 action selection 성능
- Failure monitoring과 recovery trigger의 신뢰도

중요한 점은 downstream gain이 reward fidelity를 자동으로 증명하지 않는다는 것이다.

## 5-2. What really matters in the experiments

### 1) Policy improvement and reward fidelity를 분리해야 한다

Reward A를 쓴 policy가 더 좋아졌다고 해서 A가 true progress를 정확히 측정한다고 결론 내릴 수 없다. Optimization dynamics, exploration bonus, scale가 우연히 policy에 유리했을 수 있다.

따라서 최소 두 evaluation이 필요하다.

- Standalone fidelity on held-out trajectories
- Downstream policy utility under controlled budget

### 2) Reranking에는 calibration이 덜 중요할 수 있다

Candidate 10개 중 best rollout을 고르는 용도라면 relative ordering이 중요하다. Score 0.7이 실제 70% progress를 뜻할 필요는 없다.

### 3) Planning에는 smoothness and query efficiency가 중요하다

Planner가 많은 candidate state를 평가하려면 reward surface가 locally meaningful하고 evaluation이 빨라야 한다. Expensive VLM reasoning은 accurate해도 planning inner loop에 쓰기 어려울 수 있다.

### 4) Online RL에는 exploit resistance가 중요하다

Policy는 reward model의 weakness를 적극적으로 찾는다. Static benchmark accuracy가 높아도 closed-loop optimization에서 reward hacking이 발생할 수 있다.

### 5) Non-monotonic trajectory가 real test다

Temporal-order supervision은 monotonic demonstration에서 쉽게 높은 score를 얻을 수 있다. 실제 가치가 드러나는 지점은 object를 떨어뜨리거나 wrong subgoal을 수행한 뒤 recovery하는 trajectory다.

# 6. Limitations

1. Current progress signal의 temporal resolution이 거칠다.
   - Frame-level semantic score만으로 contact transition과 fine motor control을 잡기 어렵다.

2. Visual model이 physical state를 충분히 보지 못한다.
   - Proprioception, force, tactile signal이 필요한 task가 많다.

3. Progress를 fixed-rate monotonic process로 가정하는 연구가 많다.
   - Plateau, regression, detour, recovery를 명시적으로 modeling해야 한다.

4. Foundation model inference latency가 크다.
   - Control frequency와 맞지 않을 수 있다.
   - Streaming, cache, distillation, hierarchical querying이 필요하다.

5. Long-horizon memory가 부족하다.
   - Completed, pending, invalidated subgoal을 유지하지 못하면 같은 행동을 반복하거나 이미 달성한 상태를 되돌릴 수 있다.

6. Benchmark protocol이 통일되지 않았다.
   - 다른 task, embodiment, state access, output type을 사용해 method 간 direct comparison이 어렵다.

7. Data label assumption이 결과를 지배할 수 있다.
   - Later-is-better, simulator predicate, foundation model pseudo-label은 각각 다른 bias를 가진다.

8. Reward model and policy co-adaptation이 충분히 평가되지 않는다.
   - Fixed dataset 성능과 online optimization safety 사이에 gap이 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 survey의 가장 유용한 부분은 progress reward를 하나의 scalar model family로 좁히지 않는다는 점이다. Pairwise ranking, state delta, monitoring score, executable reward program은 서로 다른 contract이며, 평가 기준도 달라야 한다.

이 관점은 robotics를 넘어 long-horizon agent에도 그대로 연결된다. Research agent나 document workflow agent에서도 terminal success만 보면 중간에 evidence가 늘었는지, 잘못된 claim을 되돌렸는지, 같은 실패를 반복하는지 알기 어렵다.

## 7-2. Reuse potential

### 1) Agent trajectory progress model

Current state가 answer completion에 가까워졌는지보다 verified evidence coverage, unresolved constraint, contradiction count를 progress feature로 사용할 수 있다.

### 2) Evidence-grounded delta reward

새 action이 support coverage를 늘렸는지, unsupported claim을 줄였는지 transition delta로 reward할 수 있다.

### 3) Pairwise rollout reranking

Absolute reward를 만들기 어렵다면 두 trajectory prefix 중 어느 쪽이 더 recoverable한지 preference model로 학습할 수 있다.

### 4) Recovery-aware benchmark

정상 trajectory뿐 아니라 wrong evidence, failed retrieval, tool error, contradictory source 이후 recovery case를 별도 split으로 만들 수 있다.

### 5) Hierarchical querying

Cheap heuristic가 대부분 step을 평가하고, ambiguous or high-impact step에만 expensive VLM or LLM progress model을 호출하는 cascade가 실용적이다.

## 7-3. Follow-up papers

- Text2Reward
- Eureka
- LIV
- VIP
- RoboCLIP
- R3M
- MineCLIP
- Preference-Based Reinforcement Learning

# 8. Summary

- Progress reward는 terminal success가 놓치는 advance, stagnation, regression, recovery를 trajectory 중간에서 측정한다.
- Survey는 interface, construction, data, evaluation evidence로 field를 통일한다.
- Foundation model scoring, temporal supervision, instruction tuning, reward program generation은 서로 다른 assumption을 가진다.
- Reward fidelity, robustness, downstream policy utility를 분리해 평가해야 한다.
- 다음 과제는 multimodal physical signal, non-monotonic trajectory, long-horizon memory, low-latency inference다.
