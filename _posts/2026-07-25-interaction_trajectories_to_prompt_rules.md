---
layout: single
title: "From Interaction Trajectories to Prompt Rules: Credit Assignment for Multi-Agent Prompt Optimization Review"
categories: Study-concept
tag: [MultiAgent, PromptOptimization, CreditAssignment, LLM, Agent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://openreview.net/pdf?id=pXF0puBofz)

[Related arXiv link checked for technical details](https://arxiv.org/abs/2605.30227)

Multi-agent LLM system을 만들면 prompt 하나를 고치는 일도 단순하지 않다. 여러 agent가 여러 round 동안 말하고, 중간 aggregation이 생기고, 마지막 output만 점수화된다. 결과가 틀렸을 때 어느 agent prompt가 문제였는지, 어느 round의 aggregation이 정보를 잃었는지, 어떤 interaction step이 최종 실패로 이어졌는지 알기 어렵다.

이 논문의 주제는 credit assignment다. Multi-agent prompt optimization을 단일 global prompt search로 다루지 않고, interaction trajectory를 round와 role 단위로 분해해 어떤 prompt rule을 고쳐야 하는지 찾는다.

> 한 줄 요약: 이 논문 계열은 multi-agent interaction trajectory에서 실패 credit을 round와 role로 나누고, 그 credit을 prompt rule update로 변환해 무차별 prompt search보다 targeted한 multi-agent prompt optimization을 하려는 접근이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같다.

- Agent system이 복잡해질수록 최종 score만으로 prompt를 고치는 방식은 variance가 너무 크다.
- Multi-agent debate, planner-solver-critic, tool-use workflow에서는 어느 component가 문제인지 알아야 한다.
- Prompt optimization을 black-box search가 아니라 diagnostic loop로 보는 관점을 준다.
- Agent 운영에서 prompt가 static instruction이 아니라 계속 업데이트되는 policy artifact가 되는 흐름과 연결된다.

# 1. Problem Setting

## 1-1. Problem definition

문제는 multi-agent system의 final output score가 너무 sparse하다는 점이다. 여러 agent와 여러 round가 함께 하나의 trajectory를 만들고, 마지막에만 정답 여부나 reward가 나온다. 이때 잘못된 final answer를 보고 모든 prompt를 한꺼번에 고치면 다음 문제가 생긴다.

- 어느 role이 잘못했는지 모른다.
- 어느 round에서 정보가 손실됐는지 모른다.
- Aggregator가 문제인지, solver가 문제인지, critic이 문제인지 분리하기 어렵다.
- 좋은 component까지 같이 수정해 regression을 만들 수 있다.
- Prompt search space가 agent 수와 round 수에 따라 커진다.

간단히 말하면 multi-agent prompt optimization은 credit assignment 문제다. 최종 실패를 interaction trajectory 안의 local decision과 local prompt rule로 되돌려야 한다.

## 1-2. Why previous approaches are insufficient

기존 prompt optimizer는 보통 prompt를 하나의 text block으로 보고, execution feedback을 바탕으로 rewrite한다. Single-agent setting에서는 이 방식이 어느 정도 통한다. 하지만 multi-agent setting에서는 prompt가 하나가 아니다.

예를 들어 planner, solver, critic, aggregator가 있는 system을 생각해보자. Final answer가 틀렸다고 해서 solver prompt만 고치면, 실제 문제는 planner가 잘못된 subproblem을 만들었기 때문일 수 있다. Critic이 틀린 critique를 했을 수도 있고, aggregator가 좋은 evidence를 버렸을 수도 있다.

그래서 black-box optimization은 두 가지 한계를 갖는다.

1. Structural credit이 없다
   - 어느 agent role이 반복적으로 약한지 구분하지 못한다.

2. Temporal credit이 없다
   - 어느 round에서 state가 망가졌는지 구분하지 못한다.

# 2. Core Idea

## 2-1. Main contribution

확인 가능한 공개판 기준으로 핵심 아이디어는 temporal credit과 structural credit을 함께 쓰는 것이다.

- Temporal credit
  - Interaction round별로 어떤 시점이 final failure에 영향을 주었는지 본다.
  - Aggregation state가 잘못 형성된 round를 찾는다.

- Structural credit
  - Agent role별로 어떤 역할이 반복적으로 약한지 본다.
  - Role prompt를 trajectory 전체에서 공유되는 policy로 보고 업데이트한다.

- Verbalized prompt update
  - Numerical gradient가 아니라 LLM critic이 만든 natural-language feedback으로 prompt를 고친다.
  - 낮은 credit을 받은 role 또는 round만 targeted하게 수정한다.

이 접근의 핵심은 모든 prompt를 매번 다시 쓰지 않는 것이다. 잘 작동한 role과 round는 그대로 두고, 낮은 credit을 받은 부분만 고친다.

## 2-2. Design intuition

Multi-agent trajectory는 그냥 message log가 아니다. 어떤 agent가 어떤 state를 만들고, 그 state가 다음 round의 input이 되며, 마지막 decision으로 이어지는 computation graph다. 그런데 LLM output은 discrete text라서 backpropagation을 그대로 쓸 수 없다.

그래서 논문은 두 가지 inductive bias를 넣는다.

1. State-space bottleneck
   - 각 round의 agent outputs를 shared state로 aggregate한다.
   - 이 shared state를 보면 어느 round에서 정보가 손실됐는지 평가할 수 있다.

2. Stationary role policy
   - 같은 role은 여러 round에서 같은 prompt policy를 공유한다고 본다.
   - 그러면 특정 role의 반복적인 실패를 structural credit으로 모을 수 있다.

이 두 장치가 있어야 final score를 local prompt rule update로 바꿀 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Multi-agent prompt optimization에서 실패 credit을 round와 role로 분해 |
| Input | Multi-agent interaction trajectory |
| Credit axis 1 | Temporal credit over rounds |
| Credit axis 2 | Structural credit over roles |
| Optimized artifact | Role prompts and aggregation prompts |
| Update method | Verbalized block coordinate descent |
| Key benefit | 낮은 credit component만 targeted하게 수정 |

## 3-2. Module breakdown

### 1) Interaction trajectory

Trajectory는 여러 round의 agent utterance와 shared state로 구성된다. Agent output은 다음 round의 context가 되고, aggregation module은 여러 agent output을 하나의 state로 압축한다.

개념적으로는 다음처럼 볼 수 있다.

$$
trajectory = (U_1, S_1, U_2, S_2, ..., U_R, S_R)
$$

여기서 $U_t$는 round $t$의 agent outputs이고, $S_t$는 aggregation state다. 최종 answer는 $S_R$에서 나온다.

### 2) State-space bottleneck

State-space bottleneck은 round별 aggregation state를 명시적으로 둔다는 뜻이다. 모든 agent utterance가 final answer로 직접 흘러가면 credit이 흐릿하다. 반대로 매 round마다 shared state가 있으면, 어느 시점의 state가 final answer를 망쳤는지 볼 수 있다.

이 shared state는 temporal credit의 anchor가 된다. Aggregator가 좋은 evidence를 버렸는지, 잘못된 intermediate conclusion을 만들었는지, 다음 round에 noise를 넘겼는지 평가할 수 있다.

### 3) Stationary role policy

Role prompt를 round마다 따로 최적화하면 search space가 커진다. Agent 수가 $N$이고 round 수가 $R$이면 prompt block이 $N * R$개가 된다. 논문 계열은 같은 role의 prompt를 round 전체에서 공유하는 stationary policy로 두어 search space를 줄인다.

이렇게 하면 특정 role이 여러 round에서 consistently 약한지 볼 수 있다. 예를 들어 critic role이 반복적으로 shallow critique를 한다면, 그 role prompt를 고치는 것이 더 타당하다.

### 4) Verbalized BCD

방법은 두 prompt block을 번갈아 고치는 block coordinate descent 형태로 볼 수 있다.

1. Role block update
   - Aggregation prompt를 고정한다.
   - 낮은 structural credit을 받은 role prompt만 고친다.

2. Aggregation block update
   - Role prompt를 고정한다.
   - 낮은 temporal credit을 받은 round의 aggregation prompt만 고친다.

이 방식은 한 번에 모든 prompt를 바꾸는 것보다 안정적이다. 각 step에서 하나의 block만 수정하기 때문에, 어떤 변경이 성능에 영향을 주었는지 추적하기 쉽다.

# 4. Training / Data / Recipe

## 4-1. Data

확인 가능한 공개판 기준으로 평가는 multiple-choice reasoning benchmark에서 수행된다. AQuA, MedMCQA, GPQA, MMLU 같은 benchmark를 사용하고, optimization split과 test split을 분리한다.

중요한 점은 test set을 prompt search에 쓰지 않는다는 것이다. Prompt optimization은 별도 optimization examples에서 수행되고, report되는 성능은 disjoint test set에서 측정된다.

## 4-2. Training strategy

이 접근은 model weight를 학습하지 않는다. 학습되는 것은 prompt text다.

- Base LLM fixed
- Multi-agent protocol fixed
- Role prompt and aggregation prompt updated
- LLM critic이 local credit과 textual feedback 생성
- Low-credit component만 targeted update

따라서 이 방법은 fine-tuning보다 운영형 prompt optimization에 가깝다. Agent workflow를 이미 가진 팀이 로그를 모아 prompt rule을 고치는 방식으로 적용할 수 있다.

## 4-3. Engineering notes

실무적으로는 다음 포인트가 중요하다.

1. 로그 구조화가 먼저다
   - Agent output, round state, final decision을 명확히 저장해야 credit assignment가 가능하다.

2. Aggregation prompt가 독립 최적화 대상이 된다
   - 많은 agent system에서 aggregator는 단순 summarizer처럼 다뤄진다.
   - 하지만 실제로는 state transition policy에 가깝다.

3. Critic prompt 품질이 전체 품질을 좌우한다
   - Credit score가 틀리면 잘못된 role을 고칠 수 있다.
   - Human review 가능한 diagnosis format이 필요하다.

4. Good component를 보존해야 한다
   - Prompt optimization의 큰 위험은 regression이다.
   - 낮은 credit component만 수정하는 방식은 불필요한 drift를 줄인다.

# 5. Evaluation

## 5-1. Main results

공개판 기준으로 credit-guided prompt optimization은 unmodified prompt와 black-box optimization baseline보다 여러 dataset과 model family에서 일관된 개선을 보고한다. 특히 MedMCQA, GPQA, MMLU 같은 setting에서 structural-only, temporal-only, combined update를 비교하고, combined temporal plus structural update가 가장 안정적인 결과를 보인다.

## 5-2. What really matters in the experiments

이 논문의 실험에서 중요한 점은 headline accuracy보다 failure repair pattern이다.

1. Targeted update가 regression을 줄이는가
   - 좋은 prompt까지 같이 바꾸면 correct-to-wrong regression이 생길 수 있다.
   - Credit-guided update는 weak link만 수정하려고 한다.

2. Structural and temporal credit이 서로 보완되는가
   - Role이 중요한 task에서는 structural update가 더 클 수 있다.
   - Aggregation이 중요한 task에서는 temporal update가 더 클 수 있다.

3. Optimization cost가 줄어드는가
   - Black-box prompt search는 exploration cost가 높다.
   - Credit-guided 방식은 search space를 줄여 query efficiency를 높이는 방향이다.

# 6. Limitations

1. LLM critic 의존성
   - Credit score와 textual gradient가 LLM critic에 의존한다.
   - Critic이 틀리면 prompt update도 잘못된다.

2. Completed trajectory 중심
   - 이미 끝난 trajectory를 분석하는 구조라, online interaction 중 실시간 수정과는 다르다.

3. Fixed role assumption
   - Stationary role prompt는 search space를 줄이지만, open-ended agent system에서는 role 자체가 바뀔 수 있다.

4. Benchmark generalization
   - Multiple-choice reasoning에서 좋은 prompt update가 tool-use, coding, long-horizon planning으로 바로 일반화된다고 보기는 어렵다.

# 7. My Take

## 7-1. Why this matters for my work

Multi-agent system을 실제로 운영하면 성능보다 먼저 debug surface가 문제된다. 답이 틀렸을 때 어느 agent가 문제였는가, 어느 round에서 context가 망가졌는가, prompt를 어디부터 고쳐야 하는가, 고친 뒤 regression은 없는가 같은 질문에 답하지 못하면 agent prompt는 금방 관리 불가능해진다.

이 논문의 핵심은 prompt optimization을 자동 rewrite가 아니라 credit-aware debugging으로 바꾼다는 점이다. Agent workflow가 길어질수록 prompt는 instruction이 아니라 policy artifact가 된다. Policy artifact를 고치려면 local credit과 versioned update가 필요하다.

## 7-2. Reuse potential

재사용 가능한 아이디어는 다음과 같다.

1. Agent trajectory schema
   - Round, role, utterance, shared state, final output, score를 구조화해 저장한다.

2. Role-wise failure dashboard
   - Agent role별로 repeated failure pattern을 집계한다.

3. Aggregator prompt optimization
   - Summary prompt가 아니라 state transition prompt로 다룬다.

4. Prompt diff review
   - LLM이 만든 prompt update를 그대로 적용하지 말고, credit reason과 함께 human review한다.

5. Regression-aware prompt release
   - Correct-to-wrong shift를 별도 metric으로 추적한다.

## 7-3. Follow-up papers

- MAPRO
- GBC: Gradient-Based Connections for Optimizing Multi-Agent Systems
- TextGrad
- DSPy MIPRO
- Multi-agent debate and DyLAN 계열 논문

# 8. Summary

- Multi-agent prompt optimization의 핵심 병목은 final score를 local prompt update로 되돌리는 credit assignment다.
- Temporal credit은 어느 round가 문제였는지 보고, structural credit은 어느 role이 문제였는지 본다.
- State-space bottleneck과 stationary role policy는 credit을 계산 가능하게 만드는 구조적 가정이다.
- Verbalized BCD는 role prompt와 aggregation prompt를 번갈아 targeted하게 고친다.
