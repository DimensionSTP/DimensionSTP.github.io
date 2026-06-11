---
layout: single
title: "On the Scaling of PEFT: Towards Million Personal Models of Trillion Parameters Review"
categories: Study-concept
tag: [PEFT, LoRA, Personalization]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.02437)

> 한 줄 요약: 이 논문은 PEFT를 full fine-tuning의 저가 대체재가 아니라, 강한 shared base model 위에 쌓이는 persistent local state로 다시 정의한다. 핵심은 adapter를 preference, skill, tool habit, memory-like update를 담는 개인 모델 단위로 보고, Scale Up, Scale Down, Scale Out 세 축에서 PEFT가 어디까지 확장될 수 있는지 묻는 것이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- 개인화 모델을 매번 full checkpoint로 복제하는 대신, base model과 adapter catalog를 분리하는 방향을 명확하게 보여준다.
- PEFT의 질문을 parameter saving에서 system scalability와 model lifecycle management로 확장한다.
- million-scale personal model 운영에서 필요한 identity, revision, provenance, evaluation, serving residency 문제를 한 프레임에 넣는다.

이 논문을 단순히 "LoRA를 더 싸게 쓰자"로 읽으면 핵심을 놓친다. 여기서 저자들이 잡는 framing은 PEFT가 비용 절감 수단이라는 기존 관점보다 넓다. base model은 shared competence를 제공하고, 작은 adapter는 특정 사용자나 작업 인스턴스의 행동 변화를 보존한다.

**핵심 framing**

PEFT는 싸게 fine-tuning하는 기술이 아니라, 하나의 큰 base model 위에 수많은 local model state를 붙이는 운영 단위가 될 수 있다.

그래서 이 글에서는 논문을 adapter method 논문이라기보다, personal model을 대규모로 운영하기 위한 scaling argument로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

- 현재 PEFT는 보통 full fine-tuning 대비 적은 parameter만 학습하는 방법으로 소개된다.
- 하지만 개인화, domain adaptation, tool-use policy, agent habit, memory-like update를 생각하면 더 중요한 질문은 다르다.
- 문제는 한두 개의 adapter를 만드는 것이 아니라, 매우 많은 adapted instance를 안정적으로 만들고, 추적하고, 평가하고, serving하는 것이다.
- 논문은 이를 세 축으로 나눠 정리한다.
  - Scale Up: base model이 강해질수록 작은 adapter update가 더 유용해지는가.
  - Scale Down: adapter가 얼마나 작아져도 안정적인 local behavior를 담을 수 있는가.
  - Scale Out: 수많은 persistent adapter instance를 어떻게 공존시킬 것인가.

이 관점에서는 모델 하나의 score보다 lifecycle이 더 중요해진다. adapter는 training artifact이면서 동시에 product state가 된다.

## 1-2. Why previous approaches are insufficient

- full fine-tuning은 개인별 모델을 만들기에는 storage, training, deployment cost가 너무 크다.
- prompt-based personalization은 간단하지만, 긴 기간에 걸친 behavior update나 skill accumulation을 안정적으로 보존하기 어렵다.
- 기존 PEFT 논의는 흔히 parameter efficiency와 benchmark score에 집중한다.
- 그러나 million-scale personal model에서는 adapter identity, versioning, rollback, evaluation, serving residency가 함께 필요하다.
- 즉 병목은 algorithm 하나가 아니라, adapter를 durable state로 관리하는 전체 시스템이다.

**기존 관점의 한계**

PEFT를 단일 학습 기법으로만 보면, 많은 adapter가 만들어진 뒤의 운영 문제를 설명하지 못한다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 PEFT scaling을 세 축으로 재구성한다는 점이다.

1. Scale Up
   - 더 강한 shared base model 위에서 작은 adapter가 어느 정도의 추가 behavior를 담을 수 있는지 본다.
   - 이는 adapter capacity만 보는 문제가 아니라, base prior가 얼마나 많은 일을 대신해 주는가의 문제다.

2. Scale Down
   - adapter를 작게 만들수록 storage와 handoff는 쉬워지지만, local update의 안정성은 약해질 수 있다.
   - 따라서 adapter size와 reliability 사이의 경계가 중요하다.

3. Scale Out
   - adapter가 많아지면 단순히 파일이 많아지는 문제가 아니다.
   - identity, revision, provenance, evaluation, serving residency, cold loading, active working set 관리가 모두 필요해진다.

논문은 이 세 축을 통해 PEFT를 compact substrate for persistent personal models로 해석한다.

## 2-2. Design intuition

직관은 꽤 단순하다.

$$
model_i = base + adapter_i
$$

여기서 base는 공통 능력이고, $adapter_i$는 특정 사용자, task, tool habit, policy preference를 담는 local state다.

이 식이 중요한 이유는 model copy의 단위가 바뀌기 때문이다. full checkpoint를 사용자별로 복제하면 impossible에 가까운 운영 문제가 되지만, adapter를 local state로 보면 base는 공유하고 차이만 관리할 수 있다.

**설계 직관**

base model이 커질수록 모든 사용자가 공유하는 능력은 base에 남기고, 개인별 차이는 adapter에 넣는 구조가 더 자연스러워진다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | PEFT를 persistent personal model을 위한 scalable local-state substrate로 재해석 |
| Main axes | Scale Up, Scale Down, Scale Out |
| Local state | LoRA 같은 small trainable adapter |
| Shared state | expensive foundation model deployment |
| Infrastructure example | MinT style adapter identity, revision, provenance, evaluation, serving residency management |
| Key difference | PEFT를 parameter-saving trick이 아니라 model lifecycle abstraction으로 다룸 |

## 3-2. Module breakdown

### 1) Adapter as persistent local state

- 논문에서 adapter는 단순히 임시 fine-tuning 결과물이 아니다.
- adapter는 특정 instance의 preference, skill, tool habit, memory-like update를 담는 local state다.
- 이 local state는 base model과 분리되어 저장되고, 다시 불러오고, 버전 관리되고, 평가되고, serving되어야 한다.
- 이 관점에서는 adapter 하나가 작은 모델처럼 행동한다.

**중요한 변화**

fine-tuning artifact가 product state가 된다.

### 2) Scale Up: stronger shared priors

- Scale Up은 base model이 커지고 강해질수록 adapter의 의미가 어떻게 바뀌는지를 본다.
- 작은 adapter가 모든 지식을 새로 학습할 필요는 없다.
- base model이 이미 general competence를 제공하면, adapter는 특정 행동 편향이나 task-local update에 집중할 수 있다.
- 따라서 큰 base model과 작은 adapter의 조합은, 작은 model을 따로 많이 학습하는 방식과 다른 economics를 만든다.

이 축은 PEFT scaling에서 자주 과소평가된다. adapter capacity만 따로 보면 작은 update처럼 보이지만, 실제 표현력은 base model prior와 결합되어 결정된다.

### 3) Scale Down: minimal but reliable adapters

- Scale Down은 adapter를 얼마나 작게 만들 수 있는가를 묻는다.
- adapter가 작아지면 storage, transfer, serving residency가 모두 쉬워진다.
- 반대로 너무 작으면 task-specific behavior를 담지 못하거나, update가 불안정해질 수 있다.
- 그래서 이 축의 핵심은 tiny adapter를 만드는 것이 아니라, tiny adapter가 reliable local state로 남을 수 있는 조건을 찾는 것이다.

**실무적 의미**

adapter가 작아질수록 million-scale catalog가 현실적이 되지만, evaluation과 rollback 없이는 위험한 state explosion이 될 수 있다.

### 4) Scale Out: many persistent instances

- Scale Out은 수많은 adapter가 동시에 존재할 때의 문제다.
- 여기서는 training algorithm보다 metadata system과 serving system이 중요해진다.
- 각 adapter는 다음 정보를 가져야 한다.
  - adapter identity
  - base model compatibility
  - revision history
  - provenance
  - evaluation result
  - serving residency state
  - rollback path
- MinT는 이런 문제를 관리하는 infrastructure example로 제시된다.

MinT 관련 공개 요약에 따르면, base model을 resident로 유지하고 LoRA adapter revision만 rollout, update, export, evaluation, serving, rollback 경로로 이동시키는 방식이 핵심이다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 특정 단일 dataset recipe를 강조하는 논문이라기보다 PEFT scaling의 역할을 재정의하는 논문에 가깝다. 따라서 데이터 자체보다 어떤 instance-specific signal이 adapter에 들어가는지가 중요하다.

가능한 local signal은 다음과 같이 정리할 수 있다.

- preference signal
- domain skill
- tool-use habit
- agent policy trace
- memory-like update
- evaluation feedback
- user-specific correction

**데이터 관점의 핵심**

PEFT가 개인 모델의 local state가 되려면, 학습 데이터도 일회성 fine-tuning dataset이 아니라 지속적으로 갱신되는 behavior evidence가 된다.

## 4-2. Training strategy

training loop는 아래처럼 이해할 수 있다.

1. shared base model은 가능한 한 resident로 유지한다.
2. 특정 user, task, agent policy에 대해 adapter를 학습한다.
3. 학습된 adapter revision을 export한다.
4. evaluation gate를 통과한 revision만 serving 후보가 된다.
5. 문제가 생기면 이전 revision으로 rollback한다.
6. active working set과 cold catalog를 분리한다.

이 구조는 일반적인 fine-tuning pipeline과 다르다. 목표가 best checkpoint 하나를 고르는 것이 아니라, 계속 늘어나는 adapter population을 관리하는 것이기 때문이다.

## 4-3. Engineering notes

- adapter-only movement는 full checkpoint movement보다 serving과 update 경로를 단순하게 만든다.
- 많은 adapter가 존재할수록 storage cost보다 indexing, loading, residency, eviction, compatibility check가 중요해진다.
- base model이 바뀌면 adapter compatibility가 깨질 수 있으므로 base revision과 adapter revision을 함께 추적해야 한다.
- evaluation은 offline score만이 아니라 deployment safety, regression, user-specific behavior drift까지 포함해야 한다.
- MinT 같은 system layer는 PEFT 연구를 model training에서 service orchestration 문제로 확장한다.

**엔지니어링 포인트**

million personal models를 말하려면, adapter를 얼마나 잘 학습하느냐만큼 adapter를 얼마나 잘 운영하느냐가 중요하다.

# 5. Evaluation

## 5-1. Main results

공개 abstract 기준으로 이 논문이 제시하는 결과의 방향은 다음 세 가지다.

- Scale Up: stronger shared priors make small local updates more useful.
- Scale Down: small adapters can remain meaningful local state under reliability constraints.
- Scale Out: many persistent adapted instances can coexist when managed through infrastructure like MinT.

MinT 공개 요약과 연결해서 보면 다음 수치가 특히 중요하다.

| Axis | Publicly visible evidence |
| --- | --- |
| Scale Up | LoRA RL training and serving validated beyond 1T total parameters in MinT report |
| Scale Down | rank-1 setting에서 adapter가 base-model size의 1% 미만일 수 있음 |
| Scale Down | adapter-only handoff가 4B dense model에서 18.3x, 30B MoE에서 2.85x measured step reduction으로 보고됨 |
| Scale Out | tensor-parallel deployment에서 $10^6$-scale addressable catalog를 지원하고, single-engine sweep은 100K로 측정됨 |
| Scale Out | packed MoE LoRA tensor가 live engine loading을 8.5-8.7x 개선했다고 보고됨 |

다만 이 표는 PEFT scaling 논문의 abstract와 MinT report의 공개 abstract를 함께 해석한 것이다. 최종 블로그 게시 전에는 PEFT paper 본문에서 어떤 수치가 직접 인용되는지 다시 확인하는 편이 안전하다.

## 5-2. What really matters in the experiments

이 논문에서 중요한 실험적 메시지는 leaderboard 승패보다 system boundary다.

1. PEFT는 model quality만으로 평가하면 부족하다.
   - adapter가 local state라면 quality, storage, latency, update path, rollback이 함께 지표가 된다.

2. base model과 adapter를 분리하면 scaling economics가 바뀐다.
   - full checkpoint를 복제하지 않고 adapter만 이동하면, 많은 개인 모델을 운영할 가능성이 생긴다.

3. adapter catalog는 passive storage가 아니다.
   - 어떤 adapter가 active GPU working set에 있고, 어떤 adapter가 cold storage에 있는지 관리해야 한다.

4. 작은 adapter는 위험도 작다는 뜻이 아니다.
   - 작은 state라도 user-facing behavior를 바꾸므로 evaluation gate가 필요하다.

이 논문의 가장 중요한 포인트는 PEFT를 benchmark optimization method가 아니라, personalized AI system의 state management primitive로 본다는 점이다.

# 6. Limitations

1. 공개 abstract만 보면 상세 benchmark, dataset, ablation 구성은 충분히 드러나지 않는다. 원문 PDF 본문 기준으로 table과 appendix를 다시 확인해야 한다.
2. million-scale personal model이라는 표현은 강력하지만, 실제 production에서는 privacy, abuse, data governance, deletion request, compliance가 함께 붙는다.
3. base model upgrade가 adapter catalog에 미치는 영향을 깊게 다뤄야 한다. base가 바뀌면 기존 adapter behavior가 유지된다는 보장이 없다.
4. PEFT adapter가 preference나 memory-like update를 담을 수 있다고 해도, 장기 기억 시스템 전체를 대체한다고 보기는 어렵다.
5. active adapter routing과 cold loading은 user traffic distribution에 따라 성능이 크게 달라질 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

- 이 논문은 개인화 LLM을 model training 문제가 아니라 state management 문제로 바꾸어 본다.
- RAG, memory, tool habit, preference tuning을 모두 prompt layer에만 두면 state가 흩어진다.
- 반대로 adapter를 durable local state로 보면, 각 사용자 또는 agent policy마다 작은 trainable state를 둘 수 있다.
- 이는 agent personalization, enterprise deployment, tool-using assistant 운영에서 꽤 중요한 방향이다.

**개인화의 단위 변화**

prompt가 개인화의 전부가 아니라, adapter revision도 개인화의 단위가 될 수 있다.

## 7-2. Reuse potential

실무에서 바로 참고할 수 있는 부분은 아래와 같다.

1. Adapter registry design
   - adapter id, base id, revision, owner, provenance, evaluation summary를 분리해서 관리한다.

2. Evaluation gate
   - 새 adapter revision은 바로 serving하지 않고, regression test와 safety test를 통과하게 만든다.

3. Active working set management
   - 모든 adapter를 GPU에 올리지 않고, hot adapter와 cold adapter를 분리한다.

4. Rollback-first deployment
   - adapter는 빠르게 업데이트되는 state이므로, rollback path가 training보다 먼저 설계되어야 한다.

5. Base-adapter compatibility check
   - base model revision이 바뀔 때 adapter migration test를 별도 프로세스로 둔다.

## 7-3. Follow-up papers

- LoRA: Low-Rank Adaptation of Large Language Models
  - PEFT의 기본 primitive를 이해하기 위한 출발점이다.

- MinT: Managed Infrastructure for Training and Serving Millions of LLMs
  - 이 논문에서 언급되는 infrastructure example을 더 구체적으로 보기 좋다.

- Trust Region On-Policy Distillation
  - adapter나 policy update가 안정적으로 움직여야 한다는 관점에서 같이 읽기 좋다.

- LongTraceRL
  - long-context agent trajectory에서 persistent update를 어떻게 만들 것인가와 연결된다.

# 8. Summary

- 이 논문은 PEFT를 cheap fine-tuning이 아니라 persistent personal model state로 재정의한다.
- 핵심 축은 Scale Up, Scale Down, Scale Out이다.
- base model은 shared competence를 담당하고, adapter는 preference, skill, tool habit, memory-like update를 담는다.
- MinT는 adapter identity, revision, provenance, evaluation, serving residency를 관리하는 infrastructure example로 제시된다.
- 실무적으로는 adapter 학습보다 adapter lifecycle과 evaluation gate가 더 중요해지는 방향을 보여준다.
