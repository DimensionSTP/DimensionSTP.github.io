---
layout: single
title: "LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents Review"
categories: Study-concept
tag: [LatentSkill, LLM, Agent, LoRA, Hypernetwork, Skill]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.06087)

[Repository link](https://github.com/yuaofan0-oss/LatentSkill)

LatentSkill은 agent system에서 점점 커지고 있는 skill context 문제를 꽤 직접적으로 건드리는 논문이다. 이 논문의 진짜 흥미로운 지점은 단순히 prompt token을 줄였다는 데 있지 않다. 더 중요한 포인트는, 자연어로 작성된 skill document를 매 step prompt에 넣는 대신 LoRA adapter로 컴파일해서 weight space에 저장한다는 점이다.

최근 agent framework에서는 task strategy, tool-use pattern, recovery heuristic 같은 절차 지식을 skill이라는 형태로 재사용한다. 이 방식은 운영 관점에서 매우 편하다. skill file을 업데이트하고, 필요할 때 가져오고, task에 맞춰 조합할 수 있기 때문이다. 하지만 매 decision step마다 같은 skill text를 다시 prompt에 붙이면 context가 계속 비싸지고, skill 자체가 plaintext로 노출되는 문제도 남는다.

LatentSkill은 이 trade-off를 weight-space skill이라는 형태로 다시 푼다. Textual skill은 skill compiler를 통과해 LoRA update가 되고, agent 실행 시에는 원래 skill text를 prompt에 넣지 않는다. 대신 해당 LoRA를 frozen backbone에 mount해서 행동을 바꾼다.

> 한 줄 요약: LatentSkill은 textual agent skill을 hypernetwork로 plug-and-play LoRA adapter로 변환해, skill knowledge를 context space가 아니라 weight space에 저장하는 agent skill framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- agent skill library가 커질수록 반복적인 skill prompting은 prefill cost와 context interference를 동시에 키운다.
- full fine-tuning이나 skill internalization은 skill을 prompt에서 없앨 수 있지만, 개별 skill의 update, removal, scaling, composition이 어려워진다.
- LatentSkill은 zero skill token, modular LoRA loading, scaling, composition을 한 설계 안에서 같이 보려 한다.
- 실험도 단순 benchmark score에서 끝나지 않고, generated LoRA weight space의 structure, controllability, composability까지 분석한다.

이 논문의 핵심 메시지는 단순하다. Agent skill은 꼭 prompt 안에 있어야 하는 것이 아니다. 자주 쓰이는 절차 지식이라면, context에 반복 삽입하는 대신 weight-space module로 바꾸는 것이 더 나은 serving primitive가 될 수 있다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 LLM agent의 reusable skill을 어떻게 효율적이고 모듈식으로 주입할 것인가다.

Agent skill은 보통 아래 정보를 담는다.

- task-specific strategy
- tool invocation pattern
- environment interaction rule
- common failure and recovery heuristic
- domain-specific procedural knowledge

이 skill을 prompt에 직접 넣으면 구현은 쉽다. 모델이 읽을 수 있는 자연어 그대로 들어가고, skill file만 바꾸면 바로 업데이트된다. 하지만 agent가 multi-step으로 움직일수록 같은 skill text가 decision step마다 반복된다. 그러면 prefill token이 늘고, context window 안에서 실제 observation과 skill instruction이 서로 경쟁한다.

문제는 단순히 token 수만이 아니다. Skill이 plaintext로 prompt에 존재하면 proprietary procedure가 그대로 노출될 수 있고, untrusted environment observation과 같은 instruction channel을 공유한다. Agent가 web page나 tool output을 읽는 상황에서는 prompt injection과 skill extraction 같은 공격면도 자연스럽게 생긴다.

## 1-2. Why previous approaches are insufficient

기존 선택지는 크게 두 방향이다.

1. In-context skill prompting

   Skill text를 그대로 prompt에 넣는다. 업데이트, 삭제, 조합은 쉽다. 하지만 매 step context overhead가 크고, skill text가 plaintext로 노출된다. Long context 안에서 모델이 skill을 항상 안정적으로 쓰는지도 보장되지 않는다.

2. Parametric skill internalization

   Agent fine-tuning이나 curriculum learning으로 skill을 model parameter에 흡수시킨다. Inference-time skill token은 줄어든다. 하지만 skill이 backbone에 섞여 들어가므로 특정 skill만 교체하거나 제거하거나 조합하기 어렵다.

LatentSkill은 이 두 방식 사이의 빈 공간을 노린다. 목표는 prompt에서 skill token을 없애면서도, skill을 독립적인 module처럼 load, unload, replace, scale, compose할 수 있게 만드는 것이다.

이 논문의 문제 설정은 prompt compression보다 넓다. 실제 질문은 "agent skill을 runtime artifact로 볼 것인가, parameter artifact로 볼 것인가"에 가깝다. LatentSkill은 이 질문에 대해 LoRA adapter cache라는 답을 제시한다.

# 2. Core Idea

## 2-1. Main contribution

LatentSkill의 핵심 기여는 textual skill을 LoRA weight로 변환하는 skill compiler다. Skill document $s$가 주어지면 hypernetwork가 skill-specific LoRA update $DeltaW(s)$를 생성한다. Backbone LLM은 frozen 상태로 유지되고, inference에서는 아래처럼 scaled adapter가 mount된다.

$$
W_l^{new} = W_l + alpha * DeltaW_l(s)
$$

여기서 $W_l$은 target module의 original weight, $DeltaW_l(s)$는 skill document에서 생성된 LoRA update, $alpha$는 injection strength다. 중요한 점은 agent가 행동할 때 prompt 안에 skill document $s$가 들어가지 않는다는 것이다. Model input은 task history 중심으로 유지되고, skill은 weight update로 반영된다.

논문이 주장하는 LatentSkill의 장점은 세 가지로 압축된다.

1. Structured

   Generated skill LoRA들이 weight space에서 domain-level cluster를 형성한다.

2. Controllable

   LoRA scaling coefficient $alpha$를 조절하면 skill influence를 연속적으로 조절할 수 있다.

3. Composable

   Skill을 semantically aligned component로 나누면 parameter-space arithmetic으로 skill을 조합할 수 있다.

이 세 가지가 중요하다. LatentSkill이 단순히 prompt를 LoRA로 압축한 것이라면 efficiency paper에 가깝다. 하지만 weight space가 구조를 갖고, strength control이 되고, component-level composition이 가능하다면 agent skill system의 abstraction 자체를 바꾸는 이야기로 읽을 수 있다.

## 2-2. Design intuition

LatentSkill의 설계 직관은 "skill을 읽는 것"과 "skill에 의해 행동하는 것"을 분리하는 데 있다.

In-context skill prompting에서는 agent가 매 step skill text를 읽어야 한다. 즉 skill은 context token으로 존재하고, 모델은 그 token을 attention으로 참조한다. LatentSkill에서는 skill compiler가 사전에 skill text를 읽고 LoRA adapter를 만든다. Agent 실행 시점의 backbone은 skill text를 다시 보지 않고, 이미 mount된 adapter를 통해 행동 분포가 바뀐다.

이 구조는 compiler 관점으로도 볼 수 있다.

- Source: natural-language skill document
- Compiler: Transformer-based hypernetwork
- Target artifact: LoRA adapter
- Runtime: frozen LLM plus selected adapter

이 analogy가 꽤 유용하다. Prompt-based skill은 매번 source code를 interpreter에 넣는 방식에 가깝고, LatentSkill은 source를 compiled artifact로 바꿔 cache하는 방식에 가깝다. 물론 이 compiled artifact가 완전한 executable program은 아니다. 하지만 serving path에서 반복 skill token을 제거한다는 점에서는 같은 방향의 최적화다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | textual agent skill을 prompt가 아니라 LoRA weight로 주입 |
| Backbone | frozen Qwen3-8B |
| Skill compiler | Transformer-based hypernetwork |
| Skill representation | generated LoRA adapter |
| Training stage 1 | skill document pretraining with reconstruction and completion |
| Training stage 2 | trajectory-supervised fine-tuning with teacher agent trajectories |
| Inference | compile skill once, cache adapter, mount selected adapter |
| Control knob | LoRA scaling coefficient alpha |
| Composition | adapter addition or component-level adapter addition |
| Main claim | skill token overhead를 줄이면서 modularity, scaling, composition을 유지 |

## 3-2. Module breakdown

### 1) Latent skill definition

LatentSkill은 textual skill document를 직접 prompt에 넣지 않는다. 대신 skill compiler가 document를 읽고 LoRA updates를 생성한다. 각 target module에서는 standard LoRA처럼 low-rank update가 frozen weight에 더해진다.

개념적으로는 다음처럼 볼 수 있다.

$$
SkillCompiler(s) = DeltaW(s)
$$

$$
Model(y | h, s) ~= Model(y | h; W + alpha * DeltaW(s))
$$

여기서 $h$는 agent history다. In-context skill prompting에서는 $h$와 $s$가 모두 prompt에 들어간다. LatentSkill에서는 $s$가 prompt에서 빠지고, $DeltaW(s)$가 parameter conditioning으로 들어간다.

이 방식의 장점은 skill을 cache할 수 있다는 점이다. Skill library가 있고, 각 skill이 자주 재사용된다면 skill document를 매번 다시 읽을 필요가 없다. 한 번 adapter로 compile한 뒤, task에 맞춰 load하면 된다.

### 2) Skill document pretraining

첫 번째 학습 단계는 skill document pretraining이다. Compiler는 GitHub에서 수집한 skill document corpus로 학습된다. 논문은 약 171K deduplicated skill documents, 약 300M tokens를 사용한다.

Pretraining task는 두 가지다.

1. Reconstruction

   Compiler가 전체 skill document를 읽고 adapter를 만든다. Adapted backbone은 reconstruction instruction을 받아 원래 document를 재생성하도록 학습된다.

2. Completion

   Compiler가 truncated skill prefix를 읽고 adapter를 만든다. Adapted backbone은 full skill document를 완성하도록 학습된다.

이 단계의 중요한 제약은 backbone LLM은 frozen이고, compiler parameter만 update된다는 점이다. 따라서 skill document의 정보가 backbone input token으로 직접 들어가지 않고, 반드시 generated adapter를 통해 전달되어야 한다.

### 3) Trajectory-supervised fine-tuning

두 번째 단계는 teacher trajectory 기반 SFT다. 각 training example은 skill document와 teacher agent trajectory를 포함한다. Compiler는 skill document 하나에서 adapter 하나를 만들고, 이 adapter는 전체 trajectory 동안 공유된다.

이 설계가 중요하다. 만약 step마다 adapter가 달라진다면 per-step shortcut이 생길 수 있다. LatentSkill은 같은 skill adapter가 multi-step trajectory 전체의 action pattern을 설명하도록 만들기 때문에, generated LoRA가 step-specific hint가 아니라 skill-level policy bias를 담도록 유도한다.

논문은 Xia et al.의 skill library와 teacher trajectories를 사용한다. SFT에는 237 complete ALFWorld task trajectories와 500 complete Search-QA task trajectories가 쓰인다. ALFWorld 쪽은 5개 skill, Search-QA 쪽은 3개 skill을 사용한다.

### 4) Inference-time skill control

Inference에서는 skill compilation과 agent execution이 분리된다. 먼저 skill library의 각 skill을 adapter로 compile하고 cache한다. 이후 task가 들어오면 skill selector가 하나 이상의 skill을 고르고, 해당 adapter를 frozen backbone에 mount한다.

Single skill이면 scaling coefficient $alpha$가 핵심 control knob가 된다.

- $alpha = 0$이면 frozen backbone과 같다.
- 적절한 $alpha$에서는 skill behavior가 강화된다.
- 너무 큰 $alpha$에서는 backbone behavior가 깨질 수 있다.

이 점은 실험에서도 inverted-U curve로 나타난다. 즉 LatentSkill은 binary on/off skill이 아니라, strength를 조절할 수 있는 skill interface에 가깝다.

### 5) Parameter-space composition

여러 skill을 같이 쓰고 싶다면 adapter를 weight space에서 더할 수 있다.

$$
DeltaW_S = sum_{s in S} DeltaW(s)
$$

하지만 논문은 naive full-skill merging이 항상 좋은 것은 아니라고 보여준다. Look skill과 Pick skill처럼 general component와 mistakes component가 공유되는 경우, complete LoRA를 그대로 더하면 공유 behavior가 두 번 증폭될 수 있다. 그러면 target recognition은 되지만 pick-up action threshold가 깨지는 식의 interference가 생긴다.

그래서 LatentSkill은 component-level composition을 강조한다. Skill text를 semantically aligned component로 나누고, shared component는 한 번만 넣고, task-specific component만 추가하는 방식이다. 이때 composition은 단순한 weight addition이 아니라 text decomposition granularity와 weight addition granularity를 맞추는 문제로 바뀐다.

# 4. Training / Data / Recipe

## 4-1. Data

학습 데이터는 크게 두 부류다.

1. Skill document pretraining corpus

   GitHub에서 수집한 약 171K deduplicated skill documents, 약 300M tokens를 사용한다. 이 데이터는 skill compiler가 natural-language procedure를 LoRA weight로 옮기는 기본 능력을 배우는 데 쓰인다.

2. Agent trajectory SFT data

   ALFWorld와 Search-QA teacher trajectories를 사용한다. ALFWorld는 household interaction task 6개 category를 포함하고, Search-QA는 single-hop과 multi-hop search-augmented QA dataset을 포함한다.

Evaluation도 두 benchmark에서 진행된다.

| Benchmark | Task type | Metric | Note |
| --- | --- | --- | --- |
| ALFWorld | text-based embodied interaction | success rate | seen 140 episodes, unseen 134 episodes, max 50 steps |
| Search-QA | search-augmented QA | exact match | NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultihopQA, MuSiQue, Bamboogle |

Search-QA에서는 NQ와 HotpotQA가 training data로 쓰이고, 나머지 5개 dataset은 out-of-domain evaluation으로 사용된다. Bamboogle은 full set 125 examples를 평가하고, 나머지 dataset은 각각 500 examples를 sample한다.

## 4-2. Training strategy

Pretraining 설정은 다음과 같다.

| Item | Value |
| --- | --- |
| Hardware | 8 H100 GPUs |
| Epochs | 10 |
| Batch size | 64 |
| Learning rate | 5e-5 |
| Warmup steps | 200 |
| Max sequence length | 4096 |
| Optimizer | AdamW |
| Weight decay | 0.1 |

SFT 설정은 다음과 같다.

| Item | Value |
| --- | --- |
| Hardware | 8 H100 GPUs |
| Epochs | 10 |
| Batch size | 32 |
| Learning rate | 1e-5 |
| Warmup steps | 400 |
| Max sequence length | 4096 |
| Optimizer | AdamW |
| Weight decay | 0.1 |

여기서 핵심은 backbone을 frozen으로 유지한다는 점이다. Training 대상은 skill compiler다. 따라서 결과물은 task-specific fine-tuned model 하나가 아니라, skill text를 adapter로 바꾸는 compiler다.

## 4-3. Engineering notes

실무적으로 볼 때 이 논문에서 재사용할 만한 engineering point는 아래다.

1. Skill compilation을 offline cache로 분리한다.

   자주 쓰이는 skill은 미리 adapter로 compile해둘 수 있다. 그러면 runtime path에서는 skill text token을 반복해서 넣지 않고 adapter load만 하면 된다.

2. Adapter versioning이 중요해진다.

   Skill document가 바뀌면 generated adapter도 바뀐다. 따라서 skill text version, compiler checkpoint version, LoRA config version을 같이 관리해야 한다.

3. Skill selector는 별도 문제로 남는다.

   논문 실험에서는 ALFWorld skill을 task category로 match하고, Search-QA skill도 dataset과 question type에 맞춰 match한다. 실제 product에서는 어떤 skill을 언제 mount할지 결정하는 selector가 전체 system quality를 크게 좌우할 수 있다.

4. Adapter serving cost도 봐야 한다.

   Prompt token은 줄지만, dynamic LoRA loading과 composition이 serving stack에서 공짜는 아니다. 많은 user-specific skill을 동시에 운영하려면 adapter cache, batching, memory residency, hot-swap latency까지 같이 설계해야 한다.

5. Component design이 composition quality를 좌우한다.

   이 논문은 skill composition이 단순한 adapter addition이 아니라, skill document를 어떤 component로 나눌지에 의존한다는 점을 잘 보여준다.

# 5. Evaluation

## 5-1. Main results

먼저 ALFWorld 결과를 보면 LatentSkill은 In-Context Skill baseline보다 평균 success rate를 크게 올리면서 prefill token을 줄인다.

| Method | Seen Avg | Unseen Avg | Seen Prefill | Unseen Prefill |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 43.6 | 47.0 | 0.44 | 0.44 |
| In-Context Skill | 52.9 | 56.0 | 1.21 | 1.23 |
| LatentSkill | 74.3 | 69.4 | 0.44 | 0.44 |

논문이 강조하는 핵심 수치는 다음이다.

- ALFWorld seen split에서 In-Context Skill 대비 +21.4 points.
- ALFWorld unseen split에서 In-Context Skill 대비 +13.4 points.
- ALFWorld prefill overhead는 In-Context Skill 대비 64.1% 감소.
- Seen split 평균 step도 Vanilla 35.0에서 LatentSkill 28.4로 줄어든다.

Search-QA에서도 평균 EM 기준으로 LatentSkill이 가장 높다.

| Method | Avg EM | Cost |
| --- | ---: | ---: |
| Vanilla | 28.1 | 0.24 |
| Few-Shot | 31.7 | 0.94 |
| RAG | 34.4 | 0.89 |
| In-Context Skill | 32.6 | 1.10 |
| LatentSkill | 35.6 | 0.31 |

여기서도 메시지는 performance plus efficiency다. LatentSkill은 Search-QA 평균 EM을 In-Context Skill 대비 +3.0 points 올리고, context overhead를 72.2% 줄인다.

다만 이 표를 읽을 때 주의할 점도 있다. LatentSkill이 모든 Search-QA subset에서 이기는 것은 아니다. 2WikiMultihopQA와 Bamboogle에서는 In-Context Skill이 더 높다. 따라서 이 결과는 "모든 search QA behavior가 좋아졌다"보다, 평균적으로 skill을 weight space로 옮겨도 성능을 유지하거나 올릴 수 있고 token overhead를 크게 줄였다는 쪽으로 읽는 것이 맞다.

## 5-2. What really matters in the experiments

### 1) 성능 향상보다 더 중요한 것은 cost surface다

이 논문의 가장 직접적인 메시지는 skill token overhead를 없애도 agent performance가 떨어지지 않는다는 것이다. 오히려 ALFWorld에서는 크게 오른다. 이건 중요한 결과다. 단순히 prompt를 줄이는 compression이 아니라, skill이 parameter conditioning으로 들어가면서 action distribution 자체가 더 안정화됐을 가능성을 보여준다.

특히 ALFWorld에서 LatentSkill은 seen split에서 prefill 0.44k를 쓰는데, In-Context Skill은 1.21k를 쓴다. 그런데 success rate는 LatentSkill이 74.3, In-Context Skill이 52.9다. 즉 token을 더 많이 넣는다고 skill이 더 잘 쓰이는 것이 아니다.

### 2) Weight space가 실제로 semantic structure를 가진다

논문은 MDS visualization으로 generated LoRA weight를 분석한다. In-domain에서는 5개 ALFWorld skill과 3개 Search skill이 weight space에서 분리된 cluster를 만든다. SFT 후에는 inter-cluster distance가 0.0887에서 0.0704로 줄어들지만, domain-level structure는 유지된다.

OOD skill 분석도 흥미롭다. Code, Finance, Writing skill들이 domain별로 분리된 cluster를 형성하고, 각 domain의 within-domain similarity가 cross-domain similarity보다 높다. 이건 compiler가 단순히 in-domain skill label을 외운 것이 아니라, procedural text의 domain signal을 LoRA weight space에 어느 정도 반영한다는 근거로 읽을 수 있다.

### 3) alpha scaling은 단순 hyperparameter가 아니라 skill strength control이다

LatentSkill은 LoRA scaling coefficient $alpha$를 sweep한다. 결과는 inverted-U curve에 가깝다.

| Split | Frozen baseline | Best avg | Too large alpha |
| --- | ---: | ---: | ---: |
| Seen | 43.57 at alpha 0.0 | 74.29 at alpha 0.6 | 22.86 at alpha 1.2 |
| Unseen | 47.01 at alpha 0.0 | 70.90 at alpha 0.5 | 8.21 at alpha 1.2 |

이 결과가 중요한 이유는, generated skill LoRA가 너무 약하면 skill behavior가 충분히 안 들어가고, 너무 강하면 backbone의 기본 decision behavior를 망가뜨린다는 점을 보여주기 때문이다. 즉 skill adapter는 켜고 끄는 switch가 아니라, task별로 조절해야 하는 control signal이다.

### 4) Skill composition은 가능하지만 naive merging은 위험하다

Look skill과 Pick skill을 조합하는 실험에서 Component Merging은 seen 84.6%, unseen 77.8%를 기록한다. Look-Only보다 성공 episode가 늘고, Look-Only가 맞힌 episode를 잃지 않는다.

반대로 Direct Merging과 Text Merging은 interference를 만든다. Direct Merging은 shared component까지 두 번 더해져 general behavior가 과증폭된다. Text Merging은 두 skill text를 하나로 concat해서 compiler에 넣는데, compiler는 single skill text만 보며 학습됐기 때문에 OOD input이 된다.

이 실험은 LatentSkill의 장점과 한계를 동시에 보여준다. Parameter-space composition은 가능하지만, skill components가 semantic하게 align되어 있어야 한다. 아무 LoRA나 더한다고 compositional generalization이 생기는 것은 아니다.

### 5) Security 결과는 방향성은 좋지만 과장하면 안 된다

논문은 perturbation과 attack 실험도 한다. Paraphrase, Plaintext, Reorder, Noise 같은 skill text perturbation에서 LatentSkill은 In-Context Skill보다 대체로 안정적이다. Prompt-level Hijack에서도 ALFWorld 기준 In-Context Skill은 52.9에서 8.57로 떨어지지만, LatentSkill은 38.6을 유지한다.

이 결과는 skill text가 prompt에 직접 노출되지 않는 것의 장점을 보여준다. 하지만 이것을 완전한 보안으로 읽으면 안 된다. Skill behavior는 여전히 adapter weight와 model behavior에 남아 있고, model extraction이나 behavioral probing까지 막는다는 증거는 아니다. 이 논문이 보여준 것은 plaintext exposure와 prompt-level attack surface를 줄인다는 쪽에 가깝다.

# 6. Limitations

1. Evaluation scope가 아직 좁다.

   논문도 명시하듯 실험은 ALFWorld와 Search-QA에 집중되어 있다. Web browsing, software engineering, multi-agent collaboration 같은 실제 agent deployment setting에서 같은 결론이 유지되는지는 추가 검증이 필요하다.

2. Backbone과 LoRA config가 고정되어 있다.

   모든 main experiment는 frozen Qwen3-8B와 fixed LoRA configuration 위에서 수행된다. 다른 model family, 다른 model scale, 다른 adapter rank나 target module에서도 같은 skill geometry가 나오는지는 아직 알 수 없다.

3. Skill selection problem은 대부분 통제되어 있다.

   ALFWorld에서는 task category로 skill을 match하고, Search-QA에서는 dataset과 question sub-type에 따라 skill을 match한다. 실제 product에서는 skill library가 훨씬 크고, selector가 틀릴 수 있다. LatentSkill 자체의 품질과 skill retrieval or selection 품질은 분리해서 봐야 한다.

4. Hypernetwork training cost가 작지 않다.

   Pretraining에 약 171K skill documents와 300M tokens가 쓰이고, 8 H100 GPUs에서 학습한다. 즉 이 방법은 skill 몇 개를 위해 가볍게 붙이는 trick이라기보다, skill compiler를 하나의 infrastructure component로 학습하는 접근이다.

5. Composition은 component alignment에 의존한다.

   Component Merging은 잘 작동하지만, 이것은 skill text를 잘 분해했기 때문이다. Whole-skill adapter addition이나 text concatenation은 interference를 만든다. 따라서 composition을 일반화하려면 skill authoring guideline과 component schema가 필요하다.

6. Weight-space skill은 해석과 governance가 더 어려울 수 있다.

   Prompt skill은 사람이 읽고 audit하기 쉽다. 반면 LoRA weight로 변환된 skill은 직접 읽을 수 없다. Skill content의 노출은 줄지만, 어떤 behavior가 실제로 encode됐는지 검증하는 도구가 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 agent system을 볼 때 context engineering과 parameter-efficient adaptation을 분리된 주제로 보지 말아야 한다는 점을 잘 보여준다.

보통 agent skill은 prompt engineering artifact로 다룬다. 좋은 markdown skill file을 만들고, 필요한 순간 retrieval해서 넣고, prompt budget 안에서 정리한다. LatentSkill은 이 artifact를 serving artifact로 옮긴다. 자주 쓰이는 skill은 prompt에 반복 삽입하지 말고, adapter로 compile해서 cache할 수 있다는 것이다.

이 관점은 실서비스 agent에서 꽤 중요하다. Product agent가 수십 개 tool skill, domain procedure, customer-specific rule을 매번 prompt에 넣는다면 token cost, latency, leakage risk가 동시에 커진다. LatentSkill류 접근은 이 반복 instruction을 adapter cache로 바꿀 수 있는 가능성을 보여준다.

다만 이걸 곧바로 "prompt skill의 대체재"로 보면 안 된다. 더 정확히는 skill lifecycle의 일부를 weight space로 옮기는 방법이다. 빠르게 수정해야 하는 rule, compliance text, user-visible instruction은 여전히 prompt에 남기는 편이 안전할 수 있다. 반대로 반복적이고 안정적인 procedural policy는 adapter로 옮길 가치가 있다.

## 7-2. Reuse potential

재사용하고 싶은 포인트는 아래 5가지다.

1. Frequently used skill을 adapter cache로 관리한다.

   Skill text를 매번 prompt에 넣는 대신, stable skill은 compiled adapter로 저장한다. Prompt에는 task observation과 short instruction만 남긴다.

2. Skill strength를 alpha로 조절한다.

   특정 task에서 skill이 너무 강하게 개입하면 오히려 decision이 망가질 수 있다. 따라서 skill on/off가 아니라 task-specific alpha tuning이 필요하다.

3. Skill authoring을 component-aware하게 설계한다.

   나중에 composition하려면 skill file을 general component, task-specific component, mistake-avoidance component처럼 잘게 나누는 편이 좋다.

4. Evaluation metric에 token cost를 같이 넣는다.

   Agent skill의 가치는 success rate만으로 보면 부족하다. Prefill token, decode token, interaction steps, adapter loading cost를 같이 봐야 한다.

5. Skill governance를 adapter level로 확장한다.

   Skill document만 audit하는 것이 아니라, generated adapter가 어떤 behavior를 유도하는지도 probe해야 한다. Weight-space skill은 운영상 편리하지만, 그만큼 observability가 더 중요해진다.

## 7-3. Follow-up papers

- SkillBank and recursive skill evolution 계열 연구
- SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization
- SkillsInjector: Dynamic Skill Context Construction for LLM Agents
- Text-to-LoRA and Doc-to-LoRA 계열 hypernetwork adapter generation 연구
- SHINE: scalable in-context hypernetwork for context-to-LoRA mapping
- LoRA composition and adapter routing 관련 연구

# 8. Summary

- LatentSkill은 textual agent skill을 prompt token이 아니라 LoRA adapter로 변환하는 framework다.
- 핵심은 zero skill tokens in prompt와 plug-and-play modularity를 동시에 유지하려는 점이다.
- Skill compiler는 document pretraining과 trajectory SFT를 통해 procedural text를 skill-level policy adapter로 바꾼다.
- ALFWorld와 Search-QA에서는 In-Context Skill보다 평균 성능이 높고 prefill overhead도 크게 줄어든다.
- 가장 중요한 한계는 evaluation scope, fixed backbone, controlled skill matching, component-aligned composition 의존성이다.
