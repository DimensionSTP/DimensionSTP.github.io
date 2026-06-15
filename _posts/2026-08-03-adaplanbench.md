---
layout: single
title: "AdaPlanBench: Evaluating Adaptive Planning in Large Language Model Agents under World and User Constraints Review"
categories: Study-concept
tag: [LLM, Agent, Benchmark, Planning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.05622)

[Code](https://github.com/JiayuJeff/AdaPlanBench)

[Dataset](https://huggingface.co/datasets/JiayuJeff/AdaPlanBench)

AdaPlanBench는 agent planning benchmark를 조금 더 현실적인 방향으로 밀어붙이는 논문이다. 보통 planning benchmark는 task와 constraint가 처음부터 주어진다고 가정한다. 그런데 실제 agent workflow에서는 그렇지 않은 경우가 많다. 사용자는 preference를 처음에 다 말하지 않고, 환경도 모든 tool availability를 upfront로 공개하지 않는다. Agent가 plan을 내놓으면 그제야 "그 도구는 없다", "그 방식은 싫다", "그 방법은 너무 위험하다" 같은 feedback이 나온다.

이 논문의 핵심은 바로 이 상황을 benchmark protocol로 만드는 것이다. AdaPlanBench는 household planning task를 기반으로 hidden world constraint와 hidden user constraint를 만들고, agent가 violation을 일으킬 때마다 constraint를 조금씩 공개한다. Agent는 이미 받은 feedback을 기억하면서, 새로 드러난 constraint까지 반영해 plan을 계속 수정해야 한다.

> 한 줄 요약: AdaPlanBench는 307개 household planning task에서 hidden world constraint와 hidden user constraint를 점진적으로 공개하면서, LLM agent가 plan을 얼마나 잘 revise하고 유지하는지 평가하는 dynamic interactive benchmark다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Agent benchmark가 static task solving에서 dynamic re-planning으로 넘어가야 하는 이유를 명확하게 보여준다.
- World constraint와 user constraint를 분리해, agent failure가 도구/환경 문제인지 preference adaptation 문제인지 볼 수 있게 만든다.
- Accuracy만 보지 않고 VPR, repeated violation, triggered constraint 같은 interaction metric을 같이 본다.
- 강한 model도 hidden constraint가 누적되면 plan quality와 constraint consistency를 동시에 유지하기 어렵다는 점을 수치로 보여준다.

이 논문의 진짜 메시지는 "새 benchmark가 하나 더 나왔다"가 아니다. 더 정확히는 agent evaluation에서 중요한 것은 final answer correctness만이 아니라, feedback을 통해 드러난 constraint state를 누적하고, 그 상태 위에서 유효하면서도 효과적인 plan을 다시 구성하는 능력이라는 주장이다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 adaptive planning under progressively revealed dual constraints다.

여기서 dual constraints는 두 종류다.

- World constraints: 환경이 부과하는 제약이다. 예를 들어 특정 도구가 없거나, 고장났거나, 사용할 수 없는 경우다.
- User constraints: 사용자의 preference가 부과하는 제약이다. 예를 들어 fan을 쓰고 싶지 않다, heat를 내는 도구를 피하고 싶다, disposable item은 쓰지 않았으면 좋겠다는 식이다.

문제는 이런 constraint가 처음부터 모두 주어지지 않는다는 점이다. Agent는 처음에는 query만 보고 plan을 제안한다. 그 plan이 hidden constraint를 위반하면, user simulator가 그 위반에 대응하는 feedback을 준다. 이후 agent는 그 feedback을 반영해 다시 plan을 내야 한다.

따라서 AdaPlanBench의 agent는 세 가지를 동시에 해야 한다.

1. 지금까지 공개된 constraint를 기억한다.
2. 새 feedback이 어떤 world/user constraint를 의미하는지 해석한다.
3. 기존 plan을 조금 수정하는 수준을 넘어, 더 나은 alternative strategy를 찾아야 한다.

이 설정은 일반적인 planning보다 어렵다. Constraint set이 고정되어 있지 않고, agent의 proposal에 따라 다음에 공개되는 constraint가 달라지기 때문이다. 즉 benchmark가 단순 QA가 아니라 interactive process가 된다.

## 1-2. Why previous approaches are insufficient

기존 benchmark가 부족한 이유는 크게 네 가지다.

첫째, 많은 planning benchmark는 constraint를 upfront로 제공한다. 그러면 task는 "주어진 조건을 만족하는 plan을 한 번 생성하는 문제"가 된다. 하지만 실제 agent는 incomplete information에서 출발하고, interaction을 통해 조건을 알아간다.

둘째, user constraint와 world constraint를 같이 다루지 않는 경우가 많다. Tool availability만 보면 environment adaptation은 평가할 수 있지만, 사용자의 preference나 usage style을 반영하는 능력은 잘 보이지 않는다. 반대로 preference benchmark만 보면 physical feasibility나 tool availability를 놓치기 쉽다.

셋째, final answer accuracy만 보면 failure mode가 섞인다. Agent가 실패했을 때 constraint를 잊은 것인지, 새로운 alternative strategy를 찾지 못한 것인지, plan은 valid하지만 물리적으로 말이 안 되는 것인지 구분하기 어렵다.

넷째, static benchmark에서는 recency bias나 over-correction이 잘 드러나지 않는다. 실제 interaction에서는 새 feedback을 반영하다가 이전 constraint를 잊거나, 새 rubric feedback에 과하게 맞추느라 이미 만족하던 constraint를 다시 깨는 일이 생긴다. AdaPlanBench는 이 부분을 runtime protocol 안에서 직접 관찰한다.

# 2. Core Idea

## 2-1. Main contribution

AdaPlanBench의 핵심 기여는 benchmark를 세 가지 layer로 설계했다는 점이다.

1. 307개 household planning query를 만든다.
2. 각 query에 대해 world constraint와 user constraint를 자동으로 구성한다.
3. Runtime에서는 hidden constraints를 한번에 공개하지 않고, agent가 violation을 만들 때마다 feedback으로 공개한다.

이 구조 덕분에 AdaPlanBench는 단순 plan generation benchmark가 아니라, plan revision benchmark가 된다. Agent가 첫 plan을 잘 쓰는지만 보는 것이 아니라, constraint가 새로 드러날 때마다 plan을 얼마나 안정적으로 고쳐 나가는지를 본다.

## 2-2. Design intuition

이 논문의 설계 직관은 꽤 명확하다.

첫째, real-world planning은 hidden constraint discovery 문제다. 사용자가 원하는 것을 다 말하지 않았고, 환경에 없는 도구도 처음부터 catalog로 주어지지 않는다면, agent는 feedback을 통해 constraint state를 구축해야 한다.

둘째, 좋은 plan은 valid plan과 다르다. 어떤 plan은 더 이상 world/user constraint를 위반하지 않지만, 실제 task를 잘 해결하지 못할 수 있다. 그래서 AdaPlanBench는 constraint satisfaction뿐 아니라 rubric-based planning quality도 본다.

셋째, adaptive planning은 memory 문제만이 아니다. 공개된 constraint를 prompt에 다시 넣어줘도 accuracy가 크게 회복되지 않는다는 분석은 이 논문의 중요한 메시지다. Agent가 constraint를 기억하는 것과, 그 constraint 아래에서 effective alternative를 찾는 것은 다른 능력이다.

내가 보기엔 이 논문의 가장 실용적인 포인트는 여기 있다. Agent 실패를 "context를 잊었다" 하나로 설명하지 않고, constraint memory, strategy exploration, physical grounding, effectiveness를 분리해서 보게 만든다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Hidden world/user constraints가 점진적으로 공개되는 상황에서 LLM agent의 adaptive planning 능력을 평가 |
| Base data | MacGyver household planning task 기반 307개 query |
| Constraint types | World constraints와 user constraints |
| Runtime protocol | Agent proposal -> judge evaluation -> user feedback -> re-planning |
| Main metrics | Acc, VPR, Avg Turns, AWRV, AURV, ATWC, ATUC |
| Key diagnostic | Valid plan이 final success를 보장하지 않으며, constraint tracking만으로도 충분하지 않다는 점 |

## 3-2. Module breakdown

### 1) Query rewriting and filtering

AdaPlanBench는 MacGyver dataset에서 household-domain task를 가져온 뒤, query rewriting과 filtering을 수행한다. 원래 query에 명시된 tool/resource constraint는 제거하고, goal은 유지한다. 예를 들어 "available tools: ..." 같은 조건을 제거해, agent가 처음부터 resource list를 알고 있는 상황을 피한다.

그 다음 concrete household task이면서 multi-step planning이 필요한 instance만 남긴다. 이렇게 해야 이후 constraint construction에서 task solvability를 유지하면서도 hidden constraint setting을 만들 수 있다.

### 2) Constraint construction pipeline

Constraint construction은 반복적인 plan sampling으로 진행된다.

먼저 여러 planner sampler가 candidate plans를 만든다. 각 plan에서 사용된 tool과 usage pattern을 추출한다. 이 extracted tool이 world constraint와 user constraint의 원재료가 된다.

- World constraint는 tool availability나 usability를 제한한다. 예를 들어 "there is no iron at home" 같은 형태다.
- User constraint는 tool usage와 관련된 preference를 만든다. 예를 들어 "I am concerned about using tools that generate high heat" 같은 형태다.

이후 constraint를 merge, canonicalize, deduplicate하고, 마지막 validation step에서 vague constraint나 contradictory preference set을 제거한다. 예를 들어 "quiet atmosphere를 싫어한다"와 "noisy place를 싫어한다"가 같이 있으면 feasible solution space를 없앨 수 있으므로 제거한다.

이 과정을 sampling round에 따라 low, medium, high profile로 나눈다. 논문 Table 2 기준 평균 constraint 수는 다음과 같다.

| Profile | Avg world constraints | Avg user constraints |
| --- | ---: | ---: |
| Low | 9.76 | 10.91 |
| Medium | 19.61 | 21.78 |
| High | 37.73 | 41.79 |

### 3) Runtime interaction protocol

Runtime에서는 agent가 query와 지금까지의 feedback만 보고 plan을 낸다. Hidden profile 안에는 world constraint set과 user constraint set이 있지만, agent에게 처음부터 공개되지 않는다.

각 turn은 다음 순서로 진행된다.

1. Agent가 plan을 제안한다.
2. World constraint judge와 user constraint judge가 violation을 찾는다.
3. Rubric judge가 plan quality를 평가한다.
4. 위반된 constraint가 있으면 user simulator가 자연어 feedback으로 바꿔 agent에게 준다.
5. Agent는 feedback을 반영해 다음 plan을 낸다.

Trajectory는 세 가지 조건 중 하나로 종료된다.

- Valid plan found: plan이 모든 world/user constraints를 만족한다.
- Maximum turn budget reached: 최대 turn 수에 도달한다.
- Early stopping triggered: 새 constraint를 더 발견하지 못한 채 이미 공개된 constraint를 반복 위반한다.

여기서 중요한 점은 feedback이 agent의 proposal에 의해 결정된다는 것이다. Agent가 어떤 strategy를 시도하느냐에 따라 어떤 constraint가 먼저 드러나는지 달라진다. 그래서 AdaPlanBench는 passive constraint following이 아니라 active exploration 성격도 갖는다.

### 4) Rubric-based quality evaluation

AdaPlanBench는 constraint satisfaction만으로 success를 판단하지 않는다. Plan이 모든 constraint를 만족해도, 실제로 task를 해결하지 못하면 성공이라고 보기 어렵기 때문이다.

논문은 네 가지 주요 rubric dimension을 사용한다.

| Rubric dimension | What it checks |
| --- | --- |
| Tool-Use Feasibility | 사용한 tool이 household environment에서 가능한가 |
| Physical Plausibility | 그 tool usage가 물리적으로 말이 되는가 |
| Effectiveness | plan이 실제 task goal을 달성할 수 있는가 |
| Safety | 실행 시 사람에게 harm을 만들지 않는가 |

각 dimension은 1 to 5 scale로 평가되고, aggregate score가 threshold를 넘어야 한다. 원문 실험에서는 $gamma = 4$를 사용한다. 따라서 AdaPlanBench의 accuracy는 단순히 constraint를 피한 plan 비율이 아니라, constraint를 만족하면서 rubric quality까지 통과한 final plan 비율이다.

# 4. Training / Data / Recipe

## 4-1. Data

AdaPlanBench dataset은 307개 household planning query를 포함한다. Hugging Face dataset card 기준으로 각 query는 `adaplanbench_queries.json`에 들어 있으며, 각 query는 여러 environment profile을 가진다.

Dataset field는 대략 다음처럼 읽으면 된다.

| Field | Meaning |
| --- | --- |
| `query_id` | query identifier |
| `query` | agent에게 주어지는 natural-language household planning problem |
| `ban_pool` | constraint-load가 다른 environment profile list |
| `ban_pool[].tools` | unavailable or nonfunctional objects/tools |
| `ban_pool[].prefs` | user preference constraints |
| `candidate_iterative_sample_num` | constraint sampling round 또는 profile index |

Dataset card는 각 query에 6개 profile이 있고, paper main evaluation에서는 profile 1, 2, 3이 각각 low, medium, high complexity에 대응한다고 설명한다. Profile 4 to 6은 main paper experiment에는 쓰이지 않았고, community stress testing 용도로 공개되어 있다.

## 4-2. Construction and evaluation recipe

이 논문은 training paper가 아니라 benchmark/evaluation paper다. 그래서 핵심 recipe는 model training이 아니라, benchmark construction과 runtime evaluation이다.

원문 기준 construction에는 여러 LLM component가 들어간다.

- Query rewriter: raw MacGyver query에서 explicit resource constraint를 제거한다.
- Binary filter: concrete household task와 multi-step planning task만 남긴다.
- Planner samplers: 다양한 candidate plan을 만든다.
- Constraint extractor: plan에서 tool과 usage pattern을 뽑는다.
- Merge model: constraint를 canonicalize하고 deduplicate한다.
- Constraint checker: vague, invalid, contradictory constraint를 제거한다.

Appendix의 model choice 설명에 따르면 planner sampler에는 GPT-4.1, DeepSeek-V3.2, Qwen3.6-Flash가 사용되고, invalid constraint filtering에는 GPT-5.4가 사용된다. Evaluation에서는 world/user constraint judge를 GPT-5.4로 두고, rubric judge는 GPT-4.1, DeepSeek-V3.2, Qwen3.6-Flash를 independent judges로 사용한다. 이 model version 표기는 원문 기준이므로, 실제 재현 시점에는 API naming과 availability를 다시 확인해야 한다.

## 4-3. Engineering notes

GitHub repository를 보면 benchmark 실행과 분석을 어느 정도 모듈화해 두었다.

- `python -m env.run` 형태로 benchmark를 실행한다.
- `ban_iterative_sample_num`을 바꿔 low, medium, high constraint load를 선택한다.
- Runtime output은 `results/runs/` 아래에 저장된다.
- Run folder에는 `metadata.json`, `summary.json`, `index.jsonl`, `queries/*.json` 형태의 결과가 남는다.
- Standalone evaluator로 main table을 재생성할 수 있다.

실무적으로는 이 구성이 중요하다. AdaPlanBench는 leaderboard score만 보는 benchmark라기보다, query별 trajectory를 열어 failure를 디버깅하는 benchmark에 가깝다. Repeated violation, triggered constraint, turn count, final rubric score를 같이 보면, model이 constraint memory에서 깨지는지, strategy exploration에서 깨지는지, physical plausibility에서 깨지는지 분리해 볼 수 있다.

Judge reliability도 원문에서 따로 확인한다. Filter model은 30개 sampled evaluation instances에 대해 human annotation과 비교했고, runtime judge는 30개 sampled trajectories에서 166 turn-level instances를 annotation했다. 원문은 runtime judge가 human majority label과 89.76% exact match를 보였고, 166개 중 161개 turn에서 constraint count 차이가 at most one이었다고 보고한다. Benchmark가 LLM judge에 의존한다는 점을 감안하면, 이 validation은 중요한 sanity check다.

# 5. Evaluation

## 5-1. Main results

Main evaluation은 medium constraint profile에서 10개 model을 비교한다. 핵심 결과는 다음과 같다.

| Model | Acc | VPR | Avg Turns | AWRV | AURV | ATWC | ATUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 14.38 | 82.35 | 4.493 | 0.242 | 0.614 | 0.608 | 1.888 |
| Qwen3-14B | 17.26 | 73.62 | 4.785 | 0.296 | 0.821 | 0.668 | 2.042 |
| Qwen3-32B | 17.92 | 80.13 | 5.010 | 0.150 | 0.645 | 0.609 | 2.082 |
| Llama-3.3-70B-Instruct | 29.32 | 83.71 | 4.619 | 0.114 | 0.537 | 0.668 | 1.830 |
| DeepSeek-v4-Flash | 35.53 | 76.97 | 6.385 | 0.464 | 0.895 | 0.977 | 2.657 |
| Gemini-3-Flash | 43.32 | 90.23 | 5.824 | 0.065 | 0.391 | 0.756 | 2.442 |
| Gemini-3.1-Pro | 34.53 | 91.21 | 5.651 | 0.124 | 0.251 | 0.769 | 2.236 |
| GPT-5 | 67.75 | 89.58 | 6.212 | 0.199 | 0.195 | 1.191 | 3.269 |
| GPT-5-Mini | 61.89 | 85.34 | 5.886 | 0.322 | 0.322 | 1.318 | 3.391 |
| GPT-5-Nano | 42.35 | 67.75 | 5.541 | 0.971 | 0.355 | 1.089 | 2.468 |

이 표에서 가장 중요한 숫자는 GPT-5의 67.75% accuracy다. 가장 강한 model도 medium constraint profile에서 70%를 넘지 못한다. Open-weight model은 대체로 30% 이하에 머문다.

하지만 더 흥미로운 것은 Acc와 VPR의 차이다. VPR은 constraint-satisfying plan으로 종료되는 비율이다. Gemini-3.1-Pro와 Gemini-3-Flash는 VPR이 90%를 넘지만, accuracy는 각각 34.53%, 43.32%다. 즉 constraint를 만족하는 plan을 찾는 것과, 실제로 quality threshold를 통과하는 plan을 찾는 것은 다르다.

## 5-2. What really matters in the experiments

### 1) Valid plan과 good plan은 다르다

AdaPlanBench에서 VPR이 높다는 것은 world/user constraints를 피하는 데 성공했다는 뜻이다. 하지만 accuracy는 모든 constraint를 만족하면서 rubric threshold도 넘는 final plan만 성공으로 본다.

이 차이가 큰 model은 "제약을 피하는 법"은 어느 정도 배웠지만, "효과적인 해결책을 만드는 법"에는 실패한다. 실제 agent service에서도 이 차이는 중요하다. 사용자가 싫어하는 것을 피하기만 하고 task 자체를 잘 해결하지 못하면, agent는 polite하지만 useless한 plan을 낼 수 있다.

### 2) Constraint가 많아질수록 quality가 무너진다

논문은 low, medium, high profile로 constraint burden을 올려가며 성능을 본다. Figure 2의 핵심 메시지는 단순하다. Constraint load가 커질수록 accuracy와 VPR이 모두 떨어진다.

이는 model이 단일 constraint following에는 강해 보여도, constraint set이 커질수록 feasible strategy space를 다시 탐색하는 데 취약하다는 뜻이다. 특히 household task처럼 action space가 open-ended인 경우, 하나의 tool이 막히면 단순 substitution이 아니라 strategy 자체를 바꿔야 할 수 있다.

### 3) Constraint tracking만으로는 부족하다

논문은 공개된 constraint를 매 turn 명시적으로 다시 넣어주는 intervention을 한다. 결과적으로 VPR은 5% to 15% 정도 좋아지지만, accuracy 개선은 4개 model 중 3개에서 3% 미만이다.

이 결과는 중요하다. 많은 agent failure를 "memory block을 더 잘 넣으면 해결된다"고 보기 쉽지만, AdaPlanBench에서는 memory 보강이 final task success를 크게 회복하지 못한다. 문제는 constraint를 기억하지 못하는 것뿐 아니라, 그 constraint 아래에서 새로운 effective plan을 구성하지 못하는 데 있다.

### 4) Rubric feedback은 local fix를 만들지만 global consistency를 깨뜨릴 수 있다

Failed query에 대해 rubric feedback을 추가로 주고 1 to 6 refinement turns를 허용하면 accuracy는 around 10% 개선된다. 하지만 VPR은 크게 떨어진다. 원문은 open-source model에서는 약 40%, proprietary model에서는 약 20% VPR drop을 보고한다.

이 현상은 recency-biased adaptation으로 해석할 수 있다. Model은 새로 받은 rubric feedback을 고치려다 이전에 만족하던 world/user constraint를 다시 위반한다. 즉 feedback이 많아진다고 항상 좋아지는 것이 아니다. Adaptive agent에는 feedback integration policy가 필요하다.

### 5) User constraint가 특히 어렵다

논문은 world-only, user-only, both constraints setting을 비교한다. 결과적으로 user-only가 world-only보다 어렵고, both가 가장 어렵다.

직관적으로도 user constraint는 까다롭다. "no fan" 같은 world constraint는 특정 object 하나를 금지하는 형태가 많다. 반면 user preference는 tool attribute나 usage style 전체를 금지할 수 있다. 예를 들어 heat를 싫어한다는 preference는 iron, hair dryer, oven, hot water 같은 여러 strategy를 한 번에 막을 수 있다. Agent 입장에서는 feasible action space가 훨씬 크게 줄어든다.

# 6. Limitations

1. Domain coverage가 household planning에 묶여 있다.
   - AdaPlanBench는 MacGyver 기반 household task로 시작한다. 이 설정은 physical plausibility와 tool substitution을 보기 좋지만, web browsing, coding, enterprise workflow, robotics execution까지 바로 일반화되지는 않는다.

2. Constraint distribution이 construction pipeline에 의존한다.
   - World/user constraints는 LLM-based planner, extractor, checker를 거쳐 생성된다. 따라서 어떤 planner sampler를 쓰는지, user preference를 어떤 방식으로 유도하는지에 따라 benchmark difficulty와 failure mode가 달라질 수 있다.

3. LLM judge dependency가 남아 있다.
   - 원문은 human validation을 제공하지만, runtime constraint checking과 rubric scoring은 여전히 LLM judge에 의존한다. 특히 physical plausibility와 effectiveness는 judge model의 commonsense와 domain prior에 영향을 받을 수 있다.

4. Text plan evaluation이지 execution benchmark는 아니다.
   - AdaPlanBench는 plan을 실행해서 실제 state transition을 검증하는 benchmark가 아니다. 따라서 real tool execution, latency, observation noise, partial failure recovery 같은 deployment issue는 별도 평가가 필요하다.

5. Constraint disclosure protocol이 실제 user behavior를 단순화한다.
   - Benchmark에서는 violation이 생기면 corresponding constraint가 feedback으로 공개된다. 실제 user는 더 모호하게 말하거나, 잘못된 feedback을 주거나, 여러 preference를 한꺼번에 말할 수 있다.

6. Safe deployment 관점에서는 "violate to discover"가 위험할 수 있다.
   - AdaPlanBench의 interaction은 agent가 hidden constraint를 위반하면서 constraint를 알아가는 구조다. Benchmark로는 유용하지만, 의료/금융/보안 같은 domain에서는 violation 자체가 unacceptable할 수 있다. 이런 domain에서는 proactive clarification이나 constraint elicitation이 더 중요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 agent evaluation을 만들 때 어떤 metric을 넣어야 하는지에 대해 꽤 좋은 힌트를 준다.

실제 workflow agent를 만들면 실패 원인이 보통 하나로 떨어지지 않는다. Model이 이전 instruction을 잊었는지, 사용자의 preference를 잘못 일반화했는지, tool availability를 잘못 추정했는지, 아니면 모든 constraint를 만족했지만 output이 useless했는지 구분해야 한다.

AdaPlanBench의 Acc/VPR gap은 이 구분을 잘 보여준다. VPR만 높으면 "constraints를 지키는 agent"처럼 보일 수 있지만, accuracy까지 같이 보면 "constraints를 지키면서 효과적인 plan을 만드는 agent"인지 확인할 수 있다.

## 7-2. Reuse potential

이 논문의 구조는 내부 agent benchmark를 만들 때 재사용하기 좋다.

1. Internal workflow benchmark
   - 실제 업무 query를 가져오고, world constraints를 system/resource limitation으로, user constraints를 stakeholder preference로 바꿀 수 있다.

2. Constraint memory diagnosis
   - 공개된 constraints를 매 turn 다시 넣어주는 memory intervention을 만들면, failure가 memory 문제인지 planning problem인지 분리할 수 있다.

3. Feedback robustness test
   - Rubric feedback을 추가했을 때 이전 constraint를 깨뜨리는지 보면, agent가 local feedback에 overfit되는지 확인할 수 있다.

4. Preference-heavy agent evaluation
   - 고객 support, recommender, procurement, document workflow처럼 preference가 많은 task에서는 user constraints를 별도 axis로 분리하는 것이 중요하다.

5. Plan quality vs constraint validity separation
   - 실무 dashboard에서는 final score 하나보다 Acc, VPR, repeated violations, effectiveness score를 나눠 보는 편이 좋다.

## 7-3. Follow-up papers

- MacGyver: 원본 household problem formulation을 이해하기 위한 배경.
- TravelPlanner: real-world travel planning에서 constraint-heavy planning을 보는 benchmark.
- FlowBench: workflow-guided planning benchmark와 비교하기 좋음.
- UserBench: user constraints와 preference following 측면에서 같이 읽을 만함.
- ADAPT: unspecified affordance constraints 아래 commonsense planning을 다루는 benchmark.

# 8. Summary

- AdaPlanBench는 hidden world/user constraints가 점진적으로 공개되는 dynamic planning benchmark다.
- 307개 household planning query를 기반으로, low/mid/high constraint profile을 구성한다.
- Main evaluation에서 best model도 medium profile 기준 67.75% accuracy에 머문다.
- VPR이 높아도 accuracy가 낮을 수 있어, valid plan과 good plan을 구분해야 한다.
- Constraint tracking과 rubric feedback만으로는 adaptive planning 문제가 충분히 해결되지 않는다.
- User constraint는 world constraint보다 더 넓은 action space restriction을 만들 수 있어 특히 어렵다.

## Verification Notes

- 원문 재확인 필요: 현재 글은 arXiv v1 preprint 기준으로 작성했다. 추후 camera-ready 또는 revised version이 나오면 model list, judge setup, metric definition을 다시 확인해야 한다.
- 수치 검증 필요: Table 2의 low/mid/high 평균 constraint 수, Table 3의 medium profile main results, Figure 4/5의 intervention 효과는 최종 게시 전 원문 PDF와 한번 더 대조하는 것이 좋다.
- figure/table 확인 필요: Figure 1 pipeline, Figure 2 constraint burden, Figure 5 rubric refinement, Figure 6 world/user/both ablation은 블로그에 그림을 넣을 경우 caption과 setting을 함께 재확인해야 한다.
- artifact 확인 필요: Hugging Face dataset viewer는 현재 split feature extraction error를 표시하지만, dataset card는 `field="data"`로 load하는 방법을 안내한다. 실제 실행 전 local loading을 확인하는 것이 좋다.
- model version 확인 필요: GPT-5, GPT-5-Mini, GPT-5-Nano, GPT-5.4, Gemini-3.1-Pro 등 model name은 원문 표기를 그대로 따른 것이다. 공개 API 이름이나 접근 가능성은 게시 시점에 별도로 확인해야 한다.
