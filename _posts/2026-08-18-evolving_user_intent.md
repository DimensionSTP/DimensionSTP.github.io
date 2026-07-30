---
layout: single
title: "LLMs Get Lost in Evolving User Intent Review"
categories: Study-concept
tag: [LLM-Agent, User-Intent, Evaluation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2607.20734)

[Code link](https://github.com/microsoft/evolving-intent)

> 한 줄 요약: 이 논문은 static single-turn benchmark를 argument reveal, argument revision, function switch가 누적되는 multi-turn conversation으로 바꾸면서 final task의 native verifier를 그대로 유지하는 framework를 제안하고, strong LLM agent도 evolving user intent 아래에서는 single-turn 성능을 안정적으로 보존하지 못함을 보여준다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Multi-turn agent evaluation을 vague conversation quality가 아니라 verifier-backed task completion으로 만든다.
- User intent evolution을 reveal, revision, switch의 세 transition으로 분해해 failure source를 통제한다.
- GSM8K, BIRD-SQL, BrowseComp+, SWE-Bench Verified를 같은 framework에서 비교한다.
- Strong static performance가 evolving-intent robustness를 보장하지 않으며, function switch가 특히 어려운 이유를 보여준다.
- Prompt recap이나 oracle recap으로도 single-turn ceiling을 완전히 회복하지 못해, 단순 memory 부족보다 깊은 state-update 문제가 있음을 시사한다.

실제 user-agent collaboration에서 user intent는 처음부터 완성되어 있지 않다. 사용자는 필요한 constraint를 나중에 떠올리고, 이미 말한 값을 수정하고, 중간에 related task로 pivot한다. Coding agent에게 feature를 요청했다가 bug fix를 먼저 부탁할 수 있고, SQL query의 date range를 바꾸거나, search 목표를 다른 entity로 전환할 수도 있다.

그런데 많은 benchmark는 final intent를 한 번에 완전히 제공한다. Model은 clean prompt를 읽고 answer를 만든다. 이 setting은 task capability를 측정하기에는 좋지만, long-lived interaction에서 old instruction과 current intent를 분리하는 능력은 거의 보지 않는다.

Multi-turn benchmark를 직접 만들면 다른 문제가 생긴다.

- Per-turn intent annotation 비용이 크다.
- User simulator의 realism을 평가하기 어렵다.
- Final answer를 LLM-as-a-judge로 채점하게 되기 쉽다.
- Static benchmark와 난도가 달라져 direct comparison이 어렵다.

이 논문은 final turn을 기존 benchmark의 original instance에 고정하고, 그 이전 intent를 backward synthesis한다. 따라서 마지막 answer는 original verifier로 그대로 채점할 수 있다.

핵심 질문은 다음과 같다.

> Model이 final task 자체를 풀 수 있을 때도, conversation 중간에 누적된 stale intent와 transition을 올바르게 정리하지 못해 실패하는가?

실험의 답은 그렇다. 특히 function switch와 transition composition이 들어가면 model은 이전 task의 trace를 버리지 못하거나, 새 task를 old state 위에 잘못 이어 붙이거나, tool budget을 exploration에 소진한다.

# 1. Problem Setting

## 1-1. Problem definition

논문은 turn $t$에서 user intent를 structured state로 표현한다.

$$
I_t = (f_t, C_t, C_t^{\mathrm{rev}}, y_t)
$$

각 component의 의미는 다음과 같다.

| Symbol | Meaning |
| --- | --- |
| $f_t$ | 현재 user가 요청하는 function or task |
| $C_t$ | 현재 function에 필요한 full argument set |
| $C_t^{\mathrm{rev}}$ | 현재 turn까지 user가 reveal한 argument subset |
| $y_t$ | 현재 intent에 대응하는 answer or target |

Conversation에서 intent는 세 transition으로 변한다.

### 1) Argument reveal

User가 아직 말하지 않았던 argument를 추가로 공개한다.

$$
C_t^{\mathrm{rev}} \supset C_{t-1}^{\mathrm{rev}}
$$

Task function과 기존 argument value는 유지된다. Model은 incomplete specification을 점진적으로 완성해야 한다.

### 2) Argument revision

User가 이전 argument value를 새 value로 바꾼다.

$$
c_i^{(t)} \neq c_i^{(t-1)}
$$

Old value는 conversation에 남아 있지만 current answer에는 new value만 사용해야 한다.

### 3) Function switch

User가 current task를 related but different function으로 바꾼다.

$$
f_t \neq f_{t-1}
$$

일부 argument는 재사용될 수 있지만 output contract와 required reasoning이 달라진다. 이 transition이 가장 큰 failure를 만든다.

Evaluation input은 multi-turn conversation이고 output은 final intent에 대한 action or answer다. Final intent는 source benchmark의 original task와 동일하므로 native verifier를 사용한다.

## 1-2. Why previous approaches are insufficient

### 1) Single-turn benchmark는 intent tracking을 task solving과 섞지 않는다

Static setting에서 model이 실패하면 task knowledge나 reasoning capability가 부족하다고 볼 수 있다. 그러나 model이 static task를 잘 풀어도 multi-turn에서 실패하면 다음 component가 원인일 수 있다.

- Old argument suppression
- Current state reconstruction
- Function boundary detection
- Relevant history selection
- Tool trajectory reset
- Output contract replacement

기존 benchmark는 이 capability를 직접 분리하지 않는다.

### 2) Under-specification만으로는 real intent evolution을 충분히 표현하지 못한다

User가 정보를 조금씩 주는 benchmark는 이미 존재한다. 하지만 실제 collaboration에서는 정보가 추가될 뿐 아니라 바뀌고, task 자체도 전환된다.

Argument reveal은 monotonic update다. 반면 revision과 switch는 non-monotonic update다. Model은 old state를 보존하는 것이 아니라 invalidate해야 한다.

### 3) LLM-as-a-judge 기반 multi-turn evaluation은 diagnosis가 어렵다

Open-ended conversation을 judge model로 평가하면 style, helpfulness, partial completion이 score에 섞인다. Intent tracking failure인지 judge preference인지 구분하기 어렵다.

이 논문은 final native verifier를 유지함으로써 answer correctness를 source benchmark와 같은 기준으로 비교한다.

### 4) Memory summary가 current intent state를 완전히 해결하지 못한다

Conversation summary를 추가하면 old context를 압축할 수 있다. 그러나 summary가 current objective를 정확히 알려줘도 model이 tool trajectory나 output plan을 제대로 reset하지 못할 수 있다.

논문에서 oracle recap도 single-turn performance를 완전히 회복하지 못한다. 이는 failure가 단순 recall 부족보다 넓다는 evidence다.

# 2. Core Idea

## 2-1. Main contribution

논문의 contribution은 세 부분으로 나뉜다.

### 1) Static-to-dynamic benchmark transformation

Existing verifiable instance $(q,y^*)$에서 final anchor intent를 추출한다.

$$
I^* = (f^*, C^*, y^*)
$$

이 final state는 source task와 동일하다. Framework는 여기서 backward하게 earlier intent를 만든다.

- 일부 argument를 숨겨 reveal event를 만든다.
- Source argument와 다른 counterfactual value를 만들어 revision event를 만든다.
- Shared argument를 가진 predecessor function을 만들어 switch event를 만든다.

Final turn에서 모든 transition은 source intent로 수렴한다. 따라서 original answer $y^*$와 verifier가 유효하다.

### 2) Controlled intent transition taxonomy

Reveal, revision, switch를 개별 또는 조합해 transition complexity를 조절한다.

이 taxonomy는 단순히 conversation length를 늘리는 것보다 중요하다. 같은 turn 수라도 monotonic reveal과 task switch는 cognitive demand가 다르다.

### 3) Cross-domain agent evaluation

논문은 네 benchmark를 사용한다.

| Domain | Benchmark | Evaluation subset |
| --- | --- | ---: |
| Math | GSM8K | 200 |
| Text-to-SQL | BIRD-SQL | 100 |
| Agentic search | BrowseComp+ | 100 |
| Software engineering | SWE-Bench Verified | 50 |

각 domain은 다른 final verifier를 가진다.

- GSM8K: Numeric answer verification
- BIRD-SQL: SQL execution or answer correctness
- BrowseComp+: Answer matching under search task protocol
- SWE-Bench Verified: Repository test execution

### 4) Failure diagnosis beyond final accuracy

논문은 transition type, transition count, source difficulty, recap intervention, tool use를 분석한다.

특히 SWE-Bench에서 model이 execution보다 exploration에 tool call을 소진하는 behavior를 보여, evolving intent가 단순 answer memory가 아니라 action trajectory management를 망가뜨릴 수 있음을 보인다.

## 2-2. Design intuition

핵심 design은 backward construction이다.

Forward simulation으로 realistic user conversation을 만들면 final answer를 새로 annotation해야 한다. Backward construction은 known verifiable final state에서 시작한다.

1. Source task에서 final function과 arguments를 추출한다.
2. Final argument의 counterfactual predecessor를 만든다.
3. Final function과 일부 argument를 공유하는 predecessor function을 만든다.
4. Transition event를 schedule한다.
5. 각 turn에서는 full state가 아니라 state delta만 natural language로 render한다.
6. 마지막 turn은 original task state와 동일하게 만든다.

이 design의 장점은 evaluation invariance다.

$$
V_{\mathrm{dynamic}}(y) = V_{\mathrm{source}}(y)
$$

Final verifier $V$를 바꾸지 않으므로 static and dynamic performance gap을 intent evolution cost로 해석하기 쉬워진다.

또 하나의 intuition은 user utterance에 current state 전체를 매번 반복하지 않는 것이다. Renderer는 $\Delta I_t$만 표현한다. 실제 user도 매 turn full requirement를 다시 말하지 않기 때문이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Static verifiable task를 evolving-intent multi-turn evaluation으로 변환 |
| Anchor | Original benchmark instance and native verifier |
| Intent state | Function, arguments, revealed subset, answer |
| Transition types | Argument reveal, argument revision, function switch |
| Construction direction | Final anchor에서 predecessor intent를 backward synthesis |
| Scheduler | Transition order와 turn placement를 consistency rule로 결정 |
| Renderer | Intent delta만 natural language user utterance로 변환 |
| Evaluation | Final turn output을 source verifier로 채점 |
| Main analysis | Transition type, count, composition, difficulty, recap, tool behavior |

## 3-2. Module breakdown

### 1) Intent extraction

Source problem과 answer에서 function $f^*$와 argument set $C^*$를 추출한다.

예를 들어 math problem은 다음처럼 분해할 수 있다.

- Function: Total cost 계산
- Arguments: Unit price, quantity, discount
- Answer: Final numeric value

SQL task는 다음처럼 분해할 수 있다.

- Function: 특정 condition을 만족하는 row count query
- Arguments: Table, date range, category, grouping rule
- Answer: Executable SQL and result

논문은 이 extraction과 synthetic predecessor generation에 GPT-5.1을 사용한다.

### 2) Counterfactual argument generation

Revision을 만들기 위해 source value $c_i^*$와 다른 counterfactual value $c_i^{\mathrm{cf}}$를 생성한다.

Conversation early turn에서는 $c_i^{\mathrm{cf}}$가 current value다. Later turn에서 user가 이를 $c_i^*$로 수정한다.

좋은 counterfactual은 다음 조건을 가져야 한다.

- Type이 source argument와 맞는다.
- Task가 여전히 solvable하다.
- Revision이 semantically plausible하다.
- Final source answer를 accidental하게 leak하지 않는다.

### 3) Predecessor function generation

Function switch를 만들기 위해 final function $f^*$ 이전에 수행할 related function $f^{\mathrm{pre}}$를 생성한다.

두 function은 일부 argument를 공유한다.

$$
C^{\mathrm{pre}} \cap C^* \neq \emptyset
$$

Shared argument는 conversation continuity를 만들지만, model에게 혼동도 준다. Old task에서 사용하던 entity, file, table, repository를 new task에서도 재사용하면서 goal만 바뀔 수 있기 때문이다.

Predecessor generation은 recursive하게 적용할 수 있어 multiple function switch를 만든다.

### 4) Turn scheduler

Scheduler는 transition event를 turn에 배치한다. 다음 consistency를 유지해야 한다.

- Final turn은 full source intent와 일치한다.
- Reveal되지 않은 argument를 premature하게 사용하지 않는다.
- Function switch는 current function이 충분히 specified된 뒤 일어난다.
- Revision order가 logical conflict를 만들지 않는다.
- Required number of reveal, revision, switch를 만족한다.

Main setting은 각 transition type을 두 번씩 포함해 총 6 transitions를 만든다. Initial turn을 포함하면 7-turn interaction이다.

### 5) Delta renderer

Renderer는 current full state를 그대로 말하지 않고 update만 utterance로 만든다.

- Reveal: "추가로 지역은 Seattle입니다."
- Revision: "아까 말한 2024년이 아니라 2025년으로 바꿔주세요."
- Switch: "그 query는 잠시 두고, 같은 table에서 monthly average를 구해주세요."

실제 paper의 utterance는 dataset and domain에 맞게 생성된다. 위 예시는 mechanism 설명을 위한 illustration이다.

### 6) Agent evaluation wrapper

각 benchmark마다 original environment를 유지한다.

- Math model은 conversation을 읽고 final answer를 낸다.
- SQL model은 database schema와 conversation을 읽고 query를 만든다.
- Search agent는 browsing trajectory를 수행한다.
- SWE agent는 repository tool을 사용해 patch를 만든다.

이 방식으로 reasoning-only task와 tool-using task를 같은 intent transition framework에서 비교한다.

# 4. Training / Data / Recipe

이 논문은 foundation model을 training하지 않는다. Data construction과 evaluation recipe가 핵심이다.

## 4-1. Source datasets

| Dataset | Samples | Why useful |
| --- | ---: | --- |
| GSM8K | 200 | Structured arguments and exact numeric verifier |
| BIRD-SQL | 100 | Schema-grounded function and argument updates |
| BrowseComp+ | 100 | Long search trajectory and evidence gathering |
| SWE-Bench Verified | 50 | Repository-level tool use and executable verification |

Sample 수는 evaluation budget과 tool cost를 고려한 subset이다. Full benchmark performance와 직접 비교할 때는 subset selection을 확인해야 한다.

## 4-2. Construction models

Intent extraction, counterfactual value generation, predecessor function generation에 GPT-5.1을 사용한다.

이 choice는 scalable construction을 가능하게 하지만 generator bias를 만든다.

- Function decomposition quality가 GPT-5.1에 의존한다.
- Counterfactual style이 uniform할 수 있다.
- Generated predecessor가 실제 user pivot보다 clean할 수 있다.

Code release는 construction stage와 evaluation runner를 분리한다.

- `intent_construction/`
- `situated_simulation/`
- `evaluation/`

## 4-3. Main scenario

Main evolving-intent scenario는 다음 event를 포함한다.

- 2 argument reveals
- 2 argument revisions
- 2 function switches

Total 6 transitions이며 initial turn을 합치면 7 turns다.

이 composition은 one-type stress test보다 훨씬 어렵다. Reveal된 argument가 later revision되고, task가 switch된 뒤 shared argument가 다른 function에 들어갈 수 있기 때문이다.

## 4-4. Evaluated models

논문은 frontier API model과 open model을 비교한다.

- GPT-5.1
- GPT-5.2
- GPT-5.5
- Gemini 3.1 Pro
- Grok 4.20
- Kimi K2.5
- Kimi K2.6
- Mistral Large 3
- DeepSeek V3.2

평가 결과는 논문이 고정한 model snapshot과 configuration에 한정되므로, model name만으로 다른 version의 결과까지 일반화해서는 안 된다.

## 4-5. Recap interventions

논문은 memory intervention을 두 종류로 비교한다.

1. Prompt recap
   - Model이 conversation을 보고 current intent를 summary하게 한다.

2. Oracle recap
   - Ground-truth current intent를 explicit recap으로 제공한다.

Oracle recap은 memory extraction error를 제거한다. 그래도 static ceiling을 회복하지 못하면 model이 correct state를 받아도 execution plan이나 attention control에서 실패한다는 뜻이다.

# 5. Evaluation

## 5-1. Main results

### 1) Static performance가 dynamic setting에서 일관되게 하락한다

대표 결과는 다음과 같다.

| Model | Benchmark | Single-turn | Evolving intent | Drop |
| --- | --- | ---: | ---: | ---: |
| GPT-5.1 | GSM8K | 98.0 | 82.0 | -16.0 pp |
| GPT-5.1 | BIRD-SQL | 72.0 | 66.0 | -6.0 pp |
| GPT-5.1 | BrowseComp+ | 49.0 | 34.0 | -15.0 pp |
| GPT-5.1 | SWE-Bench Verified | 72.0 | 0.0 | -72.0 pp |
| GPT-5.5 | GSM8K | 99.0 | 80.5 | -18.5 pp |
| GPT-5.5 | BIRD-SQL | 80.0 | 71.0 | -9.0 pp |
| GPT-5.5 | BrowseComp+ | 65.0 | 57.0 | -8.0 pp |
| GPT-5.5 | SWE-Bench Verified | 86.0 | 80.0 | -6.0 pp |
| Gemini 3.1 Pro | GSM8K | 98.0 | 82.0 | -16.0 pp |
| Gemini 3.1 Pro | BIRD-SQL | 75.0 | 72.0 | -3.0 pp |
| Gemini 3.1 Pro | BrowseComp+ | 51.0 | 40.0 | -11.0 pp |
| Gemini 3.1 Pro | SWE-Bench Verified | 86.0 | 84.0 | -2.0 pp |

Absolute drop는 model and task에 따라 크게 다르다. 중요한 공통점은 strong static model도 dynamic setting에서 ceiling을 그대로 유지하지 못한다는 것이다.

### 2) Function switch가 가장 어렵다

Argument reveal은 new information을 추가하는 monotonic operation이다. Revision은 old value를 replace해야 한다. Function switch는 더 큰 state transition이다.

- Goal이 바뀐다.
- Output schema가 바뀔 수 있다.
- Old plan을 중단해야 한다.
- Shared argument만 선택적으로 carry over해야 한다.
- Tool trajectory를 reset or re-plan해야 한다.

논문은 function switch가 reveal or revision보다 큰 drop을 만드는 경향을 보여준다.

### 3) Transition count가 늘수록 accuracy가 내려간다

같은 transition type이라도 occurrence가 하나에서 둘로 늘면 performance가 더 떨어진다. Multiple switch는 특히 어렵다.

이는 model이 last utterance만 읽는 문제가 아님을 뜻한다. Conversation 전체에서 current intent를 reconstruct하고 stale state를 제거해야 한다.

### 4) Transition composition이 compound failure를 만든다

Reveal, revision, switch를 조합하면 개별 transition보다 어려워진다.

예를 들어 early turn에서 reveal된 argument가 later revision되고, function switch 이후에도 같은 entity가 남아 있으면 model은 다음을 구분해야 한다.

- 현재 유효한 value
- 폐기된 value
- old function에만 relevant한 constraint
- new function에도 carry over되는 argument

### 5) SWE agent는 tool budget을 잘못 사용한다

SWE-Bench analysis에서 일부 model은 evolving intent 아래에서 execution tool보다 exploration에 대부분의 tool call을 사용한다. 100 tool calls 중 execution-related call이 4개 미만인 pattern이 보고된다.

Accumulated repository trace와 old objective가 model을 distract하고, final function에 맞는 patch execution으로 수렴하지 못한다.

GPT-5.1과 Grok 4.20 같은 model은 strong single-turn score를 가지면서도 evolving-intent SWE setting에서 0%까지 떨어지는 경우가 있다. 이는 capability absence보다 trajectory control failure에 가깝다.

### 6) Recap은 부분적으로 돕지만 gap을 닫지 못한다

Prompt recap과 oracle recap은 accuracy를 개선한다. 하지만 oracle current intent를 직접 줘도 single-turn score에 완전히 도달하지 못한다.

예를 들어 GPT-5.5의 BIRD-SQL function-switch setting에서 oracle recap은 65에서 75로 올리지만 single-turn 80에는 미치지 못한다.

이 결과는 intent extraction, state update, action execution이 분리된 bottleneck임을 보여준다.

## 5-2. What really matters in the experiments

### 1) Relative gap가 main metric이다

Benchmark별 difficulty와 verifier가 다르므로 raw score보다 같은 model의 static-to-dynamic drop이 중요하다.

$$
\Delta_{\mathrm{intent}} = A_{\mathrm{static}} - A_{\mathrm{dynamic}}
$$

이 gap은 base task capability를 통제한 evolving-intent cost다.

### 2) Function switch는 memory보다 belief-state replacement 문제다

Old information을 기억하지 못해서만 실패한다면 oracle recap이 거의 해결해야 한다. 하지만 gap이 남는다.

Model은 correct current state를 알아도 old plan의 inertia, tool trace, intermediate artifact를 버리지 못할 수 있다.

### 3) Easy source task와 hard source task가 다르게 compound된다

Base task가 어려울수록 intent transition이 추가 reasoning burden을 만든다. 그러나 쉬운 task에서도 switch가 들어가면 noticeable drop이 생긴다.

따라서 evolving-intent robustness를 base capability가 충분히 높아진 뒤 자연스럽게 생길 emergent property로 보기 어렵다.

### 4) Verifier preservation이 paper의 strongest design이다

User simulation realism에는 한계가 있지만, final verifier를 유지하기 때문에 dynamic degradation을 judge noise 없이 측정할 수 있다.

# 6. Limitations

1. User utterance가 synthetic and controlled하다.
   - Persona, typo, grammar variation, emotional context, indirect request가 제한된다.
   - Real user의 ambiguous correction은 더 어려울 수 있다.

2. 한 turn에 하나의 transition을 배치한다.
   - 실제 user는 한 문장에서 argument를 바꾸고 task를 switch할 수 있다.
   - Multiple simultaneous update는 별도 평가가 필요하다.

3. Final turn만 source verifier로 정확히 평가한다.
   - Intermediate turn에서 model이 당시 intent를 제대로 수행했는지는 완전히 검증하지 않는다.
   - Final success가 earlier user satisfaction을 대표하지 않을 수 있다.

4. Construction model bias가 있다.
   - GPT-5.1이 function, argument, counterfactual, predecessor를 만든다.
   - Decomposition error가 benchmark difficulty에 섞일 수 있다.

5. Function abstraction이 domain마다 다르다.
   - GSM8K의 function과 SWE repository task의 function은 complexity가 크게 다르다.
   - 동일한 transition count가 동일한 cognitive load를 뜻하지 않는다.

6. API model version reproducibility가 제한적이다.
   - Frontier model의 system behavior, tool policy, context management는 update될 수 있다.

7. Context length와 conversation length가 아직 제한적이다.
   - 7-turn setting은 controlled analysis에 좋지만 months-long collaboration을 대표하지 않는다.

8. Mitigation study가 recap 중심이다.
   - Explicit state store, typed intent graph, planner reset, stale-artifact invalidation 같은 system intervention은 더 탐색할 수 있다.

9. Static final task와 dynamic history 사이의 natural causality가 약할 수 있다.
   - Backward-generated predecessor가 logically valid해도 실제 user journey frequency를 반영하지 않을 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 agent memory 문제를 "더 많이 기억하기"가 아니라 "현재 유효한 state만 다시 bind하기"로 바꾼다는 점이 중요하다.

Long-horizon agent에서 full transcript를 보존해도 current intent를 잘 따르는 것은 아니다. 오히려 stale instruction, old plan, intermediate tool output이 모두 context에 남아 failure를 만든다.

Document AI나 enterprise RAG workflow에서도 비슷하다.

- User가 extraction schema를 수정한다.
- 대상 document pair가 바뀐다.
- Earlier condition을 철회한다.
- Evidence selection task가 answer generation task로 전환된다.
- 같은 entity를 유지하면서 output format만 바꾼다.

이때 agent에게 필요한 것은 raw memory가 아니라 versioned intent state와 invalidation rule이다.

## 7-2. Reuse potential

### 1) Explicit intent ledger

각 turn에서 다음 구조를 유지한다.

| Field | Example role |
| --- | --- |
| Current function | 지금 수행해야 할 task |
| Active arguments | Current value only |
| Superseded values | Revision history, inactive |
| Carry-over arguments | Function switch 후 유지할 field |
| Invalidated plan | 더 이상 실행하면 안 되는 action |
| Output contract | Current response or artifact schema |

### 2) Transition-aware planner reset

- Reveal: Current plan을 extend한다.
- Revision: Affected computation과 artifact만 invalidate한다.
- Switch: Planner root와 output contract를 reset한다.

모든 update를 같은 memory append로 처리하면 function switch에 취약해진다.

### 3) Stale context masking

Conversation summary를 단순 압축하지 말고 status를 붙인다.

- ACTIVE
- REVISED
- SUPERSEDED
- TASK-LOCAL
- CARRIED-OVER

Model input에는 active state를 앞에 두고, history는 audit용으로 분리할 수 있다.

### 4) Paired static-dynamic regression test

Internal benchmark에서도 같은 final task를 두 setting으로 실행한다.

- Fully specified single-turn
- Evolving-intent multi-turn

Static score가 유지되는데 dynamic score만 떨어지면 intent state component를 우선 debug한다.

### 5) Tool-state invalidation

Coding or search agent에서는 text intent뿐 아니라 tool artifact도 versioning해야 한다.

- Old search result cache
- Old patch branch
- Old SQL draft
- Old document filter
- Old evidence shortlist

Function switch 후에도 artifact가 active하면 model이 correct recap을 받아도 wrong trajectory를 계속할 수 있다.

## 7-3. Follow-up papers

- Interactive Task Alignment as a POMDP
- SWE-INTERACT: Reimagining SWE Benchmarks as User-Driven Long-Horizon Coding Sessions
- SWE-Together: Evaluating Coding Agents in Interactive User Sessions
- Response-Aware User Memory Selection for LLM Personalization
- Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM Agents
- MemoryAgentBench and long-horizon agent memory evaluations

# 8. Summary

- 이 논문은 existing verifier-backed benchmark를 evolving-intent multi-turn environment로 변환한다.
- User intent transition은 argument reveal, argument revision, function switch로 분해된다.
- Final turn을 source task에 고정해 original verifier를 그대로 사용한다.
- Strong model도 dynamic setting에서 static performance를 보존하지 못하며 function switch가 가장 어렵다.
- Oracle recap도 gap을 완전히 닫지 못해 failure가 memory recall뿐 아니라 planning and execution state에 있음을 보여준다.
- Practical agent에는 transcript memory보다 versioned intent ledger와 stale-state invalidation이 필요하다.
