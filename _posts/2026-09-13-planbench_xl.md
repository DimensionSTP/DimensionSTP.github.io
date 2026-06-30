---
layout: single
title: "PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems Review"
categories: Study-concept
tag: [PlanBench-XL, ToolUseAgent, AgentBenchmark, Planning, LLM]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.22388)

[Code link](https://github.com/JiayuJeff/PlanBench-XL)

[Dataset link](https://huggingface.co/datasets/JiayuJeff/PlanBench-XL)

[Project page](https://planbench-xl.github.io/)

PlanBench-XL은 tool-use agent benchmark의 질문을 "tool을 잘 호출하는가"에서 "거대한 tool 생태계 안에서 필요한 tool을 찾아가며 긴 계획을 복구할 수 있는가"로 확장하는 논문이다. 이 논문이 겨냥하는 핵심 문제는 고정된 visible toolset이 아니다. 실제 agent는 enterprise MCP server, API platform, software ecosystem처럼 tool이 너무 많은 환경에서 동작한다. Context window 안에 모든 tool description을 넣을 수 없으므로, agent는 매 step마다 tool retrieval 결과만 보고 다음 행동을 정해야 한다.

이 setting이 어려운 이유는 retrieval이 planning 바깥의 전처리가 아니라 planning 자체의 일부가 되기 때문이다. Agent는 현재까지 얻은 evidence로 forward search를 하고, final goal에서 거꾸로 필요한 정보를 추론하는 backward search도 해야 한다. 또한 중간 sub-goal을 스스로 만들어야 한다. 여기에 retrieval된 tool이 noise를 포함하거나, path-critical tool이 blocked되어 explicit failure, implicit failure, semantic misleading failure를 만들 수 있다. 이 경우 agent는 "이 path가 막혔다"를 알아채고 대체 tool-use path를 찾아야 한다.

PlanBench-XL은 retail domain에서 327 query, 56 datatype, 1,665 tool을 구성하고, 각 task가 평균 약 25 turn을 요구하도록 만든다. 논문은 GPT-5.4가 block-free setting에서 51.90% accuracy를 보이지만, 가장 심한 blocking condition에서는 11.36%까지 떨어진다고 보고한다. 이는 현재 frontier model도 대규모 tool 환경의 long-horizon planning에서는 아직 견고하지 않다는 신호다.

> 한 줄 요약: PlanBench-XL은 partial tool visibility, implicit sub-goal, bidirectional exploration, noisy tool, path-preserving blocker를 결합해 LLM tool-use agent가 불완전한 대규모 tool 생태계에서 robust adaptive planning을 수행할 수 있는지 평가하는 interactive benchmark다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Tool retrieval을 전처리가 아니라 long-horizon planning loop의 일부로 평가한다.
- Agent가 evidence 기반 forward search와 goal 기반 backward search를 모두 해야 하는 setting을 만든다.
- Useful tool이 missing, failing, misleading 상태가 되는 blocking condition을 통해 runtime replanning을 stress test한다.
- 기존 tool benchmark가 잘 다루지 않는 retrieval-limited tool visibility와 unreliable tool access를 동시에 넣는다.
- GPT-5.4조차 severe blocker에서 11.36%까지 무너지는 failure mode를 보여준다.
- 향후 RL playground로도 쓸 수 있는 executable interactive environment를 제공한다.

이 글에서는 PlanBench-XL을 "tool benchmark 하나 추가"보다, **tool retrieval, planning, execution feedback, failure recovery가 한 loop 안에서 얽히는 agentic planning benchmark**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

LLM tool-use agent는 task $q$를 해결하기 위해 tool 생태계 $\mathcal{T}$ 안에서 행동 순서를 만든다. 기존 benchmark는 종종 관련 tool list가 미리 보인다고 가정한다.

$$
a_t
=
\pi_{\theta}
\left(
h_t,
\mathcal{T}_{visible}
\right)
$$

그러나 실제 environment에서는 전체 tool universe가 너무 커서, agent가 보는 것은 retrieval로 노출된 일부 tool뿐이다.

$$
\mathcal{T}_t^{ret}
=
R(h_t,\mathcal{T})
$$

Agent는 partial tool visibility 아래에서 retrieval, invocation, observation update, 최종 답변 생성을 반복한다.

$$
h_{t+1}
=
h_t
\cup
(a_t,o_t)
$$

PlanBench-XL의 핵심 질문은 다음이다.

> Agent가 전체 tool space를 보지 못하고, retrieved tool이 noisy하거나 blocked될 수 있는 환경에서, implicit sub-goal을 찾아 long-horizon task를 끝까지 해결할 수 있는가?

이 문제는 일반 function calling과 다르다. Function calling은 "어떤 API를 어떻게 호출할까"에 가깝지만, PlanBench-XL은 "어떤 중간 정보가 필요하고, 그 정보를 얻는 tool을 어떻게 찾으며, 그 path가 막히면 어떻게 복구할까"를 묻는다.

## 1-2. Why previous approaches are insufficient

### 1) Fully visible toolset

Agent에게 모든 tool description이 주어지면 tool selection은 retrieval problem이 아니라 prompt selection problem에 가까워진다. 실제 enterprise 또는 API environment에서는 이 가정이 깨진다.

### 2) One-shot tool retrieval

One-shot retrieval은 query의 표면 표현에 맞는 tool만 찾는다. Long-horizon task에서는 첫 tool output을 봐야 다음 missing datatype이나 sub-goal이 드러난다. 따라서 retrieval은 반복적으로 이루어져야 한다.

### 3) Explicit intermediate goals

많은 multi-hop tool benchmark는 intermediate sub-goal이 query에 어느 정도 드러나 있다. PlanBench-XL은 agent가 implicit sub-goal을 직접 추론해야 한다.

### 4) Perfect tool availability

현실에서는 tool이 stale, missing, unavailable, misleading 상태일 수 있다. Blocking condition이 없으면 agent의 복구 능력을 측정하기 어렵다.

# 2. Core Idea

## 2-1. Main contribution

PlanBench-XL의 contribution은 네 가지다.

1. **Massive tool ecosystem**
   - Retail domain에 56 datatype과 1,665 tool을 구성한다.
   - Agent는 complete tool list를 보지 않고 retrieval로 노출된 일부 tool만 본다.

2. **Long-horizon implicit planning**
   - 327 query가 multi-step retail workflow를 요구한다.
   - Ground-truth solution path는 최소 5 tool step을 포함하고, runtime은 평균 약 25 turn이다.

3. **Bidirectional exploration**
   - Forward anticipation: 누적된 evidence에서 다음 tool/sub-goal을 찾는다.
   - Backward anticipation: desired outcome에서 필요한 중간 정보를 역으로 추론한다.

4. **Path-preserving blockers**
   - Path-critical tool access를 explicit failure, implicit failure, semantic misleading failure로 교란한다.
   - 모든 path를 막지 않고 최소 하나의 valid solution path를 남겨 adaptive replanning을 평가한다.

## 2-2. Design intuition

이 논문의 design intuition은 다음과 같다.

```text
Tool planning은 partial visibility와 unreliable access 아래에서 평가해야 한다.
```

Tool-use agent가 실제로 어려워지는 지점은 tool call syntax가 아니다. 어려운 것은 다음 loop를 끝까지 안정적으로 수행하는 일이다.

```text
candidate tool을 retrieve한다
tool description을 확인한다
tool을 호출한다
output을 관찰한다
새 datatype 또는 sub-goal을 추론한다
다시 retrieve한다
failure 또는 misleading result를 감지한다
alternative path로 복구한다
```

따라서 benchmark도 final answer accuracy만 볼 것이 아니라, search/call balance, invalid call, noisy tool usage, turn count, datatype coverage 같은 과정 지표를 함께 봐야 한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | 대규모 retrieved tool 생태계에서 long-horizon planning 평가 |
| Domain | Retail workflow |
| Queries | 327 |
| Datatypes | 56 |
| Tools | 1,665 |
| Visibility | Retrieval을 거친 partial tool visibility |
| Core challenge | Implicit sub-goal 발견과 adaptive replanning |
| Failure mechanism | Path-preserving blocker |
| Models evaluated | 주요 LLM 10종 |
| Headline result | GPT-5.4 51.90% block-free, 11.36% severe blocker |

## 3-2. Module breakdown

### 1) Tool library construction

PlanBench-XL은 retail datatype에서 시작한다. Datatype은 product name, store name, inventory status, purchase status, customer-related attribute 같은 구체적인 domain information type이다.

각 tool $\tau$에 대해 논문은 input/output datatype set을 정의한다.

$$
\mathcal{D}_{in}(\tau)
$$

$$
\mathcal{D}_{out}(\tau)
$$

Tool construction은 체계적으로 이루어진다. Input/output datatype 조합 위에서 candidate tool을 생성한 뒤 현실성과 중복성 기준으로 filter한다. 이를 통해 규모는 크지만 통제 가능한 tool universe를 만든다.

### 2) Backend record and executable tools

각 retail case는 complete backend record를 가진다. Tool call은 input argument를 backend value에 mapping하고 output datatype을 반환한다.

이는 중요하다. Answer를 commonsense로 추론할 수 없기 때문이다. Agent는 단순 추측으로 알 수 없는 값을 얻기 위해 실제로 tool을 호출해야 한다.

### 3) Retrieval-mediated runtime

각 turn에서 agent는 tool을 retrieve하거나 available tool을 호출할 수 있다. Retriever는 subset만 노출한다. Retrieved tool에는 의미적으로 비슷해 보이지만 실제로는 unusable하거나 unreliable한 noisy sibling tool이 포함될 수 있다.

Agent state에는 발견한 tool, 신뢰할 수 있는 intermediate value와 그렇지 않은 value, executable trace, final-answer correctness가 포함된다.

### 4) Bidirectional exploration

PlanBench-XL은 두 방향의 search를 명시적으로 중요하게 본다.

| Direction | Meaning |
| --- | --- |
| Forward anticipation | 관찰된 evidence에서 다음 useful datatype/tool을 찾음 |
| Backward anticipation | desired final answer에서 출발해 missing prerequisite을 역추론함 |

Long-horizon retail workflow에는 두 방향이 모두 필요한 경우가 많다. 예를 들어 final goal이 refund eligibility를 요구하고, 이는 purchase status를 요구하며, purchase status는 order ID를 요구하고, order ID는 customer lookup을 요구할 수 있다.

### 5) Path-preserving blockers

Blocker는 최소 하나의 valid solution path를 보존하면서 selected baseline tool을 failure로 대체한다.

| Blocker type | Failure mode |
| --- | --- |
| Explicit failure | Tool이 실패 또는 unavailable 상태를 명확히 알림 |
| Implicit failure | 명확한 signal 없이 tool output이 없거나 unusable함 |
| Semantic misleading | Tool이 그럴듯해 보이지만 agent를 잘못된 방향으로 유도함 |

Implicit failure와 semantic failure는 더 어렵다. Agent가 downstream inconsistency를 통해 해당 path가 틀렸음을 추론해야 하기 때문이다.

### 6) Evaluation metrics

Evaluator는 final answer accuracy 외에도 다음을 보고한다.

| Metric | Why it matters |
| --- | --- |
| Turn count | Long-horizon 효율성 |
| Search/call balance | Agent가 충분히 탐색하는지, 또는 tool call을 과하게 반복하는지 |
| Invalid tool-call rate | 실행 신뢰성 |
| Noisy-tool usage | Distractor에 대한 견고성 |
| Ground-truth datatype coverage | 필요한 중간 정보를 찾았는지 |
| Final-answer correctness | Task 성공 여부 |

# 4. Training / Data / Recipe

## 4-1. Data

PlanBench-XL release는 retail-domain benchmark artifact를 포함한다.

| Component | Value |
| --- | ---: |
| Queries | 327 |
| Datatypes | 56 |
| Tools | 1,665 |
| Average runtime | 약 25 turn |
| Tool path length | 최소 5 tool step |

Public repository와 dataset은 benchmark configuration을 실행할 수 있도록 공개되어 있다.

## 4-2. Evaluation recipe

Evaluation loop는 interactive하다.

1. Agent가 task와 current state를 관찰한다.
2. Agent가 tool을 retrieve하거나 available tool을 호출한다.
3. Environment가 retrieved tool description 또는 tool output을 반환한다.
4. Agent가 plan과 intermediate value를 갱신한다.
5. Agent가 final answer에 도달할 때까지 이 과정을 반복한다.
6. Evaluator가 correctness와 process metric을 계산한다.

Blocker configuration은 selected baseline tool을 runtime에 대체하도록 설정해 실행할 수 있다. Config에는 target remaining path와 blocker noise type이 포함된다.

## 4-3. Engineering notes

1. **Retrieval과 planning을 함께 평가한다**
   - 둘을 분리하면 핵심 난점이 가려진다.

2. **Path-preserving blocker는 solvable하게 유지한다**
   - 모든 path가 막히면 failure가 원인을 진단할 signal을 주지 못한다.

3. **Intermediate datatype coverage를 log로 남긴다**
   - Final wrong answer는 key datatype 하나를 놓친 데서 올 수 있다.

4. **Explicit failure와 implicit failure를 분리한다**
   - Agent는 조용히 틀린 path보다 명확한 error message를 더 잘 처리한다.

5. **Process metric을 사용한다**
   - 같은 final answer accuracy가 매우 다른 exploration behavior를 숨길 수 있다.

6. **Benchmark를 RL playground로 쓸 때 주의한다**
   - 목표가 adaptive planning이라면 reward가 final answer에만 collapse되어서는 안 된다.

# 5. Evaluation

## 5-1. Main results

Abstract는 PlanBench-XL이 frontier model에게도 여전히 어렵다고 보고한다. GPT-5.4는 block-free setting에서 51.90% accuracy에 도달하지만, 가장 심한 blocking condition에서는 11.36%로 떨어진다.

논문은 대부분의 model이 default setting에서 two-thirds accuracy 아래에 머문다고도 보고한다. Retrieval-time blocking 아래에서는 performance가 급격히 떨어지며, 특히 feasible path가 하나만 남거나 가장 긴 recovery path만 보존될 때 더 그렇다.

## 5-2. Failure analysis

가장 중요한 failure pattern은 단순히 "tool이 많다"가 아니다. Agent는 다음 이유로 실패한다.

- Useful tool이 적절한 시점에 retrieve되지 않는다.
- Intermediate sub-goal이 추론되지 않는다.
- Noisy tool이 그럴듯해 보인다.
- Blocking failure가 명시적으로 signal되지 않는다.
- Recovery가 더 긴 alternative path를 요구한다.
- Agent가 초반의 plausible path를 과도하게 exploit하고 alternative를 충분히 explore하지 않는다.

이는 일반적인 tool-call accuracy보다 더 풍부한 진단 signal을 제공한다.

## 5-3. What really matters in the experiments

### 1) Severe blocker에서 collapse가 발생한다

51.90%에서 11.36%로 떨어지는 결과는 expected path가 깨질 때 현재 agent가 robust recovery를 갖추지 못했음을 보여준다.

### 2) Implicit failure는 explicit failure보다 어렵다

Tool이 "I am unavailable"이라고 말하면 replanning은 더 쉽다. Tool이 조용히 unusable 상태이거나 misleading output을 주면 agent가 inconsistency를 감지해야 한다.

### 3) 긴 alternative path가 핵심 stressor다

Recovery는 단순히 "다른 tool을 시도하는 것"이 아니다. 더 긴 intermediate datatype chain을 발견해야 할 수 있다.

### 4) Retail domain은 synthetic이지만 structured하다

Benchmark는 generated되었지만 tool은 executable하고 backend record에 grounded되어 있다. 이는 통제성과 재현성을 제공하면서도 단순하지 않은 tool dependency를 만든다.

# 6. Limitations

1. **Retail domain scope가 제한적이다**
   - Benchmark는 retail workflow에 놓여 있다.
   - Enterprise, code, finance, medical, OS-level tool은 다른 failure mode를 가질 수 있다.

2. **Synthetic tool ecosystem이다**
   - Tool은 production system에서 직접 수집된 것이 아니라 생성되고 filter된 것이다.

3. **Retriever dependency가 있다**
   - 결과는 retrieval implementation과 tool description quality에 의존한다.

4. **Metric이 여전히 answer-centric하다**
   - Process metric이 있지만 final answer accuracy가 여전히 대표 지표다.

5. **Blocking model은 designed setting이다**
   - Explicit, implicit, semantic blocker는 유용하지만 모든 real-world failure를 포괄하지는 못할 수 있다.

6. **Human user interaction이 없다**
   - User clarification과 collaborative repair는 이 benchmark의 중심이 아니다.

7. **Cost와 latency가 있다**
   - Long-horizon retrieval과 tool invocation은 비쌀 수 있지만 cost analysis가 main focus는 아니다.

8. **Benchmark gaming 가능성이 있다**
   - Datatype/tool structure가 systematic하므로 agent가 benchmark construction pattern에 overfit할 수 있다.

9. **Training result가 main focus는 아니다**
   - 논문은 이를 full RL recipe가 아니라 benchmark와 diagnostic playground로 제시한다.

10. **Tool semantics는 여전히 textual하다**
    - Real tool은 더 풍부한 side effect, stateful mutation, auth, rate limit, nondeterministic behavior를 포함할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

PlanBench-XL의 핵심은 "tool이 많아서 어렵다"가 아니다. 더 중요한 점은 **tool retrieval이 planning state를 바꾸는 action이 된다는 것**이다.

Large ecosystem의 tool-use agent는 tool을 선택하는 데 그치면 안 된다. Uncertainty 아래에서 tool graph를 explore해야 한다.

이는 real agent가 MCP server, enterprise tool, API platform에서 동작하는 방식에 가깝다.

## 7-2. Reuse potential

### Agent evaluation

PlanBench-XL-style blocker를 사용해 expected tool path가 실패할 때 agent가 recover할 수 있는지 test할 수 있다.

### Tool retrieval system design

Retrieval은 immediate semantic similarity뿐 아니라 multi-step completeness를 optimize해야 한다.

### Agent memory

Agent는 어떤 tool을 시도했는지, 어떤 value를 trust할 수 있는지, 어떤 path가 suspicious한지 기억해야 한다.

### Agentic RL

Process reward에는 datatype coverage, successful recovery, noisy-tool avoidance, final correctness가 포함될 수 있다.

## 7-3. Production considerations

- 명확한 reliability metadata를 가진 tool ecosystem을 구축해야 한다.
- Retrieved but unused tool을 log로 남겨야 한다.
- Noisy하거나 failing하는 tool에 대한 repeated call을 감지해야 한다.
- Trusted value와 untrusted value를 구분하는 plan-state representation을 추가해야 한다.
- Agent eval에 failure-type taxonomy를 사용해야 한다.
- Explicit exception뿐 아니라 silent failure 아래에서도 recovery를 test해야 한다.

## 7-4. Follow-up papers

- ToolBench
- API-Bank
- Gorilla
- MCPBench
- LiveMCPBench
- Tool Decathlon
- AgentNoiseBench
- OpaqueToolsBench
- Verification Horizon
- Agent-as-a-Router

# 8. Summary

- PlanBench-XL은 large retrieved tool ecosystem에서 long-horizon planning을 평가한다.
- 1,665 tool과 56 datatype 위의 327 retail task를 포함한다.
- Agent는 tool을 iteratively retrieve하고, implicit sub-goal을 추론하며, blocker에서 recover해야 한다.
- GPT-5.4는 block-free accuracy 51.90%에서 severe blocking 아래 11.36%로 떨어진다.
- 핵심 insight는 robust tool-use agent에는 tool-call syntax뿐 아니라 adaptive exploration과 replanning이 필요하다는 점이다.
