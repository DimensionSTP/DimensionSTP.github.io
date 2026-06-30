---
layout: single
title: "Agent-as-a-Router: Agentic Model Routing for Coding Tasks Review"
categories: Study-concept
tag: [AgentAsRouter, ACRouter, CodeRouterBench, CodingAgent, ModelRouting]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.22902)

[Homepage](https://www.omnisource.cn/agent-as-a-router)

[Code link](https://github.com/LanceZPF/agent-as-a-router)

[Dataset link](https://huggingface.co/datasets/Lance1573/CodeRouterBench)

Agent-as-a-Router를 "여러 LLM 중 하나를 고르는 router 논문" 정도로 읽으면 핵심을 놓치기 쉽다. 이 논문이 겨냥하는 문제는 static routing이 아니라 **routing policy가 deployment stream에서 실행 결과를 보고 스스로 정보를 축적해야 하는가**다.

Coding task에서 model choice는 생각보다 어렵다. 어떤 model은 repository-level bug fixing에 강하고, 어떤 model은 short algorithm problem에 싸고 빠르며, 어떤 model은 특정 language나 tool-use pattern에서 더 안정적이다. 사용자가 여러 provider와 local model을 동시에 쓸 수 있다면, always-strongest model은 비용이 너무 높고, always-cheapest model은 성능이 낮다. Per-task oracle은 매번 가장 좋은 model을 고르지만, oracle은 실행 전에 결과를 알 수 없다.

기존 LLM router는 대개 task text를 보고 한 번에 model을 고르는 static classifier로 구성된다. Agent-as-a-Router는 여기서 bottleneck이 reasoning failure가 아니라 information deficit이라고 본다. 실제로 논문은 vanilla LLM router에 task-dimension별 performance statistics만 제공해도 15.3% relative gain이 발생한다고 보고한다. 즉 router가 더 똑똑하게 생각하지 못해서가 아니라, 어떤 model이 어떤 task family에서 잘했는지에 대한 실행 기반 정보가 부족하다는 것이다.

그래서 논문은 routing을 Context-Action-Feedback loop로 다시 정의한다. Router는 현재 task와 accumulated experience를 Context로 보고, candidate model 중 하나를 Action으로 선택하고, verifier가 실제 결과와 cost를 Feedback으로 돌려준다. 이 feedback은 Memory에 저장되어 다음 routing decision의 context가 된다.

> 한 줄 요약: Agent-as-a-Router는 coding model routing을 static classification이 아니라 Context->Action->Feedback->Context loop로 보고, Orchestrator, Verifier, Memory로 구성된 ACRouter와 약 10K tasks, 8 frontier LLM result matrix를 포함한 CodeRouterBench를 통해 regret-based streaming routing을 평가하는 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Model routing을 one-shot classifier가 아니라 execution-grounded online decision problem으로 정식화한다.
- Coding task에서는 "which model is best"가 아니라 "what information does the router have before execution"이 핵심 bottleneck임을 보여준다.
- CodeRouterBench를 task-by-model result matrix로 공개해 routing policy를 regret metric으로 평가할 수 있게 한다.
- Orchestrator, Verifier, Memory의 closed loop가 agent system에서 routing과 learning의 경계를 어떻게 흐리는지 보여준다.
- Claude Code, Codex, opencode 같은 commercial coding CLI에 붙일 수 있는 router integration 방향을 제시한다.
- Cost-performance frontier를 단일 model benchmark가 아니라 task stream decision quality로 평가한다.

이 글에서는 Agent-as-a-Router를 "LLM router benchmark"보다, **coding agent serving에서 model selection 자체를 feedback-accumulating agent loop로 바꾸는 paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

사용자가 여러 model backend를 사용할 수 있다고 하자. 각 coding task $t_i$마다 candidate model pool은 다음과 같다.

$$
\mathcal{M}
=
\{m_1,\ldots,m_M\}
$$

Router는 task stream $\mathcal{T}=(t_1,\ldots,t_N)$를 순서대로 처리하며 각 task마다 model index $a_i$를 선택한다.

$$
a_i
\in
\{1,\ldots,M\}
$$

선택된 model은 solution을 생성하고, verifier는 성능 score $s_i(a_i)$와 cost $\kappa_i(a_i)$를 돌려준다. User preference가 performance와 cost를 weight한다고 하면 reward는 다음처럼 쓸 수 있다.

$$
r_i(a_i)
=
\epsilon_1 s_i(a_i)
+
\epsilon_2 \kappa_i(a_i)
$$

여기서 $\epsilon_1>0$이고 $\epsilon_2<0$다. Per-task oracle은 모든 model result를 미리 알고 각 task에서 best reward model을 고른다.

$$
a_i^*
=
\arg\max_{j \in [M]}
r_i(j)
$$

Router의 cumulative regret은 oracle과의 누적 reward gap이다.

$$
\mathrm{CumReg}_N
=
\sum_{i=1}^{N}
\left(
r_i(a_i^*)-r_i(a_i)
\right)
$$

이 metric이 중요한 이유는 model routing이 streaming decision problem이기 때문이다. Average accuracy만 보면 expensive model을 많이 쓰는 router가 유리할 수 있고, cost만 보면 cheap model이 유리할 수 있다. Regret은 performance-cost trade-off 아래에서 router가 oracle에 얼마나 가까운지 본다.

## 1-2. Why previous approaches are insufficient

### 1) Always-strongest model

항상 strongest model을 쓰면 performance는 높을 수 있지만 cost가 커진다. 또한 per-task oracle과 비교하면 strongest global model도 task-specific best model보다 뒤질 수 있다.

### 2) Static heuristic router

DimensionBest처럼 task dimension별 prior를 사용하면 저렴하고 안정적이다. 하지만 deployment stream에서 새 feedback을 반영하지 못한다. OOD task가 들어오면 static prior가 빠르게 outdated될 수 있다.

### 3) Static trained policy

Trained router는 historical matrix에서 학습할 수 있지만, training 이후 state가 고정된다. 실제 deployment에서 selected model output이 pass/fail했는지, 어떤 model이 최근 similar task에서 잘했는지 반영하지 못한다.

### 4) LLM-as-a-Router

LLM에게 task prompt를 주고 "어떤 model이 좋을까"를 물으면 explainable할 수 있다. 하지만 논문 preliminary 결과는 vanilla LLM router가 oracle에 크게 못 미친다는 것을 보여준다. 단순 reasoning capability가 아니라 information access가 문제다.

# 2. Core Idea

## 2-1. Main contribution

논문의 기여는 세 가지다.

1. **Agent-as-a-Router framework**
   - Routing을 Context-Action-Feedback loop로 정의한다.
   - Verified execution feedback이 다음 decision context로 들어간다.
   - Cumulative regret을 streaming evaluation metric으로 사용한다.

2. **ACRouter implementation**
   - Orchestrator: task, prior, memory를 보고 model을 선택한다.
   - Verifier: selected model output을 sandbox, parser, task-specific checker로 평가한다.
   - Memory: chosen model, score, cost, verification trace를 누적한다.

3. **CodeRouterBench**
   - 약 10K task instances.
   - 8 frontier LLM result matrix.
   - ID task와 OOD176 agentic-programming stream.
   - Live API call 없이 offline reproduction이 가능하다.

## 2-2. Design intuition

Agent-as-a-Router의 핵심 intuition은 router가 deployment experience를 가져야 한다는 것이다.

기존 router는 다음처럼 작동한다.

```text
task -> model choice
```

Agent-as-a-Router는 다음처럼 작동한다.

```text
task + prior + memory -> model choice -> execution -> verified feedback -> memory
```

즉 router 자체가 mini-agent가 된다. Router는 행동하고, feedback을 관찰하고, context를 업데이트한다.

이 방식은 contextual bandit과 닮았다. 하지만 일반 bandit보다 coding task의 feedback이 더 풍부하다. Verifier는 단순 success/failure만 아니라 score, cost, error type, trace, sandbox evidence를 제공할 수 있다. Memory는 future routing에서 similar task evidence를 retrieve할 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Performance-cost trade-off 아래에서 coding task를 backend model로 route |
| Framework | Context-Action-Feedback loop |
| Implementation | ACRouter |
| Modules | Orchestrator, Verifier, Memory |
| Benchmark | CodeRouterBench |
| Models | Result matrix 안의 8개 frontier LLM backend |
| ID setting | 8개 model result를 가진 9,999개 task |
| Test setting | 2,919개 ID test task와 OOD176 task |
| Main metric | AvgPerf, cumulative regret, cost, performance per dollar |

## 3-2. Module breakdown

### 1) Context

Task $t_i$에서 context는 다음을 포함한다.

$$
c_i
=
(p_i,d_i,\mathcal{H}_{<i})
$$

- $p_i$: task prompt
- $d_i$: optional metadata such as task dimension
- $\mathcal{H}_{<i}$: accumulated history from previous routing decisions

핵심 design은 context가 verified execution history와 함께 커진다는 점이다.

### 2) Action

Action은 model selection이다.

$$
a_i \in [M]
$$

이 action은 현재 task를 풀 backend model을 결정한다.

### 3) Feedback

Execution 이후 verifier는 feedback을 반환한다.

$$
f_i
=
(\hat{s}_i,\hat{\kappa}_i)
$$

- $\hat{s}_i$: observed performance score
- $\hat{\kappa}_i$: cost computed from token usage and model price

이는 execution-grounded signal이다. Self-reported confidence가 아니다.

### 4) Orchestrator

Orchestrator는 dynamic context를 사용해 model을 고른다. 논문의 ACRouter는 다음을 사용한다.

- DimensionBest prior
- top-10 historical neighbors retrieved from Memory by kNN
- task metadata
- a cost-effective Qwen3.5-0.8B router model
- heuristic voting rules

중요한 점은 exact implementation만이 아니다. Orchestrator가 static prior와 dynamic experience를 통합한다는 점이다.

### 5) Verifier

Verifier는 task-specific validation tool을 unified score로 집계한다. Task type에 따라 다음을 포함할 수 있다.

- AST parsing
- sandbox execution
- unit tests
- score normalization
- verification trace logging

Verifier는 매우 중요하다. 나쁜 feedback은 memory를 오염시키기 때문이다. Verifier가 noisy하면 router는 잘못된 association을 학습한다.

### 6) Memory

Memory는 past task embedding, chosen model, score, cost, verification trace를 저장한다. Retrieval은 cosine kNN을 사용한다. Implementation은 FIFO bound를 $E=20K$로 둔다.

이 memory가 router를 "agentic"하게 만든다. Router는 similar past task를 근거로 future decision을 condition할 수 있다.

## 3-3. CodeRouterBench

CodeRouterBench는 단순 output dump가 아니라 task-by-model matrix다.

Public repository는 다음 component를 설명한다.

| Component | Description |
| --- | --- |
| ID results | 9,999 tasks x 8 models |
| ID test results | 2,919 tasks x 8 models |
| OOD176 results | 176 tasks x 8 models |
| Task metadata | ID/OOD task jsonl file |
| Model metadata | canonical model list와 USD pricing |
| Reference outputs | baseline tables, decisions, metrics |

이 구조는 router evaluation에 적합하다. 모든 routing policy를 같은 outcome matrix 위에서 offline replay할 수 있기 때문이다.

# 4. Training / Data / Recipe

## 4-1. Data

Benchmark data는 두 main phase를 갖는다.

| Split | Description |
| --- | --- |
| ID | In-distribution coding dimensions |
| OOD176 | Out-of-distribution agentic-programming tasks |

Repository는 OOD112/SWE-MiniSandbox를 legacy supplementary data로 두고, current public benchmark를 OOD176으로 설명한다.

각 result row는 task id, model, score 또는 pass signal, cost, 가능한 경우 token/latency나 verifier metadata를 기록한다.

## 4-2. Router training

ACRouter는 Qwen3.5-0.8B 기반 router LoRA adapter를 포함한다. 논문과 repository는 이를 cost-effective router model로 설명한다. 다만 순수 black-box classifier로만 쓰는 것이 아니라 prior와 heuristic을 함께 결합한다.

중요한 training lesson은 router model만으로는 충분하지 않다는 것이다. Deployment에서는 memory와 verifier feedback이 필요하다.

## 4-3. Engineering notes

1. **Offline router evaluation에는 outcome matrix를 사용한다**
   - 이렇게 하면 router comparison 중 live API randomness와 cost를 피할 수 있다.

2. **Average performance뿐 아니라 regret을 평가한다**
   - Router는 streaming decision policy다.

3. **Verifier quality가 핵심이다**
   - 나쁜 verifier feedback은 나쁜 memory를 만든다.

4. **Actual token usage로 cost를 추적한다**
   - Cost accounting이 없는 routing은 serving policy가 아니라 model selection benchmark에 가까워진다.

5. **ID와 OOD를 분리한다**
   - Static prior는 ID에서는 좋아 보이지만 OOD에서는 실패할 수 있다.

6. **Gateway 또는 proxy level에서 통합한다**
   - Routing은 request가 선택된 backend model로 보내지기 전에 일어나야 한다.

# 5. Evaluation

## 5-1. Preliminary bottleneck result

논문의 key diagnostic table은 같은 task set에서 router들을 비교한다.

| Router | AvgPerf% | PerfPerUSD |
| --- | ---: | ---: |
| Oracle | 57.00 | 8.20 |
| DimensionBest | 47.50 | 3.69 |
| Vanilla LLM router | 41.41 | 1.97 |
| +Dimension | 41.18 | 1.81 |
| +Perf stats | 47.74 | 1.71 |

중요한 결과는 vanilla LLM router에 performance statistics를 추가했을 때 +15.3% relative gain이 나온다는 점이다. 이는 information deficit hypothesis를 뒷받침한다.

## 5-2. Reproduction outputs

Public repository는 expected headline output을 다음처럼 보고한다.

| Setting | n | AvgPerf% | CumReg | CostUSD | PerfPerUSD |
| --- | ---: | ---: | ---: | ---: | ---: |
| ID | 2919 | 50.14 | 202.0 | 22.31 | 2.25 |
| ACRouter-OOD176 | 176 | 73.30 | 15.9 | 86.72 | 0.85 |

이 수치는 reproduction anchor로 유용하다. Publication 전에 final paper table과 다시 대조해야 한다.

## 5-3. What really matters in the experiments

### 1) Routing은 information-limited 문제다

핵심 결과는 특정 small router model이 마법처럼 좋다는 것이 아니다. Router는 relevant performance evidence를 가질 때 개선된다.

### 2) OOD가 진짜 stress test다

Static dimension prior는 ID에서 잘할 수 있다. OOD agentic-programming task는 router가 memory와 verification을 사용해 generalize할 수 있는지 시험한다.

### 3) Cost와 performance는 함께 보고해야 한다

항상 expensive model을 고르는 router는 success가 높을 수 있지만 deployment value는 낮을 수 있다.

### 4) Memory는 routing을 classifier에서 agent로 바꾼다

Memory는 router가 actual execution history를 사용하게 만든다. 이것이 conceptual shift다.

# 6. Limitations

1. **Benchmark matrix는 finite하다**
   - Result matrix는 fixed task-model outcome을 담고 있다.
   - Real deployment에서는 model version, latency, cost가 계속 바뀐다.

2. **Verifier quality가 learning을 제한한다**
   - ACRouter는 verifier score에서 학습한다.
   - Verifier가 약하면 routing memory도 unreliable해진다.

3. **Model pool이 제한적이다**
   - CodeRouterBench는 8개 frontier LLM을 사용한다.
   - 새로운 local 또는 proprietary model은 다시 평가해야 한다.

4. **Cost는 바뀔 수 있다**
   - Model metadata의 USD pricing은 시간이 지나면 stale해질 수 있다.

5. **Routing overhead가 있다**
   - Memory retrieval, router inference, verifier execution은 overhead를 추가한다.

6. **Task representation이 중요하다**
   - kNN memory retrieval은 embedding과 metadata에 의존한다.

7. **OOD176은 여전히 작다**
   - 176개 OOD task는 유용하지만 모든 agentic coding을 포괄하기에는 부족하다.

8. **Production integration complexity가 있다**
   - Gateway/proxy integration은 auth, privacy, logging, fallback을 처리해야 한다.

9. **Regret metric에는 user preference weight가 필요하다**
   - 사용자마다 performance와 cost에 부여하는 weight가 다를 수 있다.

10. **Static outcome replay는 live model nondeterminism을 숨긴다**
    - Offline matrix는 reproducible하지만 live provider drift를 포착하지 못한다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 핵심은 "routing으로 cost를 줄인다"가 아니다. 더 중요한 점은 **router가 verifier와 memory를 가지면 그 자체가 agent가 된다**는 점이다.

Coding workflow에서 model choice는 static metadata classification 문제가 아니다. Uncertainty 아래에서 반복되는 decision 문제다. 매 execution은 어떤 backend가 어떤 task shape에 좋은지에 대한 evidence를 만든다.

이 evidence를 버리면 router는 계속 근거가 부족한 상태에서 선택한다. ACRouter는 그 feedback loop를 명시적으로 만든다.

## 7-2. Reuse potential

### Coding assistant gateway

여러 coding assistant를 쓰는 team은 repository type, language, failing test pattern, cost budget, previous outcome을 기준으로 task를 route할 수 있다.

### Agent platform orchestration

Model routing은 shared runtime service가 될 수 있다. Task, selected model, result, cost, verifier evidence를 logging할 수 있기 때문이다.

### Evaluation infrastructure

CodeRouterBench-style full result matrix는 expensive live call을 반복하지 않고 router를 평가할 수 있게 해주는 유용한 design이다.

### Personal cost-performance policy

사용자마다 performance와 cost에 대한 $\epsilon_1,\epsilon_2$ trade-off를 다르게 정의할 수 있다.

## 7-3. Production considerations

- Model price와 result matrix를 versioning한다.
- Prompt를 router memory에 저장하기 전에 privacy guardrail을 둔다.
- Verifier가 실패하면 strong model로 fallback할 수 있게 한다.
- Model provider가 update되므로 outcome을 주기적으로 refresh한다.
- Task domain별 regret을 따로 monitoring한다.
- Model choice audit를 위해 router explainability를 유지한다.

## 7-4. Follow-up papers

- FrugalGPT
- RouteLLM
- LLMRouterBench
- CASTER
- TCAndon-Router
- SWE-Bench Verified
- Terminal-Bench
- Verification Horizon
- Agent model routing and contextual bandit literature

# 8. Summary

- Agent-as-a-Router는 model routing을 Context-Action-Feedback loop로 본다.
- Bottleneck은 단순 router reasoning이 아니라 information deficit이다.
- ACRouter는 Orchestrator, Verifier, Memory를 결합한다.
- CodeRouterBench는 regret-based routing evaluation을 위한 task-by-model matrix를 제공한다.
- 핵심 교훈은 routing이 deployment 중 execution-grounded feedback에서 학습해야 한다는 점이다.
