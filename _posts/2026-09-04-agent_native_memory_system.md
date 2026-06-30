---
layout: single
title: "Are We Ready For An Agent-Native Memory System? Review"
categories: Study-concept
tag: [AgentMemory, MemorySystem, DataManagement, RAG, AIAgent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.24775)

[Code link](https://github.com/OpenDataBox/MemoryData)

Are We Ready For An Agent-Native Memory System?은 agent memory를 algorithm component가 아니라 **data management system**으로 다시 보는 논문이다. 이 관점 전환이 중요하다. Agent memory는 이제 단순히 "embedding DB에서 top-k를 가져오는 기능"이 아니다. Persistent storage, extraction, retrieval, update, consolidation, versioning, latency, cost를 모두 포함하는 system layer가 되었다.

기존 memory benchmark는 대개 end-to-end task score를 본다. F1, BLEU, QA accuracy, conversation quality 같은 metric이다. 하지만 memory system이 왜 잘했는지, 어떤 module이 bottleneck인지, update robustness는 어떤지, query latency는 얼마인지, stale fact를 어떻게 처리하는지는 잘 보이지 않는다.

이 논문은 memory system을 네 모듈로 분해한다.

1. Memory representation and storage, 즉 memory를 어떤 형태로 표현하고 저장할 것인가
2. Memory extraction, 즉 원본 상호작용에서 저장할 정보를 어떻게 뽑아낼 것인가
3. Memory retrieval and routing, 즉 query에 맞는 memory를 어떻게 찾고 어디로 보낼 것인가
4. Memory maintenance, 즉 오래된 정보, 충돌, 중복을 어떻게 관리할 것인가

그리고 대표적인 memory system 12개와 reference baseline 2개를 5개 benchmark workload, 11개 dataset에서 평가한다. 결론은 단순하지 않다. 모든 상황에서 이기는 단일 architecture는 없고, workload의 병목과 memory structure가 맞아야 한다.

> 한 줄 요약: 이 논문은 agent memory를 persistent data management infrastructure로 보고, representation/storage, extraction, retrieval/routing, maintenance 네 module로 분해한 뒤 12개 memory systems를 5개 workload와 11 datasets에서 평가해, workload별 trade-off, update robustness, long-horizon stability, cost-performance bottleneck을 분석한 systems evaluation paper다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Agent memory를 RAG component가 아니라 database-like system으로 다룬다.
- End-to-end score만 보지 않고 retrieval fidelity, dynamic update robustness, long-horizon stability, operational cost를 분리한다.
- Memory architecture를 stream, tiered, graph, composite hybrid 등으로 비교한다.
- Temporal distance가 retrieval quality를 어떻게 망가뜨리는지 보여준다.
- Graph methods, append-only stores, hybrid systems의 장단점을 workload별로 분해한다.
- Agent memory 제품/연구를 설계할 때 어떤 module을 따로 benchmark해야 하는지 checklist를 제공한다.

이 글에서는 이 논문을 "어떤 memory system이 제일 좋다"보다, **agent-native memory system을 평가하기 위한 data management lens를 제안한 paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

LLM agent memory는 single inference step을 넘어 누적되는 state를 관리한다. 대화 history, tool logs, user preferences, episodic facts, procedural rules, environment observations가 모두 memory candidate가 된다.

논문은 agent memory system을 네 module의 tuple로 본다.

$$
M_{\mathrm{sys}}
=
(R,S,Q,U)
$$

| Module | Meaning |
| --- | --- |
| $R$ | Memory representation and storage |
| $S$ | Memory extraction |
| $Q$ | Memory retrieval and routing |
| $U$ | Memory maintenance |

이 분해가 핵심이다. 같은 end-to-end score라도 실패 원인이 다를 수 있다.

- Representation이 부실해서 evidence가 사라질 수 있다.
- Extraction이 잘못된 fact를 만들 수 있다.
- Retrieval이 필요한 memory를 찾지 못할 수 있다.
- Maintenance가 stale fact를 지우지 못할 수 있다.
- Cost가 너무 높아 production에 못 쓸 수 있다.

## 1-2. Why previous approaches are insufficient

### 1) RAG는 상태를 갖지 않는다

RAG는 보통 고정된 corpus에서 query-time retrieval을 수행한다. Agent memory는 다르다. Agent-specific information이 계속 쓰이고, 갱신되고, 충돌하고, 통합된다. 즉 memory는 read-only 저장소가 아니라 계속 바뀌는 상태 관리 계층이다.

### 2) Context engineering은 단기 context 문제에 가깝다

Context engineering은 현재 context window를 어떻게 채울지 다룬다. Agent memory는 context window 밖에 persistent state를 저장하고, 그 state의 lifecycle을 관리한다.

### 3) End-to-end metric은 불투명하다

System A가 system B보다 높은 QA score를 얻었다고 해도, 그 이유는 바로 보이지 않는다. Retrieval이 좋아서일 수도 있고, extraction이 정확해서일 수도 있고, maintenance가 충돌을 잘 처리해서일 수도 있으며, 단순히 answer model의 hallucination이 적어서일 수도 있다. 그래서 이 논문은 module-level analysis가 필요하다고 주장한다.

### 4) Operational cost가 자주 무시된다

Memory system은 index를 만드는 시간, query latency, LLM extraction 비용이 매우 클 수 있다. Production memory system은 accuracy만이 아니라 비용 대비 성능까지 함께 평가해야 한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 기여는 다음과 같다.

1. **Agent memory taxonomy**
   - Representation/storage, extraction, retrieval/routing, maintenance라는 네 core module로 memory를 나눈다.
   - Stream, tiered, graph, composite hybrid system의 특징을 정리한다.

2. **Unified benchmark study**
   - 대표 memory system 12개를 비교한다.
   - Reference baseline 2개를 함께 둔다.
   - 11개 dataset에 걸친 5개 benchmark workload에서 평가한다.

3. **Fine-grained ablations**
   - Representation fidelity, retrieval precision, update correctness, long-horizon stability를 측정한다.

4. **Cost-performance analysis**
   - 구조화된 memory system은 비용이 커질 수 있음을 보여준다.
   - 필요한 부분만 고치는 localized maintenance가 전체를 다시 정리하는 방식보다 비용 효율적일 수 있음을 보인다.

## 2-2. Design intuition

논문의 직관은 agent memory를 data system처럼 평가해야 한다는 것이다. Data system은 answer correctness만으로 평가되지 않는다. 다음이 모두 중요하다.

- data model
- indexing
- query planning
- update semantics
- versioning
- latency
- storage cost
- maintenance policy
- robustness to stale or conflicting data

Agent memory도 같은 문제를 가진다. 다만 query가 semantic하고 update가 불확실하다는 차이가 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Agent memory를 data management system으로 평가 |
| Core modules | Representation/storage, extraction, retrieval/routing, maintenance |
| Systems | 12개 memory system과 2개 baseline |
| Workloads | 5 benchmark workloads, 11 datasets |
| Evaluation axes | Task effectiveness, retrieval fidelity, update robustness, long-horizon stability, operational cost |
| Main finding | 모든 scenario를 지배하는 단일 architecture는 없음 |
| Key design point | Memory structure는 workload bottleneck에 맞아야 함 |

## 3-2. Module breakdown

### 1) Memory representation and storage

이 module은 memory가 논리적으로, 물리적으로 어떻게 저장되는지를 정의한다.

Logical representation은 다음을 포함할 수 있다.

- raw text stream
- key-value or schema facts
- vector memory
- knowledge graph
- tree hierarchy
- composite hybrid object

Physical storage는 in-context registers, vector DB, graph DB, keyword index, multi-engine backend가 될 수 있다.

Representation은 무엇을 retrieve할 수 있는지와 abstraction 과정에서 정보가 얼마나 손실되는지를 결정한다.

### 2) Memory extraction

Extraction은 raw interaction stream을 memory object로 바꾼다.

| Extraction strategy | Risk |
| --- | --- |
| Raw sequence concatenation | Recall은 높지만 cost와 noise가 큼 |
| Schema-free semantic extraction | 유연하지만 hallucination이나 detail 누락 위험이 있음 |
| Schema-constrained extraction | 더 일관적이지만 schema 밖의 fact를 놓칠 수 있음 |
| Fine-grained LLM extraction | Precision은 높일 수 있지만 multi-hop reasoning이 약해질 수 있음 |

논문은 abstraction layer가 추가될 때마다 information이 버려질 수 있음을 지적한다.

### 3) Retrieval and routing

Retrieval은 semantic kNN, keyword search, graph traversal, query planning, hybrid execution을 사용할 수 있다.

핵심 finding은 명시적인 query planning과 균형 잡힌 hybrid search가 contextual relevance를 높인다는 점이다. 하지만 evidence와 query 사이의 시간적 거리가 커질수록 retrieval accuracy는 떨어진다. 이는 similarity-based retrieval의 약점을 드러낸다.

### 4) Maintenance

Maintenance는 memory의 lifecycle을 관리한다.

| Submodule | Role |
| --- | --- |
| Conflict resolution and versioning | 충돌과 overwrite를 처리 |
| Capacity management | Memory를 evict하거나 우선순위를 조정 |
| Semantic consolidation | 중복 fact를 병합하거나 요약 |

논문은 graph-based method가 knowledge update를 더 안정적으로 처리하는 반면, append-only store는 오래된 fact를 반환할 수 있다고 본다. 필요한 부분만 고치는 localized maintenance는 전체를 다시 정리하는 방식보다 비용 효율적이다.

## 3-3. Memory architecture families

논문은 대표적인 memory architecture family를 다음처럼 정리한다.

| Family | Example style | Strength | Weakness |
| --- | --- | --- | --- |
| Stream and reflection | MemoryBank-like | Episodic stream과 summary 구조가 단순함 | Recall이 오래되거나 흐려질 수 있음 |
| Hierarchical tiered | MemGPT-like | Memory level과 이동 규칙이 명시적임 | 관리 복잡도가 큼 |
| Knowledge graph | Mem0g, Zep-like | Entity, relation, update 처리에 강함 | Temporal reasoning과 cost 문제가 있음 |
| Composite hybrid | A-MEM-like | Vector, graph, keyword, runtime state를 함께 route함 | System complexity가 큼 |

이 taxonomy가 유용한 이유는 memory system이 여러 technique을 섞는 경우가 많기 때문이다. Module decomposition은 component별 비교를 가능하게 한다.

# 4. Training / Data / Recipe

## 4-1. Evaluation data

논문은 11개 dataset에 걸친 5개 benchmark workload에서 평가한다. Workload는 conversational QA, factual recall, dynamic update, long-horizon reasoning, agentic execution scenario를 포함한다.

정확한 dataset list와 system별 결과는 재현 전에 paper table에서 확인해야 한다.

## 4-2. Evaluation axes

논문은 다섯 research question으로 평가한다.

| Axis | Question |
| --- | --- |
| Task effectiveness | Memory가 downstream task success를 실제로 개선하는가 |
| Retrieval fidelity | 저장된 evidence를 정확히 찾아오는가 |
| Dynamic update robustness | 충돌하거나 갱신된 knowledge를 올바르게 처리하는가 |
| Long-horizon stability | Evidence가 멀어진 뒤에도 memory가 유효하게 작동하는가 |
| Operational cost | Indexing time, query latency, maintenance cost가 어느 정도인가 |

이것이 논문의 main contribution이다. Memory evaluation을 하나의 final score가 아니라 system-level profile로 바꾼다.

## 4-3. Fine-grained ablation

논문은 module 하나씩 바꾸며 component-level effect를 측정한다.

예시는 다음과 같다.

- representation fidelity
- retrieval precision
- update correctness
- long-horizon stability
- localized maintenance vs global reorganization
- delayed flushing vs conservative consolidation

정확한 controlled variant는 original table에서 확인해야 한다.

## 4-4. Engineering notes

1. **Memory module을 분리해서 평가한다**
   - Final QA score만 보고하지 않는다.

2. **Temporal distance를 추적한다**
   - Evidence가 오래될수록 retrieval 성능이 나빠질 수 있다.

3. **Update semantics를 명시한다**
   - New fact가 old fact를 overwrite하는지, 대체하는지, 충돌하는지, version으로 보존하는지 명확해야 한다.

4. **Cost를 측정한다**
   - Production에서는 index build time과 query latency가 중요하다.

5. **Aggressive global consolidation을 기본값으로 두지 않는다**
   - Chronological cue를 잃을 수 있다.
   - 필요한 부분만 고치는 maintenance가 더 비용 효율적인 경우가 많다.

6. **Raw long context를 baseline으로 둔다**
   - Time-dependent query에서는 raw long-context retrieval이 memory-backed approach보다 나을 수 있다.

# 5. Evaluation

## 5-1. End-to-end findings

Headline result는 모든 상황에서 가장 좋은 단일 architecture가 없다는 것이다.

논문은 다음을 보고한다.

- Composite hybrid system은 conversational QA에서 앞선다.
- Graph-based method는 single-hop factual recall에서 강하다.
- Graph method는 temporal reasoning에서는 약할 수 있다.
- Effective memory system은 answer generation 전에 evidence localization을 외부화하므로 LLM backbone이 달라져도 비교적 안정적이다.

이는 memory architecture를 global leaderboard 순위가 아니라 workload에 맞춰 선택해야 함을 시사한다.

## 5-2. Retrieval fidelity

명시적인 query planning과 균형 잡힌 hybrid search는 contextual relevance를 높인다. 하지만 evidence와 query 사이의 시간적 거리가 커질수록 retrieval accuracy가 크게 떨어진다.

이는 가장 중요한 finding 중 하나다. Memory system은 evidence를 저장하고도, 그 evidence가 오래되거나 시간적으로 멀어지면 retrieve에 실패할 수 있다.

## 5-3. Dynamic updates

Graph-based method는 entity와 relation을 explicit하게 표현하기 때문에 knowledge update를 더 안정적으로 처리한다. Append-only store와 fact-extraction plugin은 targeted overwrite가 필요할 때 오래된 fact를 반환할 수 있다.

Production implication은 분명하다. Lifecycle management가 없는 memory는 과거 정보를 현재 사실처럼 되살릴 수 있다.

## 5-4. Long-horizon stability

많은 append-only memory store는 evidence가 멀어질수록 성능이 나빠진다. Time-dependent query에서는 raw long-context retrieval이 대부분의 memory-backed approach보다 나을 수 있다. 일반적인 semantic consolidation은 chronological cue를 파괴할 수 있다.

이는 시간 정보를 함부로 요약해 없애면 안 된다는 강한 경고다.

## 5-5. Operational cost

Highly structured system은 lightweight store보다 index construction time과 query latency가 몇 자릿수 더 클 수 있으며, 그에 비례한 accuracy gain을 주지 않을 수 있다. 필요한 부분만 고치는 localized maintenance는 전체를 다시 정리하는 방식보다 비용 효율적이다.

따라서 memory architecture는 accuracy만의 선택이 아니라 비용 대비 성능 설계 문제다.

## 5-6. What really matters in the experiments

### 1) Workload alignment

Memory system은 bottleneck에 맞춰 선택해야 한다. Temporal reasoning, factual recall, personalization, multi-hop synthesis는 서로 다른 structure를 요구한다.

### 2) Update correctness를 핵심 평가 요소로 봐야 한다

오래된 fact를 교체하거나 version으로 보존하지 못하는 memory는 유용하지 않다.

### 3) 시간 정보는 쉽게 손상된다

Summary와 semantic extraction은 chronology를 약하게 만들 수 있다. 하지만 많은 agent task는 시간에 의존한다.

### 4) 비용이 전체 판단을 좌우할 수 있다

Production memory에는 task score뿐 아니라 latency와 maintenance budget도 필요하다.

# 6. Limitations

1. **Benchmark coverage가 넓지만 완전하지는 않다**
   - 5개 workload와 11개 dataset이 모든 production agent workload를 포괄하지는 않는다.

2. **System이 빠르게 변한다**
   - Mem0, Zep, Letta, A-MEM 같은 memory system은 빠르게 변한다.

3. **Implementation choice가 중요하다**
   - 특정 memory family의 구현이 부실하면 그 family의 가능성을 과소평가할 수 있다.

4. **LLM backbone dependency가 남아 있다**
   - System이 evidence를 외부화하더라도 answer quality는 backbone에 의존한다.

5. **Cost measurement는 환경 의존적이다**
   - Latency와 index time은 hardware, API, database setting, caching에 의존한다.

6. **Privacy와 deletion 논의가 부족하다**
   - Agent memory production에는 retention policy와 right-to-delete support가 필요하다.

7. **Security risk**
   - Persistent memory는 prompt injection, poisoned fact, sensitive data를 저장할 수 있다.

8. **Human preference와 utility**
   - Benchmark QA는 user trust와 personalization quality를 완전히 포착하지 못한다.

9. **Parametric memory는 깊게 다루지 않는다**
   - 이 논문의 focus는 external memory system이다.

10. **Universal recipe는 없다**
   - 논문은 taxonomy와 finding을 제공하지, 바로 채택할 하나의 architecture를 제안하지는 않는다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 중요한 메시지는 "graph memory가 좋다"나 "hybrid memory가 좋다"가 아니다. 핵심은 **agent memory를 retrieval trick이 아니라 lifecycle-governed data system으로 봐야 한다는 점**이다.

Agent memory는 single vector index보다 database, query planner, versioned knowledge layer가 결합된 system에 더 가깝다.

이 관점은 production agent에 직접 중요하다.

## 7-2. Reuse potential

### Personal assistant memory

User preference는 시간에 따라 바뀐다. Memory system은 versioning, conflict resolution, deletion, scoped retrieval을 지원해야 한다.

### Coding agent memory

Repository convention, failed attempt, test result, issue history는 서로 다른 memory representation이 필요하다. Raw log, semantic fact, graph relation, procedural memory가 모두 중요하다.

### Enterprise agent memory

Policy, workflow, tool output, user feedback은 update correctness와 audit log를 요구한다. 오래된 policy가 active하게 남으면 append-only memory는 위험할 수 있다.

### Research agent memory

Experiment history와 claim revision은 chronology에 의존한다. Aggressive summarization은 중요한 causal order를 파괴할 수 있다.

## 7-3. Production architecture

실용적인 agent memory는 다음 layer를 포함해야 한다.

| Layer | Role |
| --- | --- |
| Raw event log | 전체 chronology를 보존 |
| Semantic memory | Fact와 preference를 추출 |
| Graph memory | Entity, relation, version을 추적 |
| Vector index | Fuzzy context를 retrieve |
| Query planner | Workload type에 따라 route |
| Maintenance policy | Update, consolidate, evict, delete 수행 |
| Audit log | 어떤 memory가 사용됐는지 설명 |

논문은 이 hybrid view를 지지한다. 하지만 hybrid system은 비용을 통제할 수 있어야 한다. Complexity만으로 accuracy가 보장되지는 않는다.

## 7-4. Follow-up papers

- Mem0
- MemGPT / Letta
- Zep
- A-MEM
- MemoryBank
- LongMemEval
- LoCoMo
- EvoArena / EvoMem
- GraphRAG
- Agent memory system surveys

# 8. Summary

- Agent memory는 data management infrastructure로 평가해야 한다.
- 논문은 memory를 representation/storage, extraction, retrieval/routing, maintenance로 분해한다.
- 12개 memory system과 2개 baseline을 5개 workload, 11개 dataset에서 평가한다.
- 어떤 architecture도 모든 상황에서 가장 좋지는 않으며, workload bottleneck이 적절한 design을 결정한다.
- Temporal reasoning, dynamic update, operational cost가 핵심 stress test다.
