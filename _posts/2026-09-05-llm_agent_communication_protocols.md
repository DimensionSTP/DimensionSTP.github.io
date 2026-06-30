---
layout: single
title: "A Technical Taxonomy of LLM Agent Communication Protocols Review"
categories: Study-concept
tag: [AgentProtocol, MultiAgentSystem, Interoperability, A2A, AgentInfrastructure]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.19135)

A Technical Taxonomy of LLM Agent Communication Protocols는 "agent protocol survey"로만 읽으면 너무 얕게 읽게 되는 논문이다. 이 논문이 실제로 다루는 문제는 agent가 서로 말할 수 있느냐가 아니라, **agent ecosystem이 서로 다른 communication contract를 어떤 축으로 분해하고 비교해야 하는가**다.

LLM agent stack은 빠르게 protocol화되고 있다. Agent가 tool을 부르고, external context를 읽고, 다른 agent에게 task를 넘기고, session state를 유지하고, schema를 negotiate하는 일이 점점 많아진다. 하지만 현재 protocol landscape는 매우 fragment되어 있다. 어떤 protocol은 agent-to-agent communication을 강조하고, 어떤 protocol은 agent-to-tool 또는 agent-to-context connection을 강조한다. 어떤 protocol은 session state를 강하게 갖고, 어떤 protocol은 stateless request-response에 가깝다. 어떤 protocol은 fixed schema를 쓰고, 어떤 protocol은 runtime schema negotiation을 지원한다.

이 논문은 이런 차이를 marketing name이나 product boundary가 아니라 taxonomy로 정리한다. 저자들은 actively maintained open-source protocol 9개를 대상으로 iterative taxonomy method를 적용하고, 5개 dimension을 제시한다.

1. Counterparty
2. Payload
3. Interaction state
4. Discovery mechanism
5. Schema flexibility

> 한 줄 요약: 이 논문은 LLM agent communication protocol을 counterparty, payload, interaction state, discovery mechanism, schema flexibility의 5차원 taxonomy로 분해하고, sampled protocol들이 short-term에는 agent-to-agent와 agent-to-context communication을 통합하는 압력을 받지만 long-term에는 federated layered protocol stack으로 갈 가능성이 높다고 분석한 protocol infrastructure 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Agent protocol을 "표준이 필요하다"는 구호가 아니라 비교 가능한 technical dimension으로 분해한다.
- Agent-to-agent와 agent-to-context, 즉 tool/data communication이 장기적으로 어떻게 수렴하거나 분리될지 생각하게 한다.
- Payload, session state, schema flexibility처럼 실제 implementation에서 문제가 되는 축을 taxonomy 안에 넣는다.
- Sampled agent-to-agent protocol들이 hybrid payload와 session-state persistence를 함께 쓰는 구체적 pattern을 보고한다.
- Decentralized discovery가 아직 드물다는 점을 짚어 agent network infrastructure의 빈 구역을 드러낸다.
- Versatility, efficiency, portability를 동시에 최대화하는 단일 protocol은 나오기 어렵다는 현실적인 결론을 제시한다.

이 글에서는 이 논문을 "어떤 agent protocol을 써야 하나"보다, **agent protocol을 선택하거나 설계할 때 무엇을 비교해야 하는가를 정리한 technical taxonomy**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Multi-agent LLM system에서 communication protocol은 단순 transport가 아니다. Protocol은 agent가 무엇을, 누구에게, 어떤 state에서, 어떤 schema로 말할 수 있는지와 상대 endpoint를 어떻게 발견하는지를 정의한다.

Protocol $P$를 다음의 5차원 design point로 볼 수 있다.

$$
P
=
(C, L, S, D, F)
$$

| Symbol | Dimension | Meaning |
| --- | --- | --- |
| $C$ | Counterparty | 누가 누구와 통신하는가 |
| $L$ | Payload | 어떤 정보가 교환되는가 |
| $S$ | Interaction state | session이 유지되는가, 어떻게 유지되는가 |
| $D$ | Discovery mechanism | 상대 endpoint를 어떻게 찾는가 |
| $F$ | Schema flexibility | message schema가 얼마나 고정적이거나 협상 가능한가 |

이 분해가 중요한 이유는 agent protocol failure가 한 축에서만 생기지 않기 때문이다.

- Counterparty가 불명확하면 agent-to-agent와 agent-to-tool boundary가 섞인다.
- Payload가 충분히 명시되지 않으면 instruction, tool call, artifact, provenance가 모두 free text로 흘러간다.
- State model이 약하면 long-running collaboration, delegation, recovery가 어렵다.
- Discovery가 중앙화되어 있으면 open agent network가 형성되기 어렵다.
- Schema가 rigid하면 interoperability는 쉬워지지만 task diversity가 줄고, flexible schema는 portability와 validation cost를 높인다.

## 1-2. Why previous approaches are insufficient

### 1) Protocol 이름만으로 technical compatibility를 알 수 없다

두 protocol이 모두 agent communication을 지원한다고 주장하더라도 state model, schema negotiation, payload type, discovery model은 완전히 다를 수 있다. 따라서 이름이나 adoption만으로 compatibility를 추론할 수 없다.

### 2) 기존 survey category는 너무 거칠 수 있다

하나의 protocol이 agent-to-agent와 agent-to-context communication을 동시에 지원할 수 있다. Natural-language message와 structured tool schema를 함께 운반할 수도 있고, 어떤 flow에서는 sessionful이지만 다른 flow에서는 stateless일 수도 있다. 따라서 유용한 taxonomy에는 서로 직교하는 여러 dimension이 필요하다.

### 3) Design axis 없이 standardization을 논의하면 이르다

하나의 standard를 주장하기 전에, field는 어떤 trade-off를 표준화하려는지 먼저 알아야 한다. Versatility, efficiency, portability는 서로 충돌할 수 있다. 단일 universal protocol이 최적이라는 보장은 없다.

### 4) Runtime governance가 충분히 명시되어 있지 않다

Agent가 organization boundary나 trust boundary를 넘어서 통신한다면 privacy, policy enforcement, schema negotiation, discovery는 선택 사항이 아니다. Protocol taxonomy는 이런 gap을 드러내야 한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 기여는 다음과 같다.

1. **Technical taxonomy**
   - LLM agent communication protocol을 분류하기 위한 5개 dimension을 제안한다.
   - Empirical-to-conceptual, conceptual-to-empirical refinement를 반복해 taxonomy를 만든다.

2. **Classification of nine protocols**
   - Demonstrable adoption이 있고 actively maintained되는 open-source protocol에 집중한다.
   - 반복해서 나타나는 architectural pattern을 식별한다.

3. **Convergence analysis**
   - Short-term에는 agent-to-agent와 agent-to-context communication을 통합하려는 압력이 있다.
   - Long-term에는 하나의 universal protocol보다 federated layered protocol stack이 예상된다.

4. **Research gap identification**
   - Decentralized discovery는 아직 드물다.
   - Privacy와 policy enforcement에는 더 명시적인 protocol support가 필요하다.

## 2-2. Design intuition

핵심 직관은 agent communication protocol이 library detail이 아니라 infrastructure가 되고 있다는 점이다.

Protocol은 최소한 다섯 질문에 답해야 한다.

```text
누가 말하는가?
무엇이 교환되는가?
어떤 state가 유지되는가?
Discovery는 어떻게 일어나는가?
Schema는 어떻게 합의되는가?
```

이 질문이 implicit하게 남으면 multi-agent system은 brittle해진다. 하나의 framework 안에서는 작동해도 vendor, team, language, runtime, trust domain을 넘으면 실패할 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | LLM agent communication protocol을 technical dimension으로 분류 |
| Method | Iterative taxonomy development |
| Sample | Adoption이 확인되는 actively maintained open-source protocol 9개 |
| Dimensions | Counterparty, payload, interaction state, discovery, schema flexibility |
| Main observed pattern | A2A protocol은 hybrid payload와 session-state persistence를 결합하는 경향 |
| Short-term forecast | Agent-to-agent와 agent-to-context communication을 통합하려는 압력 |
| Long-term forecast | Federated layered protocol stack |

## 3-2. Dimension breakdown

### 1) Counterparty

Counterparty는 endpoint type을 정의한다.

가능한 category는 다음을 포함한다.

- Agent-to-agent
- Agent-to-tool
- Agent-to-context
- Agent-to-human
- Agent-to-runtime
- Hybrid endpoint

논문은 short-term convergence pressure가 agent-to-agent와 agent-to-context communication을 통합하려는 protocol에서 나온다고 본다.

### 2) Payload

Payload는 natural-language text, structured data, tool invocation, artifact, embedding, state delta, policy, hybrid message가 될 수 있다.

Text만 보내는 protocol은 단순하지만 reliable automation에는 약하다. Structured payload를 갖는 protocol은 validation이 더 쉽지만 schema discipline을 요구한다.

Hybrid payload가 중요한 이유는 agent가 language와 structured action을 모두 필요로 하기 때문이다.

### 3) Interaction state

Interaction은 stateless일 수도 있고 session-stateful일 수도 있다.

Session-state persistence가 중요한 경우는 다음과 같다.

- delegated task
- long-running workflow
- multi-turn negotiation
- failure 이후의 recovery
- context와 artifact continuity
- policy-scoped interaction

논문은 sampled agent-to-agent protocol이 모두 hybrid payload와 session-state persistence를 결합한다고 보고한다.

### 4) Discovery mechanism

Discovery는 agent가 다른 agent, tool, context server, capability provider를 어떻게 찾는지를 답한다.

Discovery 방식은 다음이 될 수 있다.

- manually configured
- centralized registry based
- directory or manifest based
- decentralized
- runtime negotiated

논문은 decentralized discovery가 아직 드물다고 지적한다. Open agent network를 원한다면 app-specific integration을 넘어서는 discovery layer가 필요하다.

### 5) Schema flexibility

Schema flexibility는 message가 어떻게 구조화되는지를 결정한다.

| Schema model | Strength | Weakness |
| --- | --- | --- |
| Fixed schema | Validation과 portability가 쉬움 | Task diversity가 제한됨 |
| Multiple predefined schemas | 실용적인 절충안 | Versioning complexity가 커짐 |
| Runtime negotiation | 유연하고 adaptive함 | Validation과 security가 어려움 |
| Fully open text | Flexibility가 가장 큼 | Interoperability가 약함 |

논문은 sampled protocol 대부분이 multiple predefined schemas를 지원하고, 두 protocol은 runtime에서 schema를 negotiate한다고 보고한다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 model training paper가 아니다. Data는 protocol sample이다. Demonstrable adoption이 있는 actively maintained open-source protocol 9개를 분석한다.

논문은 established iterative taxonomy development method를 사용한다. Taxonomy purpose, meta-characteristic, ending condition을 정의하고 5번의 iteration을 수행한다.

| Iteration type | Count |
| --- | ---: |
| Empirical-to-conceptual | 3 |
| Conceptual-to-empirical | 2 |
| Total | 5 |

## 4-2. Analysis strategy

분석은 다음처럼 진행된다.

1. Purpose와 meta-characteristic을 정의한다.
2. Activity와 adoption criteria를 만족하는 protocol을 선택한다.
3. 관찰 가능한 protocol property를 추출한다.
4. Taxonomy dimension을 반복적으로 만들고 refine한다.
5. Protocol을 taxonomy에 따라 classify한다.
6. Protocol 간 pattern과 gap을 식별한다.
7. Short-term과 long-term architectural implication을 도출한다.


## 4-3. Engineering notes

1. **Protocol을 popularity만으로 고르지 않는다**
   - Counterparty, payload, state, discovery, schema requirement에 mapping해야 한다.

2. **A2A와 A2C requirement를 구분한다**
   - Tool/context protocol과 agent collaboration protocol은 겹치지만 동일하지 않다.

3. **State model을 주의 깊게 본다**
   - Stateless transport만으로 long-running agent collaboration은 부족하다.

4. **Schema negotiation을 governance issue로 다룬다**
   - Flexible schema에는 validation, versioning, policy control이 필요하다.

5. **Discovery는 scaling bottleneck이다**
   - Discovery가 없으면 multi-agent network는 사람이 직접 연결한 상태에 머문다.

6. **Unification보다 layering이 나을 수 있다**
   - Universal protocol보다 federated layered stack이 더 현실적일 수 있다.

# 5. Evaluation

## 5-1. Main taxonomy result

Taxonomy는 다섯 dimension을 갖는다.

| Dimension | Main question |
| --- | --- |
| Counterparty | 누가 누구와 통신하는가 |
| Payload | 무엇을 운반하는가 |
| Interaction state | Session persistence가 있는가 |
| Discovery mechanism | 상대 endpoint를 어떻게 찾는가 |
| Schema flexibility | Schema가 얼마나 고정적이거나 협상 가능한가 |

이것이 논문에서 가장 재사용 가능한 artifact다. 엔지니어링 팀이 protocol selection checklist로 사용할 수 있다.

## 5-2. Observed protocol patterns

논문은 다음을 보고한다.

- Sampled agent-to-agent protocol은 모두 hybrid payload와 session-state persistence를 결합한다.
- 대부분의 sampled protocol은 multiple predefined schema를 지원한다.
- 두 protocol은 runtime에서 schema를 negotiate한다.
- Decentralized discovery는 여전히 드물다.
- Short-term에는 agent-to-agent와 agent-to-context communication을 통합하려는 압력이 있다.
- Long-term에는 versatility, efficiency, portability를 동시에 최대화하는 단일 protocol이 나오기 어렵다.

## 5-3. What really matters in the experiments

### 1) Protocol choice는 workload-dependent다

Local tool calling용 protocol과 inter-organizational agent delegation용 protocol은 우선순위가 다르다.

### 2) Schema flexibility는 양날의 검이다

Runtime schema negotiation은 generality를 높이지만 policy, validation, security 부담도 키운다.

### 3) Discovery는 아직 덜 발달했다

Protocol ecosystem은 아직 integration-centric하다. Open decentralized agent network에는 더 강한 discovery layer가 필요하다.

### 4) 단일 protocol보다 federated stack이 더 그럴듯하다

이는 냉정한 결론이다. Agent infrastructure는 하나의 거대한 universal standard보다 internet protocol layering에 더 가까울 가능성이 높다.

# 6. Limitations

1. **표본 크기가 작다**
   - 9개 protocol을 분석한다.
   - Protocol landscape는 빠르게 변하고 있다.

2. **Open-source와 adoption filter가 있다**
   - Closed commercial protocol은 충분히 반영되지 않을 수 있다.

3. **Taxonomy는 성능 벤치마크가 아니다**
   - 논문은 design을 classify하지만 latency, security, reliability, usability를 직접 벤치마크하지 않는다.

4. **정식 interoperability test는 없다**
   - Classification이 두 protocol의 interoperability를 증명하지는 않는다.

5. **Security와 privacy는 여전히 gap이다**
   - 논문은 이를 open research issue로 식별하지만 해결하지는 않는다.

6. **Protocol semantics는 바뀔 수 있다**
   - Maintained protocol은 논문 시점 이후에도 변할 수 있다.

7. **Implementation quality가 중요하다**
   - 같은 taxonomy cell 안에서도 engineering quality는 크게 다를 수 있다.

8. **Layered stack 전망은 예측이지 증명은 아니다**
   - Long-term federated stack 주장은 근거 있는 분석이지, 반드시 그렇게 된다는 경험적 증명은 아니다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 핵심은 "어떤 agent protocol이 이길까"가 아니다. 더 중요한 점은 **agent communication을 interface design, state management, schema governance, discovery problem으로 나눠 봐야 한다는 것**이다.

Agent system은 distributed system이 되고 있다. 여러 agent, tool, memory server, evaluation service, user proxy가 상호작용하면 communication contract는 product boundary이자 safety boundary가 된다.

## 7-2. Reuse potential

### Agent platform design

Internal agent platform을 만든다면 5개 dimension을 protocol selection matrix로 사용할 수 있다.

| Need | Relevant dimension |
| --- | --- |
| Multi-agent delegation | Counterparty, state |
| Tool and data integration | Payload, schema |
| Workflow recovery | Interaction state |
| Plugin ecosystem | Discovery |
| Enterprise governance | Schema flexibility, policy enforcement |

### Evaluation and audit

Protocol log도 같은 dimension으로 audit할 수 있다. 어떤 state가 persisted되었는가? 어떤 payload type이 trust boundary를 넘었는가? 어떤 schema가 accepted되었는가? Counterparty는 어떻게 discovered되었는가?

### Standardization discussion

이 taxonomy는 막연한 "agent protocol이 필요하다" 논쟁을 피하게 한다. 올바른 질문은 어떤 layer를 standardize하고 어떤 layer를 flexible하게 둘 것인가다.

## 7-3. Production considerations

- Explicit session identity와 lifecycle을 요구한다.
- Tool call과 artifact에는 typed payload를 사용한다.
- Negotiation과 explanation에는 natural language payload를 유지하되, 모든 machine action을 free text로 두지 않는다.
- Schema negotiation을 security-sensitive한 작업으로 다룬다.
- Discovery path와 trust metadata를 기록한다.
- Cross-boundary delegation 전에 policy enforcement hook을 둔다.
- Protocol schema와 agent capability manifest를 versioning한다.

## 7-4. Follow-up papers

- A Survey of AI Agent Protocols
- LLM Agent Communication Protocol position papers
- Model Context Protocol documentation
- Agent-to-Agent protocol documentation
- OpenAPI, JSON Schema, AsyncAPI
- Distributed systems protocol design papers
- Capability discovery and service registry literature

# 8. Summary

- 이 논문은 LLM agent communication protocol을 위한 5차원 taxonomy를 제안한다.
- Dimension은 counterparty, payload, interaction state, discovery mechanism, schema flexibility다.
- Sampled A2A protocol은 hybrid payload와 session-state persistence를 결합하는 경향이 있다.
- Decentralized discovery는 아직 드물다.
- 하나의 universal protocol보다 federated layered protocol stack이 더 그럴듯하다.
