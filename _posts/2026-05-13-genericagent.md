---
layout: single
title: "GenericAgent Review"
categories: Study-concept
tag: [LLM, Agent, Memory, GenericAgent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.17091)

[Project link](https://github.com/lsdefine/GenericAgent)

GenericAgent는 agent를 더 큰 context window로 밀어 넣는 논문이라기보다, 제한된 context 안에 무엇을 남길 것인가를 시스템 설계 원칙으로 정리한 논문이다. 논문 제목에 있는 Contextual Information Density Maximization이라는 표현이 핵심이다. 긴 작업에서 agent가 망가지는 이유를 단순히 context length 부족으로 보지 않고, context 안에 decision-relevant information이 얼마나 높은 밀도로 유지되는가의 문제로 본다.

이 관점은 꽤 실무적이다. 실제 agent 시스템은 tool description, browser DOM, terminal output, retrieved memory, intermediate error log, user instruction, generated script, historical skill 같은 요소가 빠르게 쌓인다. 문제는 이 모든 정보가 길어진다는 점만이 아니다. 더 큰 문제는 중요한 정보와 노이즈가 같은 window 안에 섞이면서, 다음 action을 고르는 LLM의 입력 품질이 떨어진다는 점이다.

GenericAgent는 이 문제를 네 가지 장치로 푼다. 최소 원자 도구 집합, 계층적 on-demand memory, 검증된 trajectory를 SOP와 code로 바꾸는 self-evolution, 그리고 long execution 중 context를 자르고 압축하는 layer다. 흥미로운 점은 이 네 장치가 따로 노는 feature가 아니라, 모두 context information density라는 하나의 목표에 연결된다는 점이다.

> 한 줄 요약: GenericAgent는 long-horizon LLM agent의 병목을 context length가 아니라 context information density로 정의하고, minimal tools, hierarchical memory, self-evolution, truncation and compression을 결합해 token-efficient self-evolving agent를 만드는 시스템 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- agent 경쟁이 prompt와 tool 수 싸움에서 memory, skill reuse, context management 싸움으로 넘어가고 있다.
- long-horizon task에서 더 많은 token을 쓰는 것이 항상 더 좋은 reasoning으로 이어지지 않는다는 점을 명확히 드러낸다.
- 실무 agent를 만들 때 필요한 memory hierarchy, SOP crystallization, tool surface minimization, browser and OS control의 trade-off를 한 번에 볼 수 있다.

GenericAgent의 핵심 메시지는 "agent에게 더 많은 것을 보여주자"가 아니라 "agent가 지금 결정을 내리는 데 필요한 정보만 고밀도로 유지하자"다. 이 관점은 RAG, memory agent, browser agent, coding agent 모두에 꽤 직접적으로 옮겨갈 수 있다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 문제는 long-horizon LLM agent가 긴 상호작용 속에서 context를 어떻게 유지하고 재사용할 것인가이다.
- 긴 작업에서는 tool call 결과, 오류 로그, 웹 페이지 내용, 과거 메모리, 임시 파일 상태, user preference가 계속 누적된다.
- 단순히 context window를 늘리면 일부 문제는 완화되지만, active context 안의 signal-to-noise ratio가 낮아지는 문제는 그대로 남는다.
- agent는 매 step마다 LLM forward pass 하나로 다음 action을 고른다. 따라서 그 시점의 context가 noisy하면 tool이 많거나 memory가 커도 decision quality가 떨어진다.
- GenericAgent는 이 문제를 context information density optimization으로 재정의한다.

논문 관점에서 long-horizon agent의 핵심 병목은 두 가지다.

1. Context pollution
   - tool output과 raw feedback이 계속 들어와서 중요한 instruction과 state를 밀어낸다.
   - 긴 context는 더 많은 정보를 담지만, 동시에 더 많은 noise와 stale state를 담는다.

2. Experience non-reuse
   - 이전 episode에서 얻은 경험이 다음 episode에 구조적으로 재사용되지 않는다.
   - 같은 종류의 task를 반복해도 매번 탐색, 설치, 디버깅, browser navigation을 다시 수행한다.

결국 이 논문이 묻는 질문은 다음과 같다.

- agent는 어떻게 현재 작업에 필요한 정보만 남길 수 있는가?
- 과거 성공 trajectory를 어떻게 reusable procedure로 바꿀 수 있는가?
- tool surface를 줄이면서도 agent의 실행 범위를 넓게 유지할 수 있는가?
- context budget이 유한할 때, memory를 어떻게 넣고 빼야 하는가?

## 1-2. Why previous approaches are insufficient

기존 agent 시스템의 한계는 크게 네 가지로 볼 수 있다.

1. Tool-heavy design
   - 많은 agent framework는 task별 tool을 많이 붙이는 방식으로 확장된다.
   - tool이 많아지면 tool description 자체가 context를 차지하고, 매 step의 action selection space도 커진다.
   - 즉 tool 수 증가는 capability를 늘릴 수 있지만, decision cost도 같이 늘린다.

2. Flat memory injection
   - memory를 만들더라도 중요한 정보와 덜 중요한 정보를 한 번에 context에 넣는 경우가 많다.
   - 이런 방식은 memory가 커질수록 active context를 잠식한다.
   - memory는 많아지지만, 정작 지금 필요한 memory를 찾는 비용도 커진다.

3. Stateless repetition
   - coding agent나 browser agent가 한 task에서 얻은 workflow를 다음 task에 안정적으로 재사용하지 못하면, 매번 처음부터 탐색한다.
   - 이 경우 token cost는 task 수에 따라 거의 선형으로 증가한다.
   - 더 큰 문제는 반복 탐색 과정에서 새로운 오류가 계속 생긴다는 점이다.

4. Context length as a proxy for intelligence
   - 많은 시스템이 더 긴 context를 넣으면 더 똑똑해질 것처럼 설계된다.
   - 하지만 long-horizon execution에서는 더 긴 context가 오히려 stale state, duplicated logs, irrelevant DOM, old reasoning을 함께 보존한다.
   - GenericAgent는 이 점에서 context length보다 context information density가 더 본질적이라고 주장한다.

이 문제 설정은 최근 agent 연구에서 꽤 중요한 전환이다. 모델을 더 크게 하거나 context window를 더 늘리는 것보다, agent system이 LLM에게 어떤 상태 표현을 제공하는가가 성능과 비용을 동시에 좌우하기 때문이다.

# 2. Core Idea

## 2-1. Main contribution

GenericAgent의 핵심 기여는 새로운 foundation model이 아니라, self-evolving LLM agent를 위한 system architecture다. 논문이 제안하는 중심 원칙은 다음과 같다.

> Long-horizon performance is determined not by raw context length, but by how much decision-relevant information is maintained within a finite context budget.

이를 구현하기 위해 GenericAgent는 네 가지 구성 요소를 결합한다.

1. Minimal atomic tool set
   - task-specific tool을 많이 늘리는 대신, 파일, 코드 실행, 웹 제어, memory 관리, human-in-the-loop 같은 원자 기능만 둔다.
   - 복잡한 능력은 tool 자체가 아니라 tool composition과 learned skill로 만든다.

2. Hierarchical on-demand memory
   - 모든 memory를 active context에 넣지 않는다.
   - L1 index처럼 작은 high-level view만 기본으로 보여주고, 필요할 때 더 깊은 memory를 retrieve한다.

3. Self-evolution through SOP and code
   - 성공하거나 검증된 execution trajectory를 reusable SOP와 executable code로 바꾼다.
   - agent가 같은 작업을 다시 할 때 raw trajectory를 다시 읽는 것이 아니라, 압축된 procedure를 호출한다.

4. Context truncation and compression
   - tool output과 오래된 message를 무작정 유지하지 않는다.
   - active context가 작업 진행 중에도 일정한 정보 밀도를 유지하도록 truncation, compression, eviction, working memory anchor를 사용한다.

## 2-2. Design intuition

GenericAgent의 설계 직관은 단순하지만 강하다. LLM agent의 각 step은 결국 "현재 context를 보고 다음 action을 고르는 문제"다. 그러면 agent architecture의 목적은 더 많은 정보를 모으는 것이 아니라, 다음 action selection에 필요한 정보를 더 잘 정리해서 제공하는 것이 된다.

이 관점에서 각 module은 다음 역할을 한다.

- Minimal tools는 action space를 줄인다.
- Hierarchical memory는 knowledge space를 압축한다.
- Self-evolution은 repeated trajectory를 reusable program으로 바꾼다.
- Context compression은 active context의 noise를 줄인다.

중요한 점은 self-evolution이 모델 weight update가 아니라는 점이다. GenericAgent에서 진화하는 것은 base LLM의 parameter가 아니라, agent가 사용하는 skill, SOP, script, memory structure다. 그래서 이 논문은 reinforcement learning 논문이라기보다, agent runtime이 경험을 어떻게 외부 구조물로 축적하는지에 대한 systems paper에 가깝다.

내 해석으로는 GenericAgent가 제안하는 self-evolution은 다음과 같은 pipeline이다.

1. 처음 보는 task에서는 탐색한다.
2. tool을 호출하고, 오류를 만나고, recovery를 수행한다.
3. 성공한 path와 중요한 debugging knowledge를 memory에 남긴다.
4. 반복 가능한 부분을 SOP나 script로 정리한다.
5. 다음 유사 task에서는 raw exploration 대신 skill을 호출한다.

이 과정에서 context information density는 두 번 올라간다. 첫째, active context에는 필요한 memory만 올라온다. 둘째, 과거 경험은 긴 transcript가 아니라 짧은 procedure로 압축된다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Long-horizon task에서 token cost와 interaction cost를 줄이면서 agent의 task completion 능력을 유지하거나 높이는 것 |
| Central principle | Contextual information density maximization |
| Key modules | Minimal atomic tool set, layered memory, self-evolution, context truncation and compression |
| What evolves | LLM weight가 아니라 SOP, scripts, skills, memory structure |
| Difference from prior work | 더 많은 specialized tool을 넣기보다, 작은 tool surface와 memory hierarchy로 experience reuse를 만든다 |

## 3-2. Module breakdown

### 1) Minimal atomic tool set

GenericAgent는 많은 task-specific tool을 미리 넣는 대신, 작은 원자 도구 집합을 사용한다. 공개 README와 기술 리포트 요약 기준으로 핵심은 다음과 같다.

- file operation
- code execution
- web interaction
- memory management
- human-in-the-loop confirmation

GitHub README에서는 GenericAgent가 약 3K lines의 core code와 약 100-line agent loop를 갖고, 9 atomic tools를 통해 browser, terminal, filesystem, keyboard and mouse, screen vision, mobile device via ADB까지 다룬다고 설명한다.

여기서 중요한 것은 "도구가 적다"가 아니라 "도구의 abstraction level이 낮다"는 점이다. 예를 들어 특정 website용 tool을 계속 추가하는 대신, web scan과 browser control을 이용해 agent가 직접 procedure를 만들게 한다. 특정 API workflow를 hard-code하는 대신, code execution과 file write를 통해 script를 생성하고 이를 skill로 남긴다.

이 설계는 두 가지 trade-off를 만든다.

- 장점: tool description이 짧고, 새로운 task에 대한 compositional generalization이 가능하다.
- 단점: 첫 실행에서는 exploration cost가 커질 수 있고, tool 사용의 안전성, 권한, sandbox 정책이 중요해진다.

내가 보기엔 GenericAgent의 minimal tool design은 ReAct-style tool use의 반대 방향이라기보다, tool granularity를 낮추는 접근이다. agent에게 많은 완제품 tool을 주는 대신, 적은 수의 primitive를 주고 skill을 직접 만들게 한다.

### 2) Hierarchical on-demand memory

GenericAgent의 memory는 flat buffer가 아니라 계층 구조다. 공개 README 기준으로 다음과 같은 layer가 설명된다.

| Layer | Role | Blog interpretation |
| --- | --- | --- |
| L0 | Meta Rules | agent의 core behavioral rule과 system constraint |
| L1 | Insight Index | 빠른 routing과 recall을 위한 작은 index |
| L2 | Global Facts | 장기적으로 안정적인 factual memory |
| L3 | Task Skills and SOPs | 특정 task type을 해결하는 reusable workflow |
| L4 | Session Archive | 완료된 session에서 나온 archive record |

이 구조에서 핵심은 L1이다. 모든 memory를 context에 넣지 않고, L1에는 어떤 memory가 존재하는지에 대한 pointer만 둔다. agent는 L1을 보고 필요한 경우 L2, L3, L4를 더 깊게 retrieve한다.

이 방식은 RAG의 index-retrieve pattern과 비슷해 보이지만, agent memory에서는 의미가 조금 다르다. RAG는 주로 외부 knowledge를 찾기 위한 장치지만, GenericAgent의 memory는 agent 자신의 경험을 축적하고 재사용하기 위한 장치다. 즉 검색 대상은 일반 문서가 아니라, 과거 실행에서 검증된 fact, procedure, recovery pattern이다.

좋은 점은 memory growth가 active context growth로 직결되지 않는다는 것이다. L2와 L3가 커져도 L1이 bounded하게 유지되면, agent는 여전히 작은 context로 route를 결정할 수 있다. 이는 memory를 많이 쌓아도 다음 decision step의 입력을 과도하게 오염시키지 않는다는 점에서 중요하다.

### 3) Self-evolution through SOP and executable code

GenericAgent에서 self-evolution은 명시적이고 inspectable한 process다. 진화하는 대상은 원자 tool 자체가 아니라, task-specific strategy다. 성공한 trajectory는 다음과 같은 형태로 축적된다.

- reusable SOP
- precondition and checklist
- known failure mode
- debugging and recovery strategy
- executable script
- saved skill entry

이 구조가 중요한 이유는 raw trajectory를 그대로 저장하는 것보다 훨씬 밀도가 높기 때문이다. 예를 들어 처음에는 20만 token을 쓰면서 browser navigation, package installation, code writing, debugging을 수행할 수 있다. 하지만 다음에는 그 결과를 SOP 또는 script로 호출하면 훨씬 적은 token으로 동일한 workflow를 실행할 수 있다.

즉 self-evolution은 다음과 같은 compression이다.

- transcript to SOP
- exploration to reusable workflow
- repeated reasoning to executable code
- noisy debugging log to recovery rule

이 부분은 agent system에서 매우 중요하다. 인간 개발자도 매번 같은 작업을 처음부터 reasoning하지 않는다. shell script, notebook template, checklist, internal wiki, runbook으로 경험을 외부화한다. GenericAgent는 이 외부화를 agent runtime 내부의 memory and skill mechanism으로 구현하려는 시도다.

### 4) Context truncation and compression

GenericAgent는 context를 늘리는 대신 active context를 관리한다. 공개 기술 리포트 요약 기준으로 다음 네 가지 granularity의 pruning이 언급된다.

| Mechanism | Purpose |
| --- | --- |
| Tool output truncation | 단일 tool output이 context를 과도하게 차지하지 않도록 제한 |
| Tag-level compression | 오래된 message에서 낮은 가치의 fragment 제거 |
| Message eviction | 전체 budget이 초과되면 오래된 message 제거 |
| Working memory anchor | eviction 후에도 task-critical information이 보이도록 유지 |

이 설계는 agent가 긴 작업을 할 때 특히 중요하다. browser agent는 HTML, DOM, screenshot description, search result, form state 같은 정보를 매우 빠르게 늘린다. coding agent는 traceback, generated code, patch history, dependency installation log가 계속 쌓인다. 이런 정보를 그대로 유지하면 context window는 빨리 가득 차고, 더 중요한 user goal과 current state는 상대적으로 약해진다.

GenericAgent의 compression layer는 단순한 token trimming이 아니다. 목표는 "짧게 만들기"가 아니라 "decision에 필요한 signal을 남기기"다. 이 차이가 중요하다. 무작정 줄이면 중요한 정보가 사라지고, 무작정 남기면 noise가 늘어난다. GenericAgent의 설계는 memory hierarchy와 truncation을 결합해, active context가 task execution 중에도 일정한 density를 유지하도록 한다.

### 5) Execution loop

README 기준으로 GenericAgent의 core loop는 약 100 lines로 설명된다. 공개 설명상 loop는 다음 흐름이다.

1. Perceive environment state
2. Reason about task
3. Execute tools
4. Write experience to memory
5. Repeat

이 loop 자체는 새롭지 않아 보일 수 있다. 대부분의 agent도 perceive, plan, act, observe를 반복한다. 차이는 loop가 memory update와 skill crystallization을 system primitive로 포함한다는 점이다. 즉 한 task가 끝난 뒤 agent가 그냥 종료되는 것이 아니라, 다음 task를 위해 자신의 execution trace를 압축하고 남긴다.

이것이 GenericAgent를 단순 tool-use agent가 아니라 self-evolving agent로 만드는 지점이다.

# 4. Training / Data / Recipe

## 4-1. Data

GenericAgent는 model training 논문이 아니므로, 일반적인 의미의 pretraining dataset이나 supervised dataset을 중심으로 읽으면 핵심을 놓치기 쉽다. 이 논문에서 더 중요한 것은 agent가 runtime 중 어떤 experience data를 만들고, 어떤 형태로 남기는가다.

GenericAgent의 experience data는 대략 다음과 같이 볼 수 있다.

- raw session trace
- tool call and output
- execution error and recovery step
- user confirmation and feedback
- successful workflow
- reusable SOP
- reusable script or skill
- global facts and constraints

이 data는 L4 session archive에 더 원본에 가까운 형태로 남고, L3에는 더 압축된 task skill과 SOP로 올라간다. L2에는 장기적으로 안정적인 facts가 남고, L1에는 이 memory들이 존재한다는 compact index가 남는다.

즉 GenericAgent의 data pipeline은 offline dataset construction보다 online experience distillation에 가깝다. 이 점이 Mem0 같은 long-term memory agent 논문과도 연결된다. 다만 Mem0가 memory extraction, update, retrieval 자체에 더 초점을 둔다면, GenericAgent는 memory와 tool execution, skill reuse, context compression을 하나의 agent loop로 묶는다.

## 4-2. Training strategy

GenericAgent의 self-evolution은 gradient-based learning이 아니라 procedural learning이다. 따라서 "training strategy"를 다음과 같이 해석하는 편이 맞다.

1. Cold start
   - agent는 minimal tool set과 base rule만 가진 상태에서 task를 시작한다.
   - 필요한 package를 설치하거나 script를 작성하고, browser 또는 OS를 직접 조작한다.

2. Exploration
   - agent는 시행착오를 통해 task-specific path를 찾는다.
   - 이 과정에서 실패, error log, workaround가 생긴다.

3. Verification
   - task completion이 확인되거나, subgoal completion이 명확해지는 지점에서 trajectory가 reusable knowledge 후보가 된다.
   - GenericAgent는 low-level raw trace가 그대로 high-level skill이 되지 않도록 explicit consolidation을 둔다.

4. Consolidation
   - 성공 path를 SOP, checklist, recovery rule, executable script로 정리한다.
   - 필요한 경우 L3 skill 또는 L2 fact로 저장한다.

5. Reuse
   - 다음 유사 task에서는 L1 index를 통해 관련 skill을 찾고, 깊은 memory를 retrieve한 뒤 재사용한다.

이 방식은 RL이라기보다 runbook learning에 가깝다. agent가 reward signal을 통해 policy parameter를 업데이트하는 것이 아니라, 성공한 절차를 외부 memory와 code artifact로 남긴다. 그래서 검증 가능성과 디버깅 가능성이 상대적으로 높다.

## 4-3. Engineering notes

GenericAgent를 실무 관점에서 볼 때 중요한 engineering note는 다음과 같다.

1. Tool surface는 작게 유지한다.
   - tool을 많이 붙이면 agent가 강해지는 것처럼 보일 수 있지만, tool description과 selection cost가 커진다.
   - GenericAgent는 원자 도구를 적게 두고, 반복되는 작업은 skill로 올린다.

2. Memory는 default-injection이 아니라 on-demand retrieval이어야 한다.
   - 장기 memory가 커질수록 active context에 모두 넣는 방식은 실패한다.
   - index layer와 retrieval route가 필요하다.

3. Self-evolution은 raw trace 저장이 아니라 consolidation이 핵심이다.
   - 단순히 conversation history를 저장하는 것은 memory가 아니다.
   - 성공한 workflow를 실행 가능한 procedure로 바꾸어야 token cost가 줄어든다.

4. Context compression은 단순 요약보다 state preservation이 중요하다.
   - 오래된 message를 줄일 때 user goal, current plan, unresolved issue, file path, credential constraint 같은 핵심 상태가 사라지면 안 된다.
   - working memory anchor는 이 문제를 해결하기 위한 장치로 볼 수 있다.

5. 권한은 capability와 safety를 동시에 결정한다.
   - GenericAgent는 real browser, terminal, filesystem, keyboard and mouse, ADB까지 다룰 수 있다고 설명된다.
   - 이는 강력한 범용성을 만들지만, production system에서는 sandbox, permission boundary, audit log, human confirmation policy가 필수다.

# 5. Evaluation

## 5-1. Main results

공개 arXiv abstract는 GenericAgent가 task completion, tool use efficiency, memory effectiveness, self-evolution, web browsing에서 leading agent systems보다 적은 token과 interaction으로 더 좋은 결과를 보인다고 요약한다. 공개 기술 리포트 요약 기사와 GitHub 설명에는 다음과 같은 결과가 정리되어 있다.

| Evaluation area | Reported observation | Interpretation |
| --- | --- | --- |
| SOP-bench | GenericAgent가 100% accuracy를 보고 | SOP reuse와 task procedure memory의 효과를 보는 축 |
| Lifelong AgentBench | GenericAgent가 100% task completion을 보고 | repeated and lifelong setting에서 memory reuse를 보는 축 |
| RealFinBench | GenericAgent가 65% accuracy를 보고 | 실제 금융 태스크와 가까운 task setting에서의 성능 축 |
| Token efficiency | 같은 task에서 token consumption이 주요 비교 agent의 15% to 35% 수준이라고 보고 | context information density가 비용 절감으로 연결되는지 보는 축 |
| Repeated task | 5회 반복 시 time 102s to 66s, token 200K to 100K로 감소한다고 보고 | self-evolution이 동일 task 반복 비용을 줄이는지 보는 축 |
| Long-term evolution | 1회차 7m30s, 32 LLM calls, 222K tokens에서 9회차 1m38s, 5 calls, 23K tokens로 감소한다고 보고 | SOP와 code로 experience가 압축되는지 보는 축 |
| BrowseComp-ZH | accuracy 0.60 vs baseline 0.20, token cost는 약 1/3이라고 보고 | noisy web context에서 compression과 memory가 효과적인지 보는 축 |

여기서 중요한 것은 absolute score보다 curve다. GenericAgent의 claim은 단일 task에서 한 번 잘했다는 것이 아니라, 반복 사용 중 token cost와 interaction count가 감소한다는 점이다. 즉 시스템이 task experience를 축적하면서 더 싸고 빠르게 같은 종류의 문제를 풀어야 한다.

## 5-2. What really matters in the experiments

이 논문의 evaluation에서 진짜 봐야 하는 지표는 단순 accuracy가 아니다. Agent 논문에서는 accuracy만 보면 중요한 trade-off를 놓치기 쉽다. GenericAgent에서 봐야 하는 지표는 다음 네 가지다.

1. Task success rate
   - agent가 실제로 task를 끝냈는가.
   - 하지만 이 지표만으로는 부족하다. token을 10배 쓰고도 성공하면 실무에서는 비용 문제가 된다.

2. Token consumption
   - 같은 task success를 얼마나 적은 token으로 달성했는가.
   - GenericAgent의 핵심 주장과 직접 연결된다.

3. Interaction count or LLM call count
   - agent가 몇 번의 loop로 task를 끝냈는가.
   - repeated task에서 이 수치가 줄면 skill reuse가 실제로 작동한다고 볼 수 있다.

4. Improvement across repetitions
   - 동일하거나 유사한 task를 반복할 때 비용이 줄어드는가.
   - 이 지표가 self-evolving agent와 stateless agent를 가르는 핵심이다.

내가 보기엔 GenericAgent의 가장 중요한 실험은 benchmark score table보다 repeated execution curve다. 왜냐하면 self-evolution이라는 claim은 한 번의 성공이 아니라 두 번째, 세 번째, 아홉 번째 실행에서 더 적은 비용으로 수렴하는지를 봐야 검증되기 때문이다.

## 5-3. Interpreting the numbers conservatively

다만 숫자는 보수적으로 읽어야 한다. 기술 리포트와 요약 기사에서 보고된 수치는 강한 메시지를 주지만, publish 전에 원문 PDF의 table, baseline setting, task definition, token accounting 방식을 다시 확인하는 것이 좋다.

특히 다음 질문이 중요하다.

- Claude Code, OpenClaw 등 baseline의 version과 설정은 무엇인가?
- token count는 input만인지, input plus output인지, tool output serialization을 포함하는지?
- 같은 model backend를 썼는지, 아니면 각 agent system의 default model을 썼는지?
- self-evolution 반복 실험에서 이전 run의 memory를 어떤 방식으로 허용했는지?
- BrowseComp-ZH의 task sampling과 scoring이 어떻게 정의되었는지?

이런 조건에 따라 agent benchmark의 해석은 크게 달라진다. 그래도 GenericAgent가 던지는 message는 유효하다. long-horizon agent에서 더 많은 token은 더 많은 reasoning이 아니라, 관리되지 않은 context의 증상일 수 있다.

# 6. Limitations

1. Baseline comparability
   - Agent system benchmark는 baseline 설정에 매우 민감하다. model backend, tool permission, browser environment, memory persistence, retry policy가 조금만 달라도 결과가 달라진다.
   - 따라서 GenericAgent의 수치는 baseline setup을 원문 table에서 다시 확인해야 한다.

2. Safety and permission boundary
   - GenericAgent는 local computer에 대한 broad control을 강점으로 내세운다.
   - 하지만 terminal, browser, file system, keyboard and mouse, ADB control은 production 환경에서 강력한 risk surface다.
   - 실제 서비스 적용에는 sandbox, permission grant, human confirmation, audit trail, rollback policy가 필요하다.

3. Self-evolution quality control
   - 성공 trajectory를 SOP나 script로 남기는 것은 강력하지만, 잘못 일반화된 skill이 저장되면 이후 task에서 반복적으로 오류를 만들 수 있다.
   - skill validation, versioning, deprecation, conflict resolution이 중요하다.

4. Memory retrieval failure
   - Hierarchical memory는 L1 index와 routing이 잘 작동할 때 유효하다.
   - L1이 잘못된 pointer를 제공하거나, 필요한 skill이 있는데 retrieve하지 못하면 agent는 다시 탐색해야 한다.

5. Domain transfer
   - 웹 브라우징, 파일 조작, 코드 실행, 모바일 제어에서는 강점이 있을 수 있지만, enterprise workflow, regulated domain, private data setting에서는 별도의 제약이 생긴다.
   - 특히 user data와 credential을 다루는 agent는 보안 설계가 논문 성능만큼 중요하다.

6. Evaluation opacity
   - 공개 요약에서 많은 수치가 제시되지만, 최종 게시 전에는 원문 Figure/Table을 기준으로 benchmark 구성과 token accounting을 확인해야 한다.
   - agent benchmark는 재현성이 아직 어려운 영역이라, 수치를 그대로 marketing claim으로 쓰면 위험하다.

# 7. My Take

## 7-1. Why this matters for my work

GenericAgent는 agent를 만들 때 무엇을 architecture로 보고 무엇을 prompt로 볼지 다시 생각하게 만든다. 많은 agent prototype은 prompt, tool list, memory store를 따로 붙인다. 하지만 실제 long-horizon task에서는 이 셋이 독립적이지 않다.

- Prompt는 active context policy다.
- Tool list는 action space policy다.
- Memory는 past experience compression policy다.
- Skill library는 repeated work amortization policy다.

GenericAgent의 가치가 있는 지점은 이 네 가지를 context information density라는 하나의 설계 목표로 묶는다는 점이다. 이는 실무 agent에서 매우 중요하다. 예를 들어 사내 문서 agent나 개발 자동화 agent를 만들 때도, 모든 문서를 넣는 것이 아니라 현재 decision에 필요한 pointer와 procedure를 주는 것이 더 중요할 수 있다.

특히 좋게 본 부분은 self-evolution을 model weight update로 설명하지 않는 점이다. 실제 제품에서는 매번 모델을 fine-tune하기 어렵다. 반면 SOP, script, checklist, memory index는 훨씬 빠르게 업데이트하고 검증할 수 있다. GenericAgent는 agent learning을 parameter learning이 아니라 operational knowledge accumulation으로 보는 쪽에 가깝다.

## 7-2. Reuse potential

실무에서 바로 가져갈 만한 아이디어는 다음과 같다.

1. Memory를 L1 index와 deep memory로 나누기
   - 모든 memory를 prompt에 넣지 말고, active context에는 compact index만 넣는다.
   - 필요한 경우 deep memory를 tool call로 retrieve한다.

2. Repeated workflow를 SOP로 승격하기
   - agent가 성공한 작업은 transcript로만 남기지 않는다.
   - precondition, steps, known error, recovery, output check를 가진 SOP로 바꾼다.

3. SOP에서 executable code로 승격하기
   - 반복되는 manual step은 script나 function으로 만든다.
   - 다음 실행에서는 reasoning이 아니라 execution으로 처리한다.

4. Tool output truncation policy 만들기
   - browser DOM, logs, search result는 무조건 prompt에 넣지 않는다.
   - tool별 maximum output과 summary rule을 둔다.

5. Working memory anchor 유지하기
   - context pruning 후에도 user goal, current status, next action, unresolved blocker가 항상 남아야 한다.

6. Skill versioning과 validation 넣기
   - self-evolution은 memory가 쌓이는 만큼 부채도 쌓일 수 있다.
   - skill마다 created date, source task, validation condition, last used, failure count를 관리하는 것이 좋다.

## 7-3. Follow-up papers

- Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
  - long-term memory extraction, update, retrieval을 production 관점에서 더 자세히 볼 수 있다.

- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
  - parameter update 없이 textual artifacts를 진화시키는 관점에서 함께 읽기 좋다.

- SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents
  - self-evolving agent와 memory, policy optimization의 연결을 더 RL 쪽에서 볼 수 있는 비교축이다.

- SEA-Eval: A Benchmark for Evaluating Self-Evolving Agents
  - self-evolving agent를 어떻게 평가할지에 대한 benchmark 관점에서 후속으로 볼 만하다.

# 8. Summary

- GenericAgent는 long-horizon agent의 병목을 context length가 아니라 context information density로 재정의한다.
- 핵심 구성은 minimal atomic tools, hierarchical on-demand memory, self-evolution through SOP and code, context truncation and compression이다.
- self-evolution은 LLM weight update가 아니라, 성공 trajectory를 reusable skill과 executable code로 바꾸는 procedural learning에 가깝다.
- 논문의 중요한 실험 축은 단일 benchmark score보다 반복 실행에서 token cost와 LLM call count가 줄어드는지다.
- 실무적으로는 memory index, SOP crystallization, tool output truncation, working memory anchor를 agent runtime 설계에 바로 가져갈 수 있다.
