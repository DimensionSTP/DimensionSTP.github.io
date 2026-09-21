---
layout: single
title: "ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement Review"
categories: Study-concept
tag: [AI-Agent, AgentHarness, RecursiveSelfImprovement]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/pdf/2609.14857)

[Code link](https://github.com/IQuestLab/ModularRSI)

Agent 성능은 foundation model뿐 아니라 harness에 크게 의존한다. Harness는 model response를 tool call로 바꾸고, environment observation을 정리하고, context를 압축하고, retry를 결정하고, task가 끝났는지 판단한다.

최근 harness RSI는 agent가 자신의 execution log를 보고 harness code를 수정하게 한다. 하지만 benchmark score가 올랐다고 generalizable improvement라고 말하기는 어렵다.

- Evaluation benchmark 자체로 harness를 evolve하면 benchmark adaptation일 수 있다.
- Single failed trajectory에서 얻은 fix는 instance-specific patch일 수 있다.
- Monolithic harness를 한 번에 수정하면 어떤 mechanism이 좋아졌는지 알기 어렵다.
- Multiple changes가 서로 interference를 일으킬 수 있다.

ModularRSI는 이 문제를 benchmark-disjoint, contrastive, modular라는 세 원칙으로 푼다.

1. Downstream benchmark와 분리된 2,000 executable tasks에서 experience를 수집한다.
2. Same task의 successful and failed trajectories를 contrast해 recurring deficiency를 찾는다.
3. Harness를 five modules로 나누고 restricted scope 안에서 독립적으로 evolve한다.
4. Program check, diff review, runtime validation을 통과한 change만 채택한다.
5. 마지막 integration stage에서 module conflict를 해결한다.

> 한 줄 요약: ModularRSI는 benchmark-disjoint task pool에서 same-task success and failure를 contrast하고, agent harness를 five functional modules로 분리해 scoped code change and validation을 수행함으로써 unseen task, domain, model로 transfer되는 harness improvement를 목표로 하는 recursive self-improvement framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Harness RSI의 핵심 위험을 benchmark overfitting and instance patching으로 명확히 정의한다.
- Trajectory 하나가 아니라 same-task contrast and cross-task aggregation을 사용한다.
- Agent loop, tool, observation, context, completion detection을 independent evolvable unit으로 만든다.
- Accuracy뿐 아니라 interaction step reduction and cross-model transfer를 평가한다.
- Non-modular and joint evolution이 baseline보다 나빠질 수 있다는 직접적인 interference evidence를 제시한다.

# 1. Problem Setting

## 1-1. Harness를 왜 별도 학습 대상으로 보는가

Foundation model $f_{\theta}$와 harness $h$가 task environment $e$에서 trajectory를 만든다고 하자.

$$
\tau
=
\operatorname{Rollout}(f_{\theta},h,e)
$$

Task success는 model weight만의 함수가 아니다.

$$
R(\tau)
=
R(f_{\theta},h,e)
$$

같은 model이라도 harness가 다음 failure를 만들 수 있다.

- Tool schema를 잘못 parse한다.
- Long output을 그대로 context에 넣어 중요한 state를 밀어낸다.
- Error observation을 반복해서 보여준다.
- Solved task를 계속 실행한다.
- Incomplete task를 너무 일찍 종료한다.
- Retry state를 잃고 같은 action을 반복한다.

Harness code를 개선하면 model retraining 없이 behavior를 바꿀 수 있다.

## 1-2. Benchmark-coupled evolution의 문제

Evaluation task or subset에서 직접 harness를 evolve하면 다음 information이 code에 스며들 수 있다.

- Specific file name
- Expected output pattern
- Benchmark command
- Common package name
- Task-specific constant
- Hidden evaluator artifact

Score가 올라가도 reusable execution mechanism인지 benchmark patch인지 구분하기 어렵다.

## 1-3. Single-trajectory diagnosis의 문제

한 failed trajectory에는 두 종류의 원인이 섞인다.

1. Instance-specific reasoning failure
2. Recurring harness mechanism failure

예를 들어 package install command를 틀린 것이 model reasoning 문제인지, tool error를 truncation한 observation module 문제인지 한 trajectory만으로 판단하기 어렵다.

Same task에서 success and failure를 비교하면 공통 task difficulty를 통제하고 behavior difference를 볼 수 있다.

## 1-4. Monolithic modification의 문제

Harness 전체를 한 agent가 자유롭게 수정하면 다음 문제가 생긴다.

- Tool parsing fix가 context contract를 깨뜨린다.
- Observation compression change가 completion detection signal을 지운다.
- Similar logic가 여러 file에 중복된다.
- Improvement attribution이 불가능하다.
- Rollback unit이 너무 크다.

ModularRSI는 modification scope를 module boundary 안으로 제한한다.

# 2. Core Idea

## 2-1. Three design principles

### 1) Benchmark-disjoint evolution

Evolution data와 downstream evaluation benchmark를 분리한다. Harness는 Terminal-Bench 2.0 or SWE-Bench Verified result를 보지 않은 채 frozen된다.

### 2) Contrastive trajectory diagnosis

Same task에서 multiple rollouts을 모으고 outcome pattern에 따라 세 group으로 분류한다.

- Positive: 모든 rollout이 success
- Contrastive: success and failure가 함께 존재
- Negative: 모든 rollout이 failure

Contrastive group은 same task에서 success and failure behavior를 직접 비교할 수 있어 가장 강한 diagnosis signal을 준다.

### 3) Modular scoped evolution

Harness를 five functional modules로 나누고 한 번에 한 module scope만 수정한다.

- Agent Loop
- Tool Use
- Observation Management
- Context Management
- Task Completion Detection

## 2-2. Cross-task recurring evidence

Trajectory analyzer는 single anecdote를 바로 code patch로 바꾸지 않는다. Module별 finding을 structured form으로 저장하고 여러 task에서 반복되는 deficiency를 aggregate한다.

Finding은 대략 다음 field를 가진다.

- Evidence span
- Target module
- Failure mechanism
- Why it is systematic
- Proposed change
- Expected benefit
- Risk and validation plan

Task-specific content를 제거하고 function-level behavior로 추상화하는 것이 핵심이다.

# 3. Architecture / Method

## 3-1. Five harness modules

| Module | Main responsibility | Typical failure |
| --- | --- | --- |
| Agent Loop | Reason-act-observe cycle, retry, state transition | Repetition, bad recovery, wrong phase transition |
| Tool Use | Parse action, select and invoke tool, handle error | Invalid call, parse failure, silent tool error |
| Observation Management | Filter, format, compress environment output | Noise flooding, missing error, bad truncation |
| Context Management | Maintain history and compact long context | State loss, stale context, summary drift |
| Task Completion Detection | Decide solved, failed, or continue | Premature stop, endless execution, false success |

Repository implementation에서는 module directory name이 `agent_loop`, `observation`, `tools`, `context_mgmt`, `verification`으로 정리되어 있다. Paper의 Task Completion Detection과 repository `verification` naming은 release version에서 확인할 필요가 있다.

## 3-2. Independent experience collection

Each evolution task에서 $K$개의 rollout을 생성한다.

$$
\mathcal{E}_i
=
\{(\tau_{i1},r_{i1}),\ldots,(\tau_{iK},r_{iK})\}
$$

Task $i$의 outcome pattern에 따라 analysis strategy가 달라진다.

### Contrastive task

Successful trajectory and failed trajectory를 pairwise compare한다.

- 어떤 tool sequence가 달랐는가.
- Observation이 어떻게 정리되었는가.
- Context loss가 언제 발생했는가.
- Completion decision이 왜 달랐는가.

### Negative task

Current epoch에 success가 없으면 historical successful trajectory가 있는지 찾는다. 없다면 failure-only diagnosis를 사용하되 confidence를 낮춘다.

### Positive task

Correctness fix보다 efficiency and robustness를 찾는다.

- Unnecessary steps
- Repeated tool calls
- Redundant context
- Late completion detection

## 3-3. Trajectory memory

Evolution은 epoch across memory를 유지한다. Same task가 later epoch에서 다른 outcome을 보이면 이전 success or failure를 새로운 contrast pair로 사용할 수 있다.

Memory는 다음 역할을 한다.

- Rare success를 잃지 않는다.
- Previous modification이 어떤 behavior를 바꿨는지 추적한다.
- Repeated deficiency evidence를 누적한다.
- Oscillation and regression을 발견한다.

## 3-4. Module-level proposal and implementation

Finding을 target function별로 consolidate한 뒤 code modification agent가 scoped patch를 만든다.

Modification constraint는 다음과 같다.

- Current target module 밖의 file은 수정하지 않는다.
- Task-specific literal and benchmark identifier를 넣지 않는다.
- Existing interface를 유지한다.
- Change rationale and expected failure coverage를 기록한다.
- Previous generation은 immutable하게 보존한다.

## 3-5. Three validation gates

### 1) Program verification

- Syntax
- Import
- Interface
- Protocol
- AST structure

### 2) Diff review

- Task-specific name
- File path
- Expected answer
- Ad hoc constant
- Benchmark heuristic
- Unrelated module change

### 3) Execution validation

Modified harness를 sampled evolution tasks에서 실행한다.

- Runtime error
- Tool protocol failure
- Broken state transition
- Completion detector failure
- Severe regression

Gate를 통과하지 못하면 change를 rollback한다.

## 3-6. Integration stage

Five modules를 독립적으로 evolve한 뒤 하나의 harness로 합친다. Integration은 단순 file concatenation이 아니다.

- Duplicate function merge
- Conflicting responsibility resolution
- Interface reconciliation
- Shared state contract alignment
- Task-aware function composition

Independent module gain이 merged system에서 유지되는지 다시 validation한다.

# 4. Training / Data / Recipe

## 4-1. Benchmark-disjoint evolution dataset

공개 dataset은 2,000 executable tasks로 구성된다.

| Domain | Count | Source examples |
| --- | ---: | --- |
| Terminal-oriented tasks | 1,000 | Linux documentation and executable environments |
| Software-engineering tasks | 1,000 | GitHub, Hugging Face, Kaggle repositories |

Main experiment에서는 각 domain에서 120 tasks를 표시해 사용한다. Downstream Terminal-Bench 2.0 and SWE-Bench Verified instance와 task-specific information은 evolution pool에서 제외한다.

## 4-2. Dataset quality checks

Evolution task는 다음 조건을 확인한다.

- Environment completeness
- Reference solution execution
- Evaluator validity
- No-op failure
- Task difficulty
- Manual correctness
- Semantic distance from downstream benchmark

Benchmark-disjoint는 exact instance exclusion만으로 충분하지 않다. Similar repository and template가 섞일 수 있으므로 semantic overlap audit가 중요하다.

## 4-3. Evolution recipe

Main setup은 다음과 같다.

| Item | Setting |
| --- | --- |
| Base harness | Terminus-2 on Harbor |
| Evolution model | DeepSeek-V4-Flash variants |
| Main training tasks | 120 terminal plus 120 SWE |
| Evolution epochs | 3 |
| Batch size | 10 tasks |
| Model throughput limit | 2M TPM reported |
| Downstream harness | Frozen before evaluation |

Harness RSI는 gradient training이 아니라 trajectory analysis and code modification loop다. 따라서 reproducibility에는 model API version, prompt, environment image, code generation and validation log가 중요하다.

## 4-4. Downstream evaluation

- Terminal-Bench 2.0: 89 tasks
- SWE-Bench Verified: 500 tasks
- Harbor execution environment

Metric은 accuracy뿐 아니라 다음을 포함한다.

- Pass@3
- Pass^3 consistency
- Average interaction steps

Accuracy가 같아도 step reduction and consistency improvement를 분리할 수 있다.

## 4-5. Engineering notes

### 1) Module interface를 schema로 고정해야 한다

Observation object, context state, tool result, completion status를 typed schema로 두지 않으면 independent evolution이 interface conflict를 만든다.

### 2) Change provenance를 저장해야 한다

각 function revision에 source findings, task IDs, code diff, validation result, parent generation을 연결해야 한다.

### 3) Regression suite가 module별로 필요하다

Tool module patch는 tool parse tests, context module patch는 summary preservation tests처럼 targeted unit test를 유지하는 것이 좋다.

### 4) Benchmark keyword audit만으로는 부족하다

Literal leakage가 없어도 task family-specific heuristic이 들어갈 수 있다. Cross-domain and cross-model transfer가 stronger evidence다.

### 5) Integration test를 별도 budget으로 봐야 한다

Module별 pass가 merged harness pass를 보장하지 않는다. Shared state and stop condition conflict를 통합 stage에서 다시 평가해야 한다.

### 6) API model drift를 기록해야 한다

Closed or hosted model version이 바뀌면 same evolution prompt가 다른 patch를 만들 수 있다. Model identifier, date, temperature, response hash를 저장해야 한다.

# 5. Evaluation

## 5-1. Main downstream result

DeepSeek V4 setting에서 reported result는 다음과 같다.

| Benchmark | Baseline harness | ModularRSI harness | Delta |
| --- | ---: | ---: | ---: |
| Terminal-Bench 2.0 | 47.57 | 52.43 | +4.86 |
| SWE-Bench Verified | 73.40 | 76.45 | +3.05 |

Evolution task와 downstream benchmark가 disjoint하므로 in-sample patch보다 reusable harness mechanism improvement에 가까운 evidence다.

## 5-2. Cross-model transfer

Evolved harness를 different foundation model에 적용해도 improvement가 보고된다.

| Model | Baseline | Evolved harness |
| --- | ---: | ---: |
| GLM-5.2 | 59.55 | 61.80 |
| MiniMax-2.5 | 41.57 | 44.94 |
| DeepSeek V4 | 47.57 | 52.43 |

Same code change가 multiple model에서 positive direction을 보인다는 점은 benchmark-specific prompt patch보다 system-level transfer를 지지한다.

## 5-3. Modularity ablation

Terminal-Bench 2.0에서 evolution strategy를 비교한다.

| Strategy | Accuracy |
| --- | ---: |
| Baseline | 47.57 |
| Non-modular evolution | 46.44 |
| Joint all-module evolution | 44.19 |
| ModularRSI | 52.43 |

Monolithic freedom이 더 큰데도 performance가 낮다. Search space가 넓고 change interference가 커지면 self-modification capability가 오히려 불안정해질 수 있음을 보여준다.

## 5-4. Single-module contribution

각 module만 evolve해도 baseline보다 improve한다고 보고한다.

- Agent Loop는 largest individual accuracy gain을 만든다.
- Observation Management는 average interaction steps를 34.70에서 22.50으로 줄인다.
- Other modules도 smaller but positive gains를 보인다.

이 결과는 accuracy and efficiency contribution이 module별로 다를 수 있음을 보여준다.

## 5-5. Other harness RSI comparison

Benchmark-disjoint unified setup에서 reported Terminal-Bench result는 다음과 같다.

| Method | Accuracy |
| --- | ---: |
| Baseline | 61.79 |
| Meta-Harness | 62.92 |
| AHE | 62.54 |
| ModularRSI | 67.42 |

이 table은 main 47.57 to 52.43 table과 evaluation setup이 다르므로 absolute score를 직접 연결하면 안 된다. Controlled comparison within each table만 해석해야 한다.

## 5-6. Data difficulty analysis

SWE evolution data는 medium-difficulty task 중심 구성에서 strongest result를 보인다.

- 너무 easy한 task는 failure contrast가 부족하다.
- 너무 hard한 task는 success example이 부족하다.
- Mixed success and failure가 나오는 task가 diagnosis에 가장 유용하다.

이는 contrastive harness RSI의 data selection principle을 보여준다.

# 6. Limitations

1. Contrastive analysis 자체의 isolated ablation이 부족하다.
   - Paper도 dedicated ablation 부재를 limitation으로 언급한다.

2. 2,000 tasks 중 main evolution은 subset을 사용한다.
   - 각 domain 120 tasks로 cost를 제한하므로 full pool scaling behavior는 열려 있다.

3. Module boundary가 사람이 설계한 prior다.
   - Five-module decomposition이 다른 harness architecture에도 optimal한지 보장되지 않는다.

4. Same-task success가 없는 hard task에서는 contrast signal이 약하다.
   - Historical success도 없으면 failure-only diagnosis에 의존한다.

5. Code modification model에 의존한다.
   - Different model or API version에서 same patch quality가 재현되는지 확인이 필요하다.

6. Benchmark-disjoint가 capability-disjoint를 보장하지 않는다.
   - External tasks가 downstream benchmark와 같은 terminal and SWE ecosystem을 공유할 수 있다.

7. Integration conflict resolution의 독립 검증이 더 필요하다.
   - Module improvement가 merged system에서 어떤 interaction을 만드는지 pairwise ablation이 제한적이다.

8. Security and governance 문제가 남는다.
   - Self-modifying harness가 tool permission, sandbox, logging rule을 바꾸지 못하도록 immutable boundary가 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

ModularRSI의 핵심은 agent가 자기 code를 고칠 수 있다는 사실보다, self-modification search space를 어떻게 제한해야 generalization이 생기는지 보여준 데 있다.

Free-form whole-system editing은 강력해 보이지만 credit assignment와 regression control이 어렵다. Module scope, contrastive evidence, validation gate를 둔 쪽이 더 잘 transfer된다는 결과는 agent infrastructure를 연구할 때 매우 실용적인 lesson이다.

## 7-2. Reuse potential

### 1) Inference pipeline RSI

Preprocessing, routing, tool execution, evidence filtering, answer verification, stop decision을 module로 나누고 run log에서 recurring failure를 찾을 수 있다.

### 2) Document AI harness evolution

OCR retry, page selection, table parser routing, evidence composition, grounding validation을 independent module로 evolve할 수 있다.

### 3) Training pipeline diagnostics

Data loader, sampler, rollout worker, verifier, optimizer scheduler를 직접 수정하기보다 module-level suggestion and regression test loop를 만들 수 있다.

### 4) Contrastive trajectory dataset

Same task의 success and failure pair를 저장하면 postmortem, reward model, tool policy training에도 재사용할 수 있다.

### 5) Immutable safety layer

Evolvable module과 non-evolvable policy boundary를 분리해 credential, filesystem scope, network permission은 code agent가 수정하지 못하게 해야 한다.

## 7-3. Follow-up papers

- Meta-Harness
- Automated Harness Engineering
- HarnessDev
- OpenForgeRL
- Darwin Godel Machine
- Automated Design of Agentic Systems
- TerminalTraj

# 8. Summary

- ModularRSI는 benchmark-disjoint task에서 harness를 evolve해 unseen benchmark transfer를 목표로 한다.
- Same task의 success and failure trajectory를 contrast해 systematic deficiency를 찾는다.
- Harness를 Agent Loop, Tool Use, Observation, Context, Completion Detection의 five modules로 나눈다.
- Scoped patch and three validation gates가 monolithic and joint evolution보다 안정적인 result를 만든다.
- Main limitation은 human-defined module boundary, limited evolution subset, contrastive analysis ablation 부족이다.
