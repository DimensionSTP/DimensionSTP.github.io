---
layout: single
title: "HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness? Review"
categories: Study-concept
tag: [AIAgent, AgentHarness, Benchmark, SelfImprovement]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/pdf/2609.01437)

[Project page](https://self-developing-agents.github.io)

Agent benchmark는 보통 model과 harness를 하나의 system으로 평가한다. 같은 model weight라도 system prompt, tool schema, context compaction, retry logic, checkpointing, verifier, stopping policy가 달라지면 score는 크게 달라질 수 있다. 그런데 benchmark table에서는 이 외부 infrastructure가 고정된 전제처럼 취급되는 경우가 많다.

HarnessDev는 평가 단위를 task answer에서 runnable agent infrastructure로 옮긴다. Model에게 이미 완성된 coding agent or research agent를 주는 대신, weak seed와 몇 개의 development case만 주고 complete harness를 만들게 한다. 이후 자신의 harness가 downstream task에서 만든 feedback을 보고 harness를 다시 수정하게 한다.

여기서 harness는 단순 prompt가 아니다.

- Agent loop and execution control
- Tool invocation and observation handling
- Context construction and compaction
- Persistent state and recovery
- Verification and artifact checking
- Stopping and final submission

논문의 질문은 두 개다.

1. Current LLM은 runnable harness를 처음부터 만들 수 있는가.
2. Downstream feedback으로 harness를 개선했을 때 그 변화가 held-out task and another executor에도 남는가.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Agent performance를 model weight가 아니라 model-external system capability까지 포함해 분해한다.
- Creation and Evolution을 분리해 zero-to-runnable and feedback-driven improvement를 각각 평가한다.
- Creator model과 executor model을 분리해 harness-model co-adaptation을 측정한다.
- Task success뿐 아니라 executor-token cost를 함께 기록한다.
- Self-improvement가 visible feedback에서는 좋아 보여도 held-out and fixed-executor setting에서 쉽게 무너진다는 강한 negative evidence를 제공한다.

> 한 줄 요약: HarnessDev는 LLM이 weak seed에서 runnable agent harness를 만들고 downstream feedback으로 이를 발전시킬 수 있는지 평가하며, 현재 model은 writing and ML experimentation에서는 유용한 harness를 만들지만 code and search에서는 human reference에 뒤처지고, Evolution gain은 held-out task와 fixed executor로 충분히 안정적으로 전이되지 않음을 보여준다.

# 1. Problem Setting

## 1-1. What is an agent harness

Agent harness는 foundation model과 environment 사이의 execution contract다. Model이 같은 policy를 가지고 있어도 harness가 무엇을 보여주고 어떤 action을 허용하는지에 따라 trajectory가 달라진다.

HarnessDev는 harness를 여섯 control dimension으로 본다.

| Dimension | Role |
| --- | --- |
| Execution | Main loop, turn scheduling, subprocess control, concurrency |
| Tools | Tool definitions, permissions, argument validation, result handling |
| Context | Prompt construction, observation filtering, compaction, retrieval |
| State | Persistent memory, workspace metadata, progress and recovery state |
| Lifecycle | Initialization, checkpointing, retry, timeout, stopping |
| Verification | Tests, artifact validation, scoring, final-state checks |

Weak seed는 basic IO, passive tool access, configuration, logging, artifact writing을 제공하지만 robust execution loop, decomposition, context management, persistent state, verifier, recovery and stopping logic은 제공하지 않는다.

Creation agent는 domain specification과 1 to 3 development case를 보고 seed를 complete system으로 바꿔야 한다.

## 1-2. Why existing agent benchmarks are insufficient

### 1) Harness가 fixed hidden variable로 남는다

Model comparison이 실제로는 model-harness pair comparison인데, score는 model name만으로 보고되는 경우가 많다. Harness improvement가 model capability improvement처럼 보일 수 있다.

### 2) Runnable code와 good harness는 다르다

Harness가 실행된다는 사실은 verification, state recovery, context management가 제대로 작동한다는 뜻이 아니다. Dead code or never-triggered mechanism이 많을 수 있다.

### 3) Self-evaluation은 co-adaptation을 숨긴다

Creator가 만든 prompt, step limit, tool protocol이 creator 자신의 behavior에 맞게 조정될 수 있다. 같은 harness를 다른 executor가 사용하면 score가 급락할 수 있다.

### 4) Visible feedback gain은 generalization이 아니다

Evolution agent가 feedback task score를 반복해서 보면 해당 subset에 맞춘 patch를 만들 수 있다. Held-out task에서 gain이 유지되는지 별도로 평가해야 한다.

### 5) Capability score만으로 efficiency를 알 수 없다

비슷한 success rate를 얻어도 executor token이 7x or 19x 더 들 수 있다. Agent harness는 quality and cost를 함께 평가해야 한다.

# 2. Core Idea

## 2-1. Creation: Build a harness from a weak seed

Creation stage는 harness authoring capability를 평가한다.

1. Creator LLM receives a domain brief.
2. Weak seed repository and 1 to 3 development cases are provided.
3. Creator edits code, prompts, tools, state and verification logic.
4. Resulting harness is frozen.
5. Executor LLM runs held-out benchmark tasks inside the frozen harness.
6. External evaluator scores actual artifacts and executor-token cost.

중요한 점은 evaluator가 harness가 출력한 self-reported success를 믿지 않는다는 것이다. Code task에서는 repository diff and tests, ML task에서는 scorer-readable artifact, research task에서는 answer evidence를 실제로 검사한다.

## 2-2. Evolution: Revise the created harness from feedback

Evolution stage는 Creation output $H_0$에서 시작한다. Agent는 downstream execution result를 보고 new version을 만든다.

Harness version을 $H_k$, visible feedback set score를 $F(H_k)$, held-out score를 $G(H_k)$라고 하자.

Ideal evolution은 다음을 원한다.

$$
F(H_{k+1}) > F(H_k)
$$

and

$$
G(H_{k+1}) > G(H_k)
$$

하지만 실제로는 visible score and held-out score가 자주 다른 방향으로 움직인다. HarnessDev는 이 gap 자체를 주요 result로 보고한다.

Evolution protocol은 code domain에 집중한다.

- Feedback: 100 SWE-Pro tasks plus 89 Terminal-Bench tasks
- Held-out: 630 SWE-Pro tasks
- Official candidate evaluation budget: 10 paired evaluations
- Between pairs: at most two small 5-task probes
- Creator never sees held-out task result during development

## 2-3. Separate creator and executor

HarnessDev에는 두 model role이 있다.

- Creator $L_C$
  - Harness code and configuration을 만든다.

- Executor $L_E$
  - Frozen harness 안에서 downstream task를 수행한다.

Self-Eval은 $L_C=L_E$인 setting이다. Unified-Eval은 executor를 Gemini 3.1 Pro로 고정한다.

이 비교는 harness quality를 model-specific compatibility와 분리하려는 장치다.

$$
S(H, L_E)
$$

Harness quality는 $H$만의 scalar가 아니라 executor와의 interaction에 의존한다. 특정 creator에서 높은 $S(H,L_C)$가 다른 $L_E$에서도 높다고 보장할 수 없다.

## 2-4. Evaluate both capability and efficiency

Capability는 benchmark-specific task success or score다. Efficiency는 executor token consumption을 중심으로 측정한다. Creator가 harness를 만드는 데 사용한 token은 final runtime efficiency에서 제외한다.

이 distinction은 deployment 관점에서 중요하다.

- Creation cost: one-time or occasional system-development cost
- Execution cost: every task에 반복되는 serving cost

Harness가 복잡해질수록 runtime token and tool-call overhead가 커질 수 있다.

# 3. Architecture / Method

## 3-1. Benchmark composition

HarnessDev Creation은 4 domains and 5 downstream benchmarks, 2,207 unique instances를 포함한다.

| Domain | Benchmark | Instances | Main output |
| --- | --- | ---: | --- |
| Code | SWE-Pro | 731 | Repository patch and test success |
| Code | Terminal-Bench 2.1 | 89 | Terminal task completion |
| ML experimentation | MLE-bench | 75 | Competition-style artifact and medal rate |
| Writing | EQ-Bench3 | 46 | Rubric-scored long-form output |
| Search and research | BrowseComp | 1,266 | Evidence-grounded research answer |

Each creator-benchmark cell은 three independent harness creation attempts를 수행하고 avg@3를 보고한다.

## 3-2. Creator models

Creation에는 six frontier creator models가 포함된다.

- Opus 4.8 High
- GPT-5.5 High
- Gemini 3.1 Pro High
- DeepSeek V4 Pro High
- Qwen 3.7 Max
- Seed 2.0 Pro High

Creator는 existing mature agent harness를 복사하는 것이 아니라 weak seed를 편집한다. Domain brief and development case만 보고 architecture and implementation decision을 내린다.

## 3-3. Self-Eval and Unified-Eval

### Self-Eval

Creator model이 executor도 담당한다.

- Advantage: creator-harness pair가 의도한 behavior를 최대한 발휘한다.
- Risk: model-specific prompt and control protocol에 과적합할 수 있다.

### Unified-Eval

모든 generated harness를 Gemini 3.1 Pro executor로 실행한다.

- Advantage: cross-harness comparison에서 executor를 통제한다.
- Risk: Gemini-compatible harness가 구조적으로 유리할 수 있다.

두 setting을 함께 봐야 co-adaptation and portability를 이해할 수 있다.

## 3-4. Artifact-grounded evaluation

Harness가 final message에 `success`라고 적는 것만으로는 score가 오르지 않는다.

- Code: actual patch and tests
- Terminal: environment state
- MLE: saved model or submission artifact
- Writing: external rubric evaluation
- BrowseComp: answer and evidence quality

Source repository, execution traces, prompts, artifacts are frozen for audit. 이 design은 harness가 evaluator protocol을 self-report로 우회하는 것을 줄인다.

## 3-5. Implementation analysis

Paper는 score뿐 아니라 generated harness code를 분석한다.

- Declared state and memory mechanism이 runtime에서 실제 trigger되는지 확인한다.
- Added line count, self-test count, revision calls를 측정한다.
- Evolution version diff를 execution, tools, context, state, lifecycle, verification category로 분류한다.
- Failure가 executor reasoning or harness defect에서 왔는지 분류한다.

Creation code harness 18개는 모두 runnable했지만, code에 존재하는 state or memory mechanism 일부는 formal execution에서 사용되지 않았다. Runnable and active mechanism을 분리한 분석이다.

# 4. Training / Data / Recipe

## 4-1. This is harness development, not model training

HarnessDev의 main benchmark에서는 foundation-model weight를 update하지 않는다. Optimization variable은 repository artifact다.

$$
H^*
=
\operatorname*{argmax}_{H \in \mathcal{H}}
S(H,L_E)
$$

Creator는 code, prompt, configuration, tool wrapper, context policy, verifier를 바꾼다. 따라서 결과를 model finetuning improvement와 구분해야 한다.

## 4-2. Creation recipe

Creation agent에게 필요한 process는 대략 다음과 같다.

1. Domain requirement and weak seed inspection
2. Development-case execution
3. Failure diagnosis
4. Harness architecture design
5. Implementation
6. Self-test and revision
7. Final artifact freeze

Paper analysis에서 code volume or self-test count는 downstream score와 강하게 연결되지 않았다. 반면 revision call은 score와 더 높은 correlation을 보였다. Test를 많이 작성하는 것보다 test result를 diagnosis and targeted revision으로 연결하는 과정이 중요하다는 해석이다.

## 4-3. Evolution recipe

Evolution은 official candidate pair and small probe budget 안에서 진행한다.

- Current harness를 feedback task에 실행한다.
- Failure trace and score를 수집한다.
- Creator가 harness를 수정한다.
- New immutable version을 저장한다.
- Paired visible evaluation을 수행한다.
- Final declaration 이후 held-out 630 tasks에 post-freeze evaluation을 수행한다.

이 protocol은 unlimited leaderboard hill-climbing을 막는다. 그러나 one trajectory per creator-runtime lineage and limited official pairs 때문에 variance estimate는 충분하지 않다.

## 4-4. Engineering notes

### 1) Harness version을 immutable artifact로 저장해야 한다

Prompt, tool schema, dependency lock, environment image, context policy, stop condition이 모두 versioned되어야 한다. Code diff만으로는 same behavior를 재현하기 어렵다.

### 2) Creator and executor identity를 metadata에 넣어야 한다

Harness score를 기록할 때 creator, executor, model version, reasoning setting을 함께 저장해야 한다. Pair가 바뀌면 score meaning도 달라진다.

### 3) Visible and held-out feedback channel을 분리해야 한다

Evolution agent가 hidden score or task content를 볼 수 없도록 access boundary를 명확히 해야 한다. Otherwise self-improvement evaluation이 benchmark tuning으로 바뀐다.

### 4) Dead mechanism test가 필요하다

State, memory, retry, verifier code가 존재하는지보다 actual trajectory에서 trigger되는지 측정해야 한다. Coverage and event log가 useful하다.

### 5) Cost-normalized frontier를 봐야 한다

Success score가 같다면 executor token, wall-clock, tool calls, API cost가 낮은 harness가 더 practical하다. Pareto frontier로 보는 편이 좋다.

### 6) Harness-level unit test와 task-level evaluation을 분리해야 한다

Tool parsing and retry logic unit test는 필요하지만 downstream capability를 대체하지 않는다. Internal test count를 success proxy로 쓰면 안 된다.

# 5. Evaluation

## 5-1. Creation results

Self-Eval에서 main results는 다음과 같다. Average는 SWE-Pro, Terminal-Bench, EQ-Bench3, BrowseComp의 unweighted mean이며 metric이 다른 MLE-bench medal rate는 average에서 제외한다.

| Creator | SWE-Pro | Terminal | MLE medal | EQ-Bench3 | BrowseComp | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Opus 4.8 | 69.3 | 64.8 | 32.9 | 84.6 | 52.4 | 67.8 |
| GPT-5.5 | 32.8 | 52.1 | 19.1 | 83.0 | 52.6 | 55.1 |
| Gemini 3.1 Pro | 43.6 | 68.8 | 32.4 | 74.8 | 35.2 | 55.6 |
| DeepSeek V4 Pro | 28.9 | 35.6 | 19.6 | 75.4 | 40.9 | 45.2 |
| Qwen 3.7 Max | 33.5 | 41.3 | 3.1 | 68.7 | 32.3 | 44.0 |
| Seed 2.0 Pro | 10.8 | 6.0 | 5.3 | 71.1 | 3.2 | 22.8 |
| Human-engineered reference | 80.0 | 88.8 | 24.0 | 83.7 | 92.2 | 86.2 |

Interpretation은 domain별로 다르다.

- Writing: Opus 84.6 versus selected reference 83.7
- ML experimentation: Opus 32.9 and Gemini 32.4 versus selected reference 24.0
- Code: best generated harness는 mature reference에 뒤처진다.
- Search and research: best generated score 52.6 versus reference 92.2로 gap이 가장 크다.

Human reference row는 각 domain의 mature harness-model pair를 사용하므로 strict same-executor comparison이 아니다. Directional upper reference로 읽어야 한다.

## 5-2. Harness defect analysis

MLE-bench failed tasks의 77.8%가 harness defect로 분류된다. 즉 executor가 ML reasoning을 못해서만 실패한 것이 아니다.

Typical defects는 다음과 같다.

- Required artifact를 wrong path에 저장
- Scorer-readable format을 만들지 못함
- Long-running job state를 보존하지 못함
- Validation 없이 corrupted result를 final로 제출
- Timeout and retry logic failure
- Context or observation loss

이 result는 agent benchmark failure를 model intelligence 하나로 해석하면 안 된다는 점을 보여준다.

## 5-3. Executor dependence

Unified-Eval에서 same harness의 score가 크게 달라진다.

대표적으로 Opus-created SWE-Pro harness는 다음 변화가 있다.

| Evaluation | SWE-Pro score |
| --- | ---: |
| Self-Eval with Opus executor | 69.3 |
| Unified-Eval with Gemini executor | 33.0 |

반대로 Qwen-created BrowseComp harness는 Gemini executor에서 self setting보다 좋아진다. Harness and model compatibility가 bidirectional하다는 뜻이다.

따라서 좋은 harness를 다음처럼 정의하기 어렵다.

$$
Q(H) = \text{single scalar}
$$

더 현실적인 평가는 executor distribution에 대한 expected score다.

$$
Q(H)
=
\mathbb{E}_{L_E \sim \mathcal{P}}
[S(H,L_E)]
$$

## 5-4. Runtime cost variation

비슷한 score에서도 executor-token cost가 크게 다르다. MLE-bench에서 GPT harness는 19.1 medal rate를 약 29.3M tokens로 얻고, DeepSeek harness는 19.6을 약 208.4M tokens로 얻는 사례가 보고된다. Score는 비슷하지만 token cost는 약 7x 차이다.

전체 benchmark에서도 generated harness 사이 runtime overhead가 크게 벌어진다. Quality-only ranking은 deployment choice를 왜곡할 수 있다.

## 5-5. Evolution and held-out generalization

Evolution visible feedback에서는 대부분 lineage가 improvement를 보인다. 하지만 held-out transition agreement는 약하다.

- Comparable version switches: 64
- Feedback and held-out direction agreement: 34 of 64, 53.1%
- Declared final version이 held-out best인 lineage: 2 of 9

즉 visible improvement가 coin-flip에 가까운 frequency로 held-out direction과 맞는다. Agent가 final version을 고르는 selection ability도 제한적이다.

Self-runtime lineage의 declared final은 held-out에서 +1.43 to +4.44 point gain을 보이지만, fixed Gemini executor evolution에서는 transfer가 더 약하다. Four fixed-executor lineages 중 Opus-created lineage만 held-out gain을 유지하고, GPT lineage는 -10.32 points까지 하락한다.

## 5-6. What really matters in the experiments

### 1) Creation success보다 transfer gap이 더 중요하다

Zero-to-runnable harness를 만든 것은 의미 있다. 그러나 self-eval score alone으로 reusable agent infrastructure라고 부르기 어렵다.

### 2) Writing and MLE comparison은 reference scope를 봐야 한다

Generated harness가 selected reference를 넘은 domain이 있지만 reference executor and protocol이 완전히 matched control은 아니다. Human-level harness engineering을 이겼다는 broad claim으로 확장하면 안 된다.

### 3) Evolution result는 instability evidence다

Positive held-out mean gain보다 version-level direction agreement and final-selection failure가 더 중요하다. Improvement process가 reliable하지 않음을 보여준다.

### 4) Code size is not quality

18 generated code harness가 많은 line을 추가했지만 line count와 score는 강하게 연결되지 않는다. Gemini harness는 비교적 적은 code로 Terminal-Bench 최고 generated score를 기록한다.

# 6. Limitations

1. **Human reference is heterogeneous.**
   - Mature reference는 different harness-model pair를 사용한다.
   - Creation table의 reference gap은 same-executor causal comparison이 아니다.

2. **Evolution is limited to code domain.**
   - Held-out post-freeze evidence는 SWE-Pro에 집중된다.
   - Writing, MLE, research harness evolution generalization은 직접 검증되지 않는다.

3. **One trajectory per lineage.**
   - Nine evolution trajectories의 stochastic variance를 분리하기 어렵다.
   - Repeated creator runs and confidence interval이 부족하다.

4. **Held-out set is same benchmark family.**
   - SWE-Pro 100 feedback versus 630 held-out은 useful split이지만 cross-domain transfer는 아니다.

5. **Unified executor is one model.**
   - Gemini fixed executor result가 arbitrary executor population을 대표하지 않는다.
   - Multiple executor family matrix가 필요하다.

6. **High evaluation cost.**
   - 2,207 tasks, frontier creator models, repeated execution, token logging이 필요하다.
   - Small lab에서 full reproduction cost가 크다.

7. **Public code release scope is limited.**
   - 2026-09-21 기준 paper and project page는 확인되지만 full benchmark repository and all frozen harness artifact 공개 여부는 추가 확인이 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

HarnessDev의 핵심은 LLM이 harness를 만들 수 있다는 headline보다 evaluation unit을 바꾼 데 있다. Agent failure를 model, harness, executor-harness fit, task distribution, cost의 다섯 축으로 나누지 않으면 improvement source를 잘못 해석한다.

Research agent or coding agent를 개발할 때도 다음 matrix가 필요하다.

| Axis | Question |
| --- | --- |
| Creator quality | Harness authoring and diagnosis을 잘하는가 |
| Executor quality | Frozen harness 안에서 task를 잘 푸는가 |
| Portability | 다른 executor에서도 behavior가 유지되는가 |
| Generalization | Visible feedback gain이 held-out에 남는가 |
| Efficiency | 같은 success에 token and tool cost가 얼마나 드는가 |

## 7-2. Reuse potential

### 1) Internal agent platform benchmark

회사 내부 harness version을 frozen artifact로 만들고, model family matrix에서 score and cost를 비교할 수 있다.

### 2) Harness pull-request gate

Every harness change를 visible regression suite and sealed held-out suite에 모두 실행하고, score direction agreement를 기록할 수 있다.

### 3) Trigger coverage for state and verifier

Declared memory, retry, verifier mechanism이 actual trajectory에서 호출되는지 event coverage를 CI metric으로 둘 수 있다.

### 4) Cost-aware model routing

Harness-executor pair별 success per million tokens를 추정해 task domain에 따라 model을 route할 수 있다.

### 5) Evolution selection research

Agent가 new version을 만드는 것보다 어느 version을 keep or revert할지 판단하는 meta-verifier 연구가 중요하다. HarnessDev의 2 of 9 result가 이 gap을 직접 보여준다.

## 7-3. Follow-up papers

- Automated Design of Agentic Systems
- HarnessOpt-Bench
- Harness Updating Is Not Harness Benefit
- ModularRSI: modular and generalizable harness self-improvement
- OpenForgeRL: training harness-native agents
- Harness-of-Harness: multi-day autonomous software development

# 8. Summary

- HarnessDev는 task answer가 아니라 runnable agent harness를 평가 대상으로 둔다.
- Creation은 weak seed and few development cases에서 complete harness를 만들게 한다.
- Evolution은 visible execution feedback으로 harness를 반복 수정하게 한다.
- Six creator models, four domains, five benchmarks, 2,207 instances를 평가한다.
- Generated harness는 writing and ML experimentation에서 강하지만 code and research에서는 mature reference에 뒤처진다.
- Self-Eval gain은 fixed executor로 쉽게 무너지며 harness-model co-adaptation이 크다.
- Evolution change 64개 중 visible and held-out direction이 일치한 것은 34개다.
- Agent self-improvement의 병목은 change generation뿐 아니라 validation, selection, portability다.
