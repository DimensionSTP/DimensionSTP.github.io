---
layout: single
title: "OpenForgeRL: Train Harness-native Agents in Any Environment Review"
categories: Study-concept
tag: [Agentic-RL, Agent-Harness, Reinforcement-Learning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2607.21557)

> 한 줄 요약: OpenForgeRL은 Claude Code, Codex, OpenClaw 같은 stateful inference harness를 단순한 training-time simulator로 다시 구현하지 않고, 실제 harness가 내는 model call을 proxy로 기록하고 remote container에서 rollout해 standard SFT 및 RL stack과 연결하는 framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Modern agent의 성능이 base model뿐 아니라 harness, tool loop, context management, subagent orchestration에 크게 의존한다.
- 기존 open RL stack이 표현하기 어려운 stateful, multi-process harness rollout을 standard trajectory로 변환한다.
- Training environment와 deployment harness를 동일하게 유지해 train-deploy mismatch를 줄인다.
- Tool or claw agent와 multimodal GUI agent를 같은 infrastructure pattern으로 학습한다.
- Harness choice가 agent learning difficulty와 generalization에 미치는 영향을 실제 RL experiment로 분석한다.

최근 agent system은 model endpoint 하나로 구성되지 않는다. 실제 deployment에는 다음 요소가 함께 들어간다.

- System prompt와 role policy
- Tool schema와 MCP server
- Shell, browser, desktop, file system
- Context compaction과 observation filtering
- Retry, validation, reflection loop
- Subagent dispatch와 result aggregation
- Environment-specific timeout와 termination logic

이 전체를 inference harness라고 볼 수 있다. Claude Code, Codex, OpenClaw 같은 harness는 model이 어떤 observation을 보고 어떤 action을 만들며, failure 이후 무엇을 다시 시도하는지 결정한다.

문제는 open training stack이 보통 이런 harness를 직접 학습하지 못한다는 점이다. PPO나 GRPO framework는 model generation과 environment step이 trainer 내부에서 비교적 단순하게 반복된다고 가정한다. 반면 실제 harness는 별도 process를 띄우고, 여러 model call을 만들고, tool execution과 context management를 내부에서 수행한다.

그래서 연구자는 흔히 deployment harness를 축약한 training-only agent loop를 다시 만든다. 이 경우 model은 실제 deployment와 다른 prompt, tool semantics, context policy, retry behavior를 보고 학습한다.

OpenForgeRL은 이 문제를 interface 관점에서 해결한다.

1. Harness는 원래 방식대로 실행한다.
2. Harness가 model API를 호출할 때 proxy가 request와 response를 가로챈다.
3. Proxy가 call sequence를 standard training trajectory로 복원한다.
4. 각 rollout은 독립 remote container에서 실제 environment와 함께 실행한다.
5. Trainer는 harness 내부 구현을 몰라도 recorded token trajectory와 reward를 받아 SFT 또는 RL을 수행한다.

핵심은 complex harness를 RL framework 안으로 옮기는 것이 아니라, harness와 trainer 사이에 thin compatibility layer를 두는 것이다.

# 1. Problem Setting

## 1-1. Problem definition

Harness-native agent training은 다음 objective를 가진다.

- Training 때 사용한 harness와 deployment harness가 같아야 한다.
- Harness가 몇 번의 model call을 만드는지 미리 고정하지 않아도 된다.
- Tool, browser, GUI, subagent가 섞인 stateful process를 지원해야 한다.
- Rollout environment를 GPU training node와 분리할 수 있어야 한다.
- Standard SFT, PPO, GRPO backend가 소비할 수 있는 token trajectory로 변환되어야 한다.

Harness가 만드는 trajectory를 다음처럼 나타낼 수 있다.

$$
\tau
=
\left(
(s_0^H, a_0, r_0),
(s_1^H, a_1, r_1),
\ldots,
(s_T^H, a_T, r_T)
\right)
$$

여기서 $s_t^H$는 harness가 구성한 model input state이고, $a_t$는 model response다. State에는 system prompt, prior tool result, compacted context, subagent message가 포함될 수 있다.

Environment reward가 episode 끝의 terminal reward $r_T$만 제공한다면 각 model call에 다음처럼 return을 배정할 수 있다.

$$
r_t
=
\gamma^{T-t} r_T
$$

논문 setting에서는 보통 $\gamma=1$을 사용한다. 즉 하나의 successful rollout 안에서 여러 model response가 같은 terminal reward를 공유한다.

Group-based RL에서는 같은 task에서 여러 rollout을 생성한 뒤 group 내부 상대 성과로 advantage를 계산한다. 그러나 harness마다 call count와 trajectory length가 다르므로 standard single-turn formulation보다 mask와 token alignment가 중요해진다.

논문의 problem setting은 다음과 같이 정리할 수 있다.

| Question | OpenForgeRL의 답 |
| --- | --- |
| Harness를 누가 실행하는가 | 원래 harness process |
| Model call은 어디로 가는가 | Lightweight proxy |
| Training sample은 어떻게 얻는가 | Prompt-response call을 기록하고 trajectory로 재구성 |
| Environment는 어디서 도는가 | Rollout별 remote container |
| Scale-out은 어떻게 하는가 | Kubernetes orchestrator |
| Trainer는 무엇을 보는가 | Standard token trajectory, mask, reward |
| 어떤 task를 다루는가 | Tool/claw, browser, desktop GUI |
| 어떤 mismatch를 줄이는가 | Simplified training harness와 deployment harness의 차이 |

## 1-2. Why previous approaches are insufficient

### 1) Bare-model training은 harness capability를 학습하지 못한다

Agent benchmark에서 같은 model이라도 harness가 바뀌면 성능이 크게 달라질 수 있다. Tool naming, context formatting, action verification, retry policy가 model behavior를 바꾸기 때문이다.

Bare model에 tool-call data를 넣는 것만으로는 실제 harness의 multi-step control flow를 재현하기 어렵다.

### 2) Simplified training harness는 train-deploy mismatch를 만든다

Training stack에 맞춰 ReACT loop를 간단히 구현하면 rollout은 쉬워진다. 하지만 deployment가 OpenClaw나 Codex라면 다음 차이가 생긴다.

- System prompt가 다르다.
- Tool surface가 다르다.
- Intermediate context와 memory가 다르다.
- Error message formatting이 다르다.
- Subagent와 validator call이 빠질 수 있다.
- Termination semantics가 다르다.

Model은 training에서 보지 못한 harness behavior를 deployment에서 만나게 된다.

### 3) Complex environment를 GPU node에 co-locate하기 어렵다

GUI desktop, browser, MCP server, task-specific software는 CPU, RAM, network, container image를 요구한다. Training GPU node에 모든 environment를 함께 띄우면 resource interference와 security 문제가 커진다.

또한 task마다 Dockerfile과 dependency가 다를 수 있다. Rollout을 remote container로 분리해야 elastic scaling과 fault isolation이 가능하다.

### 4) Harness trajectory는 fixed turn으로 자르기 어렵다

어떤 harness는 하나의 high-level action을 여러 model call로 나눈다. 다른 harness는 긴 context 하나에서 큰 code action을 생성한다.

Turn count를 공통 budget으로 쓰면 harness마다 불공정하다. 논문은 wall-clock timeout을 사용해 harness-specific internal turn semantics를 강제하지 않는다.

### 5) Infrastructure failure와 policy failure를 구분해야 한다

Rollout container crash, network timeout, tool server failure를 모두 reward 0으로 기록하면 model은 자기 책임이 아닌 실패에서 negative signal을 받는다.

OpenForgeRL은 infrastructure 또는 harness error가 발생한 trajectory를 discard한다. 이는 wrong credit assignment를 줄이지만, 실제 deployment reliability를 학습 대상으로 포함하지 못한다는 trade-off도 있다.

# 2. Core Idea

## 2-1. Main contribution

### 1) Harness model call proxy

Proxy는 harness가 호출하는 model endpoint처럼 동작한다.

Harness 입장에서는 일반 API server를 호출한다. Proxy는 request를 policy inference server로 전달하고 response를 다시 harness에 돌려준다. 동시에 다음 정보를 기록한다.

- Full prompt 또는 message sequence
- Generated response
- Tokenization 결과
- Call order
- Rollout id와 task id
- Harness metadata
- Action mask 또는 loss mask
- Final reward와 termination status

이렇게 하면 harness source code를 RL loop에 맞춰 크게 수정하지 않아도 된다.

### 2) Automatic trajectory reconstruction

한 episode에서 여러 prompt-response pair가 발생한다. 이들을 standard RL sample로 바꾸려면 context overlap을 처리해야 한다.

예를 들어 second call의 prompt에는 first call response와 tool result가 포함될 수 있다. Naive concatenation은 token을 중복하고 loss mask를 잘못 만든다.

OpenForgeRL은 recorded call sequence에서 policy-generated token만 training target으로 표시하고, harness observation과 tool output은 context로 유지한다. Trainer는 reconstructed trajectory를 일반 multi-turn rollout처럼 처리한다.

### 3) Remote container orchestrator

각 rollout은 task-specific container에서 실행된다.

- Container image에 harness와 environment dependency를 넣는다.
- Kubernetes pod로 rollout을 띄운다.
- CPU와 RAM limit를 독립적으로 설정한다.
- Model inference는 별도 GPU server를 공유한다.
- Environment artifact와 verifier가 terminal reward를 계산한다.
- 실패한 pod는 retry 또는 discard한다.

Training GPU와 environment worker를 분리하므로 GUI, browser, tool server를 elastic하게 늘릴 수 있다.

### 4) Any harness x any environment abstraction

Framework의 목표는 특정 harness를 위한 bespoke trainer가 아니다. Harness와 environment를 orthogonal한 component로 본다.

- Harness examples: ReACT, ZeroClaw, OpenClaw, Codex
- Environment examples: tool tasks, MCP tasks, browser, desktop GUI

같은 policy를 여러 harness에서 학습하고, unseen harness에서 transfer를 측정할 수 있다.

### 5) Harness-native data synthesis

RL environment를 만들기 위해 task를 다음 pipeline으로 구성한다.

1. Candidate task를 제안한다.
2. Ambiguous하거나 trivial한 task를 제거한다.
3. Executable environment와 verifier를 만든다.
4. Open model로 실제 rollout해 solvability를 확인한다.
5. Environment bug와 reward loophole을 수정한다.
6. 다시 test하고 SFT 및 RL split을 만든다.

이 과정은 natural-language prompt collection보다 비싸지만 terminal reward의 validity를 높인다.

## 2-2. Design intuition

### 1) Harness를 training code로 translate하지 말고 API boundary에서 observe한다

Complex harness는 이미 deployment behavior를 정의한다. 이를 RL framework에 맞춰 다시 구현하면 semantic drift가 생긴다.

Model API는 거의 모든 harness가 반드시 통과하는 narrow waist다. Proxy를 이 boundary에 두면 harness 내부가 multi-process인지, subagent를 쓰는지, context를 어떻게 압축하는지 몰라도 model interaction을 기록할 수 있다.

### 2) Environment execution과 policy learning을 decouple한다

Rollout은 CPU-heavy, network-heavy, stateful하다. Optimization은 GPU-heavy, synchronous 또는 asynchronous하다.

두 workload를 같은 node와 process model에 억지로 맞추지 않고, remote container queue로 분리하면 각각 독립적으로 scale할 수 있다.

### 3) 실제 deployment harness에서 학습해야 harness-specific difficulty가 드러난다

OpenClaw처럼 긴 prompt와 많은 tool surface를 가진 harness는 ZeroClaw보다 학습하기 어려울 수 있다. Simplified simulator에서는 이 차이를 볼 수 없다.

Harness-native rollout은 model이 실제로 받는 context와 error surface를 보존하므로, 어느 harness가 learnable한지 연구할 수 있다.

### 4) RL의 효과를 final score뿐 아니라 behavior로 분석한다

논문은 RL 이후 다음 behavior를 본다.

- Self-verification
- Required tool coverage
- Multi-step plan completion
- Step efficiency
- Error recovery

Agent RL이 단순한 task memorization인지, reliability improvement인지 구분하려는 분석이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Deployment harness를 그대로 사용한 end-to-end SFT 및 RL |
| Core component 1 | Model-call proxy |
| Core component 2 | Kubernetes rollout orchestrator |
| Training backend | Standard RL codebase such as veRL |
| Rollout unit | One task in one remote container |
| Environment types | Tool/claw, browser, desktop GUI |
| Agent types | LLM and VLM |
| Budget control | Wall-clock timeout |
| Reward | Executable verifier based terminal reward |
| Main benefit | Reduced train-deploy mismatch |

## 3-2. Module breakdown

### 1) Harness process

Harness는 평소 deployment와 같은 방식으로 실행된다.

- Prompt를 구성한다.
- Model endpoint를 호출한다.
- Response를 tool action으로 해석한다.
- Environment observation을 다시 context에 넣는다.
- 자체 retry, context compaction, subagent logic을 수행한다.

OpenForgeRL은 harness 내부 policy를 standardize하지 않는다. 이것이 portability의 근거다.

### 2) Proxy server

Proxy는 두 plane을 연결한다.

| Plane | Role |
| --- | --- |
| Inference plane | Harness request를 policy server로 전달 |
| Data plane | Prompt-response pair와 token metadata를 기록 |
| Control plane | Rollout id, timeout, cancellation, reward association |
| Training plane | Reconstructed trajectory를 buffer 또는 trainer로 전달 |

Proxy가 correctness를 위해 보존해야 할 것은 단순 text가 아니다.

- Exact message order
- Tokenizer version
- Sampling parameter
- Stop condition
- Assistant-generated span
- Tool-generated span
- Repeated prefix
- Truncation and context compaction result

Tokenizer mismatch가 있으면 log probability와 action mask가 어긋날 수 있다.

### 3) Trajectory reconstruction

Episode가 $T+1$ model calls로 구성될 때, 각 call의 generated span을 policy action으로 본다. Tool observation과 harness-generated instruction은 state token이다.

Loss는 policy token에만 적용한다.

$$
\mathcal{L}_{\mathrm{policy}}
=
-\sum_{t=0}^{T}
\sum_{i \in \mathcal{A}_t}
\hat{A}_t
\log \pi_\theta
\left(
a_{t,i}
\mid
s_t^H, a_{t,<i}
\right)
$$

여기서 $\mathcal{A}_t$는 $t$번째 call에서 model이 실제 생성한 token index set이고, $\hat{A}_t$는 rollout-level reward에서 만든 advantage다.

Tool output을 policy target에 포함하면 model이 environment message를 생성하도록 학습될 수 있으므로 mask가 중요하다.

### 4) Containerized environment

Task는 Docker image 또는 Dockerfile과 함께 배포된다.

Tool task는 MCP server, local API, shell program을 포함할 수 있다. GUI task는 virtual display, browser 또는 desktop application을 포함한다. 논문 setup에서는 Xvfb를 사용해 headless GUI를 제공한다.

각 pod는 다음 lifecycle을 가진다.

1. Image pull and initialization
2. Task state loading
3. Harness startup
4. Proxy connection
5. Agent rollout
6. Verifier execution
7. Artifact upload
8. Cleanup

### 5) Orchestrator and queue

Kubernetes orchestrator는 rollout request를 pod로 변환하고 status를 추적한다.

- Pending
- Running
- Successful
- Policy failure
- Environment failure
- Timeout
- Cancelled

Policy failure와 infrastructure failure를 분리해야 reward data가 오염되지 않는다. 다만 어떤 failure를 discard할지 rule이 너무 넓으면 hard negative가 사라질 수 있다.

### 6) SFT data generation

SFT trajectory는 stronger teacher harness rollout에서 얻는다. 논문은 task당 여러 rollout을 만들고 successful trajectory만 유지한다.

Claw setting에서는 MiniMax-M2.5를 teacher로 사용하고 task당 3개 rollout 중 성공한 trajectory를 선택한다. GUI setting에서는 Kimi-K2.5를 teacher로 사용한다.

SFT는 base model이 harness syntax와 environment interaction에 진입할 수 있게 한다. 이후 RL은 terminal reward로 reliability를 개선한다.

### 7) GRPO training

Claw model은 Qwen3-30B-A3B-Thinking을 backbone으로 사용하고 SFT checkpoint에서 GRPO를 수행한다. GUI model은 Qwen3-VL-8B-Thinking을 사용한다.

Group size와 batch 구성은 task당 여러 rollout을 비교하도록 설계된다. Same-task group 안에서 reward variation이 없으면 advantage가 약해질 수 있으므로 task difficulty와 rollout diversity가 중요하다.

# 4. Training / Data / Recipe

## 4-1. Dataset construction

논문은 세 environment family를 구축한다.

| Environment family | SFT trajectories | RL tasks | Example harness |
| --- | ---: | ---: | --- |
| Claw / tool | 892 | 343 | ReACT, ZeroClaw, OpenClaw, Codex |
| GUI computer | 795 | 252 | Computer-use harness |
| GUI browser | 1496 | 900 | Browser-use harness |

Task 수는 frontier pretraining에 비해 작다. 핵심은 각 task가 executable environment와 verifier를 가진다는 점이다.

Data construction의 병목은 prompt generation이 아니라 다음 항목이다.

- Environment determinism
- Credential and network control
- Verifier correctness
- Reward hacking 방지
- Reset reliability
- Task dependency versioning
- GUI state reproducibility

## 4-2. Claw agent training

Claw setting의 주요 recipe는 다음과 같다.

- Backbone: Qwen3-30B-A3B-Thinking
- Teacher for SFT: MiniMax-M2.5
- SFT trajectory selection: Task당 3회 teacher rollout 중 successful trace
- RL algorithm: GRPO
- RL backend: veRL
- Training hardware: 8 x B200
- Rollout cloud: Microsoft Azure
- Rollout unit: Task-specific Kubernetes pod
- Pod resource cap: 2 CPU, 2GB RAM
- Harnesses: Multiple claw-style harnesses

Multi-harness training은 model이 특정 prompt template만 외우지 않도록 한다. 하지만 harness distribution이 넓어지면 SFT와 RL variance도 커진다.

## 4-3. GUI agent training

GUI setting의 주요 recipe는 다음과 같다.

- Backbone: Qwen3-VL-8B-Thinking
- Teacher for SFT: Kimi-K2.5
- Environment: Browser and desktop GUI
- Headless display: Xvfb
- RL backend: veRL
- Training hardware: 8 x B200
- Rollout: Remote Azure containers

GUI agent는 text tool agent보다 state가 크다.

- Screenshot
- Cursor and viewport
- Application state
- Click or type action
- Long visual trajectory

Proxy는 VLM request와 image input도 일관되게 기록해야 한다.

## 4-4. Timeout and failure handling

논문은 common turn count보다 wall-clock timeout을 사용한다. Harness마다 한 turn의 의미가 다르기 때문이다.

이 선택에는 trade-off가 있다.

- 장점: Complex harness가 internal call 수를 자유롭게 사용한다.
- 단점: Slow harness나 network delay가 policy budget과 섞인다.
- 장점: Deployment-like runtime constraint를 반영한다.
- 단점: Same task에서 compute effort를 정확히 맞추기 어렵다.

Infrastructure failure는 학습 데이터에서 제거한다. Production에서는 이런 failure도 중요한 reliability axis이므로 별도 benchmark가 필요하다.

## 4-5. Engineering notes

### 1) Proxy는 deterministic logging을 보장해야 한다

Request retry가 발생하면 같은 call이 두 번 기록될 수 있다. Streaming response가 중간에 끊기면 partial action 처리 rule이 필요하다.

### 2) Container image는 evaluation contract의 일부다

Package version, browser version, display resolution, locale가 바뀌면 GUI task가 달라질 수 있다. Image digest와 task asset checksum을 저장해야 한다.

### 3) Reward verifier는 harness보다 먼저 검증해야 한다

Agent가 success artifact를 만들지 않고 verifier loophole을 이용할 수 있다. Synthetic task pipeline에서 verifier adversarial test가 중요하다.

### 4) Rollout throughput은 GPU utilization과 직접 연결된다

Remote environment가 policy server에 충분한 request를 공급하지 못하면 8-GPU trainer가 idle해진다. Concurrent pod 수, request batching, rollout length variance를 함께 관리해야 한다.

# 5. Evaluation

## 5-1. Main results

### 1) OpenForge-Claw

SFT와 RL을 거친 OpenForge-Claw는 다음 result를 보고한다.

| Benchmark | OpenForge-Claw SFT+RL |
| --- | ---: |
| ClawEval pass^3 | 31.7 |
| ClawEval pass@3 | 55.9 |
| QwenClawBench | 33.7 |
| MCPAtlas | 28.1 |

SFT-only checkpoint는 다음과 같다.

| Benchmark | SFT only | SFT+RL |
| --- | ---: | ---: |
| ClawEval pass^3 | 21.7 | 31.7 |
| ClawEval pass@3 | 52.1 | 55.9 |
| QwenClawBench | 32.1 | 33.7 |
| MCPAtlas | 23.6 | 28.1 |

RL gain은 metric마다 다르다. Single-run reliability 성격이 강한 pass^3와 MCPAtlas에서 상대적으로 뚜렷하고, pass@3와 QwenClawBench에서는 작다.

이 차이는 RL이 candidate diversity보다 per-rollout reliability를 높였을 가능성과 연결된다.

### 2) OpenForge-GUI

GUI model의 main result는 다음과 같다.

| Benchmark | SFT only | SFT+RL |
| --- | ---: | ---: |
| OSWorld-Verified | 34.4 | 37.7 |
| Online-Mind2Web | 57.4 | 63.0 |
| WebVoyager | 61.5 | 72.3 |

WebVoyager에서 gain이 가장 크고, OSWorld-Verified에서는 더 작다. Browser task와 general desktop task의 action complexity 차이를 고려해야 한다.

### 3) Multi-harness training

Harness별 generalization experiment는 중요한 결과다.

| Training | ZeroClaw eval | OpenClaw eval | Codex eval |
| --- | ---: | ---: | ---: |
| Base | 32.5 | 11.4 | 12.2 |
| ZeroClaw only | 46.0 | 14.7 | 16.8 |
| Multi-harness | 48.5 | 20.9 | 32.5 |

ZeroClaw-only training도 unseen harness에 일부 transfer된다. Multi-harness training은 OpenClaw와 Codex에서 더 큰 improvement를 보인다.

이는 harness-specific surface와 underlying agent skill이 완전히 분리되지는 않지만, 여러 interface를 보면 더 robust한 behavior를 학습할 수 있음을 시사한다.

### 4) RL behavior analysis

RL 이후 다음 변화가 관찰된다.

- Generic shell call 비중이 22.6%에서 13.9%로 감소
- Episode length가 약간 짧아짐
- Self-verification 증가
- Required service 또는 tool coverage 증가
- Multi-step plan completion 개선
- Step reliability 개선
- Error recovery는 여전히 약함

Agent가 무작정 shell을 호출하기보다 task-specific tool을 더 잘 선택하게 된다는 해석이 가능하다.

## 5-2. What really matters in the experiments

### 1) Framework evaluation과 model evaluation을 분리해야 한다

Score가 높아진 이유는 세 가지가 섞여 있다.

- Better task data
- Harness-native SFT
- RL
- Multi-harness exposure
- Remote environment infrastructure

Framework 자체의 value는 특정 number보다 새로운 harness를 연결할 때 얼마나 적은 modification이 필요한지에 있다.

### 2) Pass^3와 pass@3는 다른 질문이다

pass@3는 세 번 중 하나라도 성공하는가를 본다. pass^3는 반복 run reliability를 더 엄격하게 본다.

RL gain이 pass^3에서 큰 것은 agent가 occasional success보다 consistent execution에 가까워졌다는 신호일 수 있다.

### 3) Harness difficulty는 model difficulty와 별개다

OpenClaw가 ZeroClaw보다 낮은 score를 보이는 이유는 model capacity만이 아니다. Longer prompt, larger tool surface, internal orchestration complexity가 learning problem을 바꾼다.

Agent benchmark에서 harness를 고정하지 않으면 model comparison이 흔들리는 이유다.

### 4) Error recovery가 남는다는 결과가 중요하다

RL이 self-verification과 tool coverage를 높여도 이미 발생한 잘못된 action을 복구하는 능력은 약하다.

Error recovery에는 다음 data가 별도로 필요할 수 있다.

- Failed action -> diagnosis -> corrected action
- Tool timeout -> alternate path
- Partial state corruption -> reset or rollback
- Incorrect assumption -> evidence re-check
- GUI modal or focus error -> recovery sequence

### 5) Cloud-scale result의 reproducibility를 봐야 한다

Training GPU뿐 아니라 Azure container fleet, Kubernetes scheduler, task image가 필요하다. Framework가 open이어도 full experiment reproduction cost는 작지 않다.

# 6. Limitations

1. Infrastructure complexity가 높다.
   - Kubernetes, cloud container fleet, policy server, proxy, trainer, artifact store를 함께 운영해야 한다.
   - Small lab에서는 framework integration cost가 model training cost보다 클 수 있다.

2. Terminal reward의 credit assignment가 거칠다.
   - Episode의 모든 model call이 같은 reward를 공유하면 어떤 decision이 성공과 실패를 만들었는지 구분하기 어렵다.
   - Long trajectory에서는 action-level or span-level reward가 필요할 수 있다.

3. Infrastructure failure를 discard하면 deployment failure를 학습하지 못한다.
   - Wrong negative를 피하는 장점이 있지만 timeout, flaky tool, partial response에 대한 robustness는 별도 data가 필요하다.

4. Synthetic task와 verifier quality에 의존한다.
   - Environment bug나 verifier loophole은 RL reward hacking으로 이어질 수 있다.
   - Task가 실제 user workflow를 얼마나 대표하는지도 따로 검증해야 한다.

5. Benchmark protocol이 완전히 동일하지 않을 수 있다.
   - Harness, action budget, timeout, browser setup, sampling 횟수가 baseline마다 다르면 score comparison이 흔들린다.
   - Main table의 protocol note를 함께 읽어야 한다.

6. Wall-clock timeout도 완전한 fairness를 보장하지 않는다.
   - Slow harness는 같은 시간에 fewer decisions를 수행할 수 있다.
   - Fast but verbose harness와 slow but efficient harness의 compute budget을 비교하기 어렵다.

7. Security와 isolation 문제가 크다.
   - Agent가 shell, browser, MCP server를 사용하므로 secret leakage, network egress, container escape를 관리해야 한다.
   - RL exploration은 일반 inference보다 더 aggressive할 수 있다.

8. Error recovery가 여전히 약하다.
   - RL이 execution reliability를 높였지만 failure 이후의 replanning은 주요 bottleneck으로 남는다.

# 7. My Take

## 7-1. Why this matters for my work

OpenForgeRL의 가장 중요한 아이디어는 새로운 RL algorithm이 아니라 "deployment harness를 training environment로 간주한다"는 점이다.

Agent model을 개선할 때 흔히 다음 세 layer가 분리된다.

1. Model training team은 simplified tool data로 SFT 또는 RL을 한다.
2. Agent team은 별도 harness를 개발한다.
3. Product team은 deployment environment와 permission을 붙인다.

이 구조에서는 model이 실제 harness의 prompt, retry, compaction, tool surface를 보지 못한다. OpenForgeRL은 model API boundary에서 interaction을 기록해 세 layer를 다시 연결한다.

특히 real-world agent에서 harness가 capability의 상당 부분을 만든다면, harness를 benchmark metadata로만 두지 말고 training distribution의 일부로 넣어야 한다는 메시지가 강하다.

## 7-2. Reuse potential

### 1) Internal harness proxy logging

기존 agent service의 LLM endpoint 앞에 transparent proxy를 두고 다음을 기록할 수 있다.

- Request messages
- Response tokens
- Tool result
- Latency
- Error type
- Final task outcome
- User correction
- Harness version

처음부터 RL을 하지 않더라도 high-quality SFT trace와 failure taxonomy를 만들 수 있다.

### 2) Harness A/B training

같은 task와 model을 여러 harness에서 rollout한다.

- ReACT
- CodeAct
- Production harness
- Minimal harness
- Memory-enabled harness

Success rate뿐 아니라 prompt length, model call count, tool coverage, recovery rate를 비교하면 harness effect를 정량화할 수 있다.

### 3) Containerized task package

Internal workflow를 다음 contract로 만들 수 있다.

- Docker image
- Initial state
- Task prompt
- Allowed tools
- Output artifact
- Deterministic verifier
- Timeout
- Cleanup rule

이 package는 SFT data generation, RL, regression testing에 공통으로 쓸 수 있다.

### 4) Failure-aware trajectory filtering

Discard rule을 단순히 success/failure로 두지 않고 다음 category로 나눈다.

- Policy failure
- Tool misuse
- Verifier failure
- Environment crash
- Network failure
- Harness parse error
- Timeout
- Safety interruption

정책 학습과 infrastructure debugging에 필요한 trace를 분리할 수 있다.

### 5) Error recovery curriculum

OpenForgeRL 결과에서 약한 recovery를 별도 curriculum으로 만들 수 있다.

1. Environment에 recoverable fault를 inject한다.
2. Failure observation을 명시적으로 노출한다.
3. Rollback, retry, alternate tool use에 partial reward를 준다.
4. Final success와 recovery efficiency를 함께 평가한다.

## 7-3. Follow-up papers

- veRL: Volcano Engine Reinforcement Learning for LLMs
- Orchard: An Open-Source Agentic Modeling Framework
- OpenHands: An Open Platform for AI Software Developers
- SWE-Agent: Agent-Computer Interfaces Enable Automated Software Engineering
- WebGym: Scaling Training Environments for Visual Web Agents
- OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks
- MCP-Atlas: Tool-Use Competency with Real MCP Servers
- Agent Lightning: Train Any AI Agent with Reinforcement Learning

# 8. Summary

- OpenForgeRL은 actual deployment harness의 model call을 proxy로 기록해 standard SFT 및 RL trajectory로 변환한다.
- Kubernetes orchestrator는 각 rollout을 task-specific remote container에서 실행해 training GPU와 environment workload를 분리한다.
- Claw agent와 GUI agent 모두 SFT 이후 RL로 benchmark 성능과 self-verification, tool coverage, multi-step reliability가 개선된다.
- Multi-harness training은 ZeroClaw뿐 아니라 unseen OpenClaw와 Codex harness로의 transfer를 높인다.
- Framework의 핵심 가치는 특정 score보다 train-deploy mismatch를 줄이고 harness choice를 학습 변수로 만들었다는 점이다.
- Terminal reward credit assignment, cloud complexity, verifier quality, error recovery는 여전히 중요한 한계다.
