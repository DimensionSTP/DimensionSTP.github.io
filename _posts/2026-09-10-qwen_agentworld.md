---
layout: single
title: "Qwen-AgentWorld: Language World Models for General Agents Review"
categories: Study-concept
tag: [Qwen-AgentWorld, LanguageWorldModel, AgentRL, AgentSimulation, AgentEvaluation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.24597)

[Code link](https://github.com/QwenLM/Qwen-AgentWorld)

[Model link](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)

[Benchmark link](https://huggingface.co/datasets/Qwen/AgentWorldBench)

Qwen-AgentWorld는 "agent benchmark를 하나 더 만든 논문"으로 읽으면 핵심을 놓치기 쉽다. 이 논문이 실제로 겨냥하는 문제는 더 크다. **agent가 실제 environment에서 action을 실행하지 않고도, language world model이 다음 observation을 시뮬레이션하고 그 시뮬레이션을 다시 agent training과 evaluation에 쓸 수 있는가**다.

기존 LLM agent는 tool, terminal, web, OS, mobile, code environment에 직접 붙어서 학습하거나 평가된다. 이 방식은 현실적이지만 비용이 크다. 실제 environment는 느리고, 실패가 누적되며, API나 web page가 바뀌고, sandbox를 대량으로 돌려야 한다. 또한 agentic RL에서는 많은 rollout이 필요하므로, real environment만으로 scaling하기 어렵다.

Qwen-AgentWorld는 language world model, 이하 LWM이라는 방향을 택한다. Environment transition을 text로 예측하는 native world model을 학습해, MCP, Search, Terminal, SWE, Android, Web, OS의 7개 agentic domain을 하나의 model 안에서 시뮬레이션한다. Action이 들어오면 model은 next observation을 long chain-of-thought reasoning으로 예측한다. 이 predicted environment가 agentic RL, controllable perturbation, fictional-world construction, agent foundation model warm-up에 쓰인다.

> 한 줄 요약: Qwen-AgentWorld는 10M+ real-world interaction trajectories로 CPT, SFT, RL 3단계 학습을 수행한 language world model이며, 7개 agent domain의 environment transition을 text로 시뮬레이션해 AgentWorldBench 평가, simulated RL, controllable environment construction, agent warm-up에 사용하는 agent infrastructure model이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- World model을 video나 physics simulator가 아니라 language environment simulator로 정의한다.
- Agent training의 bottleneck인 real-environment rollout cost를 language simulation으로 줄이려 한다.
- MCP, Search, Terminal, SWE, Android, Web, OS를 하나의 unified domain set으로 다룬다.
- AgentWorldBench를 통해 world model의 next-observation fidelity를 Format, Factuality, Consistency, Realism, Quality로 평가한다.
- Simulator로 쓰는 것과 foundation model warm-up으로 쓰는 것을 모두 실험한다.
- Agentic RL에서 "environment도 model로 학습할 수 있는가"라는 방향을 Qwen scale에서 제시한다.

이 글에서는 Qwen-AgentWorld를 "agent benchmark model"보다, **general agent를 위한 language-level environment simulator와 pretraining substrate**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Agent는 environment state $s_t$에서 action $a_t$를 실행하고 observation $o_{t+1}$를 받는다.

$$
o_{t+1}
\sim
P_{\mathrm{env}}
\left(
\cdot
\mid
h_t,a_t
\right)
$$

여기서 $h_t$는 previous observations, actions, tool outputs, messages를 포함한 interaction history다. Language world model은 실제 environment transition $P_{\mathrm{env}}$를 language model $M_{\theta}$로 근사한다.

$$
\hat{o}_{t+1}
=
M_{\theta}
\left(
h_t,a_t
\right)
$$

이 model이 충분히 정확하면 세 가지가 가능해진다.

1. Evaluation
   - Agent가 낸 action에 대해 predicted observation quality를 평가한다.
2. Simulation
   - Real environment를 매번 호출하지 않고 rollout을 생성한다.
3. Training
   - Simulated observation으로 agentic RL이나 warm-up training을 수행한다.

문제는 agentic environment가 일반 text completion보다 훨씬 어렵다는 점이다.

- Tool output format이 정확해야 한다.
- Terminal command output은 plausible하고 stateful해야 한다.
- Web과 Android UI는 state를 보존해야 한다.
- SWE environment는 repository와 test dynamics를 따라야 한다.
- Search task는 information retrieval behavior를 simulate해야 한다.
- OS environment는 file과 app state transition을 따라야 한다.

따라서 LWM은 단순 next text model이 아니라 environment transition model이다.

## 1-2. Why previous approaches are insufficient

### 1) Real environment rollout only

Real environment rollout은 faithful하지만 expensive하다. Agentic RL에서는 수천, 수만 rollout이 필요하다. Terminal, web, SWE, Android, OS를 모두 real environment로 돌리면 latency, infrastructure, instability, cost가 병목이 된다.

### 2) Benchmark-only agent evaluation

Static benchmark는 agent가 실제 environment를 어떻게 traverse하는지 일부만 본다. Benchmark는 saturation되거나 drift될 수도 있다. LWM은 새로운 environment variation을 만들 수 있는 controllable simulator를 목표로 한다.

### 3) Narrow simulators

Domain-specific simulators는 useful하지만 MCP, Search, Terminal, SWE, Android, Web, OS를 하나의 reasoning model에서 다루지는 않는다. Cross-domain agent foundation model을 만들려면 unified environment modeling이 필요하다.

### 4) Post-hoc world modeling

기존 agent model에 나중에 environment prediction head를 붙이는 것은 world modeling을 auxiliary task로 만들 수 있다. Qwen-AgentWorld는 CPT stage부터 environment modeling을 training objective로 넣는 native world model을 강조한다.

# 2. Core Idea

## 2-1. Main contribution

Qwen-AgentWorld의 기여는 네 가지로 정리할 수 있다.

1. **Native language world models**
   - Qwen-AgentWorld-35B-A3B와 Qwen-AgentWorld-397B-A17B.
   - 7개 agentic environment domain을 long CoT reasoning으로 시뮬레이션한다.

2. **Three-stage training pipeline**
   - CPT는 state transition dynamics와 professional corpora에서 general-purpose world modeling capability를 주입한다.
   - SFT는 next-state-prediction reasoning을 활성화한다.
   - RL은 hybrid rubric-and-rule reward로 simulation fidelity를 sharpen한다.

3. **AgentWorldBench**
   - 9개 established benchmark에서 5개 frontier model의 real-world interaction으로 구성된다.
   - Format, Factuality, Consistency, Realism, Quality의 5차원 평가를 사용한다.

4. **Two usage paradigms**
   - Scalable agentic RL을 위한 decoupled environment simulator
   - Downstream tool-using agent task를 위한 unified agent foundation model warm-up

## 2-2. Design intuition

Qwen-AgentWorld의 핵심 intuition은 agent environment를 "external fixed runtime"으로만 보지 않는 것이다. Environment transition도 learnable distribution이 될 수 있다.

Agent가 action 이후의 likely observation을 상상할 수 있다면 다음이 가능해진다.

- 가능한 strategy를 저렴하게 test한다.
- Controlled perturbation 위에서 train한다.
- Robustness를 위해 fictional environment를 구성한다.
- Real tool use 전에 transition reasoning을 warm-up한다.

이 관점은 robotics world model과 유사하지만, substrate가 language다. Terminal output, web page, Android UI, MCP tool response, SWE test output은 모두 textual 또는 structured observation으로 표현될 수 있다. 그래서 language model이 world model 역할을 할 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Agentic environment transition을 language로 simulate |
| Model family | Qwen-AgentWorld-35B-A3B와 Qwen-AgentWorld-397B-A17B |
| Released model | Qwen-AgentWorld-35B-A3B |
| Domains | MCP, Search, Terminal, SWE, Android, Web, OS |
| Training data | 10M+ real-world interaction trajectories |
| Training stages | CPT, SFT, RL |
| Benchmark | AgentWorldBench |
| Evaluation dimensions | Format, Factuality, Consistency, Realism, Quality |
| Main uses | Environment simulation과 agent foundation model warm-up |

## 3-2. Module breakdown

### 1) Language world model

Model은 system prompt, interaction history, current action을 입력으로 받고 next observation을 예측한다. 예를 들어 terminal simulator prompt는 model에게 Linux terminal environment를 simulate하고 command result를 출력하라고 요구할 수 있다.

이는 agent policy와 다르다. Policy는 action을 고르고, world model은 environment response를 예측한다.

### 2) Seven unified domains

일곱 domain은 서로 다른 interaction structure를 갖는다.

| Domain | Main challenge |
| --- | --- |
| MCP | Tool protocol과 structured response fidelity |
| Search | Information retrieval과 factual grounding |
| Terminal | Command execution과 file state simulation |
| SWE | Repository state와 test behavior |
| Android | UI state와 app interaction |
| Web | Browser state와 page dynamics |
| OS | Desktop/application state transitions |

하나의 LWM은 이 domain들의 format convention과 state dynamics를 모두 학습해야 한다.

### 3) CPT stage

Continued pretraining은 general-purpose world modeling capability를 주입한다. 논문은 CPT가 state transition dynamics와 augmented professional corpora를 사용한다고 설명한다.

이 stage가 중요한 이유는 world modeling을 late instruction-tuning behavior로 취급하지 않기 때문이다. World modeling은 core pretraining의 일부가 된다.

### 4) SFT stage

Supervised fine-tuning은 next-state-prediction reasoning을 활성화한다. Model은 history와 action에 condition된 environment observation을 생성하도록 학습한다.

이 stage에서 model은 simulator로 명시적으로 유용해진다.

### 5) RL stage

RL은 hybrid rubric-and-rule reward를 사용해 simulation fidelity를 높인다. Rule reward는 structured format이나 exact task-specific constraint를 확인할 수 있고, rubric reward는 exact match가 불가능한 경우 realism, consistency, quality를 평가할 수 있다.

Environment simulation에는 deterministic aspect와 open-ended aspect가 모두 있으므로 이런 hybrid reward가 필요하다.

### 6) AgentWorldBench

AgentWorldBench는 predicted observation을 real trajectory와 비교해 평가한다. 5개 dimension의 rubric mean을 0-100 scale로 normalize해 사용한다.

Dimension은 다음과 같다.

- Format
- Factuality
- Consistency
- Realism
- Quality

Benchmark는 7개 domain에 걸쳐 9개 established benchmark에서 5개 frontier model의 real interaction을 사용한다.

# 4. Training / Data / Recipe

## 4-1. Data

논문과 official README는 7개 domain에 걸쳐 10M+ real-world interaction trajectory를 보고한다.

Trajectory는 다음을 포함한다.

$$
(h_t,a_t,o_{t+1})
$$

여기서 $h_t$는 interaction history, $a_t$는 action, $o_{t+1}$는 environment observation이다.

핵심 data property는 size만이 아니라 domain diversity다. Terminal output과 web UI transition은 서로 다른 rule을 따른다. Model은 둘 다 학습해야 한다.

## 4-2. Training strategy

Training pipeline은 다음과 같다.

| Stage | Role |
| --- | --- |
| CPT | Environment transition knowledge와 domain expertise 주입 |
| SFT | Next-state-prediction reasoning 학습 |
| RL | Hybrid reward로 simulation fidelity 개선 |

이는 agent model을 만드는 과정과 닮았지만, target은 action generation이 아니라 environment simulation이다.

## 4-3. Engineering notes

1. **System prompt가 중요하다**
   - Official repo는 domain-specific world model system prompt를 제공한다.
   - 각 domain에는 format과 behavior constraint가 필요하다.

2. **Language-model-only serving을 사용한다**
   - Released 35B-A3B checkpoint에는 visual component를 참조하는 architecture definition이 포함될 수 있지만, 현재 checkpoint는 language weight다.

3. **Agent를 training하기 전에 simulator를 benchmark한다**
   - 나쁜 simulator는 나쁜 policy를 가르칠 수 있다.

4. **Simulator fidelity와 agent policy performance를 분리한다**
   - World model이 높은 score를 받아도 rare state에서 agent를 잘못 이끌 수 있다.

5. **Domain-specific failure mode를 추적한다**
   - Search factuality, terminal state, SWE test, Android UI, web dynamics는 서로 다른 방식으로 실패한다.

# 5. Evaluation

## 5-1. AgentWorldBench

Official README는 다음 overall score를 보고한다.

| Model | Overall |
| --- | ---: |
| GPT-5.4 | 58.25 |
| Claude Opus 4.8 | 56.59 |
| Claude Opus 4.6 | 57.80 |
| Gemini 3.1 Pro | 54.57 |
| Qwen3.5-35B-A3B | 47.73 |
| Qwen3.5-397B-A17B | 54.74 |
| Qwen-AgentWorld-35B-A3B | 56.39 |
| Qwen-AgentWorld-397B-A17B | 58.71 |

README는 Qwen-AgentWorld-397B-A17B가 가장 높은 overall score 58.71을 달성하고, Qwen-AgentWorld-35B-A3B가 LWM training이 없는 Qwen3.5-35B-A3B보다 +8.66 높다고 설명한다.

이는 environment simulation에 대한 rubric score이며, 그 자체가 downstream agent task success를 의미하지는 않는다.

## 5-2. Simulated RL

README는 Qwen-AgentWorld-397B-A17B를 사용한 4k OOD OpenClaw environment simulated RL 결과를 보고한다.

| Model | Claw-Eval | QwenClawBench |
| --- | ---: | ---: |
| Qwen3.5-35B-A3B | 65.4 | 47.9 |
| + Sim RL with Qwen3.6-Plus | 66.7 | 47.8 |
| + Sim RL with Qwen-AgentWorld-397B-A17B | 69.7 | 55.0 |

이는 더 좋은 language world model이 더 유용한 simulated training environment를 만들 수 있다는 claim을 뒷받침한다.

## 5-3. Agent foundation model warm-up

README는 LWM RL warm-up이 여러 downstream agent benchmark에서 Qwen3.5-35B-A3B-SFT를 개선한다고 보고한다.

| Benchmark | Base | w/ LWM RL | Delta |
| --- | ---: | ---: | ---: |
| Terminal-Bench 2.0 | 33.25 | 39.55 | +6.30 |
| SWE-Bench Verified | 64.47 | 67.86 | +3.39 |
| SWE-Bench Pro | 42.18 | 47.42 | +5.24 |
| WideSearch F1 Item | 33.38 | 46.17 | +12.79 |
| Claw-Eval | 53.60 | 64.88 | +11.28 |
| QwenClawBench | 39.76 | 49.43 | +9.67 |
| BFCL v4 | 62.29 | 71.25 | +8.96 |

흥미로운 점은 일부 benchmark가 OOD라는 것이다. World-model training은 transferable transition reasoning을 제공하는 것으로 보인다.

## 5-4. What really matters in the experiments

### 1) Simulator fidelity를 별도로 평가한다

AgentWorldBench는 environment prediction quality를 측정한다. Simulator를 RL에 사용하기 전에 필요한 절차다.

### 2) Simulation은 controllable할 수 있다

논문과 README는 controlled perturbation과 fictional-world construction을 강조한다. Simulator는 real world를 replay하는 데 그치지 않고 targeted stress test를 만들 수 있을 때 더 유용하다.

### 3) Foundation warm-up과 decoupled simulation은 별개다

Qwen-AgentWorld는 external simulator로도, agent model의 training warm-up으로도 사용할 수 있다. 두 use case는 구분되어야 한다.

### 4) LWM도 hallucinate할 수 있다

높은 rubric score가 rare environment transition error를 없애지는 않는다. Simulator-induced policy bias는 여전히 risk다.

# 6. Limitations

1. **Simulator는 approximation이다**
   - Language world model은 그럴듯하지만 틀린 observation을 생성할 수 있다.

2. **Rubric evaluation은 rare failure를 놓칠 수 있다**
   - Format, factuality, consistency, realism, quality는 유용하지만 exhaustive하지 않다.

3. **모든 environment로 scaling하기 어렵다**
   - 7개 domain은 넓지만 real agent environment는 open-ended하다.

4. **Released model은 largest model과 다르다**
   - Official release는 Qwen-AgentWorld-35B-A3B를 포함하지만, 보고된 benchmark score는 397B-A17B가 더 강하다.

5. **Real-environment verification은 여전히 필요하다**
   - Sim RL은 real task에서 검증되어야 한다.

6. **Reward model dependency가 있다**
   - Simulator fidelity를 위한 RL은 hybrid rubric-and-rule reward에 의존한다.

7. **World model이 잘못된 policy habit을 가르칠 수 있다**
   - Simulation에 systematic bias가 있으면 그 안에서 학습한 agent가 simulator artifact에 overfit될 수 있다.

8. **Cost와 latency가 있다**
   - Large-scale rollout에 35B MoE world model을 돌리는 것은 많은 real environment보다 저렴할 수 있지만 여전히 가볍지는 않다.

9. **State explosion이 생긴다**
   - Long-horizon agent environment에는 text history에 완전히 드러나지 않는 hidden state가 있다.

10. **Benchmark contamination과 model comparison 이슈가 있다**
    - Frontier model comparison은 model version과 judge protocol에 의존한다.

# 7. My Take

## 7-1. Why this matters for my work

Qwen-AgentWorld의 핵심은 "agent benchmark를 잘 푸는 model"이 아니라, **agent environment 자체를 language model로 학습 가능한 object로 만든 것**이다.

이 점은 agent training infrastructure에 중요하다. Environment rollout을 simulate할 수 있다면 agentic RL은 real sandbox throughput을 넘어 scale될 수 있다. 하지만 simulator는 deterministic environment처럼 신뢰하면 안 되고, model처럼 audit해야 한다.

## 7-2. Reuse potential

### Agentic RL

Real-environment verification 전에 LWM을 pretraining이나 curriculum용 cheap rollout generator로 사용할 수 있다.

### Stress testing

Controlled environment perturbation은 static benchmark가 포착하지 못하는 agent weakness를 드러낼 수 있다.

### Synthetic environment construction

Fictional-world construction은 real web에만 의존하지 않고 novel search나 tool environment에서 agent를 학습하게 할 수 있다.

### Environment model evaluation

AgentWorldBench-style five-dimensional observation evaluation은 다른 language world model 평가에도 재사용할 수 있다.

## 7-3. Production considerations

- Real environment를 final verifier로 유지한다.
- Domain별 simulator error category를 추적한다.
- Simulator uncertainty나 disagreement를 사용해 real environment 호출 시점을 결정한다.
- World model과 system prompt를 versioning한다.
- Production agent를 simulated observation만으로 학습하지 않는다.
- Controllable simulation은 truth oracle이 아니라 red-team과 curriculum 용도로 사용한다.

## 7-4. Follow-up papers

- World Action Models survey
- EvoArena and EvoMem
- Terminal-Bench
- SWE-Bench
- BFCL
- Agent-as-a-Router
- Verification Horizon
- SimFoundry
- Language agent environment simulation papers

# 8. Summary

- Qwen-AgentWorld는 agentic environment simulation을 위한 language world model이다.
- MCP, Search, Terminal, SWE, Android, Web, OS의 7개 domain을 다룬다.
- Training은 10M+ interaction trajectory 위에서 CPT, SFT, RL로 구성된다.
- AgentWorldBench는 다섯 rubric dimension으로 simulation quality를 평가한다.
- 핵심 promise는 controllable language simulation으로 agentic RL과 foundation warm-up을 scale하는 것이며, real-environment validation은 여전히 필요하다.
