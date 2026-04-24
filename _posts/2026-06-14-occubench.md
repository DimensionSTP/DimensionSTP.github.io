---
layout: single
title: "OccuBench: Evaluating AI Agents on Real-World Professional Tasks via Language Environment Simulation Review"
categories: Study-concept
tag: [AI-Agent, Benchmark, Evaluation, Tool-Use, LES]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.10866)

[Project page](https://gregxmhu.github.io/OccuBench-website/)

[Code](https://github.com/GregxmHu/OccuBench)

[Data](https://huggingface.co/datasets/gregH/OccuBench)

OccuBench를 agent benchmark 하나 더 추가된 논문으로만 읽으면 아쉽다. 이 논문의 진짜 흥미로운 지점은 benchmark item 수가 아니라, **공개 환경이 없는 전문직 업무를 어떻게 평가 가능한 형태로 바꾸는가** 에 있다. 요즘 agent benchmark는 web, desktop, code, 몇몇 API domain에는 점점 강해지고 있지만, 정작 실제로 가치가 큰 의료, 제조, 물류, 공공, 금융, 에너지 같은 영역은 환경을 만들기 어렵다는 이유로 평가 바깥에 남아 있었다.

OccuBench는 이 문제를 LES, Language Environment Simulator 로 푼다. 핵심 발상은 단순하다. 환경 자체를 LLM이 stateful tool-response simulator 로 담당하게 만들면, 더 이상 benchmark coverage가 "공개 API가 있는 도메인"에 묶일 필요가 없다. 그러면 emergency triage, customs processing, production scheduling, nuclear safety monitoring 같은 시나리오도 configuration 수준에서 benchmark로 만들 수 있다.

또 하나 중요한 점은 이 논문이 benchmark를 단순 clean environment leaderboard로 끝내지 않는다는 점이다. real world environment는 timeout, missing field, truncated data, stale cache처럼 겉보기에는 정상처럼 보이는 degraded state를 자주 만든다. OccuBench는 바로 이 부분을 explicit fault, implicit fault, mixed fault로 나눠 agent robustness를 같이 본다. 그래서 이 논문은 benchmark release paper라기보다 **agent evaluation systems paper** 로 읽는 편이 더 맞다.

> 한 줄 요약: OccuBench는 LES를 이용해 100개 전문직 시나리오, 10개 산업군, 65개 세부 도메인을 평가 가능한 stateful tool-use benchmark로 만들고, clean task completion뿐 아니라 fault-injected robustness까지 함께 측정하는 agent evaluation framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- **기존 agent benchmark가 다루지 못한 untestable majority** 를 어떻게 평가로 끌어오는지 구체적인 구조를 보여준다.
- benchmark 설계의 핵심을 dataset 수집이 아니라 **environment construction, solvability verification, fault injection** 으로 옮긴다.
- 실험 결과도 흥미롭다. 평균 점수보다 **industry profile, implicit fault 취약성, simulator quality** 가 더 중요한 메시지로 나온다.

이 논문의 핵심 메시지는 단순하다. 좋은 agent evaluation은 더 많은 task를 모으는 문제가 아니라, **환경을 어떤 인터페이스로 시뮬레이션하고, 그 환경이 얼마나 믿을 만한지까지 함께 검증하는 운영 문제** 라는 것이다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 **현실적으로 가치가 큰 전문직 업무일수록 benchmark로 만들기 어렵다** 는 점이다.
- 기존 benchmark는 공개 웹사이트, 공개 API, 공개 code repo, 공개 desktop app처럼 environment가 이미 열려 있는 영역에 집중된다.
- 하지만 실제 enterprise workflow는 그 반대다. 병원 triage, 금융 심사, 생산 스케줄링, 공공 허가 처리, 물류 dispatch 같은 영역은 대부분 내부 시스템에 묶여 있다.
- 그래서 지금까지 agent benchmark가 잘 다룬 것은 "공개 환경이 있는 일"이고, 잘 못 다룬 것은 "실제로 돈이 되거나 위험한 일"이었다.
- OccuBench는 이 공백을 메우기 위해, 실제 human job role에 대응되는 professional scenario를 stateful tool-use task로 바꿔 평가한다.

## 1-2. Why previous approaches are insufficient

- WebArena, OSWorld, SWE-bench, TAU-bench 같은 기존 benchmark는 각자 의미가 있지만, **coverage가 제한적** 이다.
- 새 도메인을 추가하려면 실제 app을 배포하거나, API를 연결하거나, simulator를 수작업으로 만들어야 한다. 즉 benchmark scaling cost가 너무 크다.
- 또 기존 benchmark는 대체로 clean path 중심이다. real world에서 흔한 timeout, partial response, missing field, stale value 같은 fault를 체계적으로 넣어 보지 않는다.
- 그래서 기존 점수는 clean environment 성능은 보여주지만, **실서비스에서 장애를 견디는가** 는 잘 보여주지 못한다.
- 결국 기존 접근의 한계는 benchmark가 적다는 것이 아니라, **coverage, scaling cost, robustness evaluation** 을 동시에 잡지 못한다는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- **LES, Language Environment Simulator** 를 평가 단위로 도입한다. LLM이 tool-response level environment simulator 역할을 맡는다.
- **multi-agent synthesis pipeline** 으로 evaluation instance를 자동 생성한다. 여기서 목표는 solvable, verifiable, discriminative, diverse 라는 4가지 조건을 동시에 만족하는 task를 만드는 것이다.
- benchmark를 **두 축** 으로 평가한다.
  1. clean environment에서의 task completion
  2. fault-injected environment에서의 robustness
- 단순 agent score만 보는 것이 아니라, **simulator quality가 ranking에 미치는 영향** 도 별도로 분석한다.

## 2-2. Design intuition

이 논문의 설계 직관은 꽤 실용적이다. 실제 professional domain을 benchmark로 만들기 어려운 이유는 domain logic가 너무 복잡해서가 아니라, **그 logic를 담은 executable environment가 공개되어 있지 않기 때문** 인 경우가 많다. 그렇다면 environment construction을 software engineering 문제로 풀기보다, LLM이 domain logic를 tool response 형태로 흉내 내게 만드는 편이 더 싸고 더 빠르다.

또 하나 중요한 직관은, OccuBench가 LES를 학습용 world model이 아니라 **평가용 environment simulator** 로 쓴다는 점이다. 즉 이 논문은 agent training paper가 아니라 evaluation paper다. 그래서 관심사는 perfect physical realism이 아니라, 해당 domain에서 agent가 어떤 tool을 언제 어떻게 써야 하는지를 평가할 수 있을 만큼의 stateful interaction을 만들 수 있는가에 있다.

이 논문의 가장 큰 기여는 benchmark instance보다 **evaluation interface design** 에 있다. 환경을 LLM으로 대체하되, 시스템 프롬프트, tool schema, initial state, state description으로 behavior를 고정하고, rubric verifier와 fault injection까지 같이 묶어 **평가 apparatus 전체** 를 설계한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | 공개 환경이 없는 전문직 업무를 stateful tool-use benchmark로 평가 가능하게 만드는 것 |
| Core construct | LES, Language Environment Simulator |
| Environment inputs | system prompt, tool schema, initial state, state description |
| Benchmark scale | 100 scenarios, 10 industries, 65 specialized domains, 382 instances |
| Evaluation axes | clean task completion + fault-injected robustness |
| Main difference from prior work | 실제 환경 구현 대신 LLM 기반 simulator를 평가 apparatus로 사용 |

## 3-2. Module breakdown

### 1) LES, Language Environment Simulator

- LES는 agent action을 tool call로 받고 observation을 structured JSON tool response로 돌려주는 simulator다.
- 중요한 점은 state가 rule engine 외부에 따로 저장되는 것이 아니라, **system prompt와 interaction history 안에서 암묵적으로 유지** 된다는 점이다.
- 논문은 LES가 잘 동작하는 이유를 네 가지로 설명한다.
  1. API 문서와 tool call log에 대한 format prior
  2. 전문 도메인 operational logic에 대한 지식
  3. multi-turn state tracking
  4. edge case handling

### 2) Environment configuration

각 LES 환경은 아래 4개 요소로 구성된다.

- **System Prompt**: environment behavior rule, simulation logic, error handling, output format
- **Tool Schema**: action space 정의
- **Initial State**: 시작 상태를 담은 structured JSON
- **State Description**: state field 의미와 causal consistency 힌트

이 구조가 중요한 이유는 benchmark를 executable app이 아니라 **configuration artifact** 로 만든다는 점이다. 즉 새 domain을 추가할 때 website를 다시 구현하는 대신, 시나리오와 state transition rule을 언어적으로 정의하면 된다.

### 3) Multi-agent synthesis pipeline

OccuBench는 task를 그냥 수집하지 않는다. 먼저 각 scenario마다 **16개의 non-overlapping sub-topic** 을 설계하고, domain terminology, workflow, state variable, edge case, constraint를 담은 professional reference document를 만든다. 그 다음 Gemini-3-Flash-Preview를 LES로 사용해 아래 요소를 생성한다.

1. environment configuration
2. task instruction
3. tool definition
4. solution plan
5. verification rubric

그 후 각 task를 reference plan이 있는 경우와 없는 경우 모두 여러 번 실행해서 solvability와 difficulty를 확인한다. verifier는 rubric 기준 majority vote로 trajectory를 판정하고, 실패하면 repair module이 원인을 수정한 뒤 다시 실행한다. 마지막으로 trivially easy, unsolvable, invalid schema task를 필터링한다.

여기서 중요한 점은 benchmark data generation보다 **benchmark QA loop** 다. 논문은 문제를 만드는 것보다 문제를 solvable, discriminative, automatically verifiable 상태로 만드는 과정에 훨씬 많은 공을 들인다.

### 4) Fault injection and metrics

OccuBench는 evaluation 시점에 LES의 system prompt에 fault rule을 추가해 환경을 망가뜨린다.

- **E0**: clean environment
- **E1**: explicit fault, 예를 들어 HTTP 500, timeout, connection refused
- **E2**: implicit fault, 예를 들어 truncated data, missing field, stale value
- **E3**: mixed fault

이 fault들은 transient하고, interaction 초반에 몰리지 않으며, `fault_count` 와 `fault_duration` 으로 세기를 조절할 수 있다.

평가 metric은 두 가지다.

- **Completion Rate**: 382개 task 전체 기준 통과 비율
- **Robustness Score**: fault type 전반에서의 최악 성능을 반영하는 resilience 지표

여기서 좋은 점은 clean score와 robustness score를 분리해 본다는 점이다. 실서비스 관점에서는 둘 중 하나만 높아도 충분하지 않다.

# 4. Training / Data / Recipe

## 4-1. Data

- OccuBench는 100개의 professional task scenario, 10개의 industry category, 65개의 specialized domain을 포함한다.
- 최종 evaluation set은 **382개의 solvable instance** 로 구성된다.
- 각 task는 평균 **5.5개의 tool** 과 **16.2회의 tool call** 을 가진다.
- scenario design principle도 명확하다.
  1. real job mapping
  2. no single domain contributes more than 3 scenarios
  3. majority of scenarios are untestable by existing benchmarks
  4. all scenarios require multi-step interaction

## 4-2. Training strategy

이 논문은 model training paper라기보다 benchmark construction paper다. 그래서 여기서 중요한 것은 model pretraining recipe가 아니라 **instance synthesis recipe** 다.

- synthesis는 Gemini-3-Flash-Preview를 LES로 사용한다.
- task는 clean environment에서 먼저 생성된다.
- 각 task를 multiple execution으로 검증해 solvability와 difficulty를 calibrate한다.
- verifier는 rubric-based majority vote를 사용한다.
- repair module이 실패 원인을 고쳐 재실행한다.
- evaluation 단계에서는 default LES로 Gemini-3-Flash-Preview를 사용하고, thinking mode를 지원하는 model은 reasoning mode를 켠다.

## 4-3. Engineering notes

- public GitHub repository는 **exact internal evaluation system이 아니라 clean standalone reimplementation** 이다. 즉 repo를 그대로 돌리는 것과 논문 내부 harness가 완전히 동일하다고 보면 안 된다.
- 다만 repo는 framework-agnostic 하게 설계되어 있다. OpenAI-compatible API만 맞추면 다양한 agent framework에 연결할 수 있다.
- public repo도 fault injection, verifier, LWM or LES core, CLI evaluation path를 제공해서 실험 구조를 이해하기에는 충분하다.
- 실무적으로 가장 재사용 가치가 높은 부분은 benchmark item 자체보다, **proprietary domain을 LES config로 바꾸는 방식** 과 **fault-injected evaluation mode** 다.

# 5. Evaluation

## 5-1. Main results

먼저 clean environment, E0 기준 상위권 결과를 보면 아래와 같다.

| Model | Avg E0 | What stands out |
| --- | ---: | --- |
| GPT-5.2 | 79.6 | overall 1위, Agriculture 84, Business 86, Industrial 85, Science 94 |
| Gemini 3.1 Pro | 72.3 | Education 84로 최고 |
| Claude Opus 4.6 | 71.5 | Transportation 77, Business 78에서 강함 |
| Qwen 3.5 Plus | 69.9 | Healthcare 81, Commerce 81에서 강함 |
| DeepSeek V3.2 | 69.6 | open model 중 최상위권 |

여기서 제일 중요한 결론은 **no single model dominates** 다. GPT-5.2가 overall 1위이긴 하지만 Commerce에서는 Qwen 3.5 Plus가 더 높고, Education에서는 Gemini 3.1 Pro가 더 높고, Transportation에서는 Claude Opus 4.6이 더 높다. 즉 OccuBench가 보여주는 것은 평균 순위보다 **어떤 모델이 어떤 산업 profile을 가지는가** 에 가깝다.

Open-source model도 생각보다 강하다. Qwen 3.5 Plus와 DeepSeek V3.2는 각각 69.9와 69.6으로 4위, 5위에 올라 대부분의 Claude variant를 앞선다. 이 부분은 enterprise model selection 관점에서 꽤 실무적이다.

환경 robustness는 더 흥미롭다.

| Model or Avg | E0 | E1 | E2 | E3 | Takeaway |
| --- | ---: | ---: | ---: | ---: | --- |
| Avg | 67.5 | 62.6 | 53.4 | 54.4 | implicit fault가 가장 어렵다 |
| Gemini 3.1 Pro | 72.3 | 73.3 | 63.1 | 65.2 | robustness score 최상위권 |
| GPT-5.2 | 79.6 | 75.9 | 70.4 | 67.0 | absolute score는 가장 높다 |
| Qwen 3.5 Plus | 69.9 | 61.0 | 51.6 | 54.2 | E2에서 큰 하락 |
| Claude Opus 4.6 | 71.5 | 68.1 | 53.9 | 63.9 | implicit fault에 취약 |

평균 completion rate는 E0의 67.5에서 E2의 53.4로 **14.1 point** 떨어진다. 논문이 강조하듯, implicit fault는 explicit fault보다 어렵다. 이유는 단순하다. timeout이나 HTTP 500은 실패 신호가 노골적이라 재시도를 유도하지만, truncated data나 missing field는 겉보기에는 정상 응답처럼 보이기 때문이다.

## 5-2. What really matters in the experiments

### 1) OccuBench는 average leaderboard보다 industry profile이 더 중요하다

이 논문을 leaderboard 관점에서만 읽으면 GPT-5.2 79.6이 제일 먼저 보인다. 그런데 더 중요한 건 Figure 2와 Table 2가 보여주는 **capability shape** 다. Gemini는 Education과 Science에서 강하고, Claude Opus 4.6은 Transportation과 Business에서 강하고, Qwen 3.5 Plus는 Healthcare와 Commerce에서 강하다. 이건 single-domain benchmark로는 잘 안 보이는 정보다.

이 부분이 실무적으로 가장 중요하다. enterprise에서는 평균이 제일 높은 모델보다 **우리 도메인에 맞는 모델** 이 더 중요하다.

### 2) Implicit fault는 정말로 다른 종류의 문제다

E2가 E1보다 더 어렵다는 결과는 꽤 인상적이다. 많은 팀은 agent robustness를 retry policy나 timeout handling으로 생각한다. 하지만 OccuBench는 real world에서 더 위험한 failure는 오히려 **겉보기에 정상처럼 보이는 degraded response** 라는 점을 보여준다.

논문 후반 case study도 좋다. property valuation task에서 Claude Opus 4.6은 15개 unit 중 2개만 돌아온 응답을 보고 다시 조회하지만, Kimi K2.5는 truncated response를 전체 데이터로 받아들여 DSCR을 1.72x로 잘못 계산한다. 실제 값은 1.19x라 covenant fail인데도 말이다. 이건 benchmark failure라기보다 **production risk** 에 가깝다.

### 3) Scaling과 reasoning effort는 꽤 일관되게 도움된다

Within-family scaling도 분명하다. Gemini Pro와 Flash-Lite 차이는 11.0 point, Qwen Plus와 Flash 차이는 10.2 point, Claude Opus와 Sonnet 4.6 차이는 7.1 point다. 즉 OccuBench에서는 larger model advantage가 꽤 선명하다.

Reasoning effort ablation도 실무적이다. GPT-5.2는 none 54.7에서 xhigh 82.2까지 **27.5 point** 오른다. Claude Opus 4.6도 low 70.2에서 max 73.8로 오른다. 이건 적어도 professional multi-step task에서는 inference-time thinking budget이 꽤 직접적인 성능 레버라는 뜻이다.

### 4) Simulator quality는 benchmark 바깥이 아니라 benchmark 내부 변수다

이 논문에서 제일 중요한 caution은 아마 이것일 것이다. **strong agent is not necessarily strong simulator** 다. GPT-5.2는 agent로는 1위지만, simulator로 쓰면 전체 agent 평균이 29.3까지 떨어진다. 반면 Qwen 3.5 Plus simulator는 Gemini Flash와 **85.7% pairwise agreement** 를 보인다.

즉 LES 기반 evaluation은 작동하지만, simulator를 아무 모델로나 바꾸면 되는 것은 아니다. 평가 apparatus에 simulator가 포함되어 있기 때문이다. 이 지점은 OccuBench의 강점이자 동시에 가장 큰 주의점이다.

# 6. Limitations

1. **LES는 domain logic를 시뮬레이션할 뿐, 실제 domain data를 조회하지는 않는다.**  
   그래서 drug interaction check나 financial workflow 같은 task에서 "무엇을 확인해야 하는가"는 잘 평가할 수 있지만, cent-level numeric correctness 같은 문제는 real environment testing이 추가로 필요하다.

2. **simulator dependence가 크다.**  
   같은 agent라도 어떤 simulator를 쓰느냐에 따라 score와 rank가 달라질 수 있다. 논문도 simulator quality를 검증하거나, simulator를 바꿀 때 task solvability를 다시 확인해야 한다고 말한다.

3. **public repo와 internal harness를 동일시하면 안 된다.**  
   공개 GitHub는 reference implementation이지만, 논문 내부 evaluation system은 proprietary framework 위에 있었다고 명시한다. 따라서 재현할 때는 benchmark definition과 exact harness behavior를 분리해서 봐야 한다.

4. **production gap도 아직 남아 있다.**  
   OccuBench는 stateful decision-making과 tool use에는 강하지만, 실제 enterprise deployment에서 중요한 auth, latency budget, cost control, human escalation, policy logging 같은 운영 제약까지 직접 평가하는 것은 아니다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 중요한 이유는 agent capability를 모델 problem으로만 보지 않고, **evaluation environment problem** 으로 다시 보기 때문이다. 실제로 많은 enterprise agent 프로젝트가 막히는 지점은 model quality보다 benchmark 부재다. 공개 환경이 없으니 제대로 비교할 수도 없고, regression test도 만들기 어렵다.

OccuBench는 이 상황에서 꽤 현실적인 해법을 준다. 완벽한 simulator를 만들겠다는 게 아니라, **전문직 workflow를 LES config로 바꾸고, 그 위에서 solvability와 robustness를 측정하는 구조** 를 먼저 만들자는 것이다. 이게 논문 본문 수치보다 더 큰 기여다.

## 7-2. Reuse potential

### 1) Proprietary domain internal benchmark

사내 workflow가 외부에 공개되지 않는다면, 실제 app을 복제하는 대신 LES 스타일 configuration으로 internal benchmark를 만들 수 있다. 특히 document processing, ops triage, compliance workflow처럼 rule-heavy domain에 잘 맞는다.

### 2) Fault-injected agent QA

대부분의 agent test는 clean path만 본다. 그런데 OccuBench가 보여주듯 implicit fault가 더 위험할 수 있다. 실무에서도 E1, E2, E3 같은 fault mode를 만들어 regression test에 넣는 것이 유용하다.

### 3) Model selection by industry profile

평균 점수만 보고 model을 고르면 실제 도메인에서 손해를 볼 수 있다. OccuBench 스타일 평가는 "우리 산업에서 어떤 capability profile이 필요한가"를 기준으로 model을 고르게 해준다.

### 4) Agent score와 simulator score 분리

LES나 LWM 기반 평가를 쓸 때는 agent quality와 simulator quality를 따로 봐야 한다. 이 논문은 그 분리가 왜 중요한지를 꽤 분명하게 보여준다. 이후 비슷한 benchmark를 만들더라도 cross-simulator consistency check는 거의 필수로 넣어야 할 것 같다.

## 7-3. Follow-up papers

- WebArena
- TAU-bench
- TheAgentCompany
- Terminal-Bench
- Toolathlon
- Claw-eval

# 8. Summary

- OccuBench는 공개 환경이 없는 전문직 workflow를 LES 기반 stateful tool-use benchmark로 바꾼다.
- benchmark의 핵심은 task 수보다 environment construction, solvability calibration, fault injection 설계에 있다.
- 실험 결과는 단일 1위 모델보다 industry-specific capability profile이 더 중요하다는 점을 보여준다.
- implicit fault는 explicit fault보다 더 어렵고, real-world deployment readiness를 따로 보게 만든다.
- LES 기반 평가가 유효하려면 simulator quality 자체도 benchmark의 일부로 검증해야 한다.
