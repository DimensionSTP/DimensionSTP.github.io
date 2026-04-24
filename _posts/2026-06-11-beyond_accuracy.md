---
layout: single
title: "Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning Review"
categories: Study-concept
tag: [LLM, Tool-Use, Inference-Efficiency, KV-Cache, Reasoning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.05404)

[Code link](https://github.com/sqs-ustc/tool-reasoning-framework-PTE)

Beyond Accuracy는 "tool-integrated reasoning agent는 정확도만 보면 된다"는 가정을 정면으로 깨는 논문이다. 이 논문의 진짜 흥미로운 지점은 더 강한 browsing agent나 math agent를 만드는 데 있지 않다. 오히려 tool call이 들어가는 순간 KV-cache reuse가 끊기고, 긴 tool response가 이후 decode를 계속 비싸게 만든다는 아주 시스템적인 문제를 metric 차원에서 다시 정의한다는 데 있다.

특히 이 논문의 핵심은 **tool-use reasoning quality** 가 아니라 **tool-use reasoning latency model** 이다. 기존에는 token 수나 tool call 횟수로 대충 비용을 읽는 경우가 많았는데, 이 논문은 그것으로는 실제 wall-clock latency를 잘 설명하지 못한다고 본다. 그리고 그 차이를 PTE, 즉 Prefill Token Equivalents라는 metric으로 밀어붙인다.

> 한 줄 요약: Beyond Accuracy는 Tool-Integrated Reasoning에서 token 수나 tool call 횟수 대신, prefill-decode 비대칭과 KV-cache eviction을 반영하는 PTE를 제안하고, 그 관점에서 4가지 비효율 패턴을 정리한 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- agent benchmark를 "accuracy leaderboard"로만 읽으면 실제 서비스 비용 구조를 거의 놓치기 때문이다.
- 이 논문은 **tool use cost** 를 prompt engineering 감이 아니라 transformer inference first principle로 다시 적는다.
- math, factual QA, multi-disciplinary QA를 함께 보면서, 어떤 모델이 어떤 도구 환경에서 비효율적으로 무너지는지 패턴까지 정리한다.

이 논문의 가장 중요한 메시지는 단순하다. 좋은 TIR agent는 "정답을 맞히는 모델" 이기 전에, 같은 정답을 얼마나 싸게 내는지를 같이 봐야 한다. 그리고 그 비용은 token count보다 KV-cache와 context growth에 더 크게 좌우될 수 있다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 문제는 Tool-Integrated Reasoning, 즉 TIR 환경에서 **정확도와 실제 추론 비용을 함께 보는 방법** 이 부족하다는 점이다.
- TIR에서는 모델이 reasoning 중간에 search, visit, python 같은 외부 tool을 호출한다.
- 그런데 tool call이 들어가면 단순히 step 하나가 추가되는 수준에서 끝나지 않는다. 호출 이후 다시 LLM이 이어서 생각할 때, 이전 KV-cache가 재사용되지 못해 prefill을 다시 해야 하고, tool이 반환한 긴 텍스트가 context를 부풀리면서 이후 decode도 더 비싸진다.
- 즉, "tool을 1번 더 썼다" 와 "실제 GPU에서 latency가 얼마나 늘었는가" 는 같은 말이 아니다.
- 결국 이 논문의 문제 설정은 "TIR agent가 얼마나 잘 맞히는가" 보다, "TIR trajectory의 hardware-aware cost를 어떤 단위로 측정할 것인가" 에 가깝다.

## 1-2. Why previous approaches are insufficient

- 기존 efficiency metric은 대체로 token count 또는 tool call count에 머문다.
- 이런 metric은 context가 길어질수록 decode가 memory-bound로 더 비싸지는 구조를 직접 반영하지 못한다.
- 특히 tool call 이후에는 cache miss와 유사한 재prefill 비용이 생기는데, naive token metric은 이 비용을 잘 설명하지 못한다.
- reward engineering 계열 TIR RL 연구도 많지만, cost penalty가 종종 token 수나 tool 수 같은 거친 proxy에 묶여 있다.
- 결국 기존 방식의 한계는 "비용을 본다" 가 아니라, 어떤 비용을 실제로 보고 있는가가 너무 거칠다는 데 있다.
- 여기서 논문의 초점은 **hardware-aware cost definition** 에 있다.

# 2. Core Idea

## 2-1. Main contribution

- 이 논문의 핵심 기여는 PTE, 즉 Prefill Token Equivalents를 제안한 것이다.
- 아이디어는 단순하다. tool-use trajectory 전체의 비용을 "prefill token 몇 개를 처리하는 비용과 동등한가" 라는 하나의 단위로 바꾸자는 것이다.
- 이렇게 하면 내부 reasoning token과 외부 tool use 이후의 재prefill cost, 그리고 길어진 context로 인한 decode cost를 한 스케일 위에서 비교할 수 있다.
- 논문은 이 metric이 wall-clock latency와 naive token count보다 훨씬 잘 맞는다고 주장한다.
- 그 위에서 수천 개 trajectory를 분석해 4가지 비효율 패턴을 정리한다.
  - Confirmatory Tool Usage
  - Tool-Mixing
  - Lack of Tool Priors
  - Tool Format Collapse

## 2-2. Design intuition

- 이 논문의 설계 직관은 prefill과 decode를 같은 token 단위로 대충 합치지 않는 데 있다.
- prefill은 compute-bound에 더 가깝고, decode는 커진 KV-cache를 계속 불러와야 해서 memory-bound에 더 가깝다.
- 따라서 reasoning token 100개를 더 생성하는 비용과, tool response 때문에 context가 1000 token 길어진 뒤 decode가 느려지는 비용은 성격이 다르다.
- PTE는 바로 이 비대칭을 metric 안에 집어넣는다.
- 이 논문의 진짜 좋은 점은 cost metric을 "경제적 가격표" 가 아니라 "hardware bottleneck proxy" 로 다시 세운다는 점이다. API price ratio는 사업자 정책이지만, PTE는 transformer inference path를 근거로 한다.

개념식으로 쓰면 PTE는 아래처럼 이해하면 된다.

$$
PTE = \sum_{t=1}^{T} (C_t + gamma * Y_t * S_t)
$$

여기서 $C_t$ 는 turn $t$ 에서 다시 읽어야 하는 prefill context 크기, $Y_t$ 는 해당 turn의 decode token 수, $S_t$ 는 decode 시작 시점의 누적 context 길이, $gamma$ 는 model-hardware pair에서 decode memory cost를 prefill cost로 환산하는 계수다. 정확한 원문 기호와 식 번호는 최종 발행 전 PDF 기준으로 다시 확인하는 편이 좋다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | TIR trajectory의 실제 inference cost를 hardware-aware하게 측정하고 비효율 패턴을 찾는 것 |
| Key module | PTE metric, high-concurrency validation, 5-benchmark evaluation, inefficiency pattern mining |
| Difference from prior work | token count나 tool count 대신 prefill-decode 비대칭과 KV-cache growth를 cost에 직접 반영 |
| Tools in framework | Search, Visit, Python |
| Main output | benchmark별 accuracy-PTE trade-off와 4가지 inefficiency pattern |

## 3-2. Module breakdown

### 1) PTE cost model

- PTE는 trajectory를 turn 단위로 쪼개서 본다.
- 각 turn마다 다시 넣어야 하는 context prefill cost와, 늘어난 context 위에서 발생하는 decode cost를 합산한다.
- decode cost는 단순 decode token 수만 세지 않고, decode가 시작될 때의 누적 context 길이까지 같이 반영한다.
- 그래서 reasoning이 뒤로 갈수록, 같은 token 1개라도 더 비싸질 수 있다.
- 이 점이 token count와 가장 크게 갈리는 부분이다.

### 2) gamma as model-hardware coefficient

- 논문은 $gamma$ 를 model-hardware pair의 정적 계수로 둔다.
- 이 값은 memory-bound decode cost를 compute-bound prefill cost와 같은 단위로 바꾸는 역할을 한다.
- GQA나 MLA 같은 아키텍처 차이도 반영한다.
- 즉 PTE는 "모든 모델에 같은 상수" 가 아니라, 모델 구조와 하드웨어 특성을 같이 먹는 metric이다.

### 3) High-concurrency validation

- 저자들은 PTE가 진짜 wall-clock latency와 맞는지 검증하기 위해 high-concurrency 실험을 별도로 한다.
- DeepSeek-V3.2를 8x H200 node, vLLM TP=8, 256 parallel requests 환경에 올리고 synthetic TIR workload를 돌린다.
- 중요한 건 여기서 pure model generation latency만 기록하고, tool execution이나 network transmission 시간은 제외했다는 점이다.
- 즉 이 논문이 모델링하려는 것은 "LLM inference 자체의 cost" 이지, 전체 end-to-end user latency 전부는 아니다.

### 4) Unified TIR framework

- 코드 저장소도 단순 분석 스크립트 수준이 아니다.
- Search, Visit, Python tool을 가진 unified inference framework, rollout trace, evaluation pipeline, PTE analysis code까지 같이 공개한다.
- framework 차원에서 동일한 system prompt와 동일한 tool definition을 써서 모델 간 intrinsic TIR behavior를 비교하려는 의도가 분명하다.
- 실무적으로는 이 부분이 꽤 중요하다. tool schema나 prompt가 바뀌면 model behavior가 같이 바뀌기 때문이다.

### 5) Pattern mining

- PTE를 만든 뒤 끝나는 것이 아니라, 실제 trajectory를 pattern 관점으로 읽는다.
- 논문이 잡아낸 4가지 패턴은 "이상한 예시 몇 개" 수준이 아니다.
- heuristic으로 각 패턴을 자동 검출하고, primary setting에서 frequency와 cost multiplier까지 측정한다.
- 그래서 이 논문은 metric paper이면서 동시에 TIR behavior analysis paper로도 읽힌다.

# 4. Training / Data / Recipe

## 4-1. Data

- 이 논문은 model training paper가 아니라 evaluation and analysis paper에 가깝다.
- 벤치마크는 총 5개다.
  - MATH500
  - AIME24
  - AIME25
  - SimpleQA
  - WebInstruct-Verified
- math benchmark에는 Python tool을 제공한다.
- SimpleQA에는 Search와 Visit를 제공한다.
- WebInstruct-Verified에는 Search, Visit, Python을 모두 제공한다.
- SimpleQA와 WebInstruct-Verified는 각각 500개 random subset으로 평가한다.

## 4-2. Training strategy

- 별도의 post-training은 하지 않는다.
- 대신 동일한 agent framework 안에서 여러 open-source tool-capable model을 돌려 accuracy, token, tool use count, PTE를 수집한다.
- 핵심 비교 대상은 Qwen2.5, Qwen3, Llama-3.1, GLM-4.5, DeepSeek-V3.1-Terminus, GPT-OSS-120B, Tongyi-Deepresearch 같은 공개 모델들이다.
- 모든 모델은 vLLM 위에서 동일한 system prompt와 tool definition으로 비교한다.
- WebInstruct-Verified의 정답성 판정은 trajectory 끝에서 DeepSeek-V3 judge를 사용한다. 즉 모든 benchmark가 exact-match 계열은 아니다.

## 4-3. Engineering notes

- Search tool은 Serper API, Visit tool은 Jina API, Python tool은 open-source sandbox를 쓴다.
- framework는 각 turn마다 prefill token, decode token, 누적 sequence length를 따로 로깅한다.
- 이 로깅이 있어야 PTE를 trajectory 단위로 계산할 수 있다.
- 코드 저장소를 보면 rollout tracing, reward calculation, result_analysis가 분리돼 있어, 이후 efficiency-aware RL이나 agent eval 쪽으로 확장하기 좋다.
- 이 논문은 "새 metric" 자체보다 trace observability를 갖춘 runtime을 같이 공개한 점도 꽤 실용적이다.
- 특히 **rollout trace logging** 과 **result analysis pipeline** 을 한 저장소 안에 둔 점이 좋다.

# 5. Evaluation

## 5-1. Main results

먼저 이 논문에서 제일 중요한 숫자는 accuracy table보다 validation table이다.

| Metric | Value | Meaning |
| --- | ---: | --- |
| corr(PTE, wall-clock latency) | 0.925 | 실제 generation latency와 강한 선형 상관 |
| corr(token count, wall-clock latency) | 0.625 | naive token metric은 설명력이 훨씬 낮음 |
| Spearman rank across hardware | > 0.95 | H100, H200, A100, RTX4090, V100 간 모델 효율 ranking이 대체로 유지 |

이 표가 핵심인 이유는 분명하다. PTE는 "좋아 보이는 아이디어" 수준이 아니라, 최소한 논문이 설계한 high-concurrency setting에서는 token count보다 훨씬 실제 latency와 잘 맞는다.

성능 landscape 쪽에서 논문이 말하는 포인트는 3가지다.

1. **accuracy가 비슷해도 PTE는 크게 다를 수 있다.**
   - 논문은 같은 benchmark의 상위 정확도 구간에서도 PTE가 한 order 이상 벌어질 수 있다고 본다.
   - 실제 표를 보면 SimpleQA에서 89.2 accuracy의 Qwen2.5-72B-Instruct는 PTE 6006인데, 92.9 accuracy의 GLM-4.5는 PTE 20617이다. 정확도 차이는 작지만 비용 차이는 훨씬 크다.

2. **TIR 능력은 general skill이 아니라 task-tool specific skill에 가깝다.**
   - Qwen2.5-72B는 SimpleQA 같은 web agent task에서는 강하지만, MATH500과 AIME 같은 Python reasoner setting에서는 그렇게 강하지 않다.
   - 즉 "tool use를 잘한다" 는 하나의 능력으로 뭉뚱그리기 어렵다.

3. **thinking mode는 항상 이득이 아니다.**
   - 논문은 Qwen3-235B-Thinking이나 Qwen3-32B default config처럼 thinking mode가 켜진 모델이 어려운 math에서는 accuracy gain을 만들 수 있지만, 쉬운 factual QA에서는 accuracy를 깎거나 PTE를 크게 키운다고 해석한다.
   - 이 부분은 TIR에서도 결국 "언제 길게 생각할지" 라는 routing 문제가 남아 있다는 뜻이다.

## 5-2. What really matters in the experiments

이 논문에서 진짜 중요한 것은 leaderboard가 아니라 아래 4가지다.

### 1) token front-loading과 실제 cost escalation은 다르다

- token count 관점에서는 모델이 초반 step에 긴 reasoning을 몰아 넣는 "first-step effect" 가 강하게 보일 수 있다.
- 그런데 PTE로 보면 진짜 비용은 뒤로 갈수록 커지는 경우가 많다.
- 이유는 간단하다. context가 계속 누적되기 때문이다.
- 즉, "앞에서 많이 말한 모델" 과 "뒤에서 context를 너무 크게 만들어 놓은 모델" 을 token metric은 잘 구분하지 못한다.

### 2) 네 가지 inefficiency pattern이 모두 실무적이다

| Pattern | Main meaning | Dominant setting | Freq. | Cost multiplier |
| --- | --- | --- | ---: | ---: |
| Confirmatory Tool Usage | 먼저 답을 만들고, tool은 확인용으로만 사용 | Qwen3-235B-Instruct on MATH500 | 81% | 1.77 |
| Tool-Mixing | 한 trajectory에서 여러 toolset을 과하게 섞음 | DeepSeek-V3.1-Term on WebInstruct | 59% | 2.42 |
| Format Collapse | schema나 JSON 형식이 조금만 바뀌어도 tool call이 붕괴 | Tongyi-DeepResearch on SimpleQA | 100% | N/A |
| Lack of Tool Priors | tool 사용법 prior가 약해 빈 출력이나 execution error를 냄 | Qwen2.5-72B on AIME24 | 33% | 2.15 |

이 네 가지는 꽤 재밌다.

- Confirmatory는 "도구를 solver가 아니라 verifier로만 쓰는" 패턴이다.
- Tool-Mixing은 겉보기엔 유연해 보이지만, 비용이 많이 드는데 accuracy gain은 분명하지 않다.
- Lack of Tool Priors는 tool이 있다고 다 잘 쓰는 게 아니라, pretraining이나 alignment에서 그 tool environment를 배웠는지가 중요하다는 뜻이다.
- Format Collapse는 agent 성능이 semantic competence보다 schema brittleness에 먼저 무너질 수 있다는 경고다.

### 3) 높은 PTE는 단순히 어려운 문제를 푼 흔적만은 아니다

- 논문은 difficulty를 통제한 뒤에도 incorrect trajectory가 correct trajectory보다 더 높은 PTE를 보인다고 주장한다.
- 예를 들어 Table 10 기준으로 PTE는 wall-clock latency와 0.925 상관을 보이지만, Appendix F에서는 difficulty를 통제한 뒤에도 PTE와 accuracy 사이의 음의 관계가 남는다고 정리한다.
- 심지어 GPT-OSS는 Level-5 문제를 맞힐 때보다 Level-4 문제를 틀릴 때 더 큰 PTE를 쓰는 사례도 보고한다.
- 이건 중요한 포인트다. **더 오래 생각한 것** 과 **더 비효율적으로 헤맨 것** 은 같지 않다.

### 4) frontier model도 tool infrastructure bottleneck을 피하지 못한다

- 논문은 DeepSeek-V3.1-Terminus, GPT-OSS, Qwen3-235B-Instruct 같은 frontier 계열 모델이 높은 accuracy를 내더라도, 길고 multi-round한 tool response 때문에 efficiency가 크게 악화될 수 있다고 해석한다.
- 즉 더 좋은 model alone으로는 agent runtime 문제가 끝나지 않는다.
- 결국 serving stack에서 KV-cache eviction과 long tool response를 어떻게 다룰지가 별도 문제로 남는다.

# 6. Limitations

1. 이 논문이 모델링하는 것은 tool execution time 전체가 아니라 **LLM generation latency** 다.
   - search API latency, webpage fetch latency, python 실행 latency, network variance는 본 metric의 직접 대상이 아니다.
   - 실제 end-to-end product latency는 이보다 더 복잡하다.

2. PTE도 여전히 proxy다.
   - transformer inference first principle에 기대고 있지만, 실제 deployment에서는 batching, cache policy, kernel optimization, paged attention, request interleaving이 같이 영향을 준다.

3. WebInstruct-Verified 일부 평가는 LLM judge를 쓴다.
   - 따라서 모든 benchmark가 strict symbolic scoring은 아니다.
   - 정확도 비교를 읽을 때 judge bias 가능성은 같이 봐야 한다.

4. pattern heuristic은 단순하고 primary setting 중심이다.
   - 예를 들어 confirmatory pattern은 "tool 호출 전 이미 답 token이 생성됐는가" 로 잡는다.
   - heuristic이 단순한 만큼, 더 정교한 pattern taxonomy로 확장될 여지가 있다.

5. 논문이 제안하는 것은 진단 도구이지, 직접적인 해결책 자체는 아니다.
   - inefficiency pattern을 잘 보여주지만, 어떤 RL objective나 runtime policy가 이를 가장 잘 줄이는지는 후속 문제로 남는다.

6. 추가로 조심할 점은 "accuracy 하락 없이 PTE만 낮추자" 가 항상 좋은 목표는 아니라는 점이다.
   - 특히 AIME 같은 고난도 reasoning에서는 일정 정도의 긴 trace가 실제로 필요한 경우가 있다.
   - 그래서 PTE는 절대 최소화 target이라기보다, **difficulty-aware budget control signal** 로 쓰는 편이 더 맞아 보인다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 중요한 이유는 agent를 평가할 때 "accuracy + token count" 정도로 끝내는 습관을 바꾸게 만들기 때문이다.

실서비스에서 tool-use agent를 붙이면, 진짜 비용 문제는 아래에서 터진다.

- tool call이 잦아 turn이 잘게 쪼개짐
- 매 turn마다 context prefill이 반복됨
- tool response를 그대로 넣어 이후 decode가 비싸짐
- model이 schema mismatch나 empty tool return으로 괜히 한 바퀴 더 돎

즉 모델이 똑같이 정답을 내도, 어떤 trajectory는 productizable하고 어떤 trajectory는 너무 비싸다. 이 차이를 보여주는 metric이 있어야 RL reward 설계도, serving optimization도, prompt intervention도 제대로 된다.

이 논문의 가장 큰 가치는 "agent efficiency" 를 추상적인 cost-awareness에서 끌어내렸다는 점이다.

특히 이 논문은 **KV-cache와 context growth** 를 efficiency discussion의 중심으로 다시 올려놓는다.

## 7-2. Reuse potential

실무적으로 재사용할 수 있는 포인트는 아래와 같다.

### 1) agent eval dashboard에 PTE류 metric 추가

- accuracy, tool count, avg tokens만 보는 대시보드는 부족하다.
- turn별 prefill size, decode length, cumulative context를 같이 기록하면 훨씬 좋은 진단이 가능하다.

### 2) prompt와 schema를 pattern detector로 같이 운영

- Format Collapse는 모델이 semantic하게는 맞아도 syntax에서 바로 무너질 수 있다는 뜻이다.
- tool definition rename, argument shape change, single query vs query list 같은 perturbation test를 CI처럼 돌릴 가치가 있다.

### 3) RL reward design에 직접 연결 가능

- confirmatory usage나 useless tool-mixing은 reward penalty 후보가 될 수 있다.
- 다만 단순 tool count penalty보다 PTE increase 자체를 penalty로 쓰는 편이 더 맞다.

### 4) runtime intervention에도 연결 가능

- long tool response를 요약하거나, tool output을 structured state로 압축하거나, post-tool cache policy를 바꾸는 식의 systems optimization이 가능하다.
- 이 논문은 model weight 문제가 아니라 **runtime architecture 문제** 도 같이 건드린다.

### 5) difficulty-aware routing과 같이 봐야 한다

- 쉬운 SimpleQA와 어려운 AIME를 같은 thinking budget으로 처리하면 비효율이 커진다.
- 결국 PTE는 reasoning router나 tool router의 objective에도 들어갈 수 있다.

## 7-3. Follow-up papers

- BrowseComp
- MCP-RADAR
- SimpleTIR
- THOR
- SideQuest
- CLASSIC

# 8. Summary

- Beyond Accuracy는 TIR에서 token 수나 tool 횟수 대신, prefill-decode 비대칭과 KV-cache growth를 반영하는 PTE를 제안한다.
- 논문은 high-concurrency setting에서 PTE가 wall-clock latency와 token count보다 훨씬 잘 맞는다고 보고한다.
- 분석 결과 confirmatory usage, tool-mixing, lack of priors, format collapse라는 4가지 비효율 패턴이 드러난다.
- 높은 PTE는 단순히 어려운 문제를 푼 흔적이 아니라, 비효율적 reasoning과 co-occur하는 경우가 많다.
- 이 논문의 핵심 가치는 agent 평가를 "accuracy-first" 에서 "accuracy plus hardware-aware trajectory cost" 로 옮겨 놓은 데 있다.
