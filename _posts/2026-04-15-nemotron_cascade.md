---
layout: single
title: "Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models Review"
categories: Study-concept
tag: [Nemotron, CascadeRL, RLHF]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2512.13607)

Nemotron-Cascade는 표면적으로는 "또 하나의 reasoning model technical report"처럼 보인다. 하지만 실제로 읽어보면, 이 논문이 겨냥하는 핵심은 모델 자체보다 **RL을 여러 reasoning domain에 어떻게 안정적으로 확장할 것인가**에 더 가깝다. 특히 이 글은 뒤에 나온 Nemotron-Cascade 2가 아니라, 2025년 12월 공개된 첫 번째 Nemotron-Cascade 보고서를 기준으로 정리한다.

이 논문이 흥미로운 이유는 단순히 RL stage를 여러 개 붙였기 때문이 아니다. 저자들은 general alignment, strict instruction following, math, code, SWE를 한 번에 섞어 RL 하는 대신, **도메인별 verifier 속도, response length, reward noise, hyperparameter sensitivity가 다르다**는 점을 먼저 문제로 본다. 그리고 그 해법을 "더 좋은 하나의 reward"가 아니라 **순차적이고 domain-wise한 post-training pipeline**으로 제시한다.

여기서 내가 가장 중요하게 본 포인트는 RLHF의 위치다. 보통 RLHF는 alignment 단계로만 이해되기 쉽지만, 이 논문은 RLHF가 **verbosity와 repetition을 줄여서 이후 math/code RL의 reasoning efficiency까지 높이는 pre-step**이라고 주장한다. 즉 Cascade RL은 단순한 curriculum이라기보다, **heterogeneous RL workload를 분리해서 다루는 시스템 설계**에 가깝다.

> 한 줄 요약: Nemotron-Cascade는 multi-stage SFT 이후 RLHF -> IF-RL -> Math RL -> Code RL -> SWE RL을 순차적으로 적용하는 **domain-wise Cascade RL**을 제안해, unified/dedicated reasoning model을 안정적으로 post-train하는 open recipe를 보여준 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- reasoning model을 위한 RL을 **objective 설계**보다 **training systems / curriculum 설계** 관점에서 읽을 수 있다.
- RLHF가 단순 alignment가 아니라, 후속 reasoning RL의 token efficiency를 높이는 단계라는 해석이 흥미롭다.
- 모델, intermediate checkpoint, reward model, SFT/RL 데이터까지 공개되어 있어서 **pipeline 단위로 재현*분석하기 쉬운 편**이다.

내가 보기엔 이 논문은 "8B/14B 모델이 잘 나왔다"보다, **general-purpose reasoning model을 만들 때 joint RL이 항상 정답은 아닐 수 있다**는 점을 가장 설득력 있게 보여준다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 general-purpose reasoning model에 RL을 적용할 때 생기는 **cross-domain heterogeneity**다.
- 예를 들어 math는 symbolic rule-based verifier로 빨리 검증할 수 있지만, code와 SWE는 execution 또는 patch validation 때문에 reward 계산이 훨씬 느리다.
- response length도 도메인마다 크게 다르다. instruction following은 짧고 엄격한 답이 유리하지만, math와 competitive programming은 긴 chain-of-thought와 더 많은 test-time budget이 필요하다.
- 이 차이 때문에 mixed-domain RL은 인프라를 복잡하게 만들고, response length extension 같은 curriculum 설계와 hyperparameter tuning도 도메인별로 엇갈리게 된다.
- 저자들의 목표는 결국 두 가지다.
  1. 서로 다른 reasoning domain을 하나의 post-training recipe로 다룰 것
  2. unified model이 thinking / non-thinking 모드를 turn 단위로 오갈 수 있게 만들 것

## 1-2. Why previous approaches are insufficient

- 여러 reasoning domain을 한 번에 섞어 joint RL을 하면, verifier latency와 reward characteristics가 뒤섞이면서 RL pipeline이 비대해진다.
- unified reasoning model은 편리하지만, 실제로는 dedicated thinking model보다 reasoning benchmark에서 밀리는 경우가 많다. 이 논문은 바로 그 gap을 줄이는 문제를 정면으로 다룬다.
- instruction-following verifier는 "제약을 정확히 지켰는가"만 보지만, human preference reward model은 "전반적으로 좋은 응답인가"를 본다. 이 둘은 같은 방향이 아닐 수 있다.
- 순차 학습은 보통 catastrophic forgetting을 걱정하게 만들지만, 저자들은 RL에서는 supervised learning보다 이 문제가 덜 심할 수 있다고 주장한다. 다만 이것도 prompt overlap, reward conflict, stage order를 어떻게 통제하느냐에 달려 있다.

즉 이전 방식의 한계는 단순히 "benchmark가 부족했다"가 아니라, **서로 다른 domain을 한 objective와 한 training regime 안에 너무 많이 욱여넣으려 했다**는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- **Cascade RL**: multi-stage SFT 이후 RLHF, IF-RL, Math RL, Code RL, SWE RL을 순차적으로 적용한다.
- **Unified reasoning control**: ChatML 위에 `/think` 와 `/no_think` 플래그를 두고, unified 8B 모델이 thinking / instruct 모드를 turn 단위로 전환할 수 있게 한다.
- **Parallel response SFT**: general-domain prompt마다 thinking / non-thinking response를 병렬로 만들어 unified model 학습에 사용한다.
- **Stage-specific recipe**: 각 RL stage에 맞는 reward function, response length schedule, filtering strategy, engineering trick을 별도로 둔다.
- **Open recipe**: final model뿐 아니라 중간 checkpoint와 데이터 레시피를 공개해 stage별 효과를 추적할 수 있게 한다.

## 2-2. Design intuition

이 논문의 설계 직관은 꽤 명확하다.

1. **RLHF를 먼저 둔다**
   - RLHF가 응답의 verbosity와 repetition을 줄여준다.
   - 그 결과 reasoning token budget이 더 효율적으로 쓰이게 되고, 뒤의 Math RL / Code RL이 더 안정적으로 진행된다.

2. **그 다음엔 domain을 분리한다**
   - instruction following, math, code, SWE는 reward 계산 방식과 필요한 output behavior가 다르다.
   - 따라서 한 번에 섞기보다 domain-wise로 stage를 나누면, 각 domain에 맞는 hyperparameter와 curriculum을 적용할 수 있다.

3. **general -> specialized 순서를 따른다**
   - RLHF와 IF-RL 같은 더 일반적인 능력을 먼저 맞추고,
   - 그 위에 math, code, SWE처럼 더 specialized한 능력을 얹는다.

4. **unified model도 포기하지 않는다**
   - small model은 unified reasoning에서 밀릴 것이라는 가정을 그대로 받아들이지 않고,
   - parallel SFT data + half-half RLHF 같은 장치를 넣어 8B unified model의 gap을 줄인다.

내가 보기엔 이 논문의 핵심은 "순차 학습이 좋다"가 아니라, **domain heterogeneity를 training design의 1급 변수로 취급했다**는 점이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | heterogeneous reasoning domains에 대해 안정적으로 RL을 확장하고 unified reasoning model까지 구성하는 것 |
| Base models | Qwen3-8B-Base, Qwen3-14B-Base |
| Core pipeline | Multi-Stage SFT -> RLHF -> IF-RL -> Math RL -> Code RL -> SWE RL |
| Unified mode control | ChatML 위에서 `/think`, `/no_think` 플래그로 thinking / non-thinking 전환 |
| Main novelty | joint mixed-domain RL이 아니라 sequential, domain-wise RL과 stage-specific recipe를 채택 |
| Open artifact | final model, intermediate checkpoints, RM, SFT/RL datasets 공개 |

## 3-2. Module breakdown

### 1) Multi-Stage SFT and mode control

- SFT는 두 단계로 구성된다.
  - **Stage 1 (16K)**: general-domain + math + science + code reasoning
  - **Stage 2 (32K)**: 위 데이터에 longer reasoning, tool use, SWE data를 추가
- general-domain data는 같은 prompt에 대해 **thinking / non-thinking response를 병렬로 생성**한다.
- Chat template은 Qwen3-style ChatML을 따르되, explicit flag인 `/think`, `/no_think`만으로 모드를 제어한다.
- tool calling은 system prompt 안의 `<tools>` / `</tools>` 태그와 `<tool_call>` / `</tool_call>` 포맷을 사용한다.

### 2) Cascade RL sequence

- RL data는 SFT data와 **prompt 기준으로 strict disjoint**하게 분리된다.
- 전 stage에서 공통으로 쓰는 RL optimizer는 **strict on-policy GRPO**다.
- 매 iteration에서 현재 policy로 rollout을 만들고 바로 한 번 gradient update를 수행해 importance sampling ratio를 1로 맞춘다.
- 기본 recipe는 **no KL, token-level loss**이며, 저자들은 이 구성이 수식상으론 단순하지만 실제론 안정성과 정확도에 유리하다고 본다.

### 3) Why sequential RL may work better than expected

저자들은 sequential RL이 catastrophic forgetting에 덜 취약하다고 보는 이유를 몇 가지로 설명한다.

- RL의 training distribution은 policy-dependent라서, 완전히 과거 데이터가 사라지는 supervised learning과 구조가 다르다.
- domain 사이의 reward가 완전히 반대가 아니라, "더 정확하고 더 aligned된 출력"이라는 공통 방향을 어느 정도 공유한다.
- stage order를 general -> specialized로 정해, specialized skill이 generic behavior에 덮어써지지 않게 한다.
- math / competitive programming prompt를 RLHF에서 아예 제외하는 식으로 prompt overlap도 줄인다.

### 4) Unified 8B vs dedicated thinking model

- 이 논문의 꽤 인상적인 부분은 8B unified model을 끝까지 밀어붙인다는 점이다.
- 저자들은 unified 8B가 dedicated 8B-Thinking과 reasoning 성능 차이를 거의 좁힐 수 있다고 주장한다.
- 이를 가능하게 한 핵심은 두 가지다.
  1. **parallel SFT responses**: 같은 prompt에 대해 thinking / instruct 응답을 함께 학습
  2. **half-half RLHF**: unified model RLHF batch를 thinking / non-thinking mode로 반반 나눠 학습

즉 이 논문은 single-model unification을 포기하지 않으면서도, 그 gap을 메우기 위한 구체적 장치를 같이 제시한다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 단순히 "우리가 RL을 잘했다"에서 끝나지 않고, SFT/RL 데이터 구성을 꽤 자세히 공개한다.

- **General-domain SFT**: 2.8M samples, 3.2B tokens
- **Knowledge-intensive augmentation**: 1.2M samples, 1.5B tokens
- **Math SFT Stage 1**: 353K unique prompts, 2.77M samples
- **Math SFT Stage 2**: 163K prompts, 1.88M samples
- **Code SFT Stage 1**: 172K distinct prompts, 1.42M samples
- **Code SFT Stage 2**: 79K prompts, 1.39M samples
- **Science SFT**: 226K prompts, Stage-1 289K / Stage-2 345K samples
- **Tool calling SFT**: 310K conversations, 1.41M user-assistant turns
- **SWE SFT**: code repair 127K instances, localization 92K, test generation 31K
- **Reward modeling**: 82K preference pairs

이 데이터 설계에서 중요한 건 양 자체보다 **데이터의 역할 분담**이다.

- general-domain은 unified conversational behavior를 잡고,
- math / code / science는 thinking-mode reasoning을 길게 학습하며,
- tool / SWE는 Stage-2에서 longer context와 task structure를 추가한다.

특히 general-domain SFT에서 DeepSeek-R1-0528과 DeepSeek-V3-0324를 써서 같은 prompt의 thinking / non-thinking response를 병렬 생성한 점은, unified model 학습의 핵심 장치로 읽힌다.

## 4-2. Training strategy

### 1) RLHF

- RLHF는 Cascade RL의 첫 단계다.
- reward model은 Qwen2.5-72B-Instruct 위에 scalar head를 얹어 Bradley-Terry objective로 학습한다.
- 저자들은 RewardBench를 reward model 선택에 사용하지만, **RewardBench 점수가 가장 높다고 RLHF policy가 항상 제일 좋은 것은 아니다**라고 분명히 말한다.
- unified model에서는 RLHF를 thinking / non-thinking 반반으로 섞는 **half-half** 전략이 가장 좋았다.
- 이 단계의 핵심 효과는 alignment뿐 아니라 **verbosity와 repetition 감소**다.

### 2) IF-RL

- IF-RL은 strict instruction following을 강화하기 위한 단계다.
- 하지만 rule-based verifier만 그대로 쓰면 human alignment가 깨질 수 있다.
- 그래서 unified model은 **non-thinking mode에서만 IF-RL**을 수행하고, dedicated thinking model은 **IF verifier + RM score 결합 reward**를 사용한다.
- 저자들은 IF-RL이 reasoning trace를 짧게 만들어 token efficiency를 개선하지만, reasoning benchmark에는 일시적인 하락을 만들 수 있다고 본다.

### 3) Math RL

- Math RL은 RLHF 이후 checkpoint에서 시작한다. 저자들은 SFT checkpoint보다 RLHF checkpoint에서 출발하는 것이 훨씬 유리하다고 본다.
- reward는 `<think>` 뒤의 boxed answer를 추출해 AceMath verifier로 정답 여부를 판정하는 **strict correctness reward**다.
- 여기에 code-switching penalty를 더해 reasoning 중 언어 혼합을 줄인다.
- 핵심 엔지니어링 포인트는 **response length extension**이다.
  - 8B: 24K -> 32K -> 40K
  - 14B: 28K -> 40K
- 저자들은 특히 40K stage가 hard AIME 문제 성능을 30% -> 40% 수준으로 끌어올린다고 설명한다.
- dynamic filtering으로 너무 쉽거나 너무 어려운 문제를 epoch 단위로 거르고, hard/easy problem을 낮은 확률로 다시 섞어 안정성을 유지한다.

### 4) Code RL

- Code RL은 Math RL 다음 단계로 들어간다.
- reward는 매우 단순한 **pass-all-tests binary reward**다.
- Code RL은 sampling temperature에 민감해서, 높은 temperature가 더 좋은 exploration을 주지만 training instability도 키운다.
- 이 단계에서 눈에 띄는 건 **asynchronous reward computation**이다. code verification overhead를 줄이기 위해 VeRL의 async reward computation을 사용했고, 8 DGX H100 노드 / batch 128 / rollout 8 설정에서 평균 verification time을 1172.4초에서 416.2초로 줄였다고 보고한다.

### 5) SWE RL

- SWE RL은 Cascade RL의 마지막 단계다.
- 이 stage는 execution-heavy한 Docker-based verifier 대신, **execution-free reward model**을 사용한다.
- reward는 generated patch와 ground-truth patch 사이의 lexical / semantic similarity다.
- semantic similarity는 **Kimi-Dev-72B**에 yes/no 질문을 던지고, "YES" token probability를 reward로 사용한다.
- 또 SWE는 input context length가 매우 중요해서, RL에서 **16K -> 24K input context extension curriculum**을 둔다.
- 저자들은 직접 24K보다 더 긴 input에서 resolve rate가 떨어지는 현상도 보고하며, 이것을 base model의 long-context 한계와 SFT context mismatch로 해석한다.

## 4-3. Engineering notes

- 이 논문은 final model만 공개하지 않고, **SFT / RLHF / IF-RL / Math RL / Code RL intermediate checkpoint**도 공개한다.
- 그래서 "어떤 stage가 정확히 무슨 역할을 하는가"를 model card만으로도 역추적하기 쉽다.
- evaluation은 reasoning task에서 보통 **64K generation budget, temperature 0.6, top-p 0.95**를 사용한다.
- 8B는 YaRN factor 2.0, 14B는 benchmark에 따라 2.0 또는 3.0을 쓴다.

실무 관점에서 보면 이 논문의 진짜 강점은 단일 모델의 headline score보다, **open post-training recipe로서의 해부 가능성**이다.

# 5. Evaluation

## 5-1. Main results

이 논문의 main table은 꽤 많은 benchmark를 담고 있지만, 내 기준에서는 아래 네 줄이 핵심이다.

### 1) final 8B unified model이 생각보다 강하다

최종 Nemotron-Cascade-8B는 다음과 같은 점수가 나온다.

- MMLU-Pro 75.7
- GPQA-Diamond 66.5
- ArenaHard 87.9
- IFEval 90.2
- AIME 2025 80.1
- LiveCodeBench v5 / v6: 74.3 / 71.1
- SWE-bench Verified: 37.2

즉 unified 8B라는 점을 감안하면, instruction following과 reasoning을 꽤 균형 있게 잡은 모델이다.

### 2) 14B-Thinking은 code와 SWE에서 특히 강하다

Nemotron-Cascade-14B-Thinking의 final 결과는 다음과 같다.

- MMLU-Pro 77.0
- GPQA-Diamond 69.6
- ArenaHard 89.5
- AIME 2025 83.3
- LiveCodeBench v5 / v6: 77.5 / 74.6
- SWE-bench Verified: 43.1

특히 저자들은 14B-Thinking이 LiveCodeBench 계열에서 DeepSeek-R1-0528를 넘고, SWE-bench Verified에서도 32B specialized open models를 앞선다고 강조한다.

### 3) RLHF가 생각보다 큰 역할을 한다

이 논문에서 가장 재미있는 결과는 사실 RLHF다.

- 8B unified 기준 ArenaHard는 **70.0 -> 90.1**로 크게 오른다.
- LiveCodeBench v6도 **56.7 -> 67.2**로 오른다.
- AIME 2025도 **72.8 -> 75.0**으로 오른다.

즉 RLHF가 alignment만 고치는 게 아니라, 이후 reasoning stage를 위한 **better starting point** 역할을 한다는 주장이 table 수준에서도 꽤 설득력 있다.

### 4) 각 stage의 역할이 비교적 분명하다

- **IF-RL**: IFEval / IFBench를 크게 올린다.
- **Math RL**: AIME를 올리고 hard reasoning budget을 늘린다.
- **Code RL**: LiveCodeBench에서 가장 직접적인 성능 상승을 만든다.
- **SWE RL**: SWE-bench Verified를 가장 크게 밀어 올린다.

즉 final score만 보는 것보다, intermediate checkpoint를 함께 보면 **stage별 기여가 꽤 깨끗하게 분리**되어 있다.

## 5-2. What really matters in the experiments

### 1) Half-Half RLHF가 unified model의 핵심이다

논문 deep dive에서 unified 8B SFT model에 대해 RLHF training mode를 비교한 결과, non-thinking only나 thinking only보다 **half-half**가 ArenaHard, AIME25, LiveCodeBench v6 전반에서 더 좋다.

내 해석으로는 이 결과가 꽤 중요하다. unified model은 결국 두 모드를 한 몸에 담아야 하므로, RLHF 단계에서부터 두 mode를 모두 policy distribution 안에 넣어야 cross-mode transfer가 생긴다는 뜻이기 때문이다.

### 2) Reward model size가 실제로 중요하다

저자들은 7B~72B reward model을 비교하면서, 작은 RM이 style artifact에 더 취약하고 reward hacking 위험도 높다고 본다.

이건 꽤 현실적인 포인트다. RLHF를 해보면 policy objective보다 reward model noise가 더 큰 병목이 되는 경우가 많은데, 이 논문은 그걸 "larger RM이 필요하다"는 방향으로 꽤 명확하게 보여준다.

### 3) IF-RL은 좋지만 공짜가 아니다

IF-RL 이후 IFEval과 IFBench는 확실히 오른다. 대신 reasoning trace가 짧아지고 model entropy가 줄면서, 일부 reasoning benchmark는 일시적으로 내려간다.

논문은 이걸 단점이라기보다 **의도된 compression effect**로 해석한다. 내 기준에서도 이 해석이 맞다. IF-RL은 reasoning model을 더 똑똑하게 만든다기보다, **불필요하게 길어진 답을 더 타이트하게 만든다**에 가깝다.

### 4) Code RL temperature와 SWE long-context 분석이 좋다

- Code RL은 높은 temperature가 성능엔 유리하지만 entropy explosion을 부를 수 있다.
- SWE는 input context가 길수록 도움이 되지만, 너무 길면 training noise가 커진다.

이 두 관찰은 둘 다 "RL에서 더 많은 exploration / 더 긴 context가 항상 좋은 것은 아니다"라는 점을 잘 보여준다.

### 5) benchmark reading에서도 주의할 점이 있다

- Table 1은 unified model의 reasoning benchmark를 thinking mode로 보고한다.
- IFEval / IFBench는 thinking / non-thinking 중 더 높은 점수를 사용한다.
- baseline은 공식 reported result를 쓰는 경우가 있어 완전한 apples-to-apples re-evaluation은 아니다.
- 특히 SWE에서는 Gemini-2.5가 Agentless가 아닌 자체 scaffold를 사용한다는 주석이 붙는다.
- 또 tool-calling(BFCL V3)은 이 논문의 대표 강점은 아니다. final 8B unified는 BFCL V3 64.4로, 이 영역에서 headline win을 보여주지는 않는다.

# 6. Limitations

1. **Reward model selection은 아직 불완전하다.**  
   RewardBench를 쓰지만, 높은 RewardBench가 곧 좋은 RLHF policy를 보장하지는 않는다고 저자들 스스로 인정한다.

2. **IF-RL objective와 human preference objective가 충돌할 수 있다.**  
   strict verifier는 constraint satisfaction만 보고, reward model은 전반적 품질을 본다. 이 충돌 때문에 IF-RL이 ArenaHard를 손상시킬 수 있으며, stronger generative RM 같은 보완이 future work로 남아 있다.

3. **SWE long-context는 여전히 취약하다.**  
   저자들은 input prompt가 24K를 넘고 output이 16K일 때 code resolve rate가 크게 떨어질 수 있다고 보고한다. 이는 base model의 32K context 한계와 SFT/RL mismatch를 시사한다.

4. **Code RL은 exploration과 stability 사이의 trade-off가 뚜렷하다.**  
   높은 temperature가 성능에는 유리하지만 entropy explosion을 만들 수 있다. 즉 recipe가 아직 완전히 "plug-and-play"하지는 않다.

5. **이 논문은 순수 RL 논문이 아니라, teacher-heavy post-training recipe다.**  
   SFT data generation에 DeepSeek 계열 strong teacher를 깊게 사용한다. 그래서 이 결과를 "RL만의 승리"로 읽는 것은 과하다.

6. **tool use는 상대적으로 약한 축이다.**  
   paper headline은 general-purpose reasoning 전반이지만, BFCL V3 결과를 보면 tool-calling이 이 논문의 가장 강한 영역이라고 보긴 어렵다.

# 7. My Take

## 7-1. Why this matters for my work

내 관점에서 이 논문의 가장 큰 가치는 **reasoning model을 만드는 RL stack을 domain별로 분해해서 본다**는 점이다.

요즘 reasoning model 논문을 읽다 보면 새로운 objective나 새로운 verifier에 시선이 몰리기 쉽다. 그런데 실제로 서비스 가능한 model을 만들 때 더 중요한 건,

- 어떤 stage를 먼저 할지
- 어디까지를 alignment로 볼지
- 어디서 verbosity를 줄일지
- 어떤 domain에서 별도 curriculum을 둘지
- execution-heavy reward를 어떻게 scale할지

같은 문제들이다.

Nemotron-Cascade는 이 질문들에 대해 꽤 실용적인 답을 준다. 특히 **RLHF를 reasoning preconditioning 단계로 읽는 시각**은 실제 post-training 설계에서 재사용 가치가 크다.

## 7-2. Reuse potential

실제로 가져다 쓰기 좋은 포인트를 정리하면 아래와 같다.

- unified model용 `/think` / `/no_think` explicit flag 설계
- same prompt에 대한 parallel thinking / non-thinking SFT response 구성
- unified model RLHF에서의 half-half batching
- stage별 prompt disjoint split
- Math RL의 response length extension + dynamic filtering
- Code RL의 asynchronous verifier pipeline
- SWE RL의 execution-free reward와 staged context extension
- intermediate checkpoint 공개를 통한 stage-wise diagnosis

즉 이 논문은 단순히 "우리 모델 점수 좋음"보다, **post-training 운영 레시피를 모듈화해서 볼 수 있다**는 점에서 실무 재사용성이 높다.

## 7-3. Follow-up papers

- **Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation**
- **AceReason-Nemotron**
- **Llama-Nemotron**

내가 보기엔 이 논문 다음에는 Cascade 2를 읽는 것이 가장 자연스럽다. Cascade 1이 domain-wise sequential RL의 뼈대를 세운 문서라면, Cascade 2는 그 위에 distillation과 newer artifact를 더 얹는 형태이기 때문이다.

# 8. Summary

- Nemotron-Cascade의 핵심은 단일 objective가 아니라 **domain-wise sequential RL pipeline**이다.
- RLHF를 먼저 두어 verbosity를 줄이고 reasoning efficiency를 높인다는 해석이 이 논문의 가장 좋은 포인트다.
- unified 8B 모델은 parallel SFT responses와 half-half RLHF 덕분에 dedicated thinking model과의 gap을 꽤 좁힌다.
- Code RL과 SWE RL은 각각 execution-heavy domain에서 별도 engineering trick을 붙여 성능을 끌어올린다.
- 이 논문은 leaderboard paper라기보다, **open reasoning post-training recipe를 해부할 수 있게 만든 시스템 논문**으로 읽는 편이 더 정확하다.
