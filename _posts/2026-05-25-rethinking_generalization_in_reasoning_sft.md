---
layout: single
title: "Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability Review"
categories: Study-concept
tag: [LLM, SFT, Reasoning, Post-training, Chain-of-Thought]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.06628)

[Code link](https://github.com/Nebularaid2000/rethink_sft_generalization)

이 논문은 reasoning post-training에서 자주 나오는 명제인 "SFT memorizes, RL generalizes"를 다시 검토한다. 결론은 단순한 반박이 아니다. reasoning SFT도 cross-domain generalization을 만들 수 있지만, 그 결과는 SFT objective 하나로 결정되지 않고 optimization dynamics, training data, base model capability가 같이 맞물릴 때만 나온다는 것이다.

특히 중요한 점은 저자들이 vanilla SFT를 크게 바꾸지 않는다는 데 있다. 새로운 RL objective를 제안하는 논문이 아니라, 같은 SFT objective라도 학습이 충분했는지, long-CoT trace의 품질과 구조가 적절한지, base model이 procedural pattern을 받아들일 capability를 갖고 있는지에 따라 전혀 다른 결론이 나올 수 있음을 보인다.

> 한 줄 요약: **이 논문은 reasoning SFT의 일반화 실패를 SFT 자체의 한계로 보기보다, under-optimization, low-quality data, weak base model이 만든 조건부 현상으로 재해석하고, long-CoT SFT가 reasoning 성능을 개선하는 동시에 safety를 악화시킬 수 있음을 보여준다.**

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- RLVR 중심의 reasoning post-training 흐름에서 SFT의 역할을 다시 정리하게 만든다.
- long-CoT SFT에서 checkpoint를 너무 일찍 보면 generalization을 과소평가할 수 있다는 실무적 메시지가 강하다.
- data quality, CoT structure, model capability를 분리해 보는 controlled analysis가 깔끔하다.
- reasoning 향상이 safety degradation과 같이 온다는 점을 같은 프레임 안에서 다룬다.

이 논문의 가장 좋은 점은 SFT vs RL 구도를 너무 빨리 결론내리지 않는다는 점이다. 질문을 "SFT가 일반화하는가"에서 "어떤 조건에서, 어떤 비용으로 일반화하는가"로 바꾸면 post-training recipe를 훨씬 더 현실적으로 볼 수 있다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 다루는 문제는 long chain-of-thought supervision을 사용하는 reasoning SFT가 cross-domain generalization을 할 수 있는지다.

기존 narrative는 대략 다음과 같다.

1. SFT는 demonstration을 모방하므로 training domain 안에서는 좋아진다.
2. 하지만 unseen domain에서는 surface pattern만 따라 하거나 memorization에 머문다.
3. RL은 on-policy exploration과 reward optimization을 통해 더 넓은 generalization을 만든다.

이 논문은 이 narrative가 너무 거칠다고 본다. 특히 long-CoT reasoning SFT는 short answer SFT와 다르다. target sequence가 길고, 중간에 decomposition, backtracking, verification 같은 절차적 패턴이 들어가며, 학습 초기에 모델이 그 구조를 제대로 internalize하지 못할 가능성이 크다. 그러면 짧은 training checkpoint만 보고 "SFT는 일반화하지 않는다"고 결론내릴 수 있다.

저자들의 문제 설정은 다음처럼 정리할 수 있다.

| Axis | Question |
| --- | --- |
| Optimization | long-CoT trace를 충분히 학습하기 전에 평가한 것은 아닌가? |
| Data | 데이터가 correct answer만이 아니라 좋은 reasoning procedure를 담고 있는가? |
| Model capability | base model이 procedural pattern을 internalize할 만큼 충분히 강한가? |
| Safety | reasoning transfer가 refusal/safety behavior에는 어떤 비용을 만드는가? |

## 1-2. Why previous approaches are insufficient

기존 SFT generalization 논의에는 몇 가지 혼선이 있다.

첫째, short training protocol의 결과를 SFT objective의 본질로 해석하기 쉽다. 이 논문은 1 epoch 설정에서는 in-domain math는 좋아지지만 OOD reasoning과 general capability가 제한적이거나 악화될 수 있음을 먼저 재현한다. 하지만 training을 늘리면 여러 benchmark에서 "dip-and-recovery" pattern이 나타난다.

둘째, SFT 데이터의 quality와 structure가 섞여 있다. 정답이 맞는 데이터인지, long thinking trace가 있는지, trace 안에 backtracking과 self-verification 같은 procedure가 있는지에 따라 결과가 다르다. 단순히 SFT라는 label만으로 일반화 여부를 판단하기 어렵다.

셋째, model capability가 통제되지 않은 경우가 많다. 같은 Math-CoT-20k를 넣어도 Qwen3-14B, Qwen3-8B, Qwen3-4B, Qwen3-1.7B는 서로 다른 trajectory를 보인다. weaker model은 긴 답변 형식만 모방하고, stronger model은 절차적 reasoning pattern을 더 잘 internalize할 수 있다.

넷째, reasoning gain과 safety cost를 분리해서 보는 경우가 많다. 이 논문은 reasoning SFT가 cross-domain task에는 도움이 될 수 있지만, harmful query에 대한 refusal behavior를 약화시킬 수 있다고 본다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 새로운 algorithm이라기보다 conditional analysis다. 저자들은 vanilla SFT objective를 유지한 채 세 가지 요인을 체계적으로 바꾼다.

1. Optimization dynamics
2. Training data quality and structure
3. Base model capability

그리고 다음 결론을 제시한다.

- Apparent non-generalization은 under-optimization artifact일 수 있다.
- Long-CoT SFT는 초기에 response length가 급증하고 performance가 떨어진 뒤, 충분히 학습되면 response가 짧아지고 성능이 회복되는 dip-and-recovery pattern을 보인다.
- Verified long-CoT trace는 math뿐 아니라 code, science, broad reasoning benchmark에도 transfer를 줄 수 있다.
- Countdown처럼 좁은 arithmetic game의 long-CoT trace도 backtracking, decomposition, verification 같은 절차를 담고 있으면 일부 reasoning transfer를 만들 수 있다.
- 하지만 lower-capability model은 이런 procedure를 학습하기보다 verbosity를 모방하는 데 머무를 수 있다.
- Generalization은 asymmetric하다. reasoning은 좋아질 수 있지만 safety는 나빠질 수 있다.

## 2-2. Design intuition

내가 이해한 이 논문의 설계 직관은 다음과 같다.

SFT가 배우는 것은 반드시 content만은 아니다. Long-CoT trace에는 문제 풀이 domain knowledge뿐 아니라 procedure가 들어 있다. 예를 들어 어려운 수학 문제를 풀 때 모델은 식을 세우고, 중간 결과를 점검하고, 틀린 경로를 버리고, 다른 decomposition을 시도한다. 이런 procedure는 math domain 밖에서도 일부 재사용될 수 있다.

하지만 procedure를 배우려면 세 조건이 필요하다.

1. 충분히 오래 학습해야 한다.
2. 데이터가 실제로 좋은 procedure를 담고 있어야 한다.
3. base model이 그 procedure를 압축해서 일반화할 capability를 갖고 있어야 한다.

이 중 하나라도 부족하면 SFT는 쉽게 surface imitation으로 보인다. 긴 답변을 흉내 내지만 reasoning transfer가 없거나, in-domain math만 좋아지고 OOD task는 나빠지거나, safety behavior가 무너질 수 있다.

이 관점에서 response length는 단순 verbosity metric이 아니라 coarse diagnostic이 된다. 길이가 계속 커지거나 아직 줄어드는 중이라면, 모델이 long-CoT distribution에 적응하는 중일 수 있다. 반대로 충분히 학습된 stronger model은 더 짧고 target-oriented한 response로 수렴하는 경향을 보인다.

# 3. Architecture / Method

## 3-1. Overview

이 논문은 model architecture 논문이 아니라 experimental analysis 논문이다. 따라서 method는 새로운 block보다 controlled experiment design으로 보는 편이 맞다.

| Item | Description |
| --- | --- |
| Goal | reasoning SFT의 cross-domain generalization이 어떤 조건에서 나타나는지 분석 |
| Default data | Math-CoT-20k, 20,480 long-CoT math examples |
| Teacher | Qwen3-32B with thinking enabled |
| Base models | Qwen3-14B-Base, Qwen3-8B-Base, InternLM2.5-20B-Base, Qwen2.5 variants, Qwen3 1.7B/4B/8B/14B |
| Default objective | standard SFT negative log-likelihood on response tokens |
| Main factors | optimization schedule, data config, model scale/capability |
| Key diagnostic | benchmark trajectory plus response length trajectory |
| Main caution | reasoning gain can come with safety degradation |

SFT objective 자체는 표준 negative log-likelihood다.

$$
L_{SFT}(\theta) = - E_{(x,y) \sim D} \sum_{t=1}^{|y|} \log p_{\theta}(y_t | x, y_{<t})
$$

이 단순한 objective를 유지한다는 점이 중요하다. 이 논문은 SFT objective를 RL처럼 바꾸면 일반화된다는 주장이 아니라, 같은 SFT에서도 실험 조건을 잘못 잡으면 일반화 여부를 잘못 판단할 수 있다는 주장을 한다.

## 3-2. Module breakdown

### 1) Optimization dynamics: dip-and-recovery

가장 먼저 볼 포인트는 training trajectory다. 저자들은 1 epoch training에서는 기존 연구처럼 weak generalization pattern이 보일 수 있음을 재현한다. 하지만 8 epoch까지 확장해 checkpoint를 추적하면 performance가 non-monotonic하게 움직인다.

패턴은 대략 이렇다.

1. 초기에는 model이 long-CoT format을 따라 하면서 response length가 급증한다.
2. 이 시기에는 OOD benchmark 성능이 떨어지거나 정체될 수 있다.
3. 더 학습하면 response length가 다시 줄고, reasoning benchmark 성능이 회복된다.
4. 일부 benchmark는 base model을 넘어서며 cross-domain gain을 보인다.

이게 논문의 핵심 실무 메시지다. Long-CoT SFT에서는 **early checkpoint 하나로 generalization 여부를 판단하면 안 된다.**

### 2) Response length as optimization diagnostic

논문은 response length를 중요한 diagnostic으로 본다. 초기에 길어지는 것은 모델이 thinking-like trace의 surface pattern을 먼저 배우기 때문일 수 있다. 아직 decomposition, backtracking, self-evaluation 같은 finer reasoning pattern을 안정적으로 학습하지 못하면 긴 답변은 오히려 성능을 해칠 수 있다.

반대로 충분히 학습되면 더 짧고 목적 지향적인 response로 수렴한다. 따라서 response length가 아직 급격히 변하고 있다면, in-domain math가 좋아 보여도 model이 완전히 최적화되었다고 보기 어렵다.

### 3) Data quality and structure

두 번째 축은 data다. 저자들은 Math-CoT-20k 외에 세 가지 variant를 비교한다.

| Data config | Description |
| --- | --- |
| Math-CoT-20k | verified long-CoT math traces |
| Math-NoCoT-20k | 같은 query와 final solution을 쓰되 thinking process 제거 |
| NuminaMath-20k | 같은 query에 대해 NuminaMath-1.5의 human-crafted short solution 사용 |
| Countdown-CoT-20k | Countdown arithmetic game에서 생성한 long-CoT traces |

이 비교가 좋은 이유는 data quality와 structure를 어느 정도 분리해 볼 수 있기 때문이다. Math-CoT vs Math-NoCoT는 long thinking trace의 효과를 보고, Math-NoCoT vs NuminaMath는 no-long-CoT 조건에서 data quality 차이를 본다. Countdown-CoT는 domain-specific math content보다 procedural trace가 transfer되는지를 보는 실험이다.

### 4) Model capability

세 번째 축은 model capability다. 같은 Math-CoT-20k와 같은 training protocol을 쓰더라도 Qwen3 14B, 8B, 4B, 1.7B는 다르게 움직인다.

논문이 제시하는 해석은 명확하다.

- Stronger model은 long-CoT trace에서 transferable procedural pattern을 학습할 가능성이 높다.
- Weaker model은 긴 답변 형식, 반복, verbosity를 모방하는 단계에 머무를 수 있다.
- 따라서 optimization과 data가 좋아도 base model capability가 부족하면 generalization이 제한된다.

이 부분은 실무적으로 중요하다. 같은 SFT recipe를 작은 model에 그대로 적용했을 때, 큰 model에서 보인 transfer가 그대로 나오리라 기대하면 안 된다.

### 5) Asymmetric generalization: safety cost

마지막 축은 safety다. Long-CoT SFT는 reasoning task에서는 도움이 될 수 있지만, harmful query에 대한 refusal behavior를 약화시킬 수 있다.

논문은 HEx-PHI에서 attack success rate를 보고, Math-CoT-20k로 학습한 모델이 Math-NoCoT-20k보다 safety degradation이 훨씬 크다고 보고한다. 두 데이터는 query와 final solution을 공유하므로, 저자들은 degradation이 math content 자체보다 long-CoT trace의 procedural pattern에 더 가깝다고 해석한다.

내가 보기엔 이 부분이 매우 중요하다. Long-CoT는 문제 해결 prior를 강화한다. 하지만 harmful query에서는 그 prior가 refusal boundary를 장애물처럼 우회하려는 방향으로 작동할 수 있다.

# 4. Training / Data / Recipe

## 4-1. Data

Default training dataset은 Math-CoT-20k다.

- 총 20,480개 math reasoning examples
- query는 OpenR1-Math-220k default subset에서 sampling
- response는 Qwen3-32B thinking enabled로 생성
- 각 response는 thinking process, step-by-step final summary, answer를 포함
- 여러 response를 생성한 뒤 math-verify로 correct answer만 유지
- maximum response length는 16,384 tokens

Appendix의 data generation detail도 참고할 만하다.

- generation temperature는 0.6
- top-p는 0.95
- top-k는 20
- min-p는 0
- query마다 correct response 하나만 유지
- 여러 correct response가 있으면 하나를 random selection

데이터 variant는 위에서 본 네 가지다. 중요한 것은 저자들이 단순히 "more data"를 보는 것이 아니라, long-CoT structure, solution quality, procedural trace를 나눠서 본다는 점이다.

## 4-2. Training strategy

Default training recipe는 다음과 같다.

| Component | Setting |
| --- | --- |
| Objective | standard SFT, response-token NLL |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Batch size | 256 |
| LR schedule | cosine |
| Epochs | 8 |
| Default model examples | Qwen3-14B-Base, Qwen3-8B-Base |

저자들은 이 default recipe를 기준으로 여러 schedule을 비교한다. 특히 fixed 640-step budget 아래에서 repeated exposure와 one-pass coverage를 비교한 Table 1이 중요하다.

| Setting | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval | AlpacaEval RM | HaluEval | TruthfulQA helpful |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20k, bsz 256, ep 8 | 95.1 | 66.0 | 55.1 | 63.3 | 74.4 | 68.9 | 1.42 | 72.8 | 95.6 |
| 2.5k, bsz 32, ep 8 | 94.9 | 61.7 | 51.7 | 59.9 | 73.6 | 63.4 | 1.10 | 72.6 | 96.2 |
| 20k, bsz 32, ep 1 | 92.9 | 48.0 | 45.4 | 46.8 | 70.5 | 59.8 | 0.93 | 75.2 | 86.3 |

핵심 비교는 두 번째와 세 번째 row다. 둘 다 640 gradient steps로 맞췄지만, 2.5k examples를 8 epoch 반복한 경우가 20k examples를 1 epoch 본 경우보다 강하다. 즉 long-CoT SFT에서는 단순 coverage보다 repeated exposure가 더 중요할 수 있다. 다만 첫 번째 row가 더 강하므로 data diversity도 여전히 중요하다.

## 4-3. Evaluation setup

평가 suite는 크게 네 묶음이다.

| Group | Benchmarks | Meaning |
| --- | --- | --- |
| In-domain reasoning | MATH500, AIME24 | math training domain과 직접 연결 |
| OOD reasoning | LiveCodeBench v2, GPQA-Diamond, MMLU-Pro | code, science, broad knowledge reasoning |
| General capabilities | IFEval, AlpacaEval 2.0, HaluEval, TruthfulQA | instruction following, open-ended quality, truthfulness |
| Safety | HEx-PHI | harmful query resistance, ASR, harmfulness score |

Default decoding은 temperature 0.6, max generation length 32,768이다. 논문은 IFEval, HaluEval, MMLU-Pro는 pass@1, MATH500과 LiveCodeBench v2와 GPQA-Diamond는 avg@3, AIME24는 avg@10을 사용한다.

이 평가 설계의 장점은 reasoning gain만 보지 않는다는 점이다. Math SFT가 math benchmark만 올렸다면 그건 당연한 결과일 수 있다. 이 논문은 code, science, instruction following, truthfulness, safety까지 같이 보면서 "무엇이 transfer되고 무엇이 깨지는가"를 묻는다.

# 5. Evaluation

## 5-1. Main results

### 1) Short-epoch result는 SFT generalization을 과소평가할 수 있다

논문은 먼저 Qwen3-14B-Base를 Math-CoT-20k로 1 epoch 학습하면 in-domain math는 크게 좋아지지만, OOD reasoning과 general capability에서는 제한적이거나 negative gain이 나올 수 있음을 보인다. 이 부분은 기존 narrative를 재현한다.

하지만 training을 8 epoch까지 늘리고 checkpoint trajectory를 보면 다른 그림이 나온다. MATH500, AIME24 같은 in-domain math는 빠르게 회복하고, LCB v2, GPQA-D, IFEval, AlpacaEval 같은 benchmark도 더 깊은 dip 뒤에 recovery를 보인다. 따라서 early checkpoint만 보고 SFT generalization을 판단하면 결론이 달라질 수 있다.

### 2) Repeated exposure가 long-CoT internalization에 중요하다

Table 1에서 가장 흥미로운 부분은 2.5k examples를 8 epoch 반복한 setting이 20k examples를 1 epoch 본 setting보다 전반적으로 강하다는 점이다. 예를 들어 AIME24는 61.7 vs 48.0, GPQA-D는 59.9 vs 46.8, IFEval은 63.4 vs 59.8이다.

이 결과는 long-CoT data가 short answer label보다 fitting difficulty가 높다는 해석과 잘 맞는다. Long-CoT trace는 한 번 훑는 것보다 반복 노출을 통해 distribution shift를 완화하고 procedure를 internalize하는 과정이 필요할 수 있다.

### 3) Long-CoT trace는 content보다 procedure를 transfer할 수 있다

Table 2의 Qwen3-14B block만 보면 data structure 차이가 뚜렷하다.

| Qwen3-14B data | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval | AlpacaEval RM | HaluEval | TruthfulQA helpful |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 77.8 | 14.7 | 37.5 | 44.1 | 61.8 | 64.2 | 0.53 | 54.7 | 94.4 |
| Math-CoT | 95.1 | 66.0 | 55.1 | 63.3 | 74.4 | 68.9 | 1.42 | 72.8 | 95.6 |
| Math-NoCoT | 82.4 | 17.0 | 40.3 | 48.3 | 69.1 | 71.7 | 2.11 | 70.9 | 100.0 |
| NuminaMath | 74.8 | 14.0 | 20.4 | 38.4 | 59.0 | 52.8 | -0.45 | 62.7 | 88.6 |
| Countdown-CoT | 91.5 | 41.7 | 43.8 | 53.0 | 65.4 | 61.3 | 1.36 | 72.3 | 92.4 |

Math-CoT가 reasoning-intensive task에서 가장 강하다. Math-NoCoT는 IFEval과 AlpacaEval에서는 오히려 더 좋은 경우가 있지만, reasoning task gain은 훨씬 작다. NuminaMath는 여러 OOD task에서 약하고, 저자들은 이를 low-quality solution이 SFT utility를 크게 낮출 수 있다는 근거로 본다.

Countdown-CoT는 재미있는 결과다. 단순 arithmetic game인데도 Qwen3-14B에서 MATH500 91.5, AIME24 41.7까지 올라간다. Math-NoCoT보다 math task에서 더 좋다. 이는 domain content 자체보다 decomposition, trial-and-error, backtracking, verification 같은 procedure가 transfer될 수 있다는 해석을 가능하게 한다.

다만 이 결과는 model-dependent하다. InternLM2.5-20B에서는 Countdown-CoT의 math gain이 제한적이고, IFEval은 나빠질 수 있다. 따라서 procedural generalization은 데이터만의 속성이 아니라 base model과의 interaction으로 봐야 한다.

### 4) Model capability가 충분하지 않으면 verbosity imitation에 머문다

Qwen3 1.7B, 4B, 8B, 14B를 같은 Math-CoT-20k와 같은 protocol로 학습한 결과, 14B는 뚜렷한 dip-and-recovery와 broad gain을 보이고, 8B와 4B는 더 작은 recovery를 보이며, 1.7B는 late checkpoint에서도 marginal 또는 negative gain에 머문다.

Response length도 이와 연결된다. 작은 모델은 extended training 뒤에도 긴 response를 유지하는 경향이 있고, 큰 모델은 더 빠르게 짧고 안정적인 response로 수렴한다. 이 결과는 작은 모델이 long-CoT의 surface form을 모방하는 데 머무를 수 있음을 시사한다.

### 5) Safety는 같이 악화될 수 있다

HEx-PHI 결과에서 long-CoT SFT는 세 모델 모두에서 attack success rate를 크게 높인다. 반면 Math-NoCoT는 degradation이 훨씬 작다. 저자들은 같은 query와 final solution을 공유하는 두 데이터 사이의 차이가 thinking process의 유무라는 점을 근거로, safety drop이 math content보다 long-CoT procedural pattern과 더 관련 있다고 본다.

Case study도 인상적이다. base model은 harmful query에 짧게 거절하지만, long-CoT SFT 후 모델은 처음에는 위험성을 인식하다가 thinking 과정에서 "educational purpose" 같은 rationalization을 만들고, 결국 harmful details를 제공하는 방향으로 흐를 수 있다.

## 5-2. What really matters in the experiments

이 논문의 실험에서 진짜 중요한 지표는 단일 benchmark score가 아니다. 다음 네 가지를 함께 봐야 한다.

1. Performance trajectory

- 마지막 checkpoint만 보면 overfitting인지 recovery인지 알기 어렵다.
- Early checkpoint만 보면 under-optimization을 SFT failure로 착각할 수 있다.

2. Response length trajectory

- 초기에 length surge가 나타나고 이후 감소하는지 봐야 한다.
- 길이가 계속 늘거나 줄어드는 중이면 아직 optimization stage가 안정되지 않았을 수 있다.

3. Cross-domain benchmark mix

- MATH500과 AIME24만 보면 in-domain gain만 보게 된다.
- LCB v2, GPQA-D, MMLU-Pro까지 봐야 procedure transfer를 읽을 수 있다.

4. Safety regression

- reasoning gain만 보고 배포하면 위험하다.
- Long-CoT SFT는 refusal boundary를 약화시킬 수 있으므로 safety suite가 post-training loop 안에 들어가야 한다.

이 논문의 가장 재사용 가능한 evaluation lesson은 checkpoint selection이다. Reasoning SFT에서는 best math checkpoint가 best overall checkpoint가 아닐 수 있고, response length와 safety까지 같이 보는 multi-objective checkpointing이 필요하다.

# 6. Limitations

1. Math reasoning data 중심이다.

- 논문은 math problem이 verification하기 쉽고 high-quality long-CoT response를 rejection sampling으로 얻기 쉬워서 math-only setup을 사용한다.
- 따라서 dip-and-recovery가 code generation, scientific reasoning, multimodal reasoning에서도 같은 형태로 나타나는지는 추가 확인이 필요하다.

2. Dense model 규모가 제한된다.

- 실험은 dense model 기준 최대 20B parameter까지다.
- larger dense model이나 MoE architecture에서는 optimization dynamics와 capacity constraint가 달라질 수 있다.

3. RL method와 직접 비교하지 않는다.

- 이 논문은 SFT 자체의 조건부 generalization을 분석하는 데 집중한다.
- 따라서 "SFT가 RL보다 낫다" 또는 "RL이 필요 없다"는 결론으로 읽으면 과장이다.

4. Base model choice가 강한 confound일 수 있다.

- Qwen3와 InternLM2.5처럼 비슷한 크기라도 pretraining/mid-training pipeline이 다르면 같은 SFT data에 다르게 반응한다.
- 실제 recipe 설계에서는 parameter count보다 pretraining distribution과 reasoning prior를 같이 봐야 한다.

5. Safety evaluation은 더 넓게 확장될 필요가 있다.

- HEx-PHI와 case study는 강한 warning signal을 주지만, 실제 deployment safety는 더 다양한 policy, jailbreak, tool-use, multi-turn scenario를 포함해야 한다.

6. CoT trace 공개와 safety 사이에 긴장이 있다.

- Long-CoT trace는 procedural learning에 도움이 될 수 있지만, 같은 procedure가 refusal 우회에도 쓰일 수 있다.
- 이 문제는 단순 filtering으로 끝나기보다, SFT data design, refusal data mixing, safety-aware decoding, policy evaluation을 함께 봐야 한다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 post-training을 볼 때 "algorithm 이름"보다 "조건"을 먼저 봐야 한다는 점을 잘 보여준다. SFT, GRPO, DPO, RLVR 같은 label은 중요하지만, 실제 성능은 아래 조합으로 결정된다.

- base model이 어떤 prior를 갖고 있는가
- data가 어떤 procedure를 담고 있는가
- optimization이 그 procedure를 충분히 internalize할 만큼 진행됐는가
- evaluation이 in-domain gain과 OOD transfer와 safety regression을 모두 보는가

특히 document AI나 agentic workflow에서도 이 메시지는 그대로 적용된다. 복잡한 reasoning trace를 SFT할 때, 짧은 학습 후에 OOD generalization이 없다고 판단하면 recipe를 버리기 쉽다. 하지만 실제로는 repeated exposure, checkpoint trajectory, response length, data quality를 같이 봐야 한다.

## 7-2. Reuse potential

실무적으로 재사용할 수 있는 포인트는 다음과 같다.

1. Checkpoint trajectory를 반드시 저장

- step 0, 10, 20, 40, 80, 160, 320, 640처럼 log-scale checkpoint를 저장하면 dip-and-recovery를 볼 수 있다.
- 마지막 checkpoint 하나만으로 판단하지 않는 것이 중요하다.

2. Response length를 dashboard metric으로 추가

- 평균 token length, max length, format error, stop token miss 등을 같이 본다.
- length surge 뒤에 performance recovery가 오는지 확인한다.

3. Long-CoT와 No-CoT ablation을 pair로 구성

- 같은 query와 answer를 유지하고 thinking trace만 제거하면, procedure의 효과를 더 깨끗하게 볼 수 있다.

4. Low-quality solution을 별도로 테스트

- 정답이 맞는지뿐 아니라, 중간 reasoning step이 충분한지, missing step이 많은지, hallucinated shortcut이 있는지 봐야 한다.

5. Safety regression을 post-training loop 안에 포함

- reasoning benchmark가 올라가는 checkpoint라도 safety가 무너질 수 있다.
- 특히 harmful query에서 self-rationalization이 생기는지 qualitative case study를 같이 봐야 한다.

6. Small model에는 같은 recipe를 그대로 기대하지 않기

- 작은 모델은 long-CoT procedure를 internalize하기보다 verbosity를 흉내낼 수 있다.
- small model에는 trace compression, shorter CoT, curriculum, distillation target redesign이 필요할 수 있다.

## 7-3. Follow-up papers

후속으로 같이 읽을 만한 축은 다음과 같다.

- Chu et al., "SFT memorizes, RL generalizes" 계열 연구
- ReFT: Reasoning with Reinforced Fine-Tuning
- On the Role of Reasoning Patterns in the Generalization Discrepancy of Long Chain-of-Thought Supervised Fine-Tuning
- Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning
- Long-CoT safety, self-jailbreaking, refusal degradation 관련 논문들

# 8. Summary

- 이 논문은 reasoning SFT가 일반화하지 않는다는 명제를 조건부 문제로 바꾼다.
- Long-CoT SFT에서는 early degradation 뒤에 recovery가 오는 dip-and-recovery pattern이 나타날 수 있다.
- Verified long-CoT trace와 repeated exposure는 procedure internalization에 중요하다.
- Stronger base model은 procedural pattern을 transfer하지만, weaker model은 verbosity imitation에 머무를 수 있다.
- **Reasoning gain은 safety degradation과 같이 올 수 있으므로**, post-training 평가는 benchmark score만으로 끝나면 안 된다.
