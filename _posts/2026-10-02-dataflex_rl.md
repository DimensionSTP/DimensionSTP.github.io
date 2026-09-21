---
layout: single
title: "DataFlex-RL: An Evaluation Platform for RLVR Data Policies Review"
categories: Study-concept
tag: [RLVR, GRPO, DataPolicy]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/pdf/2609.06107)

[Code link](https://github.com/haolpku/DataFlex-RL)

[Documentation](https://haolpku.github.io/DataFlex-RL-Doc)

RLVR에서 data policy는 직관적으로 중요한 문제다. 같은 verifier와 같은 optimizer를 사용해도 어떤 prompt에 rollout budget을 쓰는지, 어떤 response를 update에 남기는지, 어떤 token에 더 큰 weight를 주는지에 따라 학습 trajectory가 달라질 수 있다.

그래서 최근에는 다음과 같은 아이디어가 자주 등장한다.

- 모든 rollout이 정답이거나 오답인 prompt group은 버린다.
- 중간 난도의 prompt를 우선한다.
- Reward variance가 큰 group을 선택한다.
- Advantage가 큰 sample이나 token에 더 큰 weight를 준다.
- 성능이 정체된 domain에 다음 batch를 더 배정한다.

문제는 각 방법이 서로 다른 model, seed, benchmark, training budget에서 보고되었다는 점이다. Method 자체의 효과와 training stack 차이를 분리하기 어렵고, seed 하나에서 보인 작은 차이를 data policy의 일반적인 이득으로 읽기 쉽다.

DataFlex-RL은 이 문제를 새 policy 하나로 해결하지 않는다. 대신 selection, reweighting, adaptive mixture를 같은 GRPO pipeline 안에서 비교할 수 있는 evaluation platform을 만들고, matched seed와 domain-balanced benchmark로 다시 측정한다.

> 한 줄 요약: DataFlex-RL은 RLVR data policy를 selection, reweighting, mixture adaptation의 세 intervention point로 통일해 비교하며, 통제된 실험에서는 uniform GRPO 자체의 개선은 크지만 개별 data policy가 uniform sampling을 재현 가능하게 이긴다는 증거는 찾지 못했다고 보고한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- RLVR data policy의 효과를 method proposal이 아니라 controlled evaluation 문제로 다룬다.
- 13개 configuration을 12 matched seeds에서 비교해 small mean gap과 seed variance를 분리한다.
- Math, logic, science를 모두 포함한 domain-balanced score를 사용해 benchmark omission의 영향을 측정한다.
- Selection, reweighting, domain mixing을 scorer plus actuator interface로 구현해 새 방법을 같은 trainer에 추가할 수 있다.
- Negative result를 단순 실패가 아니라 future data policy를 비교하기 위한 공통 infrastructure로 전환한다.

# 1. Problem Setting

## 1-1. GRPO에서 data policy가 개입하는 위치

Prompt group $g$에서 $K$개의 response를 sampling하고, verifier reward로 group-relative advantage를 만든다고 하자. Response $k$의 token loss를 개념적으로 다음처럼 쓸 수 있다.

$$
\mathcal{L}_{gk}
=
\frac{1}{L_{gk}}
\sum_{t=1}^{L_{gk}}
\ell_{gkt}^{\mathrm{GRPO}}
$$

Uniform GRPO는 sampling된 valid response를 같은 rule로 update에 사용한다. Data policy는 이 flow의 세 지점에 개입한다.

1. Selection
   - 현재 rollout 중 어떤 response or group을 update에 사용할지 결정한다.

2. Reweighting
   - Sample or token을 모두 남기되 loss contribution을 다르게 준다.

3. Mixture adaptation
   - 다음 rollout batch를 어느 domain에서 sampling할지 바꾼다.

이 세 가족은 모두 data를 바꾸지만 operating point가 다르다.

| Policy family | Action timing | 바꾸는 것 | 직접 줄일 수 있는 비용 |
| --- | --- | --- | --- |
| Selection | Rollout and advantage 이후 | Current update에 남는 sample | Update compute, 반드시 rollout compute는 아님 |
| Reweighting | Policy loss 계산 시점 | Sample or token contribution | 주로 optimization signal |
| Mixture adaptation | 다음 rollout 이전 | Future prompt domain distribution | Rollout allocation |

특히 selection은 이미 생성한 rollout을 버릴 수 있다. 따라서 final accuracy가 좋아져도 generation cost를 줄였다고 자동으로 말할 수 없다. 반대로 adaptive mixture는 future batch 자체를 바꾸므로 rollout allocation과 더 직접적으로 연결된다.

## 1-2. 왜 기존 비교로는 결론이 불안정한가

### 1) Seed variance가 method gap보다 클 수 있다

RL training은 rollout sampling, reward distribution, optimizer state에 민감하다. 한두 seed의 0.5 point gain은 다른 seed에서 쉽게 사라질 수 있다.

DataFlex-RL은 같은 starting checkpoint와 seed를 method and baseline에 대응시키는 paired comparison을 사용한다. Seed $s$에서 method와 baseline의 차이를 다음처럼 본다.

$$
\Delta_s
=
M(\theta_{s}^{\mathrm{policy}})
-
M(\theta_{s}^{\mathrm{uniform}})
$$

관심 대상은 raw score가 아니라 paired difference의 평균과 confidence interval이다.

### 2) Training stack 차이가 policy 효과를 가린다

Rollout count, response length, KL coefficient, verifier, prompt format, optimizer step 수가 다르면 data policy만의 효과를 분리하기 어렵다. DataFlex-RL은 common GRPO recipe를 고정하고 intervention만 교체한다.

### 3) Benchmark aggregation이 winner를 바꿀 수 있다

Math benchmark만 많이 넣거나 logic domain을 빼면 특정 policy가 유리해질 수 있다. 논문은 domain-balanced 12-benchmark summary와 math-heavy 6-benchmark summary를 같은 run에 적용해 ranking sensitivity를 직접 측정한다.

### 4) Data policy는 accuracy 외의 변수를 바꿀 수 있다

Selection rate, update token volume, response length, domain proportion, advantage distribution은 바뀌지만 final score는 같을 수 있다. 따라서 process metric과 outcome metric을 함께 봐야 한다.

# 2. Core Idea

## 2-1. Scorer와 actuator를 분리한다

DataFlex-RL의 software abstraction은 signal과 action을 분리하는 것이다.

- Scorer는 existing RL loop에서 signal을 읽는다.
- Actuator는 score를 selection mask, weight, or domain probability로 변환한다.

예를 들어 group solve rate라는 같은 signal도 여러 방식으로 사용할 수 있다.

- 너무 쉬운 group과 너무 어려운 group을 filter한다.
- 중간 난도 group의 weight를 높인다.
- 특정 domain의 difficulty가 높으면 다음 sampling probability를 바꾼다.

이 분리는 새 아이디어를 trainer fork가 아니라 configuration 조합으로 만들게 한다. Method family와 signal source를 분리해 어느 부분이 효과를 만드는지도 비교하기 쉽다.

## 2-2. 세 intervention family

### 1) Selection

Binary mask $m_{gk}$를 사용해 response contribution을 켜거나 끈다.

$$
\mathcal{L}_{\mathrm{select}}
=
\frac{
\sum_{g,k}m_{gk}\mathcal{L}_{gk}
}{
\sum_{g,k}m_{gk}+\epsilon
}
$$

Selection signal은 group solve rate, reward difficulty, advantage magnitude처럼 이미 rollout에서 얻는 값을 사용할 수 있다.

### 2) Reweighting

Continuous weight $w_{gk}$ or token weight $w_{gkt}$를 곱한다.

$$
\mathcal{L}_{\mathrm{reweight}}
=
\frac{
\sum_{g,k}w_{gk}\mathcal{L}_{gk}
}{
\sum_{g,k}w_{gk}+\epsilon
}
$$

Response를 버리지 않으면서 informative sample에 더 큰 update를 주는 방식이다. 하지만 weight distribution이 sharp해지면 effective batch size가 작아지고 gradient variance가 커질 수 있다.

### 3) Mixture adaptation

Domain $d$의 다음 sampling probability를 $p_{t+1}(d)$라고 하자.

$$
p_{t+1}(d)
=
\operatorname{Normalize}
\left(
A\left(S_t(d)\right)
\right)
$$

$S_t(d)$는 reward gap or difficulty 같은 domain signal이고, $A$는 reward-gap, UCB, curriculum actuator다. Adaptive mixture는 현재 batch의 loss가 아니라 future rollout distribution을 바꾼다.

## 2-3. Platform의 진짜 기여는 matched control이다

DataFlex-RL은 matched-random and matched-update-token control을 제공한다.

- Matched random은 targeted selection과 같은 selection rate를 유지하되 random으로 고른다.
- Matched update token은 method와 baseline의 update token volume을 맞춘다.

이런 control은 단순한 질문에 답한다.

- 좋은 sample을 고른 효과인가.
- Update token이 줄어서 생긴 효과인가.
- Filter strength가 달라서 생긴 효과인가.

논문의 결과는 모든 method를 하나의 explanation으로 정리하기 어렵다는 쪽에 가깝다. 어떤 policy는 targeted signal이 random보다 낫고, 다른 policy는 그렇지 않으며, update token을 맞췄을 때도 방향이 일관되지 않는다.

# 3. Architecture / Method

## 3-1. Overview

| Component | Role |
| --- | --- |
| Host trainer | verl v1 and standard GRPO-style data flow |
| Scorer | Reward, advantage, log-probability, group statistic에서 signal 생성 |
| Actuator | Signal을 mask, weight, domain mixture로 변환 |
| Mechanism | Select, reweight, mix 중 intervention point 결정 |
| Registry | Config name으로 scorer and actuator 조합 |
| Diagnostics | Kept fraction, weight statistics, domain proportion 기록 |
| Controls | Matched random, matched update token, strength controls |

DataFlex-RL은 standard `main_ppo` entry point를 유지한다. Plugin이 trainer mode와 policy configuration을 등록하고, existing policy-loss hook을 재사용한다.

## 3-2. Built-in component examples

공개 implementation에는 다음과 같은 scorer가 포함된다.

- `group_solve_rate`
- `reward_difficulty`
- `advantage_magnitude`
- `token_prob`

Actuator example은 다음과 같다.

- `threshold_band`
- `topk_fraction`
- `max_variance`
- `matched_random`
- `softmax`
- `per_advantage`
- `advantage_reweight`
- `reward_gap`
- `dump_ucb`
- `tscl`

이 목록은 paper의 13 configuration과 완전히 같은 의미로 읽기보다, platform이 어떤 data policy interface를 제공하는지 보여주는 implementation inventory로 보는 편이 안전하다.

## 3-3. Domain-balanced aggregation

Math, logic, science domain의 benchmark 평균을 각각 $M$, $L$, $S$라고 하면 primary overall score는 다음처럼 domain을 같은 비중으로 둔다.

$$
\operatorname{DB12}
=
\frac{M+L+S}{3}
$$

이 방식은 benchmark 개수나 item 수가 많은 domain이 overall을 지배하지 않게 한다. 논문은 같은 run을 benchmark-macro, item-weighted, math-heavy summary로 다시 계산해 aggregation sensitivity를 분석한다.

# 4. Training / Data / Recipe

## 4-1. Training data

공통 training corpus는 15,000 prompts로 구성되며, 세 domain에 5,000개씩 배정된다.

| Domain | Main source | Verification |
| --- | --- | --- |
| Math | math_dapo, DeepScaler, GSM8K | Boxed answer verifier |
| Logic | Procedural Knights and Knaves | Assignment checker |
| Science | SciQ multiple choice | Exact option matching |

모든 record는 source and domain metadata를 유지한다. Training prompts를 12 evaluation benchmark와 대조한 audit에서는 exact or normalized match가 없고, 13-gram Jaccard similarity 0.5를 넘는 pair가 없다고 보고한다.

## 4-2. Common GRPO recipe

주요 shared setting은 다음과 같다.

| Item | Setting |
| --- | --- |
| Rollouts per prompt | 5 |
| Prompt length limit | 1,024 tokens |
| Response length limit | 8,192 tokens |
| KL coefficient | $10^{-3}$ |
| Default optimizer steps | 300 |
| Checkpoint interval | 100 steps |
| Primary base model | Qwen2.5-7B-Base |
| Primary seeds | 12 matched seeds |

모든 method를 이 recipe 안에서 비교하고, selection and reweighting은 uniform sampling과, adaptive mixture는 fixed equal mixture와 비교한다.

## 4-3. Evaluation suite

12 benchmarks는 세 domain으로 나뉜다.

### Math

- MATH-500
- AIME-2024
- OlympiadBench
- Minerva Math
- GSM8K

### Logic

- Knights and Knaves
- BBH Logical Deduction
- BBH Object Tracking
- ZebraLogic

### Science

- MMLU-Pro Chemistry
- MMLU-Pro Physics
- GPQA-Diamond

같은 training run을 broad suite로 평가하기 때문에 policy가 한 domain에서 얻은 이득과 다른 domain에서 잃은 성능을 분리할 수 있다.

## 4-4. Engineering notes

### 1) Data policy log를 first-class artifact로 남겨야 한다

Final checkpoint만 저장하면 왜 성능이 달라졌는지 알기 어렵다. Step별 kept fraction, effective token count, weight entropy, domain proportion, reward distribution을 같이 저장해야 한다.

### 2) Paired seed를 반드시 유지해야 한다

Method and baseline이 다른 seed set을 사용하면 policy effect에 seed effect가 섞인다. Same initialization and sampling seed를 가능한 범위에서 맞추고 paired difference를 보고하는 편이 좋다.

### 3) Selection rate와 rollout cost를 분리해야 한다

Post-rollout filtering은 update cost를 줄여도 rollout cost를 이미 지불했다. Efficiency claim에는 generated tokens, retained tokens, optimizer tokens를 따로 보고해야 한다.

### 4) Aggregation rule을 사전에 고정해야 한다

Best method를 확인한 뒤 favorable benchmark subset을 고르면 conclusion이 쉽게 바뀐다. Domain-balanced score와 per-domain table을 pre-register하는 것이 안전하다.

### 5) Confidence interval과 practical equivalence를 함께 봐야 한다

CI가 zero를 포함한다고 method가 완전히 동일하다는 뜻은 아니다. 반대로 mean이 높다고 reproducible gain도 아니다. 최소 meaningful effect size를 정하고 equivalence test or power analysis를 추가하는 것이 좋다.

# 5. Evaluation

## 5-1. Uniform GRPO의 training headroom

먼저 shared GRPO recipe 자체가 base checkpoint를 개선하는지 확인한다.

| Base model | Untrained | Uniform GRPO | Delta | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B-Base | 42.01 | 49.77 | +7.76 | [7.28, 8.25] |
| Llama-3.1-8B-Base | 10.87 | 21.12 | +10.25 | [7.75, 12.33] |

이 결과는 experiment가 ceiling-saturated setting이 아님을 보여준다. Common RLVR training은 분명한 improvement를 만든다. 논문의 핵심 negative result는 RLVR이 작동하지 않는다는 뜻이 아니라, 그 위에 추가한 data policy의 incremental gain이 reproducible하지 않았다는 뜻이다.

## 5-2. Primary policy comparison

Qwen2.5-7B-Base에서 13 configuration을 12 matched seeds로 평가한 결과는 다음처럼 요약된다.

- Eight rollout-selection or reweighting methods 중 uniform sampling 대비 paired 95% CI가 zero를 제외한 방법은 없다.
- Three adaptive mixtures 중 fixed equal mixture를 같은 precision level에서 이긴 방법은 없다.
- Llama-3.1-8B-Base 12-seed extension에서도 observed mean의 consistent winner가 나타나지 않는다.

중요한 점은 method가 training process를 바꾸지 못했다는 것이 아니다. Selection ratio, response length, domain allocation, update volume은 달라진다. 하지만 이런 process change가 broad final accuracy의 안정적인 improvement로 이어지지는 않았다.

## 5-3. Benchmark composition sensitivity

Qwen2.5-7B-Instruct의 같은 9 runs를 두 summary로 ranking한다.

- DB-12: Math, logic, science를 같은 비중으로 둔 12-benchmark summary
- Math-Heavy-6: Five math benchmarks plus GPQA-Diamond, logic benchmark 없음

두 ranking의 correlation은 다음과 같다.

$$
\rho=-0.33
$$

즉 math-heavy summary에서 좋아 보이는 policy가 domain-balanced summary에서는 뒤로 갈 수 있다. 반면 전체 12 benchmarks를 유지하고 weighting만 바꾼 summary는 대체로 비슷한 ranking을 보인다.

이 결과는 작은 weighting 차이보다 domain omission이 더 큰 문제일 수 있음을 보여준다.

## 5-4. 무엇을 negative result로 읽어야 하는가

논문이 지지하는 결론은 제한적이다.

> 이 paper에서 테스트한 model, data, verifier, horizon, policy configuration, evaluation protocol에서는 uniform training을 재현 가능하게 이긴 data policy가 없었다.

다음과 같은 더 강한 문장은 지지하지 않는다.

- RLVR data policy는 항상 쓸모없다.
- Hard example mining은 절대 작동하지 않는다.
- Adaptive curriculum은 모든 scale에서 실패한다.
- Accuracy가 같으므로 response efficiency나 training stability도 같다.

# 6. Limitations

1. Tested policy space가 전체 data policy를 대표하지 않는다.
   - Platform은 broad family를 다루지만 signal, actuator, hyperparameter 조합은 훨씬 더 많다.

2. Training horizon이 비교적 짧다.
   - Default 300 optimizer steps에서 early dynamic은 보지만 long-horizon advantage가 누적되는지는 제한적으로만 확인한다.

3. Reward가 clean and verifiable한 domain 중심이다.
   - Code execution, agent trajectory, learned verifier, noisy reward에서는 policy utility가 달라질 수 있다.

4. Final objective가 domain-balanced accuracy 중심이다.
   - Response length, sample efficiency, wall-clock, token cost, robustness를 포함한 Pareto comparison은 더 필요하다.

5. Fixed rollout count 이후의 post-rollout policy가 많다.
   - 이미 생성한 response를 버리는 방식은 generation compute를 절약하지 못한다.

6. Null result와 equivalence는 다르다.
   - CI가 zero를 포함한다는 사실만으로 method가 practical하게 동일하다고 단정할 수 없다.

7. Model family and scale coverage가 제한된다.
   - Qwen2.5 and Llama base model extension은 유용하지만 frontier reasoning model or much larger model로 일반화하려면 추가 실험이 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

DataFlex-RL의 가장 큰 기여는 어떤 data policy가 이겼는지가 아니라, data policy 연구에서 무엇을 고정하고 무엇을 기록해야 하는지 명시한 데 있다.

RLVR에서는 0.5 to 1 point 차이가 의미 있어 보이지만 seed, benchmark composition, response format만 바뀌어도 ranking이 뒤집힐 수 있다. 따라서 새 policy를 제안할 때 single best run보다 paired multi-seed result, process metric, per-domain trade-off를 함께 보여주는 것이 더 중요하다.

## 7-2. Reuse potential

### 1) IF-RL data policy evaluation

Instruction-following reward에서도 constraint count, violation type, response length를 scorer로 만들고 selection and reweighting을 같은 harness에서 비교할 수 있다.

### 2) Document reasoning RLVR

Document difficulty, evidence count, OCR quality, citation coverage를 signal로 사용하되, document domain별 performance를 balanced aggregation으로 평가할 수 있다.

### 3) Multi-domain curriculum

Math, code, search, tool use를 섞을 때 overall score 하나보다 domain reward gap과 forgetting curve를 함께 기록하는 curriculum platform으로 확장할 수 있다.

### 4) Policy diagnostic dashboard

Final benchmark뿐 아니라 kept fraction, weight entropy, update tokens, average response length, reward variance를 W&B dashboard에 고정하면 failure localization이 쉬워진다.

### 5) Pre-registered evaluation contract

Training 전에 seed count, primary score, domain weights, minimal effect size를 고정해 favorable subset selection을 줄일 수 있다.

## 7-3. Follow-up papers

- DeepSeekMath
- DeepSeek-R1
- DAPO
- GFPO
- PODS
- DataFlex
- Interleaved Online Fine-Tuning for Hardest Questions

# 8. Summary

- DataFlex-RL은 RLVR data policy를 selection, reweighting, mixture adaptation으로 통일한다.
- Common GRPO recipe와 matched seeds를 사용해 method effect를 training stack에서 분리한다.
- Uniform GRPO는 base checkpoint를 크게 개선하지만, tested data policy 중 reproducible winner는 없었다.
- Logic domain을 뺀 math-heavy summary는 domain-balanced ranking과 $\rho=-0.33$을 보여 benchmark composition 위험을 드러낸다.
- 핵심 가치는 negative result보다 future data policy를 공정하게 비교할 reusable platform과 evaluation contract에 있다.
