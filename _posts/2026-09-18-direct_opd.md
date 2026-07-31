---
layout: single
title: "Weak-to-Strong Generalization via Direct On-Policy Distillation Review"
categories: Study-concept
tag: [Weak-to-Strong, On-Policy-Distillation, RLVR]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2607.05394)

> 한 줄 요약: Direct-OPD는 작은 model의 post-RL policy 자체를 모방하지 않고, post-RL checkpoint와 pre-RL reference 사이의 log-probability ratio를 implicit token reward로 읽어 strong student의 own on-policy state에 적용함으로써 weak-to-strong capability transfer를 수행한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- 작은 teacher가 더 강한 student를 어떻게 개선할 수 있는지 distillation target을 다시 정의한다.
- Expensive target-model RL 대신 cheap small-model RL에서 발견한 policy improvement direction을 재사용한다.
- Teacher endpoint imitation이 아니라 pre-RL -> post-RL policy shift를 transfer한다.
- 서로 다른 teacher pair의 shift를 sequentially compose해 improvement를 누적할 수 있음을 보인다.

Weak-to-strong distillation에는 직관적인 모순이 있다. Student가 teacher보다 이미 강하다면 teacher의 final distribution을 따라가는 것이 student를 끌어내릴 수 있다. 논문의 예에서도 R1-Distill-7B는 post-RL 1.5B teacher보다 높은 초기 성능을 가지지만, standard OPD는 teacher endpoint 쪽으로 student를 당겨 성능을 낮춘다.

Direct-OPD는 질문을 바꾼다. "작은 teacher가 무엇을 알고 있는가"가 아니라 "작은 model이 RL 전후에 어떤 action을 더 선호하게 되었는가"를 본다. RL이 발견한 direction은 endpoint model의 absolute capability보다 더 transferable할 수 있다는 가정이다.

# 1. Problem Setting

## 1-1. Problem definition

Teacher pair는 같은 model family의 두 checkpoint로 구성된다.

- Reference teacher $\pi_{T_{ref}}$: RL 이전 checkpoint
- Post-RL teacher $\pi_T$: RL 이후 checkpoint

Student $\pi_S$는 teacher보다 크거나 다른 family일 수 있다. Direct-OPD는 student가 sample한 token $y_t$에서 teacher pair의 log-ratio를 계산한다.

$$
\Delta_T(y_t\mid x,y_{<t}) = \log \pi_T(y_t\mid x,y_{<t}) - \log \pi_{T_{ref}}(y_t\mid x,y_{<t})
$$

이 값이 positive면 small-model RL이 해당 token을 reference보다 더 선호하게 만들었다는 뜻이다. Negative면 RL 이후 덜 선호하게 된 action이다.

중요한 점은 trajectory가 teacher가 아니라 student에서 나온다는 것이다. Teacher shift는 strong student가 실제로 방문하는 prefix에서 평가된다.

## 1-2. Why previous approaches are insufficient

### 1) Final teacher imitation

Standard distillation은 $\pi_S$를 $\pi_T$에 맞춘다. Teacher가 student보다 약하면 useful RL behavior와 teacher의 capacity limitation을 함께 복사한다.

### 2) Direct large-model RL

Target model에서 RL을 직접 수행하면 rollout과 training cost가 model size에 비례해 커진다. Sparse verifier reward로 improvement direction을 다시 발견해야 하므로 exploration cost도 반복된다.

### 3) Weight-space transfer

Model merge나 task arithmetic은 checkpoint parameter delta를 옮긴다. 서로 다른 architecture, tokenizer, width를 가진 teacher와 student 사이에는 직접 적용하기 어렵다.

### 4) Static reward model

Small-model RL trajectory로 별도 reward model을 학습할 수 있지만, reward approximation과 over-optimization 문제가 추가된다. Direct-OPD는 teacher/reference ratio 자체를 dense implicit reward로 사용한다.

# 2. Core Idea

## 2-1. Main contribution

Direct-OPD의 transfer object는 다음 세 가지가 아니다.

- Teacher의 generated answer
- Teacher의 absolute next-token distribution
- Teacher checkpoint의 weight delta

Transfer object는 RL이 만든 behavioral shift $\Delta_T$다.

KL-regularized RL의 optimal policy relation을 뒤집어 보면, post-RL policy와 reference의 log-ratio는 reward에 비례하는 signal로 볼 수 있다. Direct-OPD는 이 signal을 strong student의 dense token reward로 재사용한다.

## 2-2. Design intuition

Student objective는 teacher shift reward를 높이면서 initial student에서 너무 멀리 벗어나지 않도록 한다.

$$
J_{\mathrm{Direct-OPD}}(\theta) = \mathbb{E}_{x,y\sim\pi_\theta}\left[\Delta_T(y\mid x)\right] - \alpha D_{\mathrm{KL}}\left(\pi_\theta\Vert\pi_S\right)
$$

이 idealized objective의 optimal policy는 다음 형태다.

$$
\pi^*(y\mid x) \propto \pi_S(y\mid x)\left(\frac{\pi_T(y\mid x)}{\pi_{T_{ref}}(y\mid x)}\right)^{1/\alpha}
$$

즉 student의 원래 distribution을 base로 유지하면서, small teacher RL이 probability를 올린 action은 boost하고 내린 action은 suppress한다.

이 관점에서는 teacher endpoint가 student보다 약해도 문제가 되지 않는다. Teacher가 RL을 통해 발견한 local improvement direction만 유효하면 된다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Teacher input | Pre-RL reference와 post-RL checkpoint pair |
| Rollout source | Strong student의 current on-policy distribution |
| Dense signal | Teacher pair의 token log-probability difference |
| Policy anchor | Initial student에 대한 KL regularization |
| Action scope | Student top-k support |
| Estimator | Rao-Blackwellized analytical top-k policy gradient |
| Deployment | Updated strong student만 사용 |

## 3-2. Module breakdown

### 1) Student on-policy rollout

Student가 prompt에 대한 response를 직접 sample한다. Teacher가 만든 trajectory를 따라가지 않기 때문에, supervision은 student가 실제로 방문하는 state에 적용된다.

이 design은 teacher-student thinking pattern이 달라도 transfer를 가능하게 한다. Teacher와 student의 top-k token overlap이 낮아도, student candidate token 각각에 대해 teacher/reference ratio를 계산할 수 있다.

### 2) Teacher shift scoring

각 student token에 대해 post-RL teacher와 reference teacher의 log-probability를 계산한다. Raw teacher probability가 아니라 두 checkpoint의 차이를 사용하므로 teacher의 static knowledge와 style은 상당 부분 상쇄되고, RL-induced change가 남는다.

예를 들어 두 teacher가 모두 어떤 token에 낮은 probability를 주더라도 post-RL에서 relative probability가 높아졌다면 positive shift가 된다. 반대로 post-RL teacher가 절대적으로 높은 probability를 주더라도 reference와 차이가 없으면 transfer signal은 작다.

### 3) Student top-k action restriction

전체 vocabulary에 대해 policy gradient를 계산하면 비용이 크고, teacher가 student support 밖의 token에 큰 ratio를 주는 경우 noisy signal이 들어올 수 있다. 논문은 student의 top-k action support, main setting에서는 $k=16$, 안에서 analytical expectation을 계산한다.

이 방식은 sampled token 한 개만 사용하는 estimator보다 variance를 줄인다. Student가 실제로 선택할 가능성이 높은 후보들의 relative shift를 함께 비교한다.

### 4) Rao-Blackwellized update

Top-k candidate에 대해 teacher shift와 student probability를 직접 합산해 expectation을 계산한다. Sampling noise를 줄이고 dense token reward를 더 안정적으로 사용하기 위한 선택이다.

Teacher distribution을 그대로 match하는 KL distillation과 달리, 여기서는 candidate action의 reward ranking을 student policy gradient에 넣는다.

### 5) Adaptive KL

Teacher shift reward의 scale은 teacher pair마다 다르다. Fixed $\alpha$가 너무 작으면 student가 log-ratio를 exploit하며 teacher와 reference support 밖으로 drift할 수 있고, 너무 크면 transfer가 거의 일어나지 않는다.

논문은 reward statistic을 이용해 KL coefficient를 adaptive하게 조절한다. 목표는 dense reward를 무조건 크게 만드는 것이 아니라, teacher/reference comparison이 informative한 student distribution 안에 머무르게 하는 것이다.

# 4. Training / Data / Recipe

## 4-1. Data

Main experiment는 mathematical reasoning과 AIME 2024, AIME 2025 evaluation에 집중한다.

두 teacher pair를 사용한다.

1. R1-Distill-1.5B -> JustRL-1.5B
2. Nemotron-1.5B -> QuestA-Nemotron-1.5B

Student는 세 종류다.

- R1-Distill-7B
- Qwen3-1.7B
- Qwen3-4B

Teacher pair가 training pipeline, data source, base family에서 다르기 때문에 method가 하나의 checkpoint pair에만 맞는지 확인할 수 있다.

## 4-2. Training strategy

Direct-OPD transfer는 student rollout 위에서 약 300 update를 수행하는 compact stage로 구성된다. Main recipe는 다음 요소를 사용한다.

- Student top-k: 16
- Learning rate: $1\times10^{-6}$
- Global batch size: 64
- Rollout count per prompt: 4
- Main response length: 2048
- Adaptive 또는 pair-specific KL control

Evaluation은 AIME 문제당 32 sample을 생성해 avg@32를 계산한다. Temperature 0.7, top-p 0.95, 긴 generation budget을 사용한다.

## 4-3. Engineering notes

### 1) Teacher pair를 한 artifact로 관리한다

Post-RL checkpoint만 보관하면 shift를 재구성할 수 없다. Exact reference checkpoint, tokenizer, prompt template, logit precision을 함께 versioning해야 한다.

### 2) Student prefix를 teacher 두 개에 동일하게 넣는다

Teacher-generated prefix를 쓰면 다시 imitation distribution으로 돌아간다. 반드시 current student rollout token을 두 teacher에서 re-score해야 한다.

### 3) Ratio range를 monitor한다

Large positive 또는 negative $\Delta_T$가 많아지는 것은 student가 teacher support 밖으로 drift했다는 signal일 수 있다. Mean reward뿐 아니라 percentile, top-k overlap, KL, entropy를 같이 봐야 한다.

### 4) Response length는 transfer hyperparameter다

더 긴 response를 학습한다고 항상 좋아지지 않는다. 논문에서는 2K training horizon이 6K보다 validation이 좋고, short-horizon update도 16K 부근의 fixed rollout behavior를 바꾼다.

### 5) Compute accounting을 분리한다

Small-model RL cost, teacher re-scoring cost, student Direct-OPD cost를 따로 기록해야 direct large-model RL과 공정하게 비교할 수 있다.

# 5. Evaluation

## 5-1. Main results

### JustRL teacher shift

| Student | AIME24 before | AIME24 after | AIME25 before | AIME25 after |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 48.3 | 58.3 | 36.8 | 43.2 |
| Qwen3-4B | 72.5 | 77.6 | 65.6 | 68.8 |
| R1-Distill-7B | 56.7 | 63.1 | 40.5 | 48.8 |

Qwen3-4B와 R1-Distill-7B는 post-RL 1.5B teacher보다 이미 강한 starting point를 가진다. 그럼에도 성능이 오르므로 method가 teacher endpoint를 단순 imitation한 결과로 설명되기 어렵다.

### QuestA teacher shift

| Student | AIME24 before | AIME24 after | AIME25 before | AIME25 after |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 48.3 | 59.0 | 36.8 | 43.1 |
| R1-Distill-7B | 56.3 | 61.2 | 39.5 | 44.0 |

서로 다른 teacher pair에서도 gain이 유지된다.

## 5-2. What really matters in the experiments

### 1) Weak-to-strong route가 direct RL보다 compute-efficient하다

R1-Distill-1.5B를 1,500 steps RL하는 데 약 160 hours on 32 A100, R1-Distill-7B direct RL에는 약 320 hours가 걸린다. Small-model RL 이후 Direct-OPD는 약 4 hours on 8 A100을 추가한다.

Matched RL step 기준으로 small-model RL + transfer point가 direct 7B RL curve보다 높은 구간에 위치한다. 다만 hardware-hour를 단순 합산하면 parallelism과 utilization 차이가 있으므로 absolute cost comparison은 주의해야 한다.

### 2) Policy shift를 순차 합성할 수 있다

Qwen3-1.7B에 JustRL shift를 적용하면 AIME24 48.3 -> 58.3, AIME25 36.8 -> 43.2가 된다. 이어 QuestA shift를 적용하면 63.8, 46.8까지 오른다.

이는 서로 다른 RL run이 만든 improvement direction을 하나의 student에 단계적으로 쌓을 수 있다는 결과다. 하지만 composition order와 interference는 더 많은 teacher pair에서 검증할 필요가 있다.

### 3) Token overlap이 핵심 조건이 아니다

Raw OPD는 teacher와 student가 비슷한 high-probability token을 공유해야 안정적이다. Direct-OPD는 student support 안의 action을 teacher/reference ratio로 평가하므로, teacher가 동일한 reasoning trace를 생성할 필요가 없다.

### 4) Short training horizon이 long behavior를 바꾼다

2K response length로 40 steps 학습한 actor도 약 16K fixed rollout 전체에서 positive teacher-shift direction으로 움직인다. 그러나 6K setting은 shift magnitude가 더 커도 AIME validation은 45.6으로, 2K의 48.8보다 낮다. Dense shift reward를 크게 만드는 것과 task performance를 높이는 것은 같지 않다.

### 5) Teacher checkpoint choice가 중요하다

Small-model RL의 모든 checkpoint가 strong student에 같은 gain을 주지 않는다. RL step마다 발견한 direction이 다르고, student state distribution에서 의미 있는 shift인지도 다르다. Teacher selection은 endpoint score가 아니라 transfer validation으로 해야 한다.

# 6. Limitations

1. Signal validity가 student state에 조건부다.
   - Teacher/reference improvement가 student-visited prefix에서 의미가 없으면 transfer가 실패할 수 있다.

2. Main evidence가 math reasoning에 집중된다.
   - AIME 중심 결과가 coding, tool use, open-ended agent task에도 그대로 유지되는지 알 수 없다.

3. Teacher pair와 student pair마다 KL이 다르다.
   - Universal coefficient가 없고 response length도 pair-dependent하다.
   - Hyperparameter search cost가 target-model RL 절감분을 일부 상쇄할 수 있다.

4. Post-RL checkpoint가 좋은 causal reward를 보장하지 않는다.
   - RL run이 reward hacking이나 narrow benchmark shortcut을 배웠다면 log-ratio가 그 direction도 전달한다.

5. Teacher re-scoring 비용이 남는다.
   - 두 teacher forward와 student rollout을 함께 수행해야 한다.
   - Teacher가 커지거나 response가 길어지면 transfer stage도 비싸질 수 있다.

6. Sequential composition의 interference가 충분히 분석되지 않았다.
   - 두 shift가 잘 합쳐진 사례는 강하지만, 순서가 바뀌거나 contradictory objective가 들어올 때의 behavior는 열려 있다.

# 7. My Take

## 7-1. Why this matters for my work

Direct-OPD의 가장 흥미로운 관점은 RL checkpoint를 "완성된 model"이 아니라 "재사용 가능한 reward artifact"로 본다는 점이다.

작은 model에서 RL을 여러 번 돌려 domain-specific direction을 찾고, 각 reference/post-RL pair를 library로 보관한 뒤 stronger backbone에 선택적으로 transfer할 수 있다. 이 구조가 성립하면 expensive model마다 같은 verifier task를 처음부터 RL할 필요가 줄어든다.

## 7-2. Reuse potential

1. Small-model RL sandbox
   - 1B-3B model에서 reward와 data recipe를 탐색한다.
   - Stable checkpoint pair만 larger student로 transfer한다.

2. Multi-skill composition
   - Math, tool use, formatting처럼 다른 RL objective의 shift를 별도 pair로 관리한다.
   - Composition 순서와 interference를 regression benchmark로 검증한다.

3. Reward audit
   - Teacher/reference ratio가 어떤 token에 positive reward를 주는지 visualization한다.
   - Known shortcut token에 reward가 몰리면 transfer 전에 차단한다.

4. Model-family transfer
   - Weight merge가 불가능한 architecture 사이에서도 behavioral shift를 전달할 수 있다.

5. Cheap checkpoint screening
   - Endpoint teacher score가 아니라 small validation student에 transfer한 gain으로 teacher checkpoint를 고른다.

## 7-3. Follow-up papers

- On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- JustRL: Scaling a 1.5B LLM with a Simple RL Recipe
- QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation
- Distillation Scaling Laws

# 8. Summary

- Direct-OPD는 teacher endpoint가 아니라 pre-RL와 post-RL 사이의 policy shift를 transfer한다.
- Teacher log-ratio를 strong student의 own on-policy token에서 dense implicit reward로 사용한다.
- Student top-k support, Rao-Blackwellized estimator, adaptive KL로 stable update를 만든다.
- Teacher보다 강한 Qwen3-4B와 R1-Distill-7B도 개선하며, small-model RL + transfer가 matched-step direct RL보다 효율적이다.
- 핵심 한계는 shift validity가 teacher pair와 student state distribution에 조건부라는 점이다.
