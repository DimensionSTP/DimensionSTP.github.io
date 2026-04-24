---
layout: single
title: "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe Review"
categories: Study-concept
tag: [LLM, Distillation, PostTraining, Reasoning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.13016)

[Code link](https://github.com/thunlp/OPD)

Rethinking OPD는 OPD를 "teacher가 더 세면 student도 더 좋아진다" 정도로 읽으면 핵심을 놓치기 쉬운 논문이다. 이 논문이 진짜 흥미로운 이유는, 최근 여러 reasoning post-training pipeline에서 점점 중요해진 OPD를 단순한 recipe가 아니라 "언제 되고 왜 안 되는가" 의 문제로 다시 풀어낸다는 데 있다. Dense token-level supervision은 얼핏 보면 RL보다 싸고 촘촘한 free lunch처럼 보이지만, 이 논문은 그 free lunch에도 구조적 조건과 길이 한계가 있다는 점을 보여준다.

이 논문의 진짜 메시지는 **teacher의 benchmark score** 보다 **student와 teacher가 공유하는 thinking pattern** 과 **teacher가 정말 새로운 지식을 주는가** 다. 그래서 이 논문은 distillation paper라기보다 reasoning post-training dynamics paper로 읽는 편이 더 맞다.

> 한 줄 요약: 이 논문은 OPD의 성패를 가르는 조건을 **thinking-pattern consistency** 와 **new knowledge beyond the student's seen data** 로 정리하고, 성공한 OPD는 high-probability overlap token 정렬로 설명되며, 실패한 OPD는 off-policy cold start 와 teacher-aligned prompt selection 으로 일부 복구할 수 있다고 주장한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Qwen3, MiMo, GLM-5 같은 최근 post-training 흐름에서 OPD가 왜 매력적인지, 그리고 왜 자주 불안정한지 를 같이 설명한다.
- teacher selection을 "누가 더 높은 점수를 받았는가" 가 아니라 "누가 student에게 배울 만한 support를 제공하는가" 로 다시 본다.
- overlap ratio, overlap-token advantage, entropy gap 같은 동적 지표를 통해 token-level distillation dynamics를 구조적으로 해석한다.
- off-policy cold start, teacher-aligned prompts, sampled-token reward의 실용적 의미까지 이어져서 실제 recipe 설계에 참고하기 좋다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 문제는 **on-policy distillation이 왜 어떤 teacher에서는 잘 되고, 어떤 teacher에서는 강하게 실패하는가** 이다.
- OPD에서는 student가 먼저 자신의 rollout을 생성하고, teacher는 student가 실제로 방문한 prefix 위에서 per-token log-probability를 제공한다.
- 이 구조는 off-policy distillation보다 exposure bias를 줄일 수 있고, RL보다 더 dense한 supervision을 줄 수 있다는 점에서 매력적이다.
- 하지만 실제로는 더 강한 teacher가 항상 더 좋은 distillation 결과를 주지 않는다. 심지어 benchmark score가 더 높은 teacher가 student를 거의 개선하지 못하거나, 오히려 student를 teacher 이전 수준으로 되돌리는 경우도 있다.
- 결국 이 논문이 묻는 질문은 "dense reward가 많으면 왜 학습이 더 잘 되는가" 가 아니다.
- 핵심은 **dense reward가 언제 실제 gradient signal로 exploitable한가** 에 가깝다.

## 1-2. Why previous approaches are insufficient

- 기존 OPD 서사는 대체로 teacher가 stronger하면 distillation도 더 잘 될 것이라는 직관에 기대고 있었다.
- 하지만 이 논문은 benchmark score가 높아도 student가 teacher의 high-probability support 근처에 있지 않으면 token-level reward를 제대로 활용하지 못한다고 본다.
- 또 기존 distillation 비교는 보통 최종 score에 집중하고, training 중 student와 teacher 분포가 어떻게 가까워지는지 를 거의 보지 않는다.
- long reasoning 쪽에서는 dense token-level reward가 outcome reward보다 좋아 보이지만, prefix가 길어질수록 teacher가 점점 unfamiliar state를 보게 되는 문제도 충분히 점검되지 않았다.
- 그래서 기존 접근의 한계는 objective 하나보다도, **teacher-student compatibility**, **state distribution mismatch**, **trajectory depth**, **prompt alignment** 를 한 프레임에서 보지 않았다는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- 첫째, 이 논문은 OPD 성패를 가르는 조건을 두 가지로 정리한다.
  1. **thinking-pattern consistency**
  2. **higher scores != new knowledge**
- 둘째, 성공한 OPD의 token-level 메커니즘을 **progressive alignment on high-probability overlap tokens** 로 설명한다.
- 셋째, overlap ratio, overlap-token advantage, entropy gap을 동적 지표로 두고 성공한 run과 실패한 run을 비교한다.
- 넷째, practical recipe로 **off-policy cold start** 와 **teacher-aligned prompt selection** 을 제안한다.
- 다섯째, dense supervision의 한계도 같이 짚는다. moderate response length에서는 잘 작동하지만, 길이가 너무 길어지면 teacher reward quality 자체가 무너질 수 있다는 점을 보여준다.

## 2-2. Design intuition

OPD의 핵심 직관은 reverse KL이 student가 이미 방문한 상태에서 teacher distribution 쪽으로 모드를 좁혀 가는 과정이라는 데 있다.

$$
L_{OPD}(\theta) = \mathbb{E}_{x \sim D_x} \left[ D_{KL}(\pi_{\theta}(\cdot \mid x) \Vert \pi_T(\cdot \mid x)) \right]
$$

이 식만 보면 teacher가 강할수록 좋을 것 같지만, 실제로는 student가 teacher의 high-probability token region 근처에 어느 정도 살아 있어야 gradient가 잘 작동한다. 이 논문은 그 근사치를 overlap ratio와 entropy gap 같은 지표로 본다.

또 하나 중요한 직관은 teacher의 역할이다. teacher는 단순히 점수가 높은 모델이면 안 된다.

student가 이미 본 데이터와 같은 분포 위에서 scale만 큰 teacher라면, student 입장에서는 배울 만한 "new knowledge" 가 거의 없을 수 있다.

그래서 이 논문은 **strong teacher** 보다 **transferable teacher** 가 중요하다고 본다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | OPD의 성패 조건, token-level mechanism, practical recipe를 체계적으로 분석 |
| Key module | overlap ratio, overlap-token advantage, entropy gap, overlap-only ablation, cold start, teacher-aligned prompt selection |
| Core claim | stronger teacher가 아니라 compatible teacher와 new knowledge가 중요함 |
| Difference from prior work | OPD를 단순 recipe가 아니라 dynamic system으로 보고, 성공과 실패를 같은 metric으로 비교 |
| Practical output | cold start, prompt alignment, support size 선택, response length sweet spot에 대한 recipe 제시 |

## 3-2. Module breakdown

### 1) OPD formulation

논문은 OPD를 크게 3가지 granularities로 정리한다.

- sampled-token OPD
- full-vocabulary OPD
- top-k OPD

이 중에서 주요 분석 축은 student top-k OPD와 sampled-token OPD다. 학생이 실제로 높게 두는 token support를 중심으로 teacher와 비교하겠다는 설정이다. 이건 실무적으로도 자연스럽다. 전체 vocabulary KL은 가장 dense하지만 비용이 크고, sampled-token은 가장 싸지만 정보가 희박해 보일 수 있기 때문이다.

### 2) Dynamic metrics

이 논문이 좋은 이유 중 하나는 training dynamics를 볼 지표를 명확하게 둔다는 점이다.

- **overlap ratio**: student와 teacher의 top-k token 집합이 얼마나 겹치는가
- **overlap-token advantage**: 겹치는 token 내부에서 probability calibration이 얼마나 맞는가
- **entropy gap**: 같은 student-visited state에서 teacher와 student의 uncertainty profile이 얼마나 다른가

개념적으로 overlap ratio는 아래처럼 이해하면 된다.

$$
\text{OverlapRatio} = \frac{|TopK_S \cap TopK_T|}{K}
$$

이 지표가 높아진다는 것은 student가 teacher의 high-probability support를 점점 더 잘 찾아간다는 뜻이다.

### 3) Phenomenology experiments

논문은 Qwen family와 DeepSeek family에서 controlled comparison을 만든다.

- Qwen3-1.7B student vs Qwen3-4B non-thinking teacher, Qwen3-4B-Base-GRPO teacher
- DeepSeek-R1-Distill-1.5B student vs R1-Distill-7B, Skywork-OR1-Math-7B
- reverse distillation으로 JustRL-1.5B student vs R1-Distill-1.5B, R1-Distill-7B

이 구성은 teacher strength, post-RL capability, same-family scale difference, thinking-pattern consistency를 분리해서 보기에 좋다. 핵심은 **teacher가 더 세다** 와 **teacher가 OPD로 transferable하다** 가 다르다는 점이다.

### 4) Overlap-only ablation

논문은 단순히 overlap이 같이 늘어난다고 끝내지 않는다. 실제 optimization support를 overlap token으로만 제한하는 ablation을 수행한다.

- Student Top-k
- Overlap Top-k
- Non-Overlap Top-k

결과적으로 overlap-only optimization이 full Student Top-k와 거의 비슷하게 작동하고, non-overlap only는 훨씬 약하다. 즉 OPD의 gradient signal은 겹치는 high-probability region에 거의 몰려 있다는 해석이 가능해진다.

### 5) Practical recovery recipe

논문이 제안하는 복구 전략은 두 가지다.

1. **off-policy cold start**
- teacher rollout으로 student를 먼저 SFT해서 initial pattern gap을 줄인다.
- 그 다음 standard OPD를 이어서 수행한다.

2. **teacher-aligned prompt selection**
- teacher post-training data와 더 가까운 prompt template, prompt content를 사용한다.
- student가 teacher가 익숙한 state 근처를 더 많이 방문하게 만든다.

이 두 전략은 모두 본질적으로 **student가 teacher의 useful support에 빨리 들어가도록 만드는 장치** 라고 볼 수 있다.

# 4. Training / Data / Recipe

## 4-1. Data

- 기본 OPD 실험의 중심 데이터는 **DAPO-Math-17K** 다.
- Qwen3-4B-Base-GRPO teacher 학습도 processed DAPO-Math-17K를 사용한다.
- cold start 실험에서는 **OpenThoughts3-1.2M의 math subset** 에서 200K prompts를 뽑아 teacher rollout SFT data를 만든다.
- prompt content alignment 실험에서는 **DAPO-Math-17K** 와 **DAPO-Math-17K에 dedup된 DeepMath subset** 을 비교한다.
- DeepMath dedup은 exact-match와 semantic dedup 두 단계로 수행하고, semantic dedup에서는 all-mpnet-base-v2와 FAISS를 사용해 cosine similarity 0.6 이상이면 duplicate로 제거한다.

## 4-2. Training strategy

Qwen3-4B-Base-GRPO teacher의 핵심 설정은 다음과 같다.

| Item | Value |
| --- | --- |
| Base model | Qwen3-4B-Base |
| RL algorithm | GRPO |
| Training epochs | 1 |
| Rollout n | 8 |
| Max prompt length | 1024 |
| Max response length | 7168 |
| Validation max response length | 31744 |
| Learning rate | 1e-6 |
| Temperature | 1.0 |
| Top-p | 1.0 |
| KL regularization | 0.0 |

기본 OPD 설정은 아래와 같다.

| Item | Value |
| --- | --- |
| Training temperature | 1.0 |
| Global batch size | 64 |
| Mini batch size | 64 |
| Rollout number | 4 |
| LogProb top-k | 16 |
| Top-k strategy | Student Top-k |
| Max prompt length | 1024 |
| Max response length | 7168 |
| Learning rate | 1e-6 |
| Epoch | 1 |
| KL Coefficient | 0.0 |

cold start SFT는 Qwen3-4B non-thinking teacher가 OpenThoughts3 math prompts 200K개에 대해 one rollout씩 생성한 데이터를 사용한다. generation은 temperature 0.7, top-p 0.95, max generation length 12288로 수행되고, incomplete response와 degenerate repetitive response를 필터링한 뒤 student full-parameter SFT에 사용한다.

## 4-3. Engineering notes

- 이 논문은 단순히 teacher/student 조합만 바꾸는 것이 아니라, **student가 어떤 prompt state를 방문하느냐** 를 중요한 engineering variable로 본다.
- evaluation은 temperature 0.7, top-p 0.95, max validation response length 31744, avg@16 기준으로 수행된다.
- prompt template alignment 실험은 같은 math 문제에 대해 template만 바꾸므로, prompt surface form이 distillation dynamics에 미치는 영향을 분리해서 보기 좋다.
- sampled-token OPD vs top-k OPD 비교도 practical하다. full vocabulary로 갈수록 이론상 dense해 보이지만, 실제로는 student top-k support 안에서 충분한 signal이 이미 나올 수 있다는 것이다.
- 여기서 가장 재사용성이 높은 포인트는, OPD를 시작하기 전에 **initial overlap ratio** 와 **entropy gap** 을 먼저 재보는 습관이다.

# 5. Evaluation

## 5-1. Main results

### 1) thinking-pattern consistency가 benchmark score보다 먼저 온다

Qwen family 비교에서 Qwen3-4B-Base-GRPO teacher는 Qwen3-4B non-thinking teacher와 benchmark score가 broadly comparable하거나 dataset별로 엇갈리지만, distillation 결과는 훨씬 좋다. 논문 해석은 명확하다. **더 좋은 early overlap** 이 downstream distillation을 좌우한다는 것이다.

### 2) higher scores != new knowledge

같은 family에서 scale만 키운 teacher는 항상 좋은 teacher가 아니다. DeepSeek family와 Qwen family 모두, same-pipeline teacher보다 post-RL teacher가 훨씬 강한 transfer를 보인다. 특히 Qwen family에서는 gap recovery rate가 Qwen3-4B-Non-Thinking-RL-Math teacher에서 58.6%로, same-pipeline Qwen3-4B teacher의 15.6%보다 훨씬 높다.

여기서 중요한 해석은 simple scale-up이 아니라 **teacher가 student가 아직 안 본 capability를 갖고 있는가** 다.

### 3) reverse distillation이 이 논문의 핵심 반례다

JustRL-1.5B를 student로 두고 R1-Distill-1.5B, R1-Distill-7B를 teacher로 reverse distillation하면 두 경우 모두 student가 거의 pre-RL 수준으로 regression한다. 특히 7B teacher는 benchmark score가 더 높음에도 1.5B teacher와 거의 같은 regression trajectory를 만든다.

이 결과는 OPD가 단순히 stronger policy imitation이 아니라 **thinking pattern acquisition** 이라는 논문 주장에 힘을 실어준다.

### 4) successful OPD의 signature는 overlap alignment다

성공한 run에서는 overlap ratio가 72%에서 91%까지 올라가고, shared top-k token이 student와 teacher 분포 전체 probability mass의 97%-99%를 차지한다. 반면 실패한 run에서는 overlap, entropy gap, overlap-token advantage가 모두 초반부터 정체된다.

이 부분이 이 논문의 가장 중요한 메커니즘 결과다. OPD는 teacher 전체 distribution을 고르게 배우는 것이 아니라, **student가 이미 어느 정도 올라와 있는 shared high-probability region** 에서 점진적으로 정렬된다.

### 5) overlap-only optimization이 거의 충분하다

Student Top-k와 Overlap Top-k를 비교하면 overlap-only optimization이 거의 full Student Top-k와 비슷하게 작동한다. 반면 Non-Overlap Top-k는 뚜렷하게 약하다.

즉 이 논문은 "overlap이 중요해 보인다" 가 아니라 "실제로 gradients가 거의 거기서 나온다" 를 ablation으로 보여준다.

### 6) cold start와 teacher-aligned prompts는 실제로 복구에 먹힌다

- cold start에서는 OpenThoughts3 math subset에서 teacher rollout 200K개로 student를 먼저 SFT한 뒤 OPD를 이어 가면 pure OPD보다 consistently better validation trajectory를 보인다.
- prompt template alignment는 same question set에서 template만 teacher post-training style로 바꿔도 세 benchmark 모두 성능을 올린다.
- appendix 기준으로 prompt-template alignment는 student가 recover하는 teacher performance fraction을 대략 80%에서 85% 수준으로 높인다.
- prompt content alignment는 더 미묘하다. overlap ratio 자체는 낮아질 수 있지만, overlap tokens 위에 student probability mass가 더 강하게 집중되면서 downstream 성능이 좋아진다.

이건 상당히 실무적이다. OPD는 model pair만 맞추면 되는 게 아니라, **teacher가 익숙한 prompt manifold 근처로 student rollout을 유도해야** 한다.

## 5-2. What really matters in the experiments

### 1) dense reward에도 horizon limit가 있다

응답 길이를 0.5K, 1K, 3K, 7K, 10K, 15K로 바꾸면 3K와 7K가 가장 안정적이고, 10K와 15K는 plateau 또는 decline을 보인다. 너무 짧으면 supervised tokens가 부족하고, 너무 길면 teacher reward quality가 무너지기 시작한다.

### 2) instability는 suffix에서 먼저 시작된다

15K setting에서는 high entropy가 응답 끝쪽에서 먼저 생기고, training이 진행될수록 prefix 쪽으로 역전파된다. teacher entropy도 같은 suffix-to-prefix 패턴을 보인다. 즉 long horizon OPD collapse는 뒤에서 먼저 시작되는 경향이 있다.

### 3) teacher continuation gain도 prefix depth에 따라 무너진다

student rollout을 잘라서 teacher가 이어 쓰게 해 보면, teacher의 accuracy advantage는 1K prefix에서 +0.37이지만 16K prefix에서는 +0.02까지 떨어진다. 이 수치는 OPD의 dense supervision이 길이에 따라 얼마나 빨리 신뢰도를 잃는지 직관적으로 보여준다.

### 4) failing teacher도 global reward는 informative할 수 있다

JustRL-1.5B teacher와 failing R1-Distill-7B teacher 모두 correct rollout에 higher sequence mean reward를 준다. AUROC도 각각 0.73, 0.75로 비슷하다. 즉 failing teacher가 reward quality 자체가 나쁜 것은 아니다.

논문은 이 차이를 **global signal vs local exploitability** 문제로 본다. reward는 globally informative하지만, per-token gradient가 locally exploitable한 방향으로 모이지 않을 수 있다는 것이다.

### 5) sampled-token reward는 생각보다 충분하다

sampled-token OPD는 top-k OPD와 평균적으로 비슷한 성능을 낸다. Top-1만 명확하게 나쁘고, k를 4보다 크게 키워도 이득은 거의 없다. 논문 해석은 간단하다. sampled-token은 student distribution에서 unbiased하게 high-probability region을 샘플링하고, Top-1은 argmax mode에만 reward를 몰아 불안정하게 만든다.

이 결과는 practical하다. **support size를 무작정 키우는 것이 핵심이 아니다** 라는 뜻이기 때문이다.

# 6. Limitations

1. 논문이 제시한 **local optimization geometry** 가설은 아직 직접 검증된 사실이 아니다. high per-token advantage와 low gradient norm의 공존을 보고 가설을 세운 수준이므로, directional gradient structure 분석은 후속 검증이 필요하다.

2. main study는 math reasoning 중심이다. AIME 2024, AIME 2025, AMC 2023, DAPO-Math-17K 같은 설정이 중심이기 때문에, agentic multi-turn, tool use, code agent에서 같은 dynamics가 그대로 유지되는지는 아직 열려 있다.

3. teacher-aligned prompt selection은 분명히 효과가 있지만, student entropy를 과하게 낮출 수 있다. teacher post-training prompt만 너무 많이 쓰면 exploration capacity가 빠르게 줄어들 가능성이 있다.

4. OPD default setting이 epoch 1, max response length 7168, top-k 16, KL coefficient 0.0 같은 구체적 recipe에 많이 묶여 있다. 다른 model family나 다른 optimizer, 다른 length regime에서 같은 해석이 얼마나 유지되는지는 추가 확인이 필요하다.

5. long horizon limitation은 꽤 본질적이다. 논문 자체가 dense reward의 free lunch가 trajectory depth가 깊어질수록 무너진다고 보여주기 때문에, extended chain-of-thought나 multi-turn agent setting으로 그대로 scale된다고 가정하면 위험하다.

6. 추가로 주의할 점은, overlap ratio 같은 지표가 descriptive signal로는 유용하지만 production recipe에서 언제 cold start를 걸고 언제 teacher를 바꿀지 를 자동 결정하는 threshold까지 제시하지는 않는다는 점이다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 중요한 이유는 OPD를 "RL보다 싸고 dense하다" 는 한 줄 설명에서 꺼내, 실제 post-training system design 문제로 바꿔 보기 때문이다. 최근 reasoning pipeline에서 OPD를 자주 쓰는 이유는 분명하지만, 왜 어떤 teacher에서는 잘 되고 어떤 teacher에서는 안 되는지 설명이 부족했다. 이 논문은 그 빈칸을 꽤 설득력 있게 메운다.

가장 실무적인 교훈은 teacher selection criteria가 바뀐다는 점이다. 앞으로 OPD teacher를 고를 때는 benchmark rank보다 아래 질문이 더 중요해질 가능성이 크다.

- student rollout에서 teacher와 initial overlap이 충분한가
- teacher가 student가 이미 본 data recipe 밖의 new knowledge를 주는가
- teacher post-training prompt와 student prompt interface가 얼마나 aligned되어 있는가
- distillation horizon이 teacher continuation reliability를 넘지 않는가

## 7-2. Reuse potential

- OPD를 돌리기 전에 small validation set에서 **overlap ratio**, **entropy gap**, **teacher continuation gain** 을 먼저 진단할 수 있다.
- initial overlap이 낮다면 pure OPD부터 시작하지 말고 **off-policy cold start** 를 먼저 거는 전략을 고려할 수 있다.
- teacher가 학습한 prompt template과 content manifold를 어느 정도 맞춰주는 **prompt alignment** 가 생각보다 큰 차이를 만들 수 있다.
- very long response에 dense token reward를 그대로 넣기보다, 3K-7K 수준의 **moderate horizon distillation** 이 더 안정적일 수 있다.
- reward support를 무조건 넓히기보다, sampled-token이나 small top-k처럼 비용 대비 효율이 좋은 설정을 먼저 보는 것이 낫다.
- 무엇보다, benchmark score가 높은 teacher를 그대로 꽂는 자동화된 teacher routing은 꽤 위험할 수 있다. OPD teacher는 **score** 보다 **transfer geometry** 로 골라야 한다.

## 7-3. Follow-up papers

- Qwen3 Technical Report
- Learning Beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation
- On-Policy Context Distillation for Language Models
- Stable On-Policy Distillation Through Adaptive Target Reformulation
- Entropy-Aware On-Policy Distillation of Language Models
- Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

# 8. Summary

- 이 논문은 OPD를 stronger teacher imitation 문제가 아니라, **thinking-pattern compatibility** 와 **new knowledge transferability** 문제로 다시 본다.
- 성공한 OPD는 student-visited state에서 high-probability overlap token alignment가 점진적으로 커지는 동학을 보인다.
- shared top-k token은 probability mass의 97%-99%를 차지하고, overlap-only optimization도 full top-k와 거의 비슷하게 작동한다.
- practical recipe로는 **off-policy cold start** 와 **teacher-aligned prompt selection** 이 유효하다.
- 다만 dense token-level reward도 horizon limit가 있으며, teacher continuation gain은 prefix depth가 깊어질수록 급격히 줄어든다.
