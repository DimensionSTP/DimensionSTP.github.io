---
layout: single
title: "On the Geometry of On-Policy Distillation Review"
categories: Study-concept
tag: [LLM, Distillation, OPD, PostTraining, Geometry]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.07082)

On the Geometry of On-Policy Distillation은 OPD를 성능 recipe가 아니라 parameter-space geometry의 문제로 다시 읽는 논문이다. 지금까지 OPD는 reasoning model post-training에서 꽤 매력적인 방법으로 쓰였다. student가 직접 생성한 on-policy trajectory 위에서 teacher distribution을 따라가게 만들기 때문에, off-policy SFT보다 exposure mismatch가 작고, outcome RL보다 token-level signal이 dense하다는 장점이 있다.

하지만 이 장점만으로는 OPD가 왜 안정적으로 작동하는지 설명하기 어렵다. OPD는 SFT처럼 teacher trace를 복제하지도 않고, RLVR처럼 outcome reward를 통해 직접 정책을 밀지도 않는다. 그렇다면 OPD update는 parameter space에서 어디로 가는가. SFT와 RLVR 사이의 단순한 중간점인가, 아니면 완전히 다른 종류의 update channel을 만드는가.

이 논문은 바로 이 질문을 다룬다.

> 한 줄 요약: On the Geometry of OPD는 OPD update가 SFT와 RLVR 사이의 단순한 interpolation이 아니라, fewer weights를 건드리고 principal direction을 더 강하게 피하면서도 early training에서 형성된 low-dimensional channel에 빠르게 lock-in되는 독자적인 parameter-space geometry를 만든다고 주장하는 분석 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- 최근 reasoning post-training에서 OPD가 자주 등장하지만, 왜 작동하는지에 대한 geometric 해석은 아직 부족하다.
- 논문은 OPD, SFT, RLVR를 같은 parameter-space diagnostic으로 비교해 recipe가 아니라 **update dynamics** 를 본다.
- OPD의 핵심 현상을 **relaxed off-principal regime** 와 **subspace locking** 으로 정리해, 앞으로 OPD 변형을 설계할 때 봐야 할 축을 제시한다.
- token sparsification, off-policy rollout, RLVR mixing 같은 control experiment를 통해 무엇이 OPD geometry를 보존하고 무엇이 바꾸는지 분리한다.

제가 보기엔 이 논문의 핵심은 OPD를 성능 향상 방법으로만 보지 말자는 데 있다. OPD가 잘 된다는 말보다 더 중요한 것은, **OPD가 어떤 방향으로 업데이트되는지** 를 이해하는 것이다. 이 논문은 그 질문을 benchmark score가 아니라 parameter trajectory로 끌고 온다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 **OPD의 training dynamics가 무엇인지 아직 잘 모른다** 는 점이다.

OPD는 보통 다음 흐름으로 이해할 수 있다.

1. student policy가 prompt에 대해 rollout을 생성한다.
2. teacher model은 student가 실제로 방문한 prefix 위에서 token distribution을 제공한다.
3. student는 자기 trajectory 위에서 teacher distribution을 따라가도록 업데이트된다.

개념적으로 쓰면, student parameter를 $\theta$, teacher distribution을 $\pi_T$, student distribution을 $\pi_\theta$라고 할 때 OPD는 student가 방문한 state 위에서 KL 계열 loss를 줄이는 형태로 볼 수 있다.

$$
L_{OPD}(\theta) = E_{x, y_{<t} \sim \pi_\theta} [D_{KL}(\pi_\theta(\cdot | x, y_{<t}) || \pi_T(\cdot | x, y_{<t}))]
$$

이 식만 보면 OPD는 SFT와 RLVR의 중간처럼 보일 수 있다. SFT처럼 teacher signal을 쓰지만, SFT와 달리 student trajectory 위에서 학습한다. RLVR처럼 on-policy sampling을 쓰지만, RLVR와 달리 sparse outcome reward 대신 dense teacher distribution을 쓴다.

논문은 이 직관이 충분하지 않다고 본다. 핵심 질문은 다음과 같다.

- OPD update는 SFT update와 같은 principal direction을 따라가는가.
- OPD update는 RLVR처럼 constraint가 강한 sparse direction에 갇히는가.
- OPD update는 training 초기에 형성된 low-dimensional subspace에 lock-in되는가.
- OPD의 geometry는 token sparsification이나 off-policy rollout에도 유지되는가.
- OPD와 RLVR를 섞으면 같은 geometry가 유지되는가.

즉 이 논문은 최종 benchmark score보다 **parameter update trajectory 자체** 를 분석 대상으로 삼는다.

## 1-2. Why previous approaches are insufficient

기존 OPD 논의는 대체로 아래 세 가지 중 하나에 집중했다.

| 관점 | 보는 것 | 놓치기 쉬운 것 |
| --- | --- | --- |
| 성능 관점 | benchmark score가 오르는가 | 어떤 parameter direction으로 이동했는가 |
| objective 관점 | KL, CE, reward loss의 차이 | 실제 optimizer trajectory가 어떤 subspace를 쓰는가 |
| recipe 관점 | teacher, prompt, rollout 설정 | OPD가 SFT/RLVR와 구조적으로 다른가 |

이 중 어떤 것도 틀린 접근은 아니다. 하지만 OPD가 점점 많이 쓰일수록, 단순히 final score를 비교하는 것만으로는 부족해진다.

예를 들어 OPD가 SFT보다 좋은 결과를 냈다고 하자. 그 이유가 teacher trace를 더 잘 따라갔기 때문인지, student가 이미 알고 있던 reasoning path를 더 안정적으로 정렬했기 때문인지, 아니면 parameter space에서 불필요한 principal direction을 피했기 때문인지 구분해야 한다. 반대로 OPD가 RLVR보다 안정적이라고 해도, 그것이 더 촘촘한 supervision 때문인지, 아니면 RLVR보다 덜 constrained된 update geometry 때문인지 확인해야 한다.

이 논문이 보는 문제는 바로 이 구분이다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 크게 4가지로 정리할 수 있다.

1. **OPD update geometry 분석**
   - OPD, SFT, RLVR를 parameter-space diagnostic으로 비교한다.
   - 논문은 OPD가 SFT보다 fewer weights를 건드리고 principal direction을 더 강하게 피한다고 보고한다.
   - RLVR와 비교하면 OPD는 덜 tight하게 constrained된 geometry를 가진다.

2. **Relaxed off-principal regime 제안**
   - OPD는 SFT처럼 넓게 principal direction을 따라가지 않는다.
   - 동시에 RLVR처럼 강하게 제한된 방향에만 묶이지도 않는다.
   - 그래서 논문은 OPD를 relaxed off-principal regime으로 해석한다.

3. **Subspace locking 관찰**
   - OPD cumulative update는 training 초기에 narrow low-dimensional channel로 빠르게 들어간다.
   - 초기 update subspace로 training을 제한해도 OPD 성능은 유지되지만, 같은 제약은 SFT 성능을 크게 손상시킨다.
   - 이는 early OPD subspace가 OPD에는 functionally sufficient하다는 근거로 제시된다.

4. **Control experiment로 geometry 분리**
   - token sparsification과 off-policy rollout은 rank dynamics를 보존한다.
   - 반면 OPD objective에 RLVR를 섞으면 rank dynamics가 바뀐다.
   - 즉 OPD geometry는 단순히 on-policy sampling만의 산물이 아니라, OPD objective 자체와도 연결되어 있다.

## 2-2. Design intuition

이 논문의 설계 직관은 다음 문장으로 요약할 수 있다.

OPD는 teacher를 따라가지만, teacher가 만든 전체 trace를 복제하지 않는다.

이 차이가 중요하다. SFT는 주어진 target sequence를 따라가야 하므로, 모델이 기존에 가지고 있던 parameter manifold를 넓게 흔들 수 있다. 반면 OPD는 student가 실제로 방문한 state에서 teacher distribution을 참고한다. 그러면 update는 student가 이미 접근 가능한 region을 중심으로 생긴다. 그래서 OPD는 SFT보다 더 localized된 update를 만들 수 있다.

하지만 RLVR와도 다르다. RLVR는 outcome reward가 sparse하고, verifiable reward가 있는 방향으로 policy를 조인다. OPD는 teacher distribution이 token-level로 더 dense하기 때문에 RLVR보다 덜 tight한 constraint를 가진다. 논문이 OPD를 relaxed off-principal regime이라고 부르는 이유가 여기에 있다.

또 하나 중요한 직관은 **early geometry matters** 다.

training 초기에 형성된 update subspace가 이후 OPD update를 강하게 제한한다면, OPD를 이해하려면 final checkpoint만 보면 안 된다. 초기 수백 step에서 어떤 direction이 열리는지, 그 direction이 이후에도 유지되는지 봐야 한다. 이 논문은 이 현상을 subspace locking이라고 부른다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | OPD가 parameter space에서 어떤 update geometry를 만드는지 분석 |
| Compared methods | OPD, SFT, RLVR |
| Main diagnostics | localization, principal-direction avoidance, cumulative subspace rank, early-subspace constraint |
| Core phenomenon | relaxed off-principal regime and subspace locking |
| Key control | token sparsification, off-policy rollout, OPD plus RLVR mixing |
| Main claim | OPD는 SFT/RLVR 사이의 단순 중간점이 아니라 독자적인 update geometry를 가진다 |

## 3-2. Module breakdown

### 1) Parameter update trajectory

논문은 training 과정에서 parameter update를 trajectory로 본다. 초기 parameter를 $\theta_0$, step $t$의 parameter를 $\theta_t$라고 하면 cumulative update는 다음처럼 쓸 수 있다.

$$
\Delta_t = \theta_t - \theta_0
$$

이 관점에서 중요한 것은 final model이 아니라 $\Delta_t$가 어떤 방향으로 누적되는가다.

- SFT는 supervised target에 맞추기 위해 relatively broad update를 만들 수 있다.
- RLVR는 reward가 주는 constraint 때문에 더 tight한 update pattern을 보일 수 있다.
- OPD는 그 둘과 다른 위치에 놓인다.

논문은 이 차이를 parameter-space diagnostic으로 측정한다.

### 2) Principal-direction avoidance

논문 abstract 기준, OPD는 SFT보다 principal direction을 더 강하게 피한다. 여기서 핵심은 OPD가 모델의 큰 변화 방향을 그대로 따라가지 않는다는 점이다.

개념적으로 어떤 principal subspace $U$가 있다고 하면, update가 그 subspace에 얼마나 투영되는지는 아래 비율처럼 생각할 수 있다.

$$
r_t = \frac{||P_U \Delta_t||_2^2}{||\Delta_t||_2^2}
$$

$r_t$가 낮다는 것은 update가 principal subspace에 덜 놓인다는 뜻이다. 논문은 OPD가 SFT보다 이런 principal direction을 더 강하게 피한다고 보고한다. 다만 RLVR와 비교하면 OPD는 덜 tight하게 constrained된다.

이 해석은 OPD를 단순히 작은 SFT로 보는 것을 막아준다. OPD가 SFT보다 작은 update를 만든다는 말과, OPD가 SFT와 다른 방향으로 움직인다는 말은 다르다. 이 논문은 후자에 더 가깝다.

### 3) Subspace locking

가장 중요한 현상은 subspace locking이다.

논문은 OPD cumulative update가 training 초기에 narrow low-dimensional channel로 빠르게 들어간다고 설명한다. 직관적으로 말하면, OPD는 초기에 열린 몇 개의 update direction을 계속 재사용한다.

이를 개념적으로 쓰면, early step에서 얻은 update들을 모아 subspace $S_{early}$를 만들 수 있다.

$$
S_{early} = span(\Delta_1, \Delta_2, ..., \Delta_k)
$$

그 뒤 training을 이 subspace 안으로 제한했을 때 OPD 성능이 유지된다면, $S_{early}$가 OPD에는 충분한 working subspace라는 뜻이다. 논문은 이런 제약이 OPD에는 잘 맞지만 SFT에는 크게 불리하다고 보고한다.

이 결과는 꽤 중요하다. OPD가 단순히 작은 learning rate로 천천히 움직이는 것이 아니라, **초기부터 특정 functional channel을 찾아 들어간 뒤 그 안에서 학습한다** 는 해석을 가능하게 한다.

### 4) Control experiments

논문은 OPD geometry가 무엇에 의해 생기는지 분리하기 위해 control experiment를 수행한다.

- token sparsification은 update token을 줄여도 rank dynamics를 보존한다.
- off-policy rollout generation으로 옮겨도 rank dynamics가 보존된다.
- 반면 OPD objective에 RLVR를 섞으면 rank dynamics가 바뀐다.

이 결과를 그대로 해석하면, OPD의 subspace locking은 단순히 token 수가 많거나 on-policy rollout이라서 생긴 현상이 아니다. objective가 제공하는 dense teacher distribution과 student-visited state의 결합이 geometry를 만든다고 보는 편이 자연스럽다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 새로운 대규모 데이터셋을 제안하는 paper라기보다, OPD training dynamics를 분석하는 paper다. 따라서 데이터 자체보다 더 중요한 것은 동일하거나 비교 가능한 training setting에서 OPD, SFT, RLVR의 update trajectory를 비교하는 방식이다.

본문 초안 기준으로 정리하면, 이 논문에서 데이터는 다음 역할을 한다.

- SFT에서는 supervised target을 따라가는 reference trajectory 역할을 한다.
- RLVR에서는 verifiable reward를 평가할 task distribution 역할을 한다.
- OPD에서는 student rollout prefix 위에서 teacher distribution을 평가하는 state distribution 역할을 한다.

즉 데이터 자체보다 **state distribution이 누구에 의해 만들어지는가** 가 핵심이다. OPD에서는 student가 state distribution을 만들고, teacher는 그 state 위에서 dense distributional signal을 제공한다.

## 4-2. Training strategy

OPD training을 간단히 보면 아래처럼 정리할 수 있다.

1. student rollout을 생성한다.
2. teacher가 같은 prefix에 대해 token distribution을 계산한다.
3. student는 teacher distribution과의 divergence를 줄인다.
4. parameter update trajectory를 step별로 기록한다.
5. SFT와 RLVR의 trajectory와 같은 diagnostic으로 비교한다.

논문이 흥미로운 지점은 1-3보다 4-5다. 보통 distillation 논문은 final performance를 중심으로 보지만, 이 논문은 update가 어느 subspace에 쌓이는지까지 본다.

## 4-3. Engineering notes

실제로 이 분석을 재현하거나 비슷한 실험을 설계한다면 다음 점이 중요하다.

- checkpoint를 충분히 촘촘하게 저장해야 한다.
- parameter delta를 layer별 또는 block별로 비교할 수 있어야 한다.
- SFT, OPD, RLVR의 optimizer setting과 compute budget을 최대한 맞춰야 한다.
- rank 또는 effective dimension을 볼 때 random noise나 scale 차이와 구분해야 한다.
- early subspace constraint 실험은 단순 projection이 아니라 training dynamics 자체를 제한하는 ablation으로 설계해야 한다.

이 논문은 OPD를 학습시키는 새로운 recipe만 제시하는 것이 아니라, OPD를 어떻게 진단해야 하는지도 보여준다. 제 관점에서는 이 부분이 실무적으로 더 중요하다. OPD가 잘 되는지 보려면 validation accuracy만 볼 것이 아니라, update가 특정 subspace에 과도하게 lock-in되는지, 그리고 그 subspace가 실제 task performance에 충분한지 봐야 한다.

# 5. Evaluation

## 5-1. Main results

arXiv abstract 기준으로 확인되는 주요 결과는 아래와 같다.

| Finding | Meaning |
| --- | --- |
| OPD affects fewer weights than SFT | OPD update는 SFT보다 더 localized되어 있음 |
| OPD avoids principal directions more strongly than SFT | OPD는 SFT의 주요 update direction을 그대로 따라가지 않음 |
| OPD is less tightly constrained than RLVR | OPD는 RLVR보다 dense하고 relaxed한 geometry를 가짐 |
| OPD cumulative updates enter a narrow low-dimensional channel | OPD에는 subspace locking 현상이 있음 |
| Early OPD subspace preserves OPD performance | 초기에 형성된 subspace가 OPD에는 functionally sufficient함 |
| Same constraint degrades SFT | SFT는 더 넓은 update space를 필요로 함 |
| Token sparsification preserves rank dynamics | rank dynamics가 token 수만으로 설명되지 않음 |
| Off-policy rollout preserves rank dynamics | rank dynamics가 on-policy rollout만의 산물이 아님 |
| OPD plus RLVR changes rank dynamics | objective mixing은 OPD geometry 자체를 바꿈 |

가장 중요한 결과는 early subspace constraint다. 단순히 OPD update가 low-rank처럼 보인다는 관찰만으로는 부족하다. 정말 그 subspace가 기능적으로 충분한지 확인해야 한다. 논문은 early OPD subspace 안에서 training을 제한해도 OPD performance가 유지되지만, SFT는 크게 떨어진다고 보고한다. 이 대조가 subspace locking을 단순한 시각화가 아니라 functional result로 만든다.

## 5-2. What really matters in the experiments

이 논문을 읽을 때 중요한 것은 score table 자체보다 diagnostic의 해석이다.

첫째, OPD가 SFT보다 fewer weights를 건드린다는 결과는 OPD가 parameter-efficient하다는 뜻으로 바로 읽으면 안 된다. 여기서 더 중요한 것은 **같은 task improvement를 어떤 geometry로 달성하는가** 다.

둘째, OPD가 principal direction을 피한다는 결과는 OPD가 safer하거나 항상 better하다는 뜻이 아니다. principal direction을 피한다는 것은 큰 parameter 변화 방향을 덜 쓰는 것이고, 이것이 task에 따라 장점일 수도 있고 한계일 수도 있다.

셋째, subspace locking은 양날의 검이다. 좋은 쪽으로 보면 OPD는 early training에서 useful channel을 빠르게 찾는다. 나쁜 쪽으로 보면 초기에 잘못 열린 channel에 갇힐 수도 있다.

넷째, OPD plus RLVR mixing이 rank dynamics를 바꾼다는 점은 recipe 설계에서 중요하다. OPD와 RLVR를 단순히 더하면 두 objective의 장점을 모두 얻는다고 기대하기 어렵다. geometry 자체가 변한다면, mixing coefficient와 schedule은 성능뿐 아니라 update rank와 direction까지 같이 봐야 한다.

# 6. Limitations

1. **Abstract에서 확인되는 범위는 geometry 중심이다.**
   - 논문은 17 pages, 8 figures로 구성되어 있으나, 공개 abstract만으로는 benchmark별 세부 수치와 model family 설정을 모두 확인하기 어렵다.
   - 블로그 반영 전 원문 PDF의 figure/table 숫자를 재확인하는 것이 좋다.

2. **Geometry가 곧 generalization을 보장하지는 않는다.**
   - OPD가 low-dimensional channel에 lock-in된다는 것은 중요한 관찰이지만, 그 channel이 새로운 domain이나 task에서도 충분하다는 뜻은 아니다.
   - subspace locking이 좋은 inductive bias인지, early over-commitment인지는 task별로 다를 수 있다.

3. **OPD recipe로 바로 변환하려면 추가 ablation이 필요하다.**
   - 논문은 token sparsification, off-policy rollout, RLVR mixing을 비교하지만, practical recipe를 만들려면 teacher strength, rollout length, KL temperature, learning rate, prompt selection까지 함께 봐야 한다.

4. **Principal direction의 정의와 diagnostic 구현이 중요하다.**
   - principal direction을 어떤 basis로 정의했는지에 따라 해석이 달라질 수 있다.
   - layer-wise analysis와 whole-model analysis가 같은 결론을 주는지도 확인해야 한다.

5. **OPD와 RLVR를 섞는 문제는 여전히 열려 있다.**
   - 논문은 mixing이 rank dynamics를 바꾼다고 보고하지만, 이것이 항상 나쁜 변화인지, 특정 schedule에서는 유리한 변화인지까지는 추가 연구가 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 OPD를 성능 비교표가 아니라 **update geometry** 로 읽게 만든다는 점에서 가치가 있다. 최근 reasoning post-training 논문들을 보면 OPD, RLVR, SFT, rejection sampling, teacher-guided rollout이 자주 섞인다. 그런데 이들을 loss 이름으로만 비교하면 실제로 모델이 어떻게 변하는지 놓치기 쉽다.

이 논문의 메시지는 분명하다. OPD는 SFT와 RLVR 사이의 적당한 compromise가 아니다. student-visited state와 teacher distribution이 결합되면서 별도의 low-dimensional update channel을 만든다. 이 해석이 맞다면, 앞으로 OPD recipe를 설계할 때는 final score뿐 아니라 다음을 같이 봐야 한다.

- update가 어느 layer에 몰리는가.
- cumulative update rank가 언제 saturate되는가.
- early subspace가 계속 유지되는가.
- teacher를 바꾸면 subspace도 바뀌는가.
- RLVR를 섞으면 rank dynamics가 어떻게 이동하는가.

제 관점에서 가장 중요한 포인트는 subspace locking이다. OPD가 초기에 충분한 channel을 찾는다면 training은 효율적일 수 있다. 하지만 초기에 형성된 channel이 잘못되면, 더 오래 학습해도 다른 direction으로 잘 빠져나가지 못할 가능성이 있다. 그래서 OPD에서는 early phase design이 생각보다 더 중요할 수 있다.

## 7-2. Reuse potential

이 논문의 분석 프레임은 여러 곳에 재사용할 수 있다.

1. **OPD recipe debugging**
   - training loss와 validation score 외에 update rank, layer-wise delta norm, early subspace projection ratio를 함께 기록한다.

2. **Teacher selection**
   - teacher score만 보지 않고, student update가 teacher 변경에 따라 어떤 subspace로 이동하는지 본다.

3. **SFT vs OPD selection**
   - task가 넓은 behavior coverage를 요구하면 SFT가 필요할 수 있다.
   - student가 이미 어느 정도 task manifold에 들어와 있고 reasoning style만 조정하면 된다면 OPD가 더 적합할 수 있다.

4. **OPD plus RLVR scheduling**
   - OPD와 RLVR를 동시에 섞기보다, OPD로 stable channel을 만든 뒤 RLVR를 별도 phase로 넣는 schedule을 실험할 가치가 있다.

5. **Parameter-efficient analysis**
   - LoRA나 adapter training에서도 subspace locking을 측정하면, adapter rank가 충분한지 더 직접적으로 볼 수 있다.

## 7-3. Follow-up papers

- Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe
- Draft-OPD: On-Policy Distillation for Speculative Draft Models
- Trust Region On-Policy Distillation
- Trust-Region Behavior Blending for On-Policy Distillation
- Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability

# 8. Summary

- On the Geometry of OPD는 OPD를 성능 recipe가 아니라 parameter-space dynamics로 분석한다.
- 논문은 OPD가 SFT보다 fewer weights를 건드리고 principal direction을 더 피하며, RLVR보다는 덜 tight하게 constrained된다고 보고한다.
- 핵심 현상은 **subspace locking** 이다. OPD cumulative update는 초기에 low-dimensional channel로 들어가고, 그 channel은 OPD에는 functionally sufficient하다.
- token sparsification과 off-policy rollout은 rank dynamics를 보존하지만, OPD와 RLVR mixing은 rank dynamics를 바꾼다.
- 이 결과는 OPD recipe를 설계할 때 final score뿐 아니라 update rank, early subspace, objective mixing geometry를 같이 봐야 함을 시사한다.
