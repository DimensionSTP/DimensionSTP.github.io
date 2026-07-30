---
layout: single
title: "To Grok Grokking: Provable Grokking in Ridge Regression Review"
categories: Study-concept
tag: [Grokking, Ridge-Regression, Learning-Theory]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2601.19791)

[HTML link](https://arxiv.org/html/2601.19791)

> 한 줄 요약: 이 논문은 overparameterized ridge regression을 gradient descent와 weight decay로 학습할 때 training error는 빠르게 작아지지만 population error는 null-space initialization이 천천히 사라질 때까지 높게 유지되는 현상을 증명하고, grokking delay가 optimization timescale separation에서 발생할 수 있음을 보인다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Grokking을 transformer나 modular arithmetic에만 나타나는 신비한 phase transition이 아니라, 가장 단순한 linear regression에서도 증명 가능한 dynamical phenomenon으로 바꾼다.
- Training data가 볼 수 있는 parameter subspace와 볼 수 없는 orthogonal subspace의 학습 속도를 분리해 delayed generalization mechanism을 설명한다.
- Weight decay $\lambda$가 grokking time을 대략 $1/\lambda$ scale로 조절한다는 명확한 prediction을 제공한다.
- Sample size, feature dimension, initialization scale가 training-fit time과 generalization time에 서로 다르게 영향을 주는 이유를 분석한다.
- Regression loss는 부드럽게 감소해도 thresholded accuracy에서는 classic plateau와 sudden jump가 나타날 수 있음을 보여준다.

Grokking은 보통 training accuracy가 먼저 100%에 도달한 뒤 test accuracy가 오랫동안 낮게 머물다가 갑자기 상승하는 현상으로 소개된다. 이 설명은 인상적이지만, 두 가지 질문을 남긴다.

1. 무엇이 training fit과 generalization 사이의 긴 시간 간격을 만드는가?
2. Sudden jump는 실제 parameter dynamics의 phase transition인가, 아니면 metric이 만든 시각적 효과인가?

이 논문은 ridge regression이라는 fully analyzable setting에서 두 질문을 다룬다. 핵심은 model이 training data span에서는 빠르게 정답을 맞추지만, data span에 보이지 않는 initialization component는 weight decay로만 천천히 제거된다는 점이다.

# 1. Problem Setting

## 1-1. Problem definition

Teacher function은 feature map $\phi(x) \in \mathbb{R}^m$에서 linear하게 표현된다고 가정한다.

$$
N^*(x)
=
\langle \theta^*,\phi(x) \rangle
$$

Student도 같은 feature map을 사용한다.

$$
N(x;\theta)
=
\langle \theta,\phi(x) \rangle
$$

Training sample은 $n$개이며 feature dimension은 $m$이다. 논문이 관심을 두는 regime은 $m \gg n$인 overparameterized setting이다.

Ridge objective는 다음과 같다.

$$
L_n(\theta;\lambda)
=
\frac{1}{2n}
\sum_{i=1}^{n}
\left(
N(x_i;\theta)-N^*(x_i)
\right)^2
+
\frac{\lambda}{2}
\|\theta\|_2^2
$$

Gradient descent update는 다음처럼 쓸 수 있다.

$$
\theta_{t+1}
=
\theta_t
-
\eta
\nabla_\theta L_n(\theta_t;\lambda)
$$

논문은 grokking을 두 event time으로 정의한다.

- $t_1$: training error가 threshold $\epsilon$ 아래로 내려가기 직전의 마지막 time
- $t_2$: population error가 threshold $c$ 아래로 내려가는 첫 time

Grokking delay는 다음과 같다.

$$
T_{\mathrm{grok}}
=
t_2-t_1
$$

이 정의는 plot을 보고 subjective하게 plateau를 정하는 대신, train and population objective에 명시적 threshold를 둔다.

## 1-2. Why previous approaches are insufficient

Grokking에 대한 기존 설명은 여러 mechanism을 제안했다.

- Feature learning이 memorizing representation에서 generalizing representation으로 전환된다.
- Weight norm이나 representation complexity가 특정 시점에 급격히 변한다.
- Optimization이 loss landscape의 flat or low-complexity solution으로 이동한다.
- Weight decay가 late training에서 implicit bias를 강화한다.
- Fourier feature나 circuit이 단계적으로 형성된다.

이 설명들은 nonlinear neural network에서 중요한 통찰을 준다. 하지만 mechanism이 여러 개 섞여 있어 grokking의 최소 조건을 분리하기 어렵다.

이 논문은 더 제한적인 질문을 택한다.

> Feature learning이 전혀 없는 fixed linear feature model에서도 training fit과 generalization 사이의 큰 delay를 증명할 수 있는가?

가능하다면 grokking의 일부는 representation phase transition 없이도 설명된다. 즉 overparameterization, initialization, finite sample geometry, explicit regularization만으로도 delayed generalization이 나타날 수 있다.

또한 accuracy plateau는 discrete metric에 민감하다. Regression prediction error가 서서히 줄어도 threshold를 넘는 sample 수가 특정 구간에서 급격히 늘면 accuracy는 sudden transition처럼 보인다. 따라서 loss dynamics와 displayed metric을 분리해야 한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 contribution은 end-to-end provable grokking construction이다.

1. Realizable overparameterized ridge regression
   - Student와 teacher가 같은 feature space에 있다.
   - Label noise가 없는 setting에서 population error를 임의로 작게 만들 수 있다.

2. Data-span decomposition
   - Parameter를 training feature span에 평행한 component와 직교한 component로 나눈다.
   - Training error는 parallel component만 본다.
   - Population error는 orthogonal component도 본다.

3. Two-timescale theorem
   - Parallel error는 empirical Hessian의 nonzero eigenvalue에 의해 빠르게 감소한다.
   - Orthogonal initialization은 weight decay rate로만 감소한다.
   - Small $\lambda$에서 $t_2$가 $1/\lambda$ scale로 커져 큰 grokking gap을 만든다.

4. Controlled empirical extensions
   - Linear ridge setting에서 theorem prediction을 검증한다.
   - Random ReLU feature model과 two-layer nonlinear network에서도 같은 qualitative trend가 나타나는지 본다.

## 2-2. Design intuition

Training feature matrix를 $\Phi \in \mathbb{R}^{n \times m}$라고 하자. $m>n$이면 training data가 관측하는 parameter direction은 최대 $n$ dimension뿐이다.

Parameter error를 다음처럼 분해한다.

$$
e_t
=
\theta_t-\theta^*
=
e_t^{\parallel}
+
e_t^{\perp}
$$

- $e_t^{\parallel}$: row space of $\Phi$ 안의 error
- $e_t^{\perp}$: training feature span에 orthogonal한 error

Empirical prediction은 $\Phi e_t$에만 의존한다. Orthogonal component는 다음을 만족한다.

$$
\Phi e_t^{\perp}
=
0
$$

따라서 $e_t^{\perp}$가 아무리 커도 training error에는 보이지 않는다. 반면 new input feature $\phi(x)$는 training span 밖의 direction을 포함할 수 있으므로 population error는 이 component를 본다.

Gradient descent with weight decay에서 orthogonal component에는 data gradient가 없다. Update는 단순히 다음과 같다.

$$
e_{t+1}^{\perp}
=
(1-\eta\lambda)
e_t^{\perp}
+
\text{ridge-bias term}
$$

주된 initialization component는 대략 다음 속도로 감소한다.

$$
\|e_t^{\perp}\|_2
\approx
(1-\eta\lambda)^t
\|e_0^{\perp}\|_2
$$

Small $\eta\lambda$에서 exponential approximation을 쓰면 다음과 같다.

$$
(1-\eta\lambda)^t
\approx
\exp(-\eta\lambda t)
$$

따라서 population error threshold에 도달하는 시간은 다음 scale을 가진다.

$$
t_2
\propto
\frac{1}{\eta\lambda}
\log
\left(
\frac{\|e_0^{\perp}\|}{c}
\right)
$$

반면 parallel component는 empirical covariance의 nonzero eigenvalue에 의해 훨씬 빠르게 줄 수 있다. 이 속도 차이가 grokking이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Target phenomenon | Early training fit followed by delayed population generalization |
| Base model | Linear predictor on fixed feature map |
| Objective | Squared error plus ridge penalty |
| Regime | Overparameterized, $m \gg n$ |
| Optimization | Full-batch gradient descent with weight decay |
| Key decomposition | Training-data span vs orthogonal complement |
| Fast timescale | Empirical fitting in observed subspace |
| Slow timescale | Weight-decay removal of invisible initialization component |
| Main scaling | Grokking time grows approximately as $1/\lambda$ |

## 3-2. Module breakdown

### 1) Empirical and population loss

정규화 항을 제외한 학습 손실은 다음과 같이 쓸 수 있다.

$$
\widehat{L}_n(\theta)
=
\frac{1}{2n}
\|\Phi(\theta-\theta^*)\|_2^2
$$

Population loss는 다음과 같은 quadratic form이다.

$$
L(\theta)
=
\frac{1}{2}
\mathbb{E}_x
\left[
\langle \theta-\theta^*,\phi(x)\rangle^2
\right]
$$

Population covariance를 $\Sigma=\mathbb{E}[\phi(x)\phi(x)^T]$라고 하면 다음과 같다.

$$
L(\theta)
=
\frac{1}{2}
(\theta-\theta^*)^T
\Sigma
(\theta-\theta^*)
$$

Empirical loss는 $\Phi$의 row space 밖을 전혀 보지 못하지만 population covariance는 그 direction에 nonzero mass를 둘 수 있다.

### 2) Parallel dynamics

Empirical Hessian은 다음과 같다.

$$
H_n
=
\frac{1}{n}
\Phi^T\Phi
$$

Parallel component는 $H_n$의 positive eigenspace에서 update된다. Eigenvalue $s_j>0$인 direction의 contraction factor는 대략 다음과 같다.

$$
1-\eta(s_j+\lambda)
$$

$s_j$가 $\lambda$보다 크면 fitting speed는 data curvature가 지배한다. 따라서 training error는 relatively few iterations 안에 threshold 아래로 내려갈 수 있다.

### 3) Orthogonal dynamics

Null space에서는 $s_j=0$이다. 따라서 contraction factor는 다음 하나뿐이다.

$$
1-\eta\lambda
$$

$\lambda$가 작을수록 이 component는 매우 천천히 사라진다. Training prediction에는 나타나지 않지만 new sample에서 error를 만든다.

이 mechanism은 memorization을 별도 lookup table로 구현하는 것이 아니다. Model이 training constraints를 만족하는 solution manifold 안에서 initialization-dependent null-space component를 오래 유지하는 것이다.

### 4) Ridge bias and eventual generalization

Weight decay는 orthogonal error를 줄이지만 동시에 target parameter도 origin 쪽으로 shrink한다. 따라서 $\lambda$를 크게 하면 grokking delay는 짧아지지만 final estimator의 ridge bias가 커질 수 있다.

논문은 적절히 작은 $\lambda$와 충분한 training time을 선택해 다음 세 stage를 동시에 만든다.

1. Training error가 빠르게 작아진다.
2. Population error는 일정 시간 threshold 위에 남는다.
3. 이후 population error도 임의로 작은 값 아래로 내려간다.

즉 단순히 train-test gap이 오래 유지되는 underfitting example이 아니라, eventual generalization까지 포함한 grokking construction이다.

### 5) Thresholded accuracy

Regression loss는 smooth exponential decay를 보일 수 있다. Classic grokking plot처럼 long plateau 뒤 sudden jump를 보기 위해 논문은 prediction error threshold에 기반한 surrogate accuracy를 사용한다.

예를 들어 sample별 squared error가 threshold $\tau$보다 작으면 correct로 정의할 수 있다.

$$
\operatorname{Acc}_\tau(\theta)
=
\mathbb{P}
\left(
|N(x;\theta)-N^*(x)|^2
\leq
\tau
\right)
$$

Population error distribution이 threshold 근처를 통과하면 continuous loss curve가 sharp accuracy transition으로 보인다. 이는 grokking의 시각적 suddenness가 metric nonlinearity에 의해 강화될 수 있음을 보여준다.

### 6) Nonlinear extensions

논문은 theorem을 그대로 nonlinear network에 확장하지 않는다. 대신 두 empirical bridge를 둔다.

- Random ReLU features
  - Hidden weights는 고정하고 output layer만 학습한다.
  - Model은 parameter에 대해 linear하지만 input feature는 nonlinear하다.

- Two-layer ReLU network
  - Hidden layer와 output layer를 함께 학습한다.
  - Feature learning이 가능하다.

두 setting에서도 weight decay와 initialization scale에 따른 delayed generalization trend가 관찰된다. 하지만 이는 proof가 아니라 mechanism의 qualitative robustness check다.

# 4. Training / Data / Recipe

## 4-1. Data

Theoretical setting은 teacher-student regression이다.

- Input $x$는 지정된 distribution에서 sampling한다.
- Teacher parameter $\theta^*$가 noiseless label을 만든다.
- Student는 같은 or related feature family를 사용한다.
- Training sample size $n$은 feature dimension $m$보다 작다.

대표 linear ridge experiment의 default setting은 다음과 같다.

| Parameter | Default value |
| --- | ---: |
| Training samples $n$ | 100 |
| Feature dimension $m$ | 1000 |
| Initialization variance $\nu^2$ | 1 |
| Weight decay $\lambda$ | $10^{-4}$ |
| Learning rate $\eta$ | 1 |
| Train threshold $\epsilon$ | 0.01 |
| Population threshold $c$ | 0.01 |

이 setting은 $m/n=10$인 clear overparameterized regime다. Training data span 밖에 큰 orthogonal subspace가 남기 때문에 initialization component가 population error에 오래 영향을 줄 수 있다.

## 4-2. Training strategy

Linear experiment는 full-batch gradient descent를 사용한다. Stochastic gradient noise나 minibatch ordering을 제거해 subspace dynamics를 직접 본다.

주요 sweep은 다음과 같다.

1. Weight decay $\lambda$
   - Smaller $\lambda$는 slow orthogonal decay를 만들어 grokking delay를 늘린다.
   - 너무 큰 $\lambda$는 final ridge bias를 키울 수 있다.

2. Sample size $n$
   - Smaller $n$은 training constraints가 적어 empirical fit이 더 빨라질 수 있다.
   - Orthogonal subspace도 커져 train-generalization gap이 커질 수 있다.

3. Feature dimension $m$
   - Tested overparameterized range에서는 $t_1$과 $t_2$가 dimension에 크게 민감하지 않은 결과가 보고된다.
   - 이는 특정 normalization and experiment에 대한 관찰이며 universal invariance는 아니다.

4. Initialization scale $\nu$
   - Larger initialization은 orthogonal component를 키운다.
   - Decay가 exponential이므로 threshold crossing time은 initialization magnitude에 logarithmic하게 의존한다.

## 4-3. Engineering notes

### 1) Weight decay와 L2 regularization을 구분해 기록해야 한다

Simple SGD에서는 같은 식으로 보일 수 있지만 adaptive optimizer에서는 decoupled weight decay와 objective L2 penalty가 다른 dynamics를 만든다. 이 논문의 theorem은 명시된 ridge objective와 gradient descent를 기준으로 한다.

### 2) Train loss에 regularization term을 포함할지 분리해야 한다

Grokking의 training fit은 data error를 뜻한다. Ridge penalty까지 합친 objective만 보면 training error가 이미 작아졌는지 해석하기 어렵다.

### 3) Population metric을 충분히 크게 추정해야 한다

Delayed generalization curve는 finite validation set noise에 민감할 수 있다. Dense evaluation grid or large held-out set이 필요하다.

### 4) Parameter projection을 logging하면 mechanism을 직접 검증할 수 있다

- Data-span norm $\|e_t^{\parallel}\|$
- Orthogonal norm $\|e_t^{\perp}\|$
- Weight norm
- Train and population loss

이 네 curve를 함께 보면 training fit과 null-space cleanup이 분리되는지 확인할 수 있다.

### 5) Threshold choice를 여러 개 보고해야 한다

Single accuracy threshold만 사용하면 sudden transition을 과장할 수 있다. Raw regression loss와 여러 $\tau$의 thresholded accuracy를 함께 제시하는 것이 안전하다.

# 5. Evaluation

## 5-1. Main results

### 1) Provable three-stage dynamics

논문의 theorem은 적절한 data geometry, initialization, learning rate, weight decay 아래에서 다음을 보인다.

- Early time에는 training error가 threshold 아래로 내려간다.
- 같은 시점의 population error는 여전히 threshold보다 크다.
- 충분히 늦은 time에는 population error도 임의로 작은 threshold 아래로 내려간다.

따라서 overfitting 뒤 persistent failure가 아니라 delayed success가 증명된다.

### 2) Grokking time은 weight decay에 강하게 의존한다

Small $\lambda$ regime에서 generalization time은 대략 다음 scale을 따른다.

$$
t_2
=
\Theta
\left(
\frac{1}{\lambda}
\right)
$$

Learning rate를 명시하면 $1/(\eta\lambda)$ scale이다. Training-fit time $t_1$은 data curvature에 더 크게 의존하므로 $\lambda$를 줄이면 두 time 사이의 gap이 커진다.

이 result는 weight decay를 단순한 final generalization regularizer가 아니라 dynamics clock으로 해석하게 한다.

### 3) Smaller sample size가 더 큰 grokking gap을 만들 수 있다

직관적으로 data가 적으면 generalization이 나빠질 것 같지만, grokking delay 관점에서는 두 effect가 있다.

- Training constraint가 적어 empirical fit이 더 빨리 끝날 수 있다.
- Unobserved parameter subspace가 커져 population cleanup이 더 오래 필요할 수 있다.

따라서 smaller $n$이 $t_1$을 앞당기고 $t_2-t_1$을 늘릴 수 있다. 이것은 grokking gap이 model capacity만이 아니라 data geometry에 의해 만들어진다는 뜻이다.

### 4) Feature dimension effect는 tested setting에서 제한적이다

논문 실험에서는 overparameterized range에서 $m$을 바꿔도 $t_1$과 $t_2$가 크게 움직이지 않는 결과가 나타난다. 이미 $m \gg n$이고 normalization이 맞춰진다면, 추가 null-space dimension보다 initialization energy and weight decay가 dominant할 수 있다.

하지만 이 결과를 모든 feature scaling에 일반화하면 안 된다. Feature normalization, teacher distribution, initialization variance를 함께 바꾸면 dimension dependence가 달라질 수 있다.

### 5) Initialization scale은 logarithmic time shift를 만든다

Orthogonal component가 $(1-\eta\lambda)^t$로 감소하므로 initial norm을 상수 배 키우면 threshold crossing time은 logarithmic하게 이동한다.

$$
\Delta t
\approx
\frac{1}{\eta\lambda}
\log c_0
$$

여기서 $c_0$는 initialization scale ratio다. 실험은 larger initialization에서 grokking이 늦어지는 trend를 지지한다.

### 6) Nonlinear models에서도 qualitative trend가 나타난다

Random ReLU feature와 fully trained two-layer ReLU experiment에서도 다음 pattern이 관찰된다.

- Training performance가 먼저 좋아진다.
- Test performance는 weight decay에 따라 늦게 개선된다.
- Smaller weight decay가 longer delay를 만든다.

다만 nonlinear network에서 실제 hidden representation이 theorem의 fixed subspace decomposition을 그대로 따른다는 증거는 아니다.

## 5-2. What really matters in the experiments

이 논문을 읽을 때 중요한 것은 grokking curve가 예쁘게 보이는지가 아니다.

1. Raw loss and thresholded accuracy
   - Smooth dynamics와 visually sudden metric을 분리한다.

2. $t_1$ and $t_2$ scaling
   - 단일 curve보다 hyperparameter 변화에 대한 event-time law가 mechanism을 더 강하게 검증한다.

3. Parallel and orthogonal norms
   - Data fit과 invisible component decay가 실제로 다른 timescale을 가지는지 본다.

4. Final error floor
   - Faster grokking이 larger ridge bias와 교환된 것은 아닌지 확인한다.

5. Linear proof vs nonlinear evidence
   - Theorem의 claim과 neural-network experiment의 qualitative analogy를 구분한다.

이 논문은 grokking을 완전히 설명했다기보다, grokking이라는 label 아래 최소 두 현상이 섞여 있음을 보여준다. 하나는 subspace-dependent optimization delay이고, 다른 하나는 genuinely nonlinear representation change다. 전자를 제거하거나 측정하지 않으면 후자를 과대해석할 수 있다.

# 6. Limitations

1. Main proof는 realizable linear ridge regression에 한정된다.
   - Teacher가 student feature space에 있고 label noise가 없다.

2. Nonlinear network result는 empirical observation이다.
   - Hidden feature learning이 동일한 parallel-orthogonal mechanism을 따른다는 theorem은 없다.

3. Classic grokking benchmark와 loss geometry가 다르다.
   - Modular arithmetic classification의 exact accuracy plateau와 squared regression loss는 직접 동일하지 않다.

4. Thresholded accuracy가 suddenness를 만들 수 있다.
   - Sharp transition이 underlying dynamics의 true phase transition을 뜻하지 않을 수 있다.

5. Full-batch gradient descent setting이다.
   - SGD noise, adaptive optimizer, learning-rate schedule, batch normalization이 dynamics를 바꿀 수 있다.

6. Bounds가 tight하다는 보장은 없다.
   - Theorem이 grokking 존재를 보이더라도 observed $t_1$, $t_2$의 exact constant를 완전히 설명하지 않을 수 있다.

7. Non-realizable and noisy labels는 open problem이다.
   - Weight decay가 noise-fitting component와 true null-space component를 어떻게 구분하는지 추가 분석이 필요하다.

8. Feature dimension independence는 제한된 experimental regime의 결과다.
   - Scaling law로 일반화하려면 더 넓은 aspect ratio and normalization sweep이 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

Long training 뒤 late generalization이 나타났을 때 곧바로 emergent circuit or representation phase transition을 가정하기 쉽다. 이 논문은 먼저 더 단순한 diagnosis를 하라고 말한다.

- Training data가 보지 못하는 parameter or feature direction이 있는가?
- Regularization만이 그 direction을 줄이고 있는가?
- Train metric은 그 component에 blind한가?
- Test metric의 threshold가 smooth decay를 sudden jump로 보이게 하는가?
- Weight decay sweep에서 delay가 $1/\lambda$와 비슷하게 움직이는가?

이 check를 통과한 뒤에도 unexplained transition이 남을 때 nonlinear representation mechanism을 논하는 편이 더 설득력 있다.

## 7-2. Reuse potential

### 1) Subspace-aware training diagnostics

- Empirical Jacobian or feature matrix의 dominant span을 추정한다.
- Parameter update and residual을 in-span and out-of-span으로 projection한다.
- Late generalization이 null-space norm 감소와 동기화되는지 본다.

### 2) Weight-decay scaling experiment

- 여러 order의 $\lambda$를 sweep한다.
- Final score뿐 아니라 $t_1$, $t_2$, $t_2-t_1$을 기록한다.
- Delay가 $1/\lambda$ law와 맞는지 본다.

### 3) Metric sensitivity audit

- Continuous loss
- Calibration error
- Multiple threshold accuracy
- Margin distribution

여러 metric을 함께 그려 suddenness가 어디서 생기는지 확인한다.

### 4) Initialization ablation

- Initial norm과 direction을 독립적으로 바꾼다.
- Delay가 logarithmic scale shift를 보이는지 확인한다.

### 5) Regularization schedule design

- Constant small weight decay는 긴 cleanup phase를 만들 수 있다.
- Early fit 이후 weight decay를 높이는 schedule이 delay를 줄이는지 실험할 수 있다.
- 다만 final bias와 optimizer interaction을 함께 평가해야 한다.

## 7-3. Follow-up papers

- Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
- Progress Measures for Grokking via Mechanistic Interpretability
- Omnigrok: Grokking Beyond Algorithmic Data
- Towards Understanding Grokking: An Effective Theory of Representation Learning
- A Mechanistic Interpretability Analysis of Grokking
- Deep Linear Networks for Studying Optimization and Generalization

# 8. Summary

- 이 논문은 overparameterized ridge regression에서 training fit이 먼저 일어나고 population generalization이 늦게 나타나는 grokking을 end-to-end로 증명한다.
- Parameter error를 training-data span과 orthogonal complement로 나누면 training error는 parallel component만 보고 population error는 둘 다 본다.
- Parallel component는 data curvature로 빠르게 감소하지만 orthogonal initialization은 weight decay rate $(1-\eta\lambda)^t$로 천천히 사라진다.
- 이 timescale separation 때문에 grokking delay는 small $\lambda$에서 대략 $1/\lambda$ scale로 증가한다.
- Nonlinear experiment도 비슷한 trend를 보이지만, fixed-feature theorem과 representation-learning mechanism을 동일시해서는 안 된다.
