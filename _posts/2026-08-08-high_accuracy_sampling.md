---
layout: single
title: "High-accuracy sampling for diffusion models and log-concave distributions Review"
categories: Study-concept
tag: [Diffusion-Models, Sampling-Theory, Optimization]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2602.01338)

> 한 줄 요약: 이 논문은 density value를 직접 계산하지 않고 score 또는 gradient query만 사용할 수 있는 상황에서, rejection sampling을 모사하는 First-Order Rejection Sampling, FORS를 제안한다. 이를 diffusion reverse transition과 log-concave proximal sampler에 결합해 target accuracy $\delta$에 대한 query complexity를 기존의 polynomial dependence에서 polylogarithmic dependence로 개선한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Diffusion sampling의 속도를 단순한 step reduction이 아니라 target distribution error에 대한 query complexity로 분석한다.
- Score만 학습하고 density는 알 수 없는 diffusion model의 정보 제약을 정면으로 다룬다.
- Rejection sampling을 first-order oracle만으로 구현하는 FORS라는 재사용 가능한 subroutine을 제안한다.
- Ambient dimension $d$ 대신 data distribution의 intrinsic dimension $d_\star$가 complexity에 나타나는 결과를 제공한다.
- 같은 Gaussian-tilt subproblem이 diffusion reverse kernel과 log-concave proximal sampler를 연결한다.
- Image benchmark나 FID를 보고하는 논문이 아니라, high-accuracy sampling이 가능한 조건과 비용을 증명하는 theory paper다.

Diffusion model의 practical sampler는 보통 적은 network evaluation으로 좋은 sample을 만드는 데 초점을 둔다. 반면 이 논문이 묻는 질문은 더 근본적이다.

> Score function만 질의할 수 있을 때, target error를 $\delta$까지 줄이기 위해 필요한 query 수를 $\mathrm{polylog}(1/\delta)$로 만들 수 있는가?

기존 DDPM convergence analysis는 score estimation error와 discretization error를 통제하면서 target distribution에 접근한다. 그러나 minimal assumption 아래에서는 accuracy가 높아질수록 필요한 step 수가 $1/\delta$, $1/\delta^2$ 같은 polynomial rate로 증가하는 결과가 많았다. Higher-order method도 dependence를 완화하지만, 높은 차수의 smoothness나 problem parameter에 민감해질 수 있다.

이 논문은 방향을 바꾼다. Reverse process를 더 정교하게 discretize하기보다, 각 reverse transition을 Gaussian proposal과 rejection correction의 조합으로 본다. Density ratio를 직접 계산할 수 없다는 문제는 gradient path integral과 Poisson randomization으로 해결한다. 그 결과 score query만으로 Gaussian tilt를 고정밀하게 sample하는 FORS가 핵심 primitive가 된다.

# 1. Problem Setting

## 1-1. Problem definition

Data distribution을 $p_{\mathsf{data}}$라고 하고, sampler의 output distribution을 $\widehat{p}$라고 하자. 논문이 원하는 보장은 다음 형태다.

$$
D(p_{\mathsf{data}}, \widehat{p})
\leq
\delta + C_{\mathsf{apx}}\varepsilon_{\mathsf{score}}.
$$

여기서 각 항의 의미는 다음과 같다.

- $\delta$: sampling algorithm 자체가 만드는 target accuracy
- $\varepsilon_{\mathsf{score}}$: learned score와 true score 사이의 approximation error
- $C_{\mathsf{apx}}$: score error가 final distribution error로 전달되는 정도
- $D$: distribution discrepancy

핵심은 $\delta$를 작게 만들 때 필요한 score query 수다. High-accuracy regime에서는 $\delta$가 절반으로 줄 때 계산량이 몇 배씩 커지는 polynomial dependence와, log factor만 늘어나는 polylogarithmic dependence의 차이가 매우 크다.

Diffusion model은 density $p_t(x)$ 자체보다 score를 학습한다.

$$
s_t^\star(x) = \nabla \log p_t(x).
$$

따라서 sampler가 사용할 수 있는 정보는 보통 다음과 같다.

- Noise sample
- Schedule parameter
- Approximate score $s_t(x)$
- Score의 first-order evaluation

반면 standard rejection sampling이나 Metropolis-Hastings가 요구하는 unnormalized density value 또는 exact density ratio는 직접 얻기 어렵다. 이 정보 제약 아래에서 high-accuracy sampling을 구현하는 것이 논문의 problem setting이다.

논문은 최종 data distribution과 early-stopped distribution을 구분한다. Main theorem은

$$
X_1 \sim
\mathsf{N}(\alpha_0 X_0, \sigma_0^2 I),
\qquad
X_0 \sim p_{\mathsf{data}}
$$

로 정의되는 smoothed distribution $p_1$에 대한 KL guarantee를 먼저 제공한다. 이후 $\sigma_0$를 충분히 작게 선택해 $p_1$과 $p_{\mathsf{data}}$의 차이를 bounded Lipschitz metric에서 통제한다.

## 1-2. Why previous approaches are insufficient

기존 high-accuracy diffusion analysis의 병목은 크게 세 가지다.

### 1) Discretization error가 accuracy dependence를 만든다

Reverse SDE나 ODE를 finite step으로 근사하면 local error가 누적된다. $\delta$를 작게 만들수록 step size를 줄여야 하고, 필요한 score query 수가 $\mathrm{poly}(1/\delta)$로 증가하기 쉽다.

### 2) Density-free setting에서 correction이 어렵다

Proposal sample을 만든 뒤 exact target으로 보정하려면 density ratio가 필요하다. 하지만 diffusion model은 score만 학습하므로 zeroth-order density query를 사용할 수 없다. Score를 적분하면 log-density difference를 얻을 수 있지만, 이를 매 candidate마다 정확하게 계산하면 비용이 크다.

### 3) Dimension dependence가 ambient space를 따라갈 수 있다

Image나 high-dimensional data는 embedding dimension $d$가 매우 크다. 실제 data가 저차원 manifold나 finite support 근처에 있더라도 bound가 $d$에 직접 비례하면 theory가 data geometry를 반영하지 못한다.

Higher-order solver는 local approximation을 개선할 수 있지만, 높은 order의 derivative regularity가 필요하거나 hidden constant가 problem parameter에 민감해질 수 있다. 이 논문은 higher-order discretization 대신, first-order query만으로 exact rejection logic을 모사하는 방향을 택한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 네 단계로 이어진다.

### 1) First-Order Rejection Sampling, FORS

Proposal distribution $q$와 tilt $w$가 있을 때 target을 다음처럼 쓰고 싶다고 하자.

$$
\widehat{p}(x)
\propto
q(x)\exp(w(x)).
$$

Standard rejection sampling은 $w(x)$의 값을 요구한다. FORS는 $w(x)$를 직접 알지 못해도, 조건부 기대값이 $w(x)$인 bounded random estimator $W$를 sample할 수 있으면 된다.

$$
\mathbb{E}[W \mid x] = w(x),
\qquad
W \in [-B,B].
$$

Candidate $x \sim q$를 뽑은 뒤 $J \sim \mathsf{Poisson}(2B)$를 sample하고, $J$개의 independent estimator $W_j$를 만든다. Acceptance probability는 다음과 같다.

$$
\prod_{j=1}^{J}
\frac{B+W_j}{2B}.
$$

Poisson generating function을 사용하면 acceptance probability의 expectation이 $\exp(\mathbb{E}[W \mid x])$에 비례한다. 따라서 density value 없이도 $q(x)\exp(w(x))$로 tilted된 distribution을 sample할 수 있다.

### 2) Gaussian tilt를 first-order query로 sample한다

Diffusion reverse kernel과 proximal sampler에서 공통으로 나타나는 subproblem은 다음 Gaussian tilt다.

$$
\nu(x)
\propto
\exp\left(
-f(x)
-
\frac{\|x-x_0\|^2}{2\eta}
\right).
$$

$f$를 reference point $x_+$에서 first-order expansion하면 natural Gaussian proposal을 얻는다.

$$
q =
\mathsf{N}
\left(
x_0 - \eta \nabla f(x_+),
\eta I
\right).
$$

이 proposal은 target의 linear part와 quadratic Gaussian term을 흡수한다. 남은 nonlinear residual을 gradient path integral로 표현하고, 이를 bounded estimator로 만들어 FORS correction에 넣는다.

### 3) Reverse diffusion step을 Gaussian tilt로 본다

Forward diffusion의 reverse transition은 이전 noisy state의 density와 Gaussian transition factor의 곱으로 표현된다. 즉 각 backward kernel은 Gaussian tilt sampling 문제다.

저자들은 DDPM-like Gaussian proposal을 먼저 만들고, FORS를 corrector로 호출한다. 전체 reverse chain은 다음 형태가 된다.

1. Terminal noise approximation에서 $X_K$를 sample한다.
2. $k=K-1, \ldots, 1$에 대해 score 기반 Gaussian proposal을 정의한다.
3. FORS로 corrected reverse transition에서 $X_k$를 sample한다.
4. Early-stopped output $X_1$을 반환한다.

### 4) 같은 primitive를 log-concave sampling에 재사용한다

Proximal sampler의 restricted Gaussian oracle, RGO도 Gaussian tilt다.

$$
\mathsf{RGO}_{f,\eta,y}(x)
\propto
\exp\left(
-f(x)
-
\frac{\|y-x\|^2}{2\eta}
\right).
$$

따라서 RGO를 FORS로 구현하면 zeroth-order value query 없이 gradient만으로 proximal sampling을 수행할 수 있다. 논문은 이를 통해 general log-concave distribution에 대한 first-order high-accuracy guarantee를 얻는다.

## 2-2. Design intuition

FORS의 설계 직관은 "density difference를 직접 계산하지 말고, unbiased random correction을 exponential weight로 바꾸자"는 것이다.

일반적으로 gradient path integral은 함수값 차이를 준다.

$$
h(x)-h(x_0)
=
\int_0^1
\left\langle
\dot{\gamma}_r(x),
\nabla h(\gamma_r(x))
\right\rangle dr.
$$

Random path parameter를 sample하면 이 integral의 unbiased estimator를 만들 수 있다. 하지만 rejection sampling에는 $\exp(h(x)-h(x_0))$가 필요하다. 단순히 estimator를 exponentiate하면 Jensen bias가 생긴다.

FORS는 Poisson randomization을 사용해 이 문제를 우회한다. Random product의 expectation이 exponentiated conditional mean과 맞도록 설계한다. 즉 first-order estimator에서 density tilt를 직접 simulation한다.

이 논문의 가장 중요한 연결은 diffusion과 log-concave sampling이 모두 "Gaussian proposal을 만들고 nonlinear residual을 보정하는 문제"로 환원된다는 점이다. FORS는 특정 diffusion schedule 전용 trick이 아니라, Gaussian tilt가 나타나는 first-order sampling 문제의 meta-algorithm에 가깝다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Score 또는 gradient query만으로 high-accuracy sampling 수행 |
| Core primitive | First-Order Rejection Sampling, FORS |
| Proposal | First-order Gaussian approximation |
| Correction signal | Gradient path integral로 만든 bounded random estimator |
| Randomization | Poisson variable과 product-form acceptance probability |
| Diffusion application | 각 backward transition을 FORS-corrected Gaussian tilt로 sample |
| Geometry term | Ambient dimension이 아닌 intrinsic dimension $d_\star$ |
| Log-concave application | Proximal sampler의 RGO를 FORS로 구현 |
| Main metric | KL과 bounded Lipschitz discrepancy에 대한 query complexity |

## 3-2. Module breakdown

### 1) FORS acceptance mechanism

FORS의 input은 세 가지다.

- Proposal distribution $q$
- Bound $B$
- 각 candidate $x$에서 $[-B,B]$ 범위의 estimator를 생성하는 distribution $\mathcal{W}_x$

Algorithm은 candidate가 accept될 때까지 반복한다.

1. $x \sim q$
2. $J \sim \mathsf{Poisson}(2B)$
3. $W_1, \ldots, W_J \sim \mathcal{W}_x$
4. 다음 probability로 $x$를 accept

$$
A(x,W_{1:J})
=
\prod_{j=1}^{J}
\frac{B+W_j}{2B}.
$$

Theorem 3.1에 따르면 output density는 다음에 비례한다.

$$
q(x)
\exp\left(
\mathbb{E}[W_1 \mid x]
\right).
$$

한 번의 FORS call에서 필요한 estimator sample 수는 probability $1-\delta$로

$$
O\left(Be^{2B}\log(1/\delta)\right)
$$

이고, $T$번 call하면 total count는

$$
O\left(Be^{2B}(T+\log(1/\delta))\right)
$$

로 제어된다. 따라서 $B$를 constant order로 유지하는 것이 중요하다.

### 2) Gaussian proposal and residual correction

Target Gaussian tilt는

$$
\nu(x)
\propto
\exp\left(
-f(x)
-
\frac{\|x-x_0\|^2}{2\eta}
\right)
$$

이고, proposal은

$$
q(x)
=
\mathsf{N}
\left(
x_0-\eta\nabla f(x_+),
\eta I
\right)
$$

로 둔다. Proposal이 local linear behavior를 흡수하면 target-proposal log ratio는 gradient difference의 path integral로 표현할 수 있다.

논문은 path를 일반화해 Lipschitz gradient뿐 아니라 Holder-continuous gradient도 다룬다. Estimator가 너무 큰 경우에는 $[-B,B]$로 clip한다. 이 clipping은 exact unbiasedness를 깨뜨릴 수 있으므로, proof에서는 clipping event와 approximation error를 따로 제어한다.

### 3) Backward diffusion sampler

Diffusion algorithm의 input은 다음과 같다.

- Time별 score estimate $\{s_k\}_{k=1}^{K}$
- Terminal distribution approximation $\widehat{p}_K$
- Schedule $\{\alpha_k, \eta_k\}_{k=1}^{K}$

Terminal sample을 만든 뒤 모든 reverse step에서 FORS를 호출한다.

$$
X_K \sim \widehat{p}_K,
$$

$$
X_k
\leftarrow
\mathrm{FORS}
(B,q_k,\mathcal{W}_x^k),
\qquad
k=K-1,\ldots,1.
$$

$ q_k $는 current noisy state와 score estimate를 평균에 반영한 DDPM-like Gaussian proposal이다. $\mathcal{W}_x^k$는 true backward kernel과 proposal 사이의 residual tilt를 추정한다.

### 4) Intrinsic dimension

논문은 data geometry를 covering number로 정의한다. Distribution $p$의 support를 radius $r$의 ball로 덮는 최소 개수를 $N(p;r)$라고 하면,

$$
\dim_{\sigma^2}(p)
=
1 \vee
\inf_{r \geq 0}
\left(
\log N(p;r)
+
\frac{r^2}{\sigma^2}
\right)
\wedge d.
$$

이를 이용해 data intrinsic dimension $d_\star$를 정의한다. 항상 $d_\star \leq d$이며, 다음 구조를 반영할 수 있다.

- Compact low-dimensional manifold에 집중된 distribution
- 적은 수의 cluster나 support point를 가진 distribution
- Ambient coordinate는 크지만 effective geometry는 낮은 distribution

예를 들어 support가 $N$개 point 이하이면 $d_\star \leq \log N$이다. 따라서 complexity가 raw dimension보다 data support의 effective complexity를 따라갈 수 있다.

### 5) Error decomposition

Theorem 4.3의 KL bound는 다음 구조를 가진다.

$$
D_{\mathsf{KL}}
(p_1 \| \widehat{p}_1)
\lesssim
D_{\mathsf{KL}}
(p_K \| \widehat{p}_K)
+
K\delta
+
\sum_{k=1}^{K}
\eta_k
\varepsilon_{k,\mathsf{score}}^2.
$$

세 항은 각각 다음을 의미한다.

- Terminal initialization error
- Step별 FORS approximation과 failure budget의 누적
- Score estimation error의 schedule-weighted 누적

이 decomposition의 장점은 sampling algorithm error와 learned score error를 분리한다는 점이다. Sampler step을 더 늘려도 score model이 부정확하면 마지막 항이 남는다.

### 6) Log-concave proximal sampler

Proximal sampler는 두 conditional update를 반복한다.

$$
Y_n \sim
\mathsf{N}(X_n,\eta I),
$$

$$
X_{n+1}
\sim
\mathsf{RGO}_{f,\eta,Y_n}.
$$

두 번째 conditional이 바로 Gaussian tilt다. FORS로 RGO를 구현하면 $f(x)$ value를 조회하지 않고 $\nabla f(x)$만 사용해 high-accuracy correction을 수행할 수 있다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 neural diffusion model을 새로 학습하거나 image dataset에서 benchmark하는 논문이 아니다. 따라서 conventional한 training data section은 없다.

Analysis가 요구하는 data-side assumption은 다음과 같다.

- Minimal theorem에서는 $p_{\mathsf{data}}$의 finite second moment만 가정한다.
- Second moment는

$$
M_2^2
=
\mathbb{E}_{X_0 \sim p_{\mathsf{data}}}
\|X_0\|^2
$$

로 정의한다.
- Refined result에서는 denoiser 또는 score의 non-uniform smoothness condition이 추가된다.
- Log-concave application에서는 log-concavity, Poincare inequality, log-Sobolev inequality, warm-start quality 등에 따라 서로 다른 result가 나온다.

## 4-2. Training strategy

학습 recipe 대신 oracle model을 명확히 봐야 한다.

### Score oracle

각 diffusion level에서 approximate score $s_k$를 사용할 수 있다고 가정한다. Error는 pointwise worst case가 아니라 $L^2$ 평균 error로 측정한다. Main result는 대략 $\widetilde{O}(\delta)$ 수준의 score accuracy를 요구한다.

### First-order oracle

Gaussian tilt와 log-concave target에서는 $\nabla f(x)$를 query할 수 있다. $f(x)$ 자체를 조회하는 zeroth-order oracle은 요구하지 않는다.

### Schedule

Step size와 noise schedule은 FORS estimator가 bounded regime에 머물도록 선택해야 한다. Theorem에는 $\sigma_k^2/\eta_k$가 intrinsic dimension, $\log(1/\delta)$, smoothness parameter보다 충분히 큰 조건이 들어간다.

### Early stopping

Main KL theorem은 smoothed target $p_1$을 대상으로 한다. Original $p_{\mathsf{data}}$와의 gap은 bounded Lipschitz metric에서 noise scale $\sigma_0$를 통해 제어한다. Log-smoothness를 추가로 가정하면 final DDPM step을 붙여 KL guarantee를 original data distribution까지 확장하는 결과도 제시한다.

## 4-3. Engineering notes

Theory를 실제 neural sampler로 옮길 때는 다음 항목이 병목이 될 수 있다.

1. FORS acceptance rate
   - Complexity에는 $e^{2B}$가 들어간다.
   - Proof에서는 $B=\Theta(1)$로 유지하지만, practical estimator의 scale이 커지면 rejection overhead가 급격히 증가할 수 있다.

2. Gradient path estimator variance
   - 어떤 path와 reference point를 선택하는지에 따라 estimator variance와 clipping frequency가 달라진다.
   - Theory의 general path construction을 efficient tensor operation으로 구현하는 문제가 남는다.

3. Score evaluation batching
   - FORS는 candidate별 random number of first-order query를 요구한다.
   - GPU batch에서 sample마다 acceptance loop 길이가 다르면 utilization이 낮아질 수 있다.

4. Prox-like center computation
   - Gaussian proposal의 quality는 $x_+$ 또는 denoiser-based center에 달려 있다.
   - 이 center를 구하는 비용이 query count에는 단순화되어도 wall-clock에서는 중요할 수 있다.

5. Query complexity와 runtime의 차이
   - Score query 수가 적어도 control flow, rejection, memory access, random path generation 비용이 클 수 있다.
   - 실용성을 판단하려면 FLOPs, latency, acceptance rate, batch efficiency를 별도로 측정해야 한다.

논문은 implementation과 empirical evaluation을 future work로 남긴다. 따라서 현재 결과는 immediately deployable sampler recipe보다 theoretical possibility result로 읽는 것이 정확하다.

# 5. Evaluation

## 5-1. Main results

이 논문에는 FID, ImageNet, CIFAR-10 같은 empirical benchmark가 없다. Evaluation의 단위는 theorem의 error guarantee와 oracle query complexity다.

### 1) Minimal-assumption diffusion result

Finite second moment 외의 data assumption을 두지 않을 때, bounded Lipschitz error는 다음 구조로 제어된다.

$$
D_{\mathsf{BL}}^2
(p_{\mathsf{data}},\widehat{p}_1)
\lesssim
\delta^2
+
\sum_{k=1}^{K}
\eta_k
\varepsilon_{k,\mathsf{score}}^2.
$$

Total query complexity는 다음과 같다.

$$
O\left(
d_\star
\log^3
\left(
\frac{d+M_2^2}{\delta^2}
\right)
\right).
$$

Target accuracy dependence가 logarithmic power로만 증가한다는 점이 핵심이다. Data dependence도 $d$가 아니라 $d_\star$를 통해 나타난다.

### 2) Refined smoothness result

Denoiser Jacobian의 Frobenius norm이 high probability에서 $L_{\mathrm{F},\delta}$로 제어되는 non-uniform condition 아래에서는 complexity가 다음 형태로 줄어든다.

$$
O\left(
L_{\mathrm{F},\delta}
\log^3
\left(
\frac{d+M_2^2}{\delta^2}
\right)
\right).
$$

Operator-norm smoothness와 intrinsic dimension을 결합한 corollary에서는 leading term이 다음처럼 표현된다.

$$
\min\left\{
\sqrt{dL_{\mathrm{op}}},
 d_\star^{2/3}L_{\mathrm{op}}^{1/3}
\right\}
\mathrm{polylog}(1/\delta).
$$

이 결과는 worst-case ambient dimension보다 score geometry와 data geometry가 함께 complexity를 결정할 수 있음을 보여준다.

### 3) Log-concave sampling

FORS를 proximal sampler의 RGO에 넣으면 first-order query만 사용하는 high-accuracy sampler를 얻는다. 구체적인 complexity는 smoothness, Poincare 또는 log-Sobolev condition, initialization divergence에 따라 달라진다.

논문의 핵심 claim은 모든 log-concave setting에서 dimension과 condition number가 사라진다는 것이 아니다. Target accuracy $\delta$에 대한 dependence를 polylogarithmic하게 만들면서, density value query 없이 gradient oracle만 사용한다는 데 있다.

## 5-2. What really matters in the experiments

### 1) Polylog result의 target을 정확히 봐야 한다

Main KL guarantee는 early-stopped distribution $p_1$에 대해 먼저 성립한다. Original data distribution으로 가려면 smoothing bias를 추가로 통제해야 한다. Bounded Lipschitz metric은 weak convergence를 metrize하지만, TV나 perceptual quality와 같은 의미는 아니다.

### 2) Score accuracy requirement가 사라진 것은 아니다

Sampler error가 polylog step으로 줄어도 learned score error term은 남는다.

$$
\sum_k
\eta_k
\varepsilon_{k,\mathsf{score}}^2
$$

가 충분히 작아야 final error도 작다. High-accuracy sampling theorem은 inaccurate score를 자동으로 고치는 결과가 아니다.

### 3) Query complexity는 system speed와 다르다

FORS의 random rejection loop, path integral estimator, candidate별 variable workload는 GPU에서 비효율적일 수 있다. 이론적으로 score call 수가 줄어도 wall-clock latency가 줄어든다는 보장은 없다.

### 4) Intrinsic dimension은 강점이면서 추상적 quantity다

$d_\star$가 ambient dimension보다 작을 수 있다는 점은 의미가 크다. 하지만 실제 image distribution에서 해당 covering-number quantity를 추정하거나 theorem의 constant를 calibration하는 일은 쉽지 않다.

### 5) Empirical evidence가 아직 없다

논문 스스로 implementation과 experimental evaluation을 future work로 둔다. 따라서 practical sampler와 비교한 sample quality, rejection rate, memory, latency, large neural score model에서의 numerical stability는 열린 문제다.

# 6. Limitations

1. Primarily theoretical result다
   - Neural diffusion model과 실제 dataset에서 FORS를 구현한 결과가 없다.
   - Complexity improvement가 real GPU speedup으로 이어지는지 확인되지 않았다.

2. Early-stopped target과 original data distribution을 구분해야 한다
   - Main KL theorem은 smoothed distribution $p_1$에 대한 것이다.
   - $p_{\mathsf{data}}$까지의 guarantee는 bounded Lipschitz metric, small noise choice, 추가 smoothness에 따라 달라진다.

3. Oracle assumption이 practical training error와 바로 대응되지 않는다
   - Time별 $L^2$ score error를 충분히 작게 통제한다고 가정한다.
   - 실제 score network의 tail error, out-of-distribution state, finite precision은 theorem에 직접 반영되지 않을 수 있다.

4. Rejection constant가 practical overhead를 만들 수 있다
   - Query bound에 $e^{2B}$가 포함된다.
   - Estimator scale과 clipping을 constant regime에 유지하는 구현이 중요하다.

5. Bounded Lipschitz metric의 해석은 제한적이다
   - Weak convergence에는 적합하지만 image fidelity나 rare-mode coverage를 직접 의미하지 않는다.
   - 다른 divergence에서 같은 minimal-assumption polylog result가 성립하는 것은 아니다.

6. Intrinsic dimension은 직접 관측하기 어렵다
   - Covering-number definition은 theory에 유용하지만 real data에서 estimation하기 어렵다.
   - Bound가 작더라도 hidden constant와 schedule constraint가 practical cost를 좌우할 수 있다.

7. Log-concave result도 condition-free가 아니다
   - Smoothness, isoperimetry, warm start, Poincare 또는 log-Sobolev constant에 따라 complexity가 달라진다.
   - "Gradient만으로 polylog accuracy"라는 headline과 각 theorem의 전제조건을 함께 읽어야 한다.

# 7. My Take

## 7-1. Why this matters for my work

Diffusion sampler를 평가할 때 흔히 NFE와 sample metric만 본다. 이 논문은 그 아래의 세 가지 error source를 분리하게 한다.

- Score approximation error
- Reverse process discretization error
- Exact target correction을 하지 못해 생기는 bias

FORS는 세 번째 항을 first-order query만으로 보정하는 방향을 제시한다. 이 관점은 image diffusion뿐 아니라 score-based posterior sampling, energy-based inference, Bayesian computation, constrained generation에도 연결될 수 있다.

특히 method design에서 다음 질문이 유용하다.

- Learned network가 제공하는 정보는 density인가, score인가?
- Proposal이 target의 어느 부분까지 흡수하는가?
- 남은 residual을 deterministic integration과 randomized correction 중 무엇으로 처리할 것인가?
- Theoretical query count와 accelerator-friendly execution 사이의 gap은 얼마나 큰가?
- Accuracy를 높일 때 model error와 solver error 중 무엇이 먼저 floor를 만드는가?

## 7-2. Reuse potential

### 1) Predictor-corrector sampler design

Fast approximate proposal을 만든 뒤 FORS-like first-order corrector를 붙이는 구조를 검토할 수 있다. Corrector를 모든 step에 적용하지 않고 high-error region이나 마지막 noise level에 선택적으로 적용하는 hybrid design도 가능하다.

### 2) Score error budget allocation

Error bound가 $\sum_k \eta_k\varepsilon_k^2$로 나타나므로, 모든 time step에 같은 score accuracy를 요구할 필요는 없다. Schedule weight가 큰 구간에 training capacity나 evaluation budget을 더 배분할 수 있다.

### 3) Intrinsic-dimension-aware analysis

Model input dimension 대신 representation manifold, codebook support, latent dimension을 기준으로 sampler complexity를 분석하는 방향을 제공한다. Latent diffusion에서는 pixel dimension보다 latent support geometry가 더 적절할 수 있다.

### 4) Rejection diagnostics

실제 구현에서는 acceptance rate, clipping rate, estimator variance, candidate별 query count를 logging해야 한다. 평균 NFE만 보면 tail latency와 unstable region을 놓칠 수 있다.

### 5) Theory-to-system validation protocol

이 논문의 실용성을 검증하려면 다음 순서가 적절하다.

1. Low-dimensional synthetic target에서 theorem quantity와 measured error를 비교한다.
2. Analytic score를 가진 Gaussian mixture에서 FORS correction을 검증한다.
3. Small neural score model에서 batch efficiency와 rejection overhead를 측정한다.
4. Latent diffusion에 적용해 FID뿐 아니라 distributional discrepancy와 NFE를 함께 본다.
5. Same score network 아래 DDPM, higher-order solver, FORS hybrid를 비교한다.

## 7-3. Follow-up papers

- Denoising Diffusion Probabilistic Models
- Score-Based Generative Modeling through Stochastic Differential Equations
- Convergence of Score-Based Generative Modeling for General Data Distributions
- Faster High-Accuracy Log-Concave Sampling via Algorithmic Warm Starts
- Sublinear Iterations Can Suffice Even for DDPMs
- Faster Diffusion Models via Higher-Order Approximation
- Linear Convergence of Diffusion Models under the Manifold Hypothesis

# 8. Summary

- FORS는 tilt value를 직접 계산하지 않고 bounded first-order estimator와 Poisson randomization으로 rejection sampling을 모사한다.
- Diffusion reverse transition과 proximal sampler의 RGO는 모두 Gaussian tilt subproblem으로 환원된다.
- Minimal second-moment assumption 아래 bounded Lipschitz error $\delta$를 intrinsic dimension에 비례하고 $\log^3(1/\delta)$에 의존하는 query 수로 달성한다.
- Refined smoothness condition에서는 dimension dependence를 더 줄이고, log-concave sampling에도 first-order high-accuracy guarantee를 제공한다.
- 현재 결과는 strong theoretical advance이지만, neural sampler에서 acceptance rate, batch efficiency, wall-clock speed, sample quality를 검증하는 empirical work가 필요하다.
