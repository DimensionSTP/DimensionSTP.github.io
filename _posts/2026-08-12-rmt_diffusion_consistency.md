---
layout: single
title: "A Random Matrix Theory Perspective on the Consistency of Diffusion Models Review"
categories: Study-concept
tag: [Diffusion, Random-Matrix-Theory, Generative-Models]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2602.02908)

[Project page](https://animadversio.github.io/diffusion-consistency-rmt)

> 한 줄 요약: 이 논문은 서로 겹치지 않는 training split과 다른 architecture로 학습한 diffusion model이 같은 noise seed에서 비슷한 sample을 만드는 현상을 Gaussian mean and covariance가 결정하는 linear sampling map으로 설명하고, finite data가 noise scale을 $\sigma^2 \mapsto \kappa(\sigma^2)$로 renormalize해 overshrinkage와 cross-split fluctuation을 만든다는 random matrix theory를 제시한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Diffusion model의 cross-run consistency를 단순한 empirical curiosity가 아니라 data spectrum과 finite-sample statistics의 문제로 정식화한다.
- 같은 noise가 같은 semantic structure로 이어지는 이유 중 상당 부분을 deep feature가 아니라 Gaussian mean and covariance만으로 설명한다.
- Dataset size가 작을 때 low-variance mode가 과도하게 shrink되어 average-looking sample이 생기는 mechanism을 수식으로 보여준다.
- Cross-split disagreement를 anisotropy, inhomogeneity, global sample-size scaling으로 분해한다.
- Linear theory가 어디까지 UNet과 DiT를 설명하고, memorization regime에서 어디서 깨지는지 함께 보여준다.

Diffusion model을 여러 번 학습하면 weight는 완전히 다르다. Training data split도 겹치지 않을 수 있고, UNet과 DiT처럼 architecture도 다를 수 있다. 그런데 deterministic sampler에서 같은 initial noise seed를 넣으면 generated image의 pose, layout, coarse identity가 놀랄 만큼 비슷한 경우가 있다.

이 현상은 직관적으로 이상하다. GAN이나 VAE의 latent space는 rotation ambiguity가 있어 같은 latent coordinate가 run마다 다른 semantic을 가질 수 있다. 반면 diffusion model은 같은 Gaussian noise coordinate가 training run을 넘어 일정한 의미를 유지하는 것처럼 보인다.

논문은 이 consistency의 상당 부분이 high-order semantic representation보다 더 단순한 통계에서 나온다고 주장한다.

- Independent split이 같은 population mean을 공유한다.
- Independent split이 비슷한 covariance spectrum과 eigenstructure를 공유한다.
- Diffusion score의 lowest-order approximation은 이 Gaussian statistics로 결정되는 linear vector field다.
- Deterministic sampling map은 이 shared spectral structure를 따라간다.

Random matrix theory, RMT는 finite dataset의 empirical covariance가 population covariance에서 어떻게 흔들리는지를 계산한다. 이를 diffusion denoiser와 full sampling trajectory에 적용하면, 평균 sample이 어떤 bias를 갖는지와 split 사이 variance가 어디서 커지는지를 예측할 수 있다.

# 1. Problem Setting

## 1-1. Problem definition

Target data distribution을 $p_0(\mathbf{x})$라고 하자. Noise scale $\sigma$에서 noised distribution은 Gaussian convolution으로 정의된다.

$$
p(\mathbf{x};\sigma)
= p_0 * \mathcal{N}(0,\sigma^2 I)
$$

Diffusion model은 각 noise scale에서 score field를 학습한다.

$$
\mathbf{s}(\mathbf{x},\sigma)
= \nabla_{\mathbf{x}} \log p(\mathbf{x};\sigma)
$$

EDM probability-flow ODE에서는 다음 dynamics로 noise에서 data sample로 이동한다.

$$
\frac{d\mathbf{x}}{d\sigma}
= -\sigma \nabla_{\mathbf{x}} \log p(\mathbf{x};\sigma)
$$

같은 initial noise $\mathbf{x}_{\sigma_T}$를 서로 다른 training split으로 학습한 model에 넣었을 때 final sample $\mathbf{x}_0$가 비슷하다면, sampling map 자체가 dataset realization에 대해 stable하다는 뜻이다.

논문이 묻는 질문은 다음과 같다.

| Question | Meaning |
| --- | --- |
| Why are samples consistent? | 같은 distribution의 independent split이 왜 같은 noise를 비슷한 sample로 보내는가 |
| What does finite data change? | Empirical covariance estimation error가 denoiser와 sample을 어떻게 bias하는가 |
| Where does disagreement concentrate? | 어떤 eigenmode와 어떤 noise seed가 split에 더 민감한가 |
| Does linear theory explain deep networks? | Gaussian linear denoiser의 예측이 UNet과 DiT에도 남는가 |
| When does the theory fail? | Memorization과 nonlinear feature learning이 dominant한 regime은 어디인가 |

## 1-2. Why previous approaches are insufficient

### 1) Visual similarity만으로 mechanism을 알 수 없다

Same-seed image pair가 닮았다는 관찰은 강하지만, 이유는 여러 가지일 수 있다.

- Architecture inductive bias
- Dataset overlap
- Training seed correlation
- Sampler artifact
- Memorization
- Shared low-order statistics

논문은 non-overlapping split, cross-architecture comparison, nearest-neighbor analysis, Gaussian linear predictor를 결합해 가능성을 분해한다.

### 2) Population-level theory는 finite dataset variation을 놓친다

Infinite-data score는 distribution에 의해 uniquely 정해진다. 하지만 실제 model은 finite sample에서 empirical mean과 covariance를 본다.

Population covariance $\Sigma$와 empirical covariance $\hat{\Sigma}$의 차이가 small element-wise noise처럼 보여도, high-dimensional setting에서는 spectrum과 inverse operator에 체계적인 bias를 만든다.

Diffusion denoiser에는 다음과 같은 resolvent가 들어간다.

$$
\hat{\Sigma}(\hat{\Sigma}+\sigma^2 I)^{-1}
$$

High dimension에서는 이 matrix를 단순히 $\Sigma(\Sigma+\sigma^2 I)^{-1}$로 치환할 수 없다. Finite-sample correction이 필요하다.

### 3) Denoiser 한 step만 보면 full sample consistency를 설명하지 못한다

Diffusion generation은 여러 noise scale의 vector field를 integration한다. 각 scale에서 small bias가 누적되어 final sampling map을 만든다.

따라서 denoiser expectation뿐 아니라 matrix fractional power가 포함된 full trajectory를 분석해야 한다.

### 4) Global MSE는 disagreement structure를 숨긴다

두 sample의 pixel MSE가 같아도 error가 어디에 있는지는 다를 수 있다.

- Dominant eigenmode에서 큰 structural difference가 날 수 있다.
- Low-variance mode의 fine detail만 달라질 수 있다.
- 특정 initial noise seed가 split variation에 더 민감할 수 있다.

논문은 spectral anisotropy와 input-wise inhomogeneity를 분리한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 contribution은 다섯 가지다.

### 1) Linear origin of consistency

FFHQ32를 서로 겹치지 않는 30k split 두 개로 나누고, 각 split에 UNet과 DiT를 학습한다. 같은 noise seed에서 나온 sample은 split과 architecture를 넘어 비슷하다.

더 놀라운 점은 각 split의 empirical mean과 covariance만 사용하는 Gaussian linear predictor도 비슷한 coarse sample을 만든다는 것이다.

Generated sample은 training nearest neighbor보다 cross-split generated sample에 더 가깝고, linear predictor에 가까운 sample일수록 split 간에도 더 consistent하다. 논문은 이 관계에서 Pearson correlation $r=0.244$, $p=5 \times 10^{-15}$를 보고한다.

### 2) Renormalized noise scale

RMT deterministic equivalence는 empirical covariance resolvent를 population covariance와 effective regularization으로 치환한다.

$$
\hat{\Sigma}(\hat{\Sigma}+\lambda I)^{-1}
\asymp
\Sigma(\Sigma+\kappa(\lambda)I)^{-1}
$$

여기서 $\kappa(\lambda)$는 scalar self-consistency equation을 만족한다.

$$
\kappa(\lambda)-\lambda
= \gamma \kappa(\lambda)
\operatorname{tr}
\left[
\Sigma(\Sigma+\kappa(\lambda)I)^{-1}
\right]
$$

$\gamma=d/n$은 data dimension과 sample count의 ratio다.

Diffusion에서는 $\lambda=\sigma^2$이므로, finite data는 raw noise variance $\sigma^2$를 더 큰 effective noise $\kappa(\sigma^2)$로 바꾸는 것처럼 작동한다.

### 3) Overshrinkage

Population eigenvalue를 $s$라고 하면 ideal shrinkage factor는 대략 다음 형태다.

$$
\frac{s}{s+\sigma^2}
$$

Finite-data deterministic equivalent에서는 다음처럼 바뀐다.

$$
\frac{s}{s+\kappa(\sigma^2)}
$$

$\kappa(\sigma^2)>\sigma^2$인 regime에서는 signal이 더 강하게 shrink된다. 특히 small-eigenvalue direction이 dataset mean으로 과도하게 눌린다.

그 결과 limited-data sample은 다음 성질을 보인다.

- Coarse mean structure가 강하다.
- Texture와 background variation이 줄어든다.
- Mid-to-low spectral mode variance가 부족하다.
- Fine detail consistency를 얻으려면 더 많은 sample이 필요하다.

### 4) Factorized fluctuation law

Independent data split 사이 fluctuation은 세 요소로 읽을 수 있다.

1. Anisotropy
   - Probe direction이 covariance eigenbasis에서 어디에 놓이는가.

2. Inhomogeneity
   - Initial noise가 어떤 spectral direction에 얼마나 정렬되는가.

3. Global scaling
   - Dataset size $n$과 aspect ratio $\gamma=d/n$가 전체 variance를 얼마나 줄이는가.

이 decomposition은 평균 consistency만이 아니라 어떤 sample과 어떤 direction이 불안정한지를 예측한다.

### 5) Fractional-power deterministic equivalence

Linear diffusion sampling map에는 $(\hat{\Sigma}+\sigma^2 I)^{1/2}$ 같은 matrix fractional power가 들어간다.

논문은 Balakrishnan-style integral representation을 사용해 fractional matrix power를 resolvent integral로 바꾸고, RMT deterministic equivalence를 full sampling trajectory까지 확장한다.

이론적으로 가장 새로운 부분은 denoiser 한 scale을 넘어 entire deterministic sampling map의 expectation과 variance를 다룬다는 점이다.

## 2-2. Design intuition

이 논문의 직관을 아주 단순하게 줄이면 다음과 같다.

Diffusion model은 high noise에서 image의 fine detail을 아직 만들지 않는다. 이 구간에서는 data distribution의 mean, covariance, dominant principal component 같은 low-order structure가 score field를 크게 결정한다.

Independent split은 individual image는 다르지만 같은 population에서 왔으므로 dominant statistics가 비슷하다. Deterministic sampler는 같은 initial noise를 이 shared direction을 따라 움직인다. 그래서 pose, global layout, coarse appearance가 비슷해진다.

Finite data에서는 empirical covariance가 low-variance direction을 정확히 추정하지 못한다. RMT 관점에서는 이 uncertainty가 effective noise를 높인다. Model은 uncertain direction을 더 강하게 shrink하고 mean 쪽으로 당긴다.

따라서 consistency와 bias는 같은 mechanism의 두 면이다.

- Shared dominant statistics는 split 간 consistency를 만든다.
- Limited samples는 uncertain mode를 overshrink해 detail을 줄인다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Phenomenon | Independent data split과 architecture가 같은 noise에서 유사한 sample 생성 |
| Analytic model | Gaussian data + optimal affine linear denoiser |
| Random object | Empirical covariance $\hat{\Sigma}$ |
| Main RMT tool | Deterministic equivalence and resolvent analysis |
| Expectation result | $\sigma^2$가 $\kappa(\sigma^2)$로 renormalize |
| Bias | Low-variance mode overshrinkage toward mean |
| Variance result | Anisotropy x inhomogeneity x dataset-size scaling |
| Full trajectory | Fractional matrix power를 integral representation으로 분석 |
| Deep validation | EDM UNet and DiT across data sizes and datasets |

## 3-2. Module breakdown

### 1) Optimal linear denoiser

Population mean을 $\mu$, empirical mean을 $\hat{\mu}$, empirical covariance를 $\hat{\Sigma}$라고 하자.

Gaussian noise가 섞인 input $\mathbf{x}$에 대한 optimal affine denoiser는 다음과 같다.

$$
D^*_{\hat{\Sigma}}(\mathbf{x};\sigma)
= \hat{\mu}
+ (\hat{\Sigma}+\sigma^2 I)^{-1}
\hat{\Sigma}(\mathbf{x}-\hat{\mu})
$$

Covariance eigenbasis에서 보면 각 direction을 eigenvalue 크기에 따라 통과시키는 Wiener filter다.

- Large eigenvalue: signal direction으로 보고 많이 보존한다.
- Small eigenvalue: noise에 가깝다고 보고 mean 쪽으로 shrink한다.

### 2) Linear sampling map

Optimal linear denoiser를 probability-flow ODE에 넣으면 trajectory를 closed form으로 풀 수 있다.

$$
\mathbf{x}_{\hat{\Sigma}}(\sigma)
= \hat{\mu}
+ (\hat{\Sigma}+\sigma^2 I)^{1/2}
(\hat{\Sigma}+\sigma_T^2 I)^{-1/2}
(\mathbf{x}_{\sigma_T}-\hat{\mu})
$$

$\sigma \to 0$에서 final sampling map이 나온다.

이 식은 same initial noise가 covariance eigenbasis와 eigenvalue scaling을 통해 sample structure로 변환된다는 것을 보여준다.

### 3) Deterministic equivalent

High-dimensional limit에서 random empirical covariance를 deterministic population surrogate로 바꾼다.

핵심은 matrix 자체가 element-wise로 가까워진다는 뜻이 아니다. Bounded probe vector나 normalized trace로 본 observable이 deterministic expression에 수렴한다는 뜻이다.

이 distinction이 중요하다. Sample covariance의 individual eigenvalue와 eigenvector는 noisy할 수 있지만, denoiser가 사용하는 aggregate operator는 predictable할 수 있다.

### 4) Renormalization equation

$\kappa$는 population spectrum과 aspect ratio를 이용한 self-consistency equation으로 정해진다.

- $n$이 커져 $\gamma=d/n$이 작아지면 $\kappa(\sigma^2)$는 $\sigma^2$에 가까워진다.
- $n$이 작고 data dimension이 크면 renormalization이 강해진다.
- Low-noise scale에서 차이가 가장 크다.

Low-noise 구간은 fine detail을 만드는 단계이므로 limited-data effect가 detail consistency에 강하게 나타난다.

### 5) Variance probes

두 independent dataset realization에서 learned operator가 얼마나 달라지는지를 probe vector $\mathbf{v}$와 input $\mathbf{x}$에 대해 계산한다.

Variance expression은 구조적으로 다음 질문에 답한다.

- 어떤 output direction $\mathbf{v}$가 noisy한가.
- 어떤 input noise seed $\mathbf{x}_{\sigma_T}$가 split-sensitive한가.
- $n$이 증가할 때 disagreement가 어떤 rate로 줄어드는가.

### 6) Deep network validation

Theory는 linear denoiser에 대해 exact asymptotic prediction을 만든다. 이후 UNet과 DiT에서 qualitative pattern을 확인한다.

Validation target은 deep network의 exact pixel output을 예측하는 것이 아니다.

- Dataset size가 커질수록 consistency가 증가하는가.
- Limited data에서 overshrinkage가 나타나는가.
- Disagreement가 eigenmode별로 anisotropic한가.
- Seed별 disagreement를 population covariance만으로 예측할 수 있는가.
- Memorization regime에서는 correlation이 무너지는가.

# 4. Training / Data / Recipe

## 4-1. Data

Deep network experiment는 다음 dataset을 사용한다.

- FFHQ32 and FFHQ64
- AFHQ32
- CIFAR10 and CIFAR100
- LSUN Church 32 and 64
- LSUN Bedroom 32 and 64

각 dataset에서 non-overlapping split 두 개를 만든다. Dataset size는 다음 다섯 단계다.

$$
n \in \{300, 1000, 3000, 10000, 30000\}
$$

Architecture당 size별 두 split이므로 한 dataset에서 10개 training run이 구성된다.

## 4-2. Training strategy

### Common optimization

- Framework: EDM
- Objective: Denoising score matching
- Optimizer: Adam
- Learning rate: $10^{-4}$
- Training steps: 50,000
- Batch size: 256

### DiT

- Hidden size: 384
- Depth: 6 layers
- Attention heads: 6
- MLP ratio: 4
- Patch size: mainly 2

### UNet

- Base channels: 128
- Channel multipliers: 1, 2, 2, 2
- Self-attention at resolution 8

### Evaluation

- Same fixed noise seeds across split and architecture
- Heun deterministic sampler
- Sampling steps: 35
- Evaluation samples: 1,000
- Evaluation batch size: 512

### Motivating experiment

FFHQ32에서는 30k non-overlapping split 두 개에 UNet과 DiT를 학습하고, 512 initial noise에 대한 pairwise MSE를 비교한다.

Generated sample pair, linear predictor, training nearest neighbor를 함께 비교해 consistency가 memorization보다 shared map에 가깝다는 근거를 만든다.

## 4-3. Engineering notes

### 1) Same-seed pairing이 필수다

Distribution metric만 보면 sampling-map alignment를 알 수 없다. 이 논문은 같은 noise seed를 model pair에 넣고 pointwise output을 비교한다.

Reproduction에서는 seed, sampler, noise schedule, numerical precision을 고정해야 한다.

### 2) Deterministic sampler를 써야 한다

Stochastic sampler는 model difference와 sampling randomness를 섞는다. Cross-run mapping consistency를 보려면 probability-flow ODE나 deterministic solver가 적합하다.

### 3) Training nearest neighbor control이 필요하다

Cross-split sample이 닮았다는 사실만으로 memorization을 배제할 수 없다. 각 generated sample과 training/control split nearest neighbor distance를 비교해야 한다.

### 4) Population statistics estimation을 분리한다

Theory prediction에는 mean, covariance, spectrum이 필요하다. High-dimensional image covariance는 memory와 numerical stability 문제가 있으므로 PCA/eigendecomposition implementation을 명시해야 한다.

### 5) Pixel MSE와 perceptual metric을 함께 생각해야 한다

논문은 spectral theory와 직접 연결되는 pixel-space MSE를 중심으로 사용한다. 그러나 perceptual similarity와 semantic consistency는 pixel MSE와 다를 수 있다.

### 6) Memorization regime를 별도 표시한다

Linear theory는 individual training point memorization을 표현할 수 없다. $n \leq 1000$처럼 nearest-neighbor gap이 큰 구간을 theory fit에서 섞으면 conclusion이 왜곡된다.

# 5. Evaluation

## 5-1. Main results

### 1) Independent split과 architecture 사이에 strong same-seed consistency가 있다

FFHQ32의 30k split에서 UNet1, UNet2, DiT1, DiT2 sample은 같은 seed를 공유할 때 시각적으로 유사하다.

Pairwise pixel MSE에서도 generated model pair와 linear predictor가 low-distance block을 만든다. Generated sample은 training set nearest neighbor보다 다른 split의 generated sample에 더 가깝다.

이는 simple sample memorization만으로 설명하기 어렵다.

### 2) Gaussian linear predictor가 consistency의 상당 부분을 설명한다

Empirical mean과 covariance로 만든 Wiener-filter sampling map은 split 간 거의 같은 output을 만들고, deep model sample의 coarse structure도 예측한다.

Linear predictor에 더 가까운 DNN sample일수록 cross-split consistency도 더 높다. Pearson $r=0.244$는 effect size가 매우 크지는 않지만, 512 seed를 넘어 강한 statistical significance를 보인다.

### 3) Mean/covariance를 의도적으로 다르게 만들면 consistency가 약해진다

Principal component를 기준으로 data를 stratify해 split mean이나 variance를 mismatch시키는 counterfactual experiment에서 generated sample consistency가 떨어진다.

Shared Gaussian statistics가 consistency의 causal ingredient라는 주장을 강화하는 control이다.

### 4) Finite data는 low-variance direction을 overshrink한다

RMT는 $\kappa(\sigma^2)>\sigma^2$인 effective noise를 예측한다. Limited-data denoiser는 population denoiser보다 stronger shrinkage를 보인다.

Deep network에서도 $n=3000$ 부근의 generalization regime에서 face가 average face 쪽으로 매끄럽게 보이고, mid-to-low eigenmode variance가 부족하다.

Dataset size가 30k로 늘어 learned spectrum이 population spectrum에 가까워지면 이 bias가 줄어든다.

### 5) Memorization에서 renormalization으로 두 phase가 나타난다

논문은 data size에 따라 두 regime을 관찰한다.

1. Memorization phase, $n \leq 1000$
   - Generated sample이 own training split nearest neighbor에 더 가깝다.
   - Linear theory가 individual point memorization을 설명하지 못한다.

2. Renormalization phase, $n \geq 3000$
   - Training split과 control split nearest-neighbor distance가 비슷해진다.
   - Sample이 linear predictor에 접근한다.
   - Overshrinkage와 spectral fluctuation prediction이 나타난다.

Transition point는 architecture capacity와 image resolution에 따라 달라질 수 있다.

### 6) Consistency는 eigenmode에 따라 anisotropic하다

Dataset size가 늘 때 top eigenspace의 cross-split MSE가 가장 크게 감소한다. Middle과 lower eigenspace는 더 많은 sample이 필요하고, 일부 구간에서는 consistency가 크게 좋아지지 않는다.

Coarse semantic structure가 fine detail보다 먼저 reproducible해지는 spectral explanation이다.

### 7) Seed별 disagreement도 inhomogeneous하다

같은 model pair에서도 어떤 initial noise는 split 간 거의 같은 sample을 만들고, 어떤 noise는 더 크게 갈라진다.

FFHQ64, $n=30000$ UNet에서 RMT seed-wise prediction과 empirical cross-split deviation의 Spearman correlation은 0.33, $p=2.5 \times 10^{-26}$로 보고된다.

Prediction은 split identity나 network architecture를 보지 않고 population covariance와 dataset size만 사용한다.

### 8) Deep network의 absolute deviation은 linear theory보다 크다

Linear RMT는 disagreement가 집중되는 direction과 seed를 qualitative하게 맞추지만, deep network의 absolute deviation magnitude는 더 크다.

Nonlinear feature learning, architecture bias, optimization path가 추가 idiosyncrasy를 만든다는 뜻이다.

### 9) Longer training에서는 linear predictor와의 거리가 다시 늘 수 있다

Appendix의 250k-step experiment에서는 DNN sample이 training 초기에 linear predictor에 가까워졌다가, 이후 nonlinear structure를 학습하며 다시 멀어질 수 있다.

Linear stage는 final model 전체를 설명하는 완전한 theory라기보다, consistency가 형성되는 강한 baseline dynamics로 읽어야 한다.

## 5-2. What really matters in the experiments

### 1) Same-seed pointwise comparison을 했다

FID나 sample quality는 distribution 수준 metric이다. 이 논문의 질문은 latent-to-sample mapping이 run 간 정렬되는가이므로 paired seed evaluation이 맞는 metric이다.

### 2) Nearest-neighbor control로 memorization을 분리했다

Cross-split similarity와 training-set copying은 함께 일어날 수 있다. Own-split/control-split nearest neighbor distance가 theory-valid regime을 정하는 데 중요하다.

### 3) Counterfactual moment mismatch가 mechanism을 강화한다

Natural split만 비교하면 mean/covariance similarity는 correlation일 수 있다. Principal-component stratification으로 moments를 바꿨을 때 consistency가 약해지는 result가 linear-statistics explanation을 더 설득력 있게 만든다.

### 4) Exact magnitude보다 structure를 예측한다

RMT prediction은 deep network pixel error를 정확히 맞추지 않는다. 대신 어느 mode와 seed에서 deviation이 커지는지를 맞춘다. Theory evaluation의 target을 올바르게 잡은 부분이다.

### 5) Failure regime를 명시한다

Small-data memorization에서 correlation이 collapse한다. Theory가 맞는 result만 고른 것이 아니라 scope boundary를 empirical phase transition으로 보여준다.

# 6. Limitations

1. Core theorem은 Gaussian linear denoiser에 대한 것이다
   - Deep model은 higher-order statistics와 nonlinear feature를 학습한다.
   - Linear theory는 coarse consistency baseline이지 full generative semantics theory가 아니다.

2. Empirical covariance model assumption이 필요하다
   - RMT result는 high-dimensional asymptotic과 Wishart-like universality assumption에 의존한다.
   - Natural image sample의 dependence와 heavy-tail이 theorem condition을 정확히 만족하는지는 별도 문제다.

3. Mean estimation effect를 단순화한다
   - Main derivation 일부는 $\hat{\mu}=\mu$를 두고 covariance effect를 분리한다.
   - Small data에서는 empirical mean variation도 sample consistency에 영향을 준다.

4. Deterministic sampling에 초점을 맞춘다
   - Stochastic sampler, classifier-free guidance, conditional generation에서는 random noise와 condition sensitivity가 추가된다.
   - Same-seed consistency의 의미가 달라질 수 있다.

5. Low-resolution image 중심이다
   - 32x32와 64x64 dataset은 spectrum analysis에 적합하지만, modern text-to-image와 video model의 scale과 condition complexity를 대표하지 않는다.

6. Pixel MSE가 semantic consistency를 완전히 반영하지 않는다
   - Small translation이나 texture change가 MSE를 크게 만들 수 있다.
   - 반대로 perceptually 다른 image가 low-frequency structure 때문에 MSE상 가깝게 보일 수 있다.

7. Memorization transition은 architecture-dependent다
   - $n=1000$과 $n=3000$ boundary를 universal threshold로 보면 안 된다.
   - Model capacity, augmentation, training steps, resolution에 따라 바뀐다.

8. Sample quality와 consistency는 다른 목표다
   - 두 run이 같은 biased average sample을 만들면 consistency는 높지만 quality나 diversity는 낮을 수 있다.
   - Reproducibility를 무조건 desirable로 해석하면 안 된다.

9. Causal explanation은 부분적이다
   - Moment-mismatch control은 strong evidence지만, deep network consistency의 전체 원인이 Gaussian statistics라고 증명하지는 않는다.
   - Optimization and architecture alignment가 추가로 기여할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 큰 가치는 diffusion model의 reproducibility를 "seed를 고정했더니 비슷했다"에서 spectral sample complexity 문제로 바꾼 데 있다.

Model scaling 연구에서는 parameter count와 compute를 주로 보지만, 어떤 output feature가 몇 개의 training sample에서 안정화되는지도 중요하다.

- Dominant eigenspace는 적은 sample로도 split 간 정렬될 수 있다.
- Low-variance detail은 더 많은 sample을 요구한다.
- Limited data에서는 detail이 random하게 흔들리는 것뿐 아니라 mean 쪽으로 systematic하게 shrink한다.

이 관점은 image diffusion뿐 아니라 video generation에도 연결된다. Global motion, camera trajectory, scene layout 같은 high-variance structure는 run 간 안정적일 수 있지만, object identity consistency와 fine temporal detail은 더 많은 data와 stronger inductive bias를 요구할 수 있다.

또한 same-seed consistency를 model fingerprint처럼 사용할 수 있다. 두 checkpoint가 같은 population statistics를 얼마나 공유하는지, architecture change가 sampling map을 얼마나 바꾸는지 paired seed map으로 분석할 수 있다.

## 7-2. Reuse potential

### 1) Cross-checkpoint seed panel

Checkpoint pair마다 fixed noise seed bank를 공유하고 다음을 저장한다.

- Pixel MSE
- Perceptual distance
- Feature-space cosine similarity
- Nearest-neighbor distance
- Low/mid/high spectral band error

### 2) Data-split reproducibility curve

Dataset size를 늘리며 same-seed consistency가 언제 memorization regime를 벗어나는지 본다. Single final checkpoint보다 phase transition을 보는 것이 중요하다.

### 3) Spectral variance audit

Generated sample covariance를 data covariance eigenbasis에 project한다. 어떤 mode가 overshrink되거나 split-sensitive한지 확인한다.

### 4) Moment-matched and moment-mismatched control

Data curation method를 비교할 때 sample count만 맞추지 말고 mean/covariance를 맞춘 split과 intentionally mismatched split을 함께 만든다.

### 5) Seed difficulty prediction

Population covariance와 initial noise alignment로 split-sensitive seed를 미리 예측한다. Evaluation seed를 random average만 쓰지 않고 easy/hard consistency bucket으로 나눌 수 있다.

### 6) Training-time trajectory

Checkpoint마다 linear predictor distance와 cross-split consistency를 같이 측정한다. Model이 shared low-order structure를 먼저 학습한 뒤 higher-order detail로 분기하는 시점을 볼 수 있다.

### 7) Video diffusion extension

Frame stack이나 latent video feature의 covariance spectrum에서 다음을 분리할 수 있다.

- Spatial dominant modes
- Temporal low-frequency modes
- High-frequency motion detail
- Object-specific residual modes

Same noise와 same condition에서 model/data split consistency를 분석하면 temporal structure의 sample complexity를 볼 수 있다.

## 7-3. Follow-up papers

- The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
- The Geometry of Diffusion Models: Noise to Data Maps and Linear Structure
- How Much Do Diffusion Models Memorize?
- Generalization, Memorization, and Creativity in Diffusion Models
- Elucidating the Design Space of Diffusion-Based Generative Models
- High-Dimensional Asymptotics of Ridge Regression and Sample Covariance Resolvents
- Random Matrix Theory for Modern Machine Learning

# 8. Summary

- Independent data split과 다른 architecture로 학습한 diffusion model은 deterministic sampler에서 같은 noise를 비슷한 sample로 보낸다.
- Empirical mean과 covariance만 사용하는 Gaussian linear predictor가 이 consistency의 상당 부분과 coarse sample structure를 설명한다.
- Finite data effect는 raw noise $\sigma^2$를 effective noise $\kappa(\sigma^2)$로 renormalize하는 deterministic equivalence로 표현된다.
- Effective noise 증가는 low-variance direction을 과도하게 shrink해 sample을 mean 쪽으로 당기고 fine detail variance를 줄인다.
- Cross-split disagreement는 eigenmode anisotropy, seed-wise inhomogeneity, dataset-size scaling으로 분해된다.
- Linear RMT는 UNet과 DiT의 non-memorization regime에서 qualitative structure를 예측하지만, deep network의 absolute deviation과 memorization은 완전히 설명하지 못한다.
