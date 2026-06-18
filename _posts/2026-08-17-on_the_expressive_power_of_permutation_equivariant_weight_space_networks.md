---
layout: single
title: "On the Expressive Power of Permutation-Equivariant Weight-Space Networks Review"
categories: Study-concept
tag: [WeightSpaceLearning, EquivariantNetworks, Expressivity]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2602.01083)

[Code link](https://github.com/dayanadir/capacity_increase_inr_editing_experiment)

Weight-space learning은 neural network의 입력이나 출력이 image, text, token이 아니라 **다른 neural network의 parameter**인 문제를 다룬다. Model accuracy prediction, neural network classification, learned optimization, pruning, model editing처럼 이미 학습된 network를 읽고 변환하는 작업이 대표적이다.

이 setting은 일반 data learning과 다른 symmetry를 가진다. MLP의 hidden neuron 순서를 바꾸고 인접 layer의 weight를 함께 재배열해도 같은 function이 구현된다. 따라서 weight-space model은 raw parameter array의 순서가 아니라, 그 parameter가 속한 permutation orbit를 일관되게 처리해야 한다.

그동안 DWS, NP-NFN, HNP-NFN, GMN, NG-GNN, NFT처럼 서로 다른 permutation-aware architecture가 제안되었다. 각 방법은 tensor operation, graph message passing, attention 등 다른 computation을 사용한다. 자연스럽게 다음 질문이 생긴다.

> 어느 architecture가 더 expressive한가, 그리고 weight-space task의 종류에 따라 universality가 어떻게 달라지는가?

이 논문은 이 질문을 architecture comparison 하나로 끝내지 않는다. Target을 function-space functional, permutation-invariant functional, function-space operator, permutation-equivariant operator의 네 종류로 나누고, 각 setting에서 universality가 가능한 조건과 불가능한 조건을 정리한다. 특히 prominent architecture 대부분이 compact domain에서 같은 expressive power를 가지며, global universality를 막는 핵심 원인이 architecture 이름보다 permutation orbit을 구분하는 능력과 output capacity에 있을 수 있음을 보인다.

이 이론적 분석은 Output Capacity Expansion, OCE라는 간단한 design으로 이어진다. 하나의 input network에서 하나의 output network만 예측하는 대신 여러 output network를 예측하고 그 function을 평균한다. Parameter budget을 맞춘 MNIST INR editing 실험에서 OCE는 DWS와 GMN 모두에서 error를 꾸준히 낮추며, strongest reported prior baseline 대비 최대 34% 개선을 보인다.

> 한 줄 요약: 이 논문은 permutation-equivariant weight-space network의 expressivity를 네 종류의 target map으로 분해하고, architecture 간 차이보다 general position과 output function capacity가 universality를 결정하는 경우가 많음을 보인 뒤, 그 분석을 OCE라는 실용적 architecture modification으로 연결한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Weight-space architecture를 module별로 비교하는 대신 **어떤 target map을 학습하는가**로 문제를 재정의한다.
- DWS, NP-NFN, HNP-NFN, GMN, NG-GNN 같은 주요 architecture가 expressive power 관점에서 상당 부분 동등하다는 결과를 제시한다.
- Global non-universality와 general-position universality를 분리해, symmetry가 실제로 만드는 blind spot을 설명한다.
- Function-space operator에서 output architecture 자체가 bottleneck이 될 수 있음을 formal하게 보인다.
- Theory가 OCE라는 concrete design과 parameter-matched experiment로 이어진다.
- Quantization이 general-position assumption을 깨뜨릴 수 있다는 실무적 경고도 포함한다.

이 논문은 새로운 weight-space block 하나를 제안하는 paper라기보다, **weight-space task를 설계할 때 먼저 확인해야 할 expressivity map**을 제공하는 paper로 읽는 편이 유용하다.

# 1. Problem Setting

## 1-1. Weight space에는 왜 permutation symmetry가 있는가

$L$-layer MLP의 parameter를 $v$라고 하자. Hidden layer $\ell$의 width가 $d_\ell$이면, hidden neuron 순서는 $d_\ell!$가지로 바꿀 수 있다. 각 hidden layer에서 가능한 permutation을 묶으면 다음 group을 얻는다.

$$
G_A = S_{d_1} \times \cdots \times S_{d_{L-1}}.
$$

여기서 $A$는 network architecture이고, $S_{d_\ell}$은 $d_\ell$개 neuron의 permutation group이다. $g \in G_A$가 parameter에 작용한 결과를 $\rho(g)v$라고 쓰면, neuron permutation은 realized function을 바꾸지 않는다.

$$
f_{\rho(g)v} = f_v.
$$

따라서 weight-space prediction이 scalar나 class label을 출력한다면 permutation-invariant해야 한다.

$$
\Phi(\rho(g)v) = \Phi(v).
$$

반대로 pruning mask, gradient, edited weight처럼 input parameter와 같은 neuron indexing을 갖는 output을 예측한다면 permutation-equivariant해야 한다.

$$
\Phi(\rho(g)v) = \rho(g)\Phi(v).
$$

이 조건을 무시하면 같은 function을 나타내는 두 parameterization에 서로 다른 prediction을 내거나, input neuron을 재배열했는데 output update는 원래 indexing에 남는 문제가 생긴다.

## 1-2. Permutation invariance와 function invariance는 다르다

이 논문의 중요한 구분은 permutation orbit과 function equivalence class가 항상 같지 않다는 점이다.

Permutation으로 연결된 두 parameter는 반드시 같은 function을 구현한다. 하지만 같은 function을 구현하는 모든 parameter pair가 permutation만으로 연결되는 것은 아니다. Redundant neuron, zero weight, activation 특성, scaling symmetry처럼 다른 parameter redundancy도 존재할 수 있다.

따라서 다음 두 target은 서로 다른 문제다.

- **Permutation-invariant functional**: hidden neuron permutation에만 invariant하면 된다.
- **Function-space functional**: 같은 function을 구현하는 모든 parameterization에 같은 값을 내야 한다.

예를 들어 weight norm이나 Hessian statistic은 neuron permutation에는 invariant할 수 있지만, realized function만으로 결정되지 않을 수 있다. 반면 INR가 표현하는 image category나 model의 function-level behavior는 function-space quantity에 더 가깝다.

## 1-3. 논문이 분리한 네 가지 approximation setting

| Setting | Input and output | Required symmetry | Example |
| --- | --- | --- | --- |
| Function-space functional | $\mathcal{F}_A \rightarrow \mathbb{R}^m$ | 같은 function에 같은 output | INR classification, function property prediction |
| Permutation-invariant functional | $V_A \rightarrow \mathbb{R}^m$ | neuron permutation에 invariant | weight norm, curvature statistic, parameter-level quality score |
| Function-space operator | $\mathcal{F}_A \rightarrow \mathcal{F}_B$ | input representation과 무관하게 function을 function으로 변환 | INR editing, domain transformation |
| Permutation-equivariant operator | $V_A \rightarrow V_B$ | input permutation에 output indexing이 함께 변환 | gradient prediction, pruning mask, learned optimizer update |

이 구분이 필요한 이유는 universality result가 setting마다 다르기 때문이다. "Permutation-equivariant network가 universal한가"라는 질문만으로는 충분하지 않다. Input domain이 all weight space인지 general-position subset인지, output이 scalar인지 fixed-size network인지, target이 parameter-level인지 function-level인지까지 정해야 한다.

## 1-4. 기존 접근의 한계

기존 weight-space paper는 대체로 새로운 equivariant layer나 graph construction을 제안하고 downstream benchmark에서 비교한다. 이런 접근만으로는 다음을 구분하기 어렵다.

1. Architecture가 정말 target function을 표현하지 못하는가.
2. 표현은 가능하지만 optimization이 어려운가.
3. Input symmetry 때문에 어떤 architecture도 구분할 수 없는 point가 있는가.
4. Output network의 function class가 target operator의 range보다 작은가.
5. Model parameter 수보다 output representation capacity가 부족한가.

논문은 이 다섯 문제를 분리한다. 특히 architecture ranking에 앞서 **task의 symmetry class와 output range를 먼저 정의해야 한다**는 것이 전체 분석의 출발점이다.

# 2. Core Idea

## 2-1. Main contribution 1: 주요 architecture의 expressive equivalence

논문은 DWS, NP-NFN, HNP-NFN, GMN, NG-GNN 등 prominent permutation-equivariant weight-space architecture 사이의 simulation relation을 분석한다. 핵심 결과는 compact input domain에서 이 architecture들이 invariant map과 equivariant map에 대해 같은 expressive closure를 가진다는 것이다.

즉, finite width와 실제 training dynamics는 다를 수 있지만, arbitrary width와 depth를 허용한 approximation power만 보면 많은 architecture가 서로보다 본질적으로 더 강하지 않다.

NFT는 full weight space에서는 별도 취급이 필요하지만, 뒤에서 설명할 general-position condition 아래에서는 다른 architecture와 같은 expressive class에 들어간다.

이 결과의 practical implication은 명확하다.

- Benchmark 차이가 곧 fundamental expressivity gap을 의미하지 않는다.
- Architecture 선택은 optimization, memory, inductive bias, implementation cost로 이동할 수 있다.
- 어떤 target이 모든 architecture에 공통으로 불가능하다면, block을 바꾸기보다 domain assumption이나 output parameterization을 바꿔야 한다.

## 2-2. Main contribution 2: General Position이 universality를 복원한다

Global weight space에서는 서로 다른 permutation orbit을 기존 architecture가 구분하지 못하는 경우가 있다. 논문은 두 binary weight configuration $v$와 $v'$를 구성한다. 두 configuration의 middle weight matrix는 rank가 각각 3과 2이므로 permutation으로 연결될 수 없다.

$$
v \not\sim_G v'.
$$

하지만 DWS 계열의 모든 model은 이 두 input을 같은 representation으로 처리한다. High-level로 쓰면 다음과 같다.

$$
\Phi(v)=\Phi(v').
$$

따라서 두 orbit에 서로 다른 값을 주는 continuous invariant target이 존재하면, 해당 architecture family는 global weight space 전체에서 그 target을 근사할 수 없다. 이 counterexample은 단순히 finite width가 부족한 문제가 아니라 orbit separation 자체의 한계다.

논문은 이를 피하기 위해 general position, GP를 정의한다. 대표적인 GP condition은 각 hidden layer의 bias가 pairwise distinct하다는 것이다. Exclusion set을 단순화해 쓰면 다음과 같다.

$$
E_A = \left\{v : \exists \ell, i < j, (b_\ell)_i = (b_\ell)_j \right\}.
$$

$V_A \setminus E_A$에서는 bias value를 이용해 hidden neuron의 canonical order를 정할 수 있다. Canonicalization이 가능하면 arbitrary continuous map을 canonical coordinate에서 근사한 뒤 group action으로 다시 equivariant하게 옮길 수 있다.

이 condition은 continuous random initialization에서는 almost surely 만족된다. Equal bias가 정확히 발생하는 set은 measure zero이기 때문이다. 하지만 quantization처럼 parameter를 discrete grid에 투영하면 tie가 늘어나고 GP가 쉽게 깨질 수 있다. 이 점은 theory를 deployment assumption과 연결하는 중요한 부분이다.

## 2-3. Main contribution 3: Target map에 따른 universality map

논문의 결과를 high-level로 정리하면 다음과 같다.

| Target class | Global result | GP compact domain result | 핵심 bottleneck |
| --- | --- | --- | --- |
| Function-space functional | Continuous target에 대해 universal | 동일 | Realized function equivalence를 처리하는 invariant representation |
| Permutation-invariant functional | Global universality가 일반적으로 성립하지 않음 | Universal | Distinct permutation orbit을 model family가 분리하지 못하는 경우 |
| Function-space operator | Fixed underlying architecture에서는 universal하지 않음 | Input과 output에 쓰이는 architecture를 충분히 크게 두면 가능 | Output function class의 bounded capacity |
| Permutation-equivariant operator | Global universality가 일반적으로 성립하지 않음 | Universal | Invariant setting의 orbit non-separation이 equivariant setting으로 이어짐 |

여기서 가장 중요한 결과는 function-space operator다. Scalar functional은 input network를 읽고 finite-dimensional value를 내면 된다. 반면 operator는 새로운 function을 출력해야 한다. 현재 weight-space operator처럼 input과 output을 같은 fixed architecture $A$로 제한하면, target operator가 더 복잡한 function을 요구할 때 output range 자체가 부족하다.

예를 들어 input INR를 더 복잡한 image function으로 바꾸는 operator를 생각하자. Meta-network가 아무리 expressive해도 output slot이 항상 같은 작은 MLP 하나라면, 그 MLP family 밖의 function은 만들 수 없다.

> Operator learning에서는 meta-network의 capacity와 output network family의 capacity를 분리해서 봐야 한다.

## 2-4. Main contribution 4: Output Capacity Expansion

논문은 fixed-output limitation을 단순한 architecture change로 연결한다. OCE는 하나의 output network 대신 $k$개의 output network를 예측하고, 그 function output을 평균한다.

$$
\mathrm{OCE}(v) = (v_1, ..., v_k) \in V_A^k.
$$

$$
f_{\mathrm{OCE}(v)}(x)
= \frac{1}{k}\sum_{i=1}^{k} f_{v_i}(x).
$$

$k$가 커지면 output function class는 single fixed-size MLP보다 넓어진다. 중요한 점은 meta-network 전체 parameter 수를 그대로 늘리지 않았다는 것이다. Paper는 $k \in \{1, 2, 4, 8\}$을 비교하면서 internal hidden width를 조절해 $k > 1$ model의 parameter count가 $k=1$ baseline을 넘지 않도록 맞춘다.

따라서 성능 향상을 단순히 "더 큰 meta-network를 썼다"로 설명하기 어렵다. OCE는 같은 parameter budget을 **internal representation width와 output function capacity 사이에 재배분**한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Permutation-equivariant weight-space network의 expressivity를 systematic하게 분류 |
| Input | Fixed architecture $A$를 가진 MLP 또는 INR의 parameter $v \in V_A$ |
| Symmetry group | Hidden-layer neuron permutation group $G_A$ |
| Compared families | DWS, NP-NFN, HNP-NFN, GMN, NG-GNN, NFT |
| Analysis axes | Architecture equivalence, universality, GP, output capacity |
| Practical method | Output Capacity Expansion, OCE |
| Main experiment | MNIST SIREN INR dilation |
| Main metric | Pixel-space MSE, reported as MSE $\times 10^{-2}$ |
| Main message | Architecture block보다 target class, orbit identifiability, output range가 expressivity를 결정할 수 있음 |

## 3-2. Architecture equivalence를 보이는 방식

논문의 proof strategy는 각 architecture를 독립적으로 universal하다고 증명하는 데 그치지 않는다. Architecture 사이에 어떤 primitive를 simulate할 수 있는지 보여 expressivity relation을 연결한다.

High-level로 보면 다음 세 단계다.

1. **Common equivariant primitives 식별**
   - Layer와 neuron index를 따라 aggregate하고 broadcast하는 operation을 정리한다.
   - Tensor formulation과 graph formulation이 같은 orbit-level information을 전달할 수 있음을 보인다.

2. **Architecture 간 simulation**
   - 한 family의 layer가 구현하는 invariant 또는 equivariant map을 다른 family가 arbitrary precision으로 근사할 수 있음을 보인다.
   - 이를 반복해 network-level expressive closure를 비교한다.

3. **Base universality result와 결합**
   - Canonicalization 또는 separating invariant를 이용해 continuous target을 standard neural network approximation problem으로 바꾼다.
   - Equivalence relation을 통해 universality result를 여러 architecture family로 전파한다.

이 결과는 "architecture가 모두 같다"는 뜻이 아니다. Finite model에서 필요한 width, depth, optimization landscape, computational complexity는 다를 수 있다. 논문이 같다고 말하는 것은 asymptotic approximation class다.

## 3-3. Canonicalization과 separating representation

GP subset에서는 distinct bias가 neuron identifier 역할을 한다. 각 hidden layer의 bias를 정렬하면 permutation orbit마다 canonical representative를 선택할 수 있다.

개념적으로 canonicalization map $c$는 다음 property를 가져야 한다.

$$
c(\rho(g)v) = c(v).
$$

또한 GP subset에서는 서로 다른 orbit이 같은 canonical representation으로 collapse하지 않아야 한다. 이 representation 위에서 ordinary MLP로 invariant target을 근사할 수 있다.

Equivariant target은 canonical coordinate에서 output을 계산하고, original input이 canonical order로 이동할 때 사용한 permutation을 반대로 적용해 원래 indexing으로 되돌린다.

이 construction이 보여주는 것은 GP가 단순한 technical convenience가 아니라는 점이다. GP는 neuron을 안정적으로 구분할 수 있는 signal을 제공하고, symmetry quotient를 locally well-behaved하게 만든다.

## 3-4. Fixed output architecture의 barrier

Function-space operator $T$를 생각하자.

$$
T : \mathcal{F}_A \rightarrow \mathcal{F}_A.
$$

Meta-network는 input parameter $v \in V_A$를 읽고 같은 architecture의 output parameter $u = \Phi(v) \in V_A$를 만든다. 최종 output function은 $f_u$다.

문제는 $\Phi$의 expressivity와 무관하게 $u$가 항상 fixed $V_A$에 속한다는 점이다. $A$가 fixed-size network라면 $\mathcal{F}_A$는 제한된 function family다. Target operator의 range에 $\mathcal{F}_A$로 근사할 수 없는 function이 포함되면 universality는 불가능하다.

이 논리는 weight-space model만의 문제가 아니다. Hypernetwork, learned optimizer, model editor처럼 network를 output으로 만드는 모든 system에 적용된다.

- Hypernetwork가 충분히 크다고 output network가 자동으로 expressive해지지 않는다.
- Weight update를 정확히 예측해도 destination architecture가 target behavior를 담지 못할 수 있다.
- Operator benchmark의 error를 meta-model optimization failure로만 해석하면 output bottleneck을 놓칠 수 있다.

## 3-5. OCE의 parameterization

OCE는 output tensor의 feature dimension을 $k$로 늘려 같은 architecture의 parameter set을 $k$개 예측한다. 각 output network는 independently parameterized되지만, input encoding과 most internal computation은 shared meta-network에서 나온다.

INR editing에서는 각 $v_i$가 edited INR candidate가 되고, final image는 $k$개 INR output의 평균이다. Paper는 residual form도 사용한다.

$$
\hat{v}_i = v + s\Delta v_i,
$$

여기서 $\Delta v_i$는 predicted update이고 $s$는 learnable scalar다. 이 residual parameterization은 input INR가 이미 원본 digit function을 잘 표현한다는 prior를 활용한다.

OCE의 설계 의도는 ensemble diversity 자체보다 range expansion에 있다. 여러 fixed-size network의 average는 single network 하나보다 더 복잡한 function을 표현할 수 있다. 다만 실제로 $k$개 output이 서로 다른 role을 학습하는지, 단순 variance reduction인지, optimization landscape가 좋아진 것인지는 별도 분석이 필요하다.

# 4. Training / Data / Recipe

## 4-1. Data and task

Main experiment는 MNIST INR editing이다.

- 각 MNIST image를 하나의 pretrained SIREN INR로 표현한다.
- Input INR architecture는 $(2, 32, 32, 1)$이다.
- Coordinate $(x, y)$를 입력하면 grayscale pixel value를 출력한다.
- Target image는 original $28 \times 28$ digit에 $3 \times 3$ morphological dilation을 한 번 적용한 결과다.
- Meta-network는 input INR weight를 읽고 dilated image를 표현하는 edited INR parameter를 예측한다.
- Standard MNIST train and test split을 사용하고, training set 중 5,000개를 validation으로 둔다.

이 task가 좋은 이유는 output이 weight vector가 아니라 rendered function quality로 평가된다는 점이다. Edited parameter가 target parameter와 coordinate-wise로 같을 필요는 없고, 같은 dilated image function을 표현하면 된다. 따라서 function-space operator setting을 직접 평가한다.

## 4-2. Compared methods

Paper의 main comparison에는 다음 계열이 포함된다.

- NFT
- NP-NFN
- NG-GNN-64
- ScaleGMN-B
- NG-T-64
- ScaleGMN with GradMetaNet++
- DWS with $k=1$
- GMN with $k=1$
- DWS with OCE, $k \in \{2,4,8\}$
- GMN with OCE, $k \in \{2,4,8\}$

OCE experiment의 핵심 control은 parameter budget이다. $k$를 늘릴 때 internal hidden dimension을 줄여 total trainable parameter count가 $k=1$ baseline을 넘지 않도록 맞춘다.

## 4-3. Training recipe

논문에 기재된 main recipe는 다음과 같다.

| Item | Value |
| --- | --- |
| Epochs | 150 |
| Optimizer | AdamW |
| Initial learning rate | $10^{-3}$ |
| Weight decay | $10^{-3}$ |
| Batch size | 32 |
| LR schedule | Validation이 5 epoch 개선되지 않으면 learning rate를 절반으로 감소 |
| Minimum learning rate | $10^{-4}$ |
| Random seeds | 3 |
| Hardware | Single A100-SXM4-40GB per run |
| Runtime | 약 4시간에서 10시간 per run |
| Total reported compute | 약 200 A100 GPU-hours |

## 4-4. Engineering notes

### 1) Paper table과 reproduction repository 수치를 섞지 말아야 한다

공개 repository는 DWS와 ScaleGMN의 $k \in \{1,2,4,8\}$ 8개 run을 재현한다. README의 result는 각 run의 **best test loss**를 적는다. Paper Table 1과 Table 2는 3 seed mean과 standard deviation을 보고한다. Reporting convention이 다르므로 숫자를 직접 대응시키면 안 된다.

### 2) GMN naming을 확인해야 한다

Paper table은 GMN으로 표기하지만 repository implementation path는 ScaleGMN이다. Reproduction이나 code comparison에서는 exact config와 model class를 확인하는 편이 안전하다.

### 3) Output cost는 meta-network parameter count와 별도다

OCE는 parameter-matched meta-network comparison이지만 $k$개의 output INR를 evaluate하고 average한다. Training parameter budget이 같다고 end-to-end inference FLOPs나 memory가 같은 것은 아니다.

### 4) Quantized weight-space input은 GP를 다시 확인해야 한다

Full-precision pretrained weight는 distinct bias condition을 almost surely 만족할 수 있다. 하지만 8-bit 이하 quantization은 many ties를 만들 수 있다. Weight-space model을 compressed checkpoint 위에 적용한다면 permutation equivariance만 확인할 것이 아니라 GP satisfaction rate와 parameter collision도 측정해야 한다.

# 5. Evaluation

## 5-1. Main results

Paper는 pixel-space MSE를 $10^{-2}$ scale로 보고한다. 주요 결과는 다음과 같다.

| Method | OCE $k$ | MSE $\times 10^{-2}$ | Interpretation |
| --- | ---: | ---: | --- |
| NFT | 1 | $5.10 \pm 0.04$ | Attention-based weight-space baseline |
| NP-NFN | 1 | $2.55 \pm 0.00$ | NFN baseline |
| NG-GNN-64 | 1 | $2.06 \pm 0.01$ | Graph baseline |
| ScaleGMN-B | 1 | $1.89 \pm 0.00$ | Strong GMN baseline |
| NG-T-64 | 1 | $1.75 \pm 0.01$ | Transformer-style graph baseline |
| ScaleGMN with GradMetaNet++ | 1 | $1.60 \pm 0.01$ | Strongest reported prior baseline in the table |
| DWS | 1 | $2.29 \pm 0.01$ | OCE comparison baseline |
| GMN | 1 | $1.96 \pm 0.02$ | OCE comparison baseline |
| DWS with OCE | 8 | $1.36 \pm 0.03$ | DWS $k=1$ 대비 41% 감소 |
| GMN with OCE | 8 | $1.06 \pm 0.13$ | GMN $k=1$ 대비 46% 감소 |

GMN with OCE의 $1.06$은 strongest reported prior baseline $1.60$보다 약 34% 낮다. Paper의 headline improvement는 이 comparison에서 나온다.

## 5-2. OCE scaling

OCE의 더 중요한 결과는 best point 하나보다 $k$에 따른 monotonic trend다.

| $k$ | DWS MSE | DWS relative change | GMN MSE | GMN relative change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.29 | 0% | 1.96 | 0% |
| 2 | 2.01 | -12% | 1.69 | -14% |
| 4 | 1.58 | -31% | 1.19 | -39% |
| 8 | 1.36 | -41% | 1.06 | -46% |

이 table에서 볼 포인트는 세 가지다.

첫째, DWS와 GMN에서 같은 trend가 나타난다. OCE effect가 특정 architecture 하나에만 묶이지 않는다.

둘째, parameter budget을 맞춘 상태에서 $k$ 증가가 성능을 개선한다. Internal channel을 줄이더라도 output function capacity에 budget을 배분하는 편이 이 operator task에서는 유리했다.

셋째, $k=8$ GMN의 standard deviation은 $0.13$으로 다른 row보다 크다. Best mean만 볼 것이 아니라 optimization variability도 함께 봐야 한다.

## 5-3. What really matters in the experiments

### 1) Theory가 falsifiable design change로 이어진다

"Output family가 bottleneck일 수 있다"는 theorem이 $k$를 늘리는 OCE experiment로 바로 연결된다. 단순 architecture ablation보다 설계 이유가 명확하다.

### 2) Same parameter count가 same system cost를 뜻하지 않는다

Paper의 control은 trainable parameter count다. $k$개 INR를 render하는 cost, output activation memory, deployment format은 별도다. OCE를 production setting에 적용하려면 quality versus end-to-end latency curve가 필요하다.

### 3) Function-level evaluation이 적절하다

같은 image function을 표현하는 weight는 여러 개다. Parameter MSE보다 rendered pixel MSE를 사용한 것은 function-space operator의 목적과 일치한다.

### 4) General-position stress test가 theory assumption을 현실 문제로 바꾼다

Paper는 60,000개 MNIST INR에서 bias quantization 후 GP 비율을 측정한다.

| Weight representation | Bias-distinct GP rate |
| --- | ---: |
| Full precision | 100% |
| 16-bit | 97.98% |
| 12-bit | 72.66% |
| 8-bit | 0.62% |

8-bit에서 primary bias-only criterion은 거의 항상 깨진다. Paper가 여러 alternative separator를 결합한 GP5 criterion을 사용하면 8-bit에서 약 80%까지 회복되지만, aggressive quantization에서는 여전히 문제가 남는다.

이 결과는 "GP failure set은 measure zero"라는 continuous-space statement가 finite-precision system에서는 충분하지 않음을 보여준다. Training checkpoint가 quantized되거나 weight가 clipped, rounded, tied되는 pipeline에서는 empirical GP audit가 필요하다.

## 5-4. Architecture equivalence를 benchmark equality로 오해하면 안 된다

Theory가 architecture의 asymptotic expressive closure를 같다고 해도 finite experiment 성능은 다르다. Table 1에서 GMN 계열은 DWS보다 낮은 MSE를 보인다. 이는 다음 요인의 차이일 수 있다.

- Finite width에서 target을 표현하는 efficiency
- Optimization landscape
- Message passing과 tensor aggregation의 inductive bias
- Gradient scale와 normalization
- Parameter allocation 방식

따라서 theorem의 practical conclusion은 "architecture choice가 중요하지 않다"가 아니다. 더 정확한 결론은 다음과 같다.

> Architecture benchmark 차이를 fundamental impossibility로 해석하기 전에, target class, domain assumption, finite capacity, optimization을 분리해야 한다.

# 6. Limitations

1. **Theory의 중심은 MLP weight space다.** Appendix는 transformer weight space로 일부 result를 확장하지만, broad transformer experiment나 large-scale language model weight editing으로 검증하지 않는다.

2. **Expressivity는 optimization과 generalization을 보장하지 않는다.** Universal approximator라는 사실은 필요한 width, sample complexity, training stability, out-of-distribution behavior를 말해주지 않는다.

3. **GP assumption은 finite-precision pipeline에서 약해질 수 있다.** Random full-precision initialization에서는 자연스럽지만 quantization, pruning, weight tying, symmetric initialization은 bias collision과 canonization ambiguity를 만들 수 있다.

4. **Empirical evidence가 한 task에 집중된다.** Main result는 MNIST SIREN dilation이다. Classification, pruning, learned optimization, larger INR, transformer checkpoint에서도 OCE trend가 유지되는지 확인이 필요하다.

5. **OCE는 output representation cost를 늘린다.** Meta-network parameter 수는 matched되지만 $k$개 network를 저장하고 evaluate해야 한다. Single deployable checkpoint가 필요한 task에서는 average-of-functions output이 직접 적용되지 않을 수 있다.

6. **OCE improvement의 mechanism이 완전히 분리되지 않는다.** Function class expansion, ensemble averaging, optimization smoothing, implicit regularization이 각각 얼마나 기여하는지 추가 ablation이 필요하다.

7. **Symmetry scope가 permutation에 집중된다.** ReLU scaling, sign, normalization, tied parameter처럼 다른 equivalence relation까지 포함한 unified theory는 아니다.

8. **Strong prior baseline과의 comparison protocol을 주의해야 한다.** Table에는 published baseline과 authors' OCE run이 함께 있다. Dataset와 evaluation은 맞추지만 every baseline을 identical code and compute에서 재실행한 comparison으로 해석하면 안 된다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 큰 가치는 "어떤 equivariant block을 쓸 것인가"보다 앞에 놓여야 하는 design checklist를 제공한다는 점이다.

Weight-space task를 시작할 때 다음 순서로 문제를 정리할 수 있다.

1. **Target이 parameter-level인가 function-level인가**
   - Weight norm, curvature, optimizer update처럼 representation-dependent한가.
   - Model behavior, rendered image, decision boundary처럼 function만으로 정해지는가.

2. **Output이 invariant인가 equivariant인가**
   - Scalar, class, quality score인가.
   - Input indexing과 함께 움직여야 하는 weight, mask, gradient인가.

3. **Input checkpoint가 GP에 가까운가**
   - Quantization, pruning, tied weight, duplicated neuron이 GP exclusion set에 들어가는가.

4. **Output family가 target range를 담을 수 있는가**
   - Meta-network가 아니라 destination architecture가 bottleneck일 수 있는가.

5. **Parameter budget과 system budget을 구분했는가**
   - Trainable parameter가 같아도 output count, inference FLOPs, memory는 다를 수 있다.

이 checklist만 적용해도 architecture benchmark에서 발생하는 많은 혼동을 줄일 수 있다.

## 7-2. Reuse potential

### 1) Learned optimizer and gradient prediction

Gradient나 update는 permutation-equivariant operator다. GP condition과 orbit separation assumption을 확인한 뒤 DWS나 GMN family를 선택할 수 있다. Theory상 expressive closure가 비슷하다면 implementation efficiency와 optimization stability를 우선 비교할 수 있다.

### 2) Model editing and repair

Small INR나 compact neural field를 다른 function으로 바꾸는 task에서는 output architecture capacity를 명시적으로 sweep할 가치가 있다. OCE뿐 아니라 wider destination network, mixture output, basis expansion을 같은 관점에서 비교할 수 있다.

### 3) Model zoo analytics

Model accuracy, robustness, task identity를 weight에서 예측할 때 target이 function-space functional인지 단순 permutation-invariant functional인지 구분해야 한다. Data augmentation으로 neuron permutation만 넣는 것이 sufficient한지도 이 구분에 따라 달라진다.

### 4) Quantization-aware weight-space learning

Quantized model zoo를 학습 input으로 사용할 경우, bit width별 parameter collision과 GP rate를 사전 측정하는 protocol이 필요하다. GP가 깨지는 구간에서는 alternative separator, orbit-aware feature, explicit tie handling이 필요할 수 있다.

### 5) PEFT and LoRA analysis

LoRA adapter는 full MLP weight와 다른 parameterization과 symmetry를 가진다. 이 논문의 theorem을 그대로 적용할 수는 없지만, target map, group action, exclusion set, output range를 먼저 정의하는 방법론은 그대로 재사용할 수 있다.

## 7-3. Follow-up papers

- **Deep Weight Spaces: Learning from Neural Network Weights for Classification and Regression**
  - DWS architecture와 weight-space tensor symmetry를 이해하는 출발점이다.

- **Permutation Equivariant Neural Functionals**
  - Neural functional layer와 universality 논의의 기반을 제공한다.

- **Graph Metanetworks for Processing Diverse Neural Architectures**
  - Neural network를 graph로 표현해 architecture diversity를 처리하는 관점을 제공한다.

- **Neural Functional Transformers**
  - Attention 기반 weight-space processing과 NFT의 expressive behavior를 비교하는 데 필요하다.

- **GradMetaNet: An Equivariant Architecture for Learning on Gradients**
  - Gradient를 input or target으로 사용하는 equivariant meta-learning task와 main experiment baseline을 이해하는 데 유용하다.

# 8. Summary

- Hidden neuron permutation은 realized function을 바꾸지 않으므로 weight-space model에는 invariant or equivariant structure가 필요하다.
- 주요 DWS, NFN, GMN, NG-GNN 계열은 compact domain의 asymptotic expressivity 관점에서 상당 부분 동등하다.
- Global universality failure는 distinct permutation orbit을 architecture가 분리하지 못해서 발생할 수 있고, GP subset에서는 canonicalization을 통해 universality가 복원된다.
- Function-space operator에서는 fixed output architecture 자체가 bottleneck이며, meta-network capacity만 늘려서는 해결되지 않는다.
- OCE는 output network를 $k$개로 확장하고 평균해 function capacity를 높이며, parameter-matched MNIST INR editing에서 DWS와 GMN 모두를 개선한다.
- 실무에서는 target class, GP assumption, output capacity, quantization, system cost를 함께 점검해야 한다.
