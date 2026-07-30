---
layout: single
title: "Why Far Looks Up: Probing Spatial Representation in Vision-Language Models Review"
categories: Study-concept
tag: [VLM, Spatial-Reasoning, Representation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2605.30161)

[Project page](https://cheolhong0916.github.io/whyfarlooksup/)

[Code link](https://github.com/cheolhong0916/contrastive-probing)

[SpatialTunnel dataset](https://huggingface.co/datasets/cubec/spatialtunnel)

> 한 줄 요약: 이 논문은 VLM이 "far"를 3D distance로 이해하기보다 image에서 위쪽에 있는 object와 결합하는 perspective shortcut을 학습할 수 있음을 benchmark split, SpatialTunnel intervention, contrastive hidden-state probing으로 진단하고, distance representation의 coherence가 counter-heuristic robustness를 예측한다는 결과를 제시한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Spatial benchmark의 높은 average accuracy가 genuine 3D representation을 의미하지 않을 수 있음을 보여준다.
- Natural image에서 "higher means farther"라는 perspective regularity가 benchmark label distribution과 model representation에 동시에 들어가는 과정을 분석한다.
- Data scaling으로 overall score가 좋아져도 perspective-consistent and counter-heuristic gap이 남거나 커질 수 있음을 보여준다.
- Synthetic SpatialTunnel로 vertical image position과 actual depth를 독립적으로 조절한다.
- Contrastive probing으로 horizontal, vertical, distance axis의 internal geometry를 정량화하고 downstream robustness와 연결한다.

VLM spatial reasoning 평가에서는 보통 "cup이 book보다 left인가", "red object가 blue object보다 far인가" 같은 question을 준다. Model이 correct answer를 내면 spatial relation을 이해했다고 해석하기 쉽다.

하지만 natural photographs에는 강한 geometric prior가 있다. Ground plane이 보이는 image에서 farther object는 perspective projection 때문에 image plane의 더 높은 위치에 나타나는 경우가 많다. Model은 3D distance를 추론하지 않고 다음 heuristic을 사용할 수 있다.

- Above -> Far
- Below -> Close

이 heuristic은 dataset majority example에서는 잘 맞는다. 따라서 average benchmark score를 높인다. 그러나 object가 아래쪽에 있지만 실제로 더 멀거나, 위쪽에 있지만 더 가까운 counter-heuristic case에서는 무너진다.

논문은 이 문제를 세 단계로 진단한다.

1. Existing benchmark가 perspective heuristic에 얼마나 skewed되어 있는지 측정한다.
2. Vertical position과 depth를 decouple한 SpatialTunnel에서 behavior를 확인한다.
3. Hidden representation에서 vertical and distance axis가 얼마나 entangled되어 있는지 contrastive probe로 측정한다.

핵심 질문은 다음과 같다.

> VLM의 spatial score가 높아질 때, model은 cleaner 3D axis를 학습하는가, 아니면 training distribution의 shortcut을 더 강하게 internalize하는가?

논문의 답은 model and training recipe에 따라 다르다. More spatial data가 consistent sample accuracy를 올려도 counter sample gap을 없애지 못할 수 있다. 반면 auxiliary depth supervision을 강하게 받은 RoboRefer-2B나 very large-scale Qwen3-VL-235B는 더 분리된 spatial representation을 보인다.

# 1. Problem Setting

## 1-1. Problem definition

Image $x$에 두 object $a$와 $b$가 있고, question이 relation $r$을 묻는다고 하자. Relation category는 세 axis로 묶인다.

| Axis | Relation pair |
| --- | --- |
| Horizontal | left / right |
| Vertical | above / below |
| Distance | far / close |

Model output accuracy만 보면 각 relation을 맞혔는지 알 수 있다. 그러나 model이 사용하는 internal feature는 알 수 없다.

Perspective projection 아래에서 image vertical coordinate $v$와 depth $z$는 natural image distribution에서 correlated될 수 있다.

$$
P(z_a > z_b \mid v_a < v_b) > P(z_a > z_b)
$$

좌표 convention에 따라 image에서 위쪽 object의 vertical value가 작다고 두면, $v_a < v_b$일 때 $a$가 더 멀 가능성이 높다는 뜻이다.

논문은 sample을 두 group으로 나눈다.

- Consistent: Ground-truth distance relation이 perspective heuristic과 일치한다.
- Counter: Ground-truth distance relation이 perspective heuristic과 반대다.

Model이 genuine depth cue를 사용한다면 두 group의 gap이 크지 않아야 한다. Vertical shortcut을 쓰면 consistent accuracy는 높고 counter accuracy는 낮아진다.

Gap은 다음처럼 볼 수 있다.

$$
G = A_{\mathrm{consistent}} - A_{\mathrm{counter}}
$$

$G$가 클수록 perspective heuristic dependence가 강하다.

## 1-2. Why previous approaches are insufficient

### 1) Overall accuracy가 majority shortcut을 숨긴다

EmbSpatial-Bench에서 perspective-consistent question은 80.9%인데 counter-heuristic question은 10.7%다. CV-Bench-3D도 consistent sample 비중이 60.5%, counter sample이 10.8%다.

Dataset majority가 heuristic과 일치하면 shortcut model도 높은 average accuracy를 얻는다. 따라서 overall score 하나만 보면 robust 3D reasoning과 distribution matching을 구분하기 어렵다.

### 2) Natural image에서 depth cue를 완전히 통제하기 어렵다

Real image에는 다음 cue가 동시에 존재한다.

- Vertical position
- Apparent size
- Occlusion
- Ground contact
- Lighting and shadow
- Background context
- Perspective lines
- Object category prior

Counter sample에서 model이 실패해도 어떤 cue 때문인지 정확히 분리하기 어렵다.

### 3) Behavioral split만으로 internal mechanism을 알 수 없다

Consistent-counter gap이 커도 model representation이 실제로 vertical and distance를 같은 direction으로 encode하는지 직접 알 수 없다.

Model은 representation에서는 axis를 분리하지만 decoder에서 bias를 쓸 수도 있고, 반대로 representation 자체가 entangled되어 있을 수도 있다.

### 4) More data가 bias를 해결한다는 가정을 검증해야 한다

Spatial instruction data를 늘리면 score는 오를 수 있다. 그러나 dataset이 같은 perspective correlation을 반복하면 shortcut도 더 강해질 수 있다.

논문은 80k, 400k, 800k, 2M scale의 spatial fine-tuning을 비교해 data scaling이 representation structure를 어떻게 바꾸는지 본다.

# 2. Core Idea

## 2-1. Main contribution

논문의 contribution은 네 가지다.

### 1) Perspective-consistency audit

Existing spatial benchmark의 question을 relation label과 image layout에 따라 consistent and counter group으로 나눈다.

이 audit는 benchmark의 score distribution을 다시 읽게 만든다. 예를 들어 Qwen2.5-VL-3B를 2M spatial sample로 fine-tuning했을 때 EmbSpatial consistent accuracy는 60.9%, counter accuracy는 24.0%로 36.9 pp gap이 남는다.

More data가 consistent region을 잘 맞추지만 counter region을 같은 속도로 개선하지 못한다.

### 2) SpatialTunnel

SpatialTunnel은 Blender로 만든 controlled tunnel environment다.

- 두 object의 depth를 고정한다.
- Tunnel 중심축 주위 angular position을 바꾼다.
- 16 x 16 grid로 두 object의 image-plane placement를 sweep한다.
- Shape, color, size, lighting을 randomize한다.
- Background and perspective structure를 단순화한다.

이 design은 actual depth ordering을 유지하면서 vertical image position을 바꾼다. 따라서 "above means far" heuristic이 ground truth와 일치하거나 충돌하는 cell을 정확히 만들 수 있다.

### 3) Contrastive hidden-state probing

Question pair를 최소한으로 바꿔 opposite relation을 묻는다.

예를 들어 original question과 object-swapped question을 만든다.

- Original: Is cup left or right of book?
- Swapped: Is book left or right of cup?

각 question의 final-token hidden state를 $h_{q_1}$과 $h_{q_2}$라고 하면 delta vector는 다음과 같다.

$$
\delta = h_{q_2} - h_{q_1}
$$

이 delta는 object identity와 shared visual context를 상당 부분 상쇄하고 relation direction에 집중하도록 만든다.

### 4) Representation metrics

논문은 두 핵심 metric을 사용한다.

#### Axis Coherence

같은 axis에 속하는 sign-corrected delta vector가 stable direction을 이루는지 본다.

$$
\operatorname{Coh}
= \frac{2}{N(N-1)}
\sum_{i<j}
\cos(\tilde{\delta}_i, \tilde{\delta}_j)
$$

높을수록 relation axis가 일관된 hidden direction으로 encode된다.

#### Vertical-Distance Entanglement Index

Vertical and distance direction이 perspective heuristic에 맞게 결합되어 있는지 본다.

$$
\operatorname{VD\text{-}EI}
= \frac{1}{4}
[
\cos(\mu_{\mathrm{above}}, \mu_{\mathrm{far}})
+ \cos(\mu_{\mathrm{below}}, \mu_{\mathrm{close}})
- \cos(\mu_{\mathrm{above}}, \mu_{\mathrm{close}})
- \cos(\mu_{\mathrm{below}}, \mu_{\mathrm{far}})
]
$$

Positive value가 클수록 above-far and below-close coupling이 강하다. Lower value가 cleaner axis separation을 뜻한다.

## 2-2. Design intuition

논문의 strongest intuition은 counterfactual symmetry다.

Question에서 object order만 바꾸면 correct relation은 inverse가 된다. Image, object pair, lexical content는 거의 동일하다. Hidden state difference를 보면 relation encoding을 더 직접적으로 분리할 수 있다.

또 하나의 intuition은 behavior and representation을 함께 보는 것이다.

| Observation | Possible interpretation |
| --- | --- |
| High overall, high gap | Majority shortcut으로 benchmark score를 얻을 가능성 |
| High counter, high Coh_D | Stable distance representation과 robustness |
| Similar score, different VD-EI | Same behavior score가 different internal mechanism을 숨김 |
| More data, unchanged Coh_D | Scaling이 representation bottleneck을 해결하지 못함 |
| Low gap on natural benchmark only | Dataset skew 때문에 false robustness 가능 |

SpatialTunnel은 behavioral intervention이고 contrastive probe는 representational diagnosis다. 두 축을 같이 사용해 shortcut claim을 강화한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | VLM spatial benchmark 성능이 genuine axis representation인지 shortcut인지 진단 |
| Main bias | Vertical image position and distance entanglement |
| Behavioral audit | Perspective-consistent vs counter-heuristic split |
| Controlled benchmark | SpatialTunnel |
| Probe unit | Original and object-swapped question hidden-state difference |
| Representation metrics | Axis coherence and VD-EI |
| Main models | Molmo-7B-O, NVILA-Lite-2B, Qwen2.5-VL-3B |
| Reference models | RoboRefer-2B-SFT, Qwen3-VL-235B-A22B |
| Data scaling | 80k, 400k, 800k, 2M spatial samples |
| Main finding | Distance axis is weakest and predicts counter robustness |

## 3-2. Module breakdown

### 1) Benchmark skew analyzer

각 sample에서 두 object의 image position과 ground-truth relation을 사용해 perspective consistency를 판정한다.

Distance question에 대해 다음이 heuristic-consistent하다.

- Higher object is farther.
- Lower object is closer.

반대면 counter-heuristic이다.

이 분류는 benchmark를 새로 만드는 것이 아니라 existing sample의 hidden distribution structure를 드러낸다.

### 2) Spatial data scaling

Base VLM을 five-dataset mixture로 fine-tuning한다.

- SAT
- RoboSpatial
- SPAR-7M
- RefSpatial
- PRISM

Total sample scale은 80k, 400k, 800k, 2M이다.

Scaling curve에서 overall score뿐 아니라 consistent and counter accuracy를 따로 본다. Bias가 줄려면 counter improvement가 consistent improvement를 따라가야 한다.

### 3) SpatialTunnel renderer

SpatialTunnel은 single-point perspective corridor 안에 two objects를 배치한다.

각 configuration에서 다음을 통제한다.

- Object depth
- Angular position
- Image vertical and horizontal placement
- Shape
- Color
- Size
- Lighting

Phase-variation split은 16 x 16 angular grid를 사용해 3,072 rows를 제공한다. Contrastive probing split은 balanced question pair 1,200 rows를 제공한다. 공개 dataset에는 size-variation 1,100 rows도 포함된다.

### 4) Hidden-state extraction

각 VLM에서 original and swapped question을 같은 image와 함께 forward한다.

- Every transformer layer에 hook을 건다.
- Final answer token 직전 또는 final question token의 hidden state를 추출한다.
- Per-layer delta vector를 저장한다.
- Axis coherence, pairwise cosine matrix, PCA를 계산한다.

Paper-reported metric은 model별 selected layer $L^*$에서 계산한다. Layer는 axis coherence가 plateau에 도달하고 VD-EI가 stable하며, next-token prediction에 특화된 last few layers는 피하는 기준으로 고른다.

### 5) Sign alignment

Left and right, above and below, far and close는 inverse relation이다. Delta direction의 sign을 맞추지 않으면 같은 axis가 반대 방향으로 나타나 coherence가 낮아진다.

따라서 relation pair를 canonical direction으로 sign-correct한 뒤 mean and cosine을 계산한다.

### 6) Cross-benchmark correlation

Representation metric이 SpatialTunnel에만 맞는 diagnostic인지 확인하기 위해 EmbSpatial and CV-Bench-3D counter accuracy와 correlation을 본다.

Distance coherence와 counter accuracy 사이에 높은 Spearman correlation을 보고한다.

- EmbSpatial: $\rho = 0.759$
- CV-Bench-3D: $\rho = 0.804$
- Both: $p < 10^{-3}$

Correlation은 causation을 증명하지 않지만, internal structure가 robustness marker로 유용함을 보여준다.

# 4. Training / Data / Recipe

## 4-1. Models

Main family는 다음과 같다.

| Model | Scale | Role |
| --- | ---: | --- |
| Molmo-7B-O | 7B | Base and spatial fine-tuning study |
| NVILA-Lite-2B | 2B | Efficient VLM scaling study |
| Qwen2.5-VL-3B | 3B | Base and 2M spatial data study |
| RoboRefer-2B-SFT | 2B | Large RGB-D and spatial supervision reference |
| Qwen3-VL-235B-A22B | MoE | Very large-scale reference |

RoboRefer와 Qwen3-VL은 training recipe가 main three families와 동일하지 않다. 따라서 result는 controlled single-factor comparison보다 reference point로 읽어야 한다.

## 4-2. Fine-tuning data

Five spatial dataset mixture를 total count에 맞춰 scale한다.

- 80k
- 400k
- 800k
- 2M

Data composition과 sampling ratio는 appendix를 확인해야 한다. 중요한 것은 same family 안에서 data scale에 따라 behavioral gap and representation metric을 추적한다는 점이다.

## 4-3. Probe recipe

공개 code 기준 practical pipeline은 다음과 같다.

1. Model and processor를 load한다.
2. Original and swapped question을 만든다.
3. Every transformer layer에 forward hook을 등록한다.
4. 두 forward의 final-token hidden state를 저장한다.
5. Delta vector를 계산한다.
6. Relation direction을 sign-align한다.
7. Per-layer coherence와 6 x 6 similarity matrix를 계산한다.
8. VD-EI, PCA, cross-scale plot을 만든다.
9. Stable plateau에서 paper layer를 선택한다.

이 probe는 linear classifier를 새로 training하지 않는다. Hidden vector geometry를 직접 계산하는 lightweight diagnostic다.

## 4-4. SpatialTunnel configs

공개 dataset은 다음 config를 가진다.

| Config | Purpose | Rows |
| --- | --- | ---: |
| contrastive_probing | Balanced relation pair와 hidden-state probe | 1,200 |
| phase_variation | 16 x 16 angular position grid | 3,072 |
| size_variation | Apparent-size shortcut analysis | 1,100 |

Paper의 main narrative는 vertical-distance entanglement에 집중하지만, release는 size-distance cue 분석도 확장할 수 있게 구성되어 있다.

# 5. Evaluation

## 5-1. Main results

### 1) Existing benchmark가 perspective-consistent sample에 치우쳐 있다

| Benchmark split | Consistent | Counter |
| --- | ---: | ---: |
| EmbSpatial-Bench | 80.9% | 10.7% |
| CV-Bench-3D | 60.5% | 10.8% |

Remaining sample은 heuristic과 neutral하거나 classification이 어려운 case일 수 있다. Main point는 counter sample이 매우 적다는 것이다.

### 2) 모든 model에서 consistent-counter gap이 나타난다

Project page의 representative result에서 model별 gap은 상당하다.

| Model | EmbSpatial counter | EmbSpatial consistent | Gap |
| --- | ---: | ---: | ---: |
| Molmo-7B | 34.9 | 63.5 | -28.6 pp |
| NVILA-Lite-2B | 27.1 | 49.0 | -21.9 pp |
| RoboRefer-2B | 59.7 | 87.0 | -27.3 pp |
| Qwen2.5-VL-3B | 32.6 | 54.7 | -22.1 pp |
| Qwen3-VL-235B | 41.7 | 73.3 | -31.6 pp |

CV-Bench-3D에서는 RoboRefer and Qwen3-VL의 gap이 각각 3.5 pp and 7.3 pp로 작아, model마다 cross-benchmark robustness가 다르다.

### 3) Data scaling이 gap을 자동으로 제거하지 않는다

Qwen2.5-VL-3B with 2M spatial samples는 EmbSpatial에서 consistent 60.9%, counter 24.0%로 36.9 pp gap을 보인다.

Scaling은 consistent region의 score를 꾸준히 올리지만, counter region은 불안정하거나 느리게 개선된다. SpatialTunnel heatmap에서도 consistent cell은 좋아지는 반면 counter cell은 계속 어렵다.

이 결과는 more spatial instruction data가 shortcut-free representation을 보장하지 않음을 보여준다.

### 4) Distance axis coherence가 가장 약하다

Horizontal and vertical relation은 비교적 stable direction을 형성한다. Distance delta는 origin 근처에 모이거나 vertical cluster와 overlap하는 경향이 있다.

RoboRefer-2B의 representative coherence는 다음과 같다.

| Axis | Coherence |
| --- | ---: |
| Horizontal | 0.649 |
| Vertical | 0.830 |
| Distance | 0.182 |

Best reference model에서도 distance가 가장 약한 axis다.

### 5) Internal geometry가 비슷한 benchmark score 사이를 구분한다

Molmo, NVILA, Qwen fine-tuned model은 horizontal and vertical cluster는 분리되지만 distance vector가 vertical direction과 겹친다.

RoboRefer-2B는 three axis가 더 clean하게 분리되고, Qwen3-VL-235B도 scale을 통해 comparable structure를 보인다.

이는 same average accuracy를 가진 model도 different failure mechanism을 가질 수 있음을 보여준다.

### 6) Coh_D가 counter accuracy와 상관한다

SpatialTunnel or model-scale sweep에서 distance coherence가 높을수록 counter accuracy가 높다.

- EmbSpatial correlation: $\rho = 0.759$
- CV-Bench-3D correlation: $\rho = 0.804$

Representation metric이 benchmark-specific score보다 robust model selection signal이 될 가능성을 보여준다.

### 7) RGB-D supervision and scale are two routes

RoboRefer-2B는 large-scale RGB-D spatial supervision을 사용해 높은 robustness를 보인다. Qwen3-VL-235B는 targeted recipe가 같지 않지만 enormous pretraining scale에서 separated axis가 나타난다.

둘은 동일 causal mechanism을 증명하지 않는다. 다만 clean spatial representation이 auxiliary depth signal 또는 very large-scale learning에서 나타날 수 있다는 reference를 준다.

## 5-2. What really matters in the experiments

### 1) Counter accuracy를 따로 보고해야 한다

Overall score 개선만 보고하면 majority heuristic을 강화한 model을 더 좋은 spatial reasoner로 선택할 수 있다.

Practical leaderboard에는 다음을 같이 넣는 것이 좋다.

- Overall accuracy
- Consistent accuracy
- Counter accuracy
- Gap
- Axis coherence
- VD-EI

### 2) Synthetic benchmark는 replacement가 아니라 diagnostic이다

SpatialTunnel은 real-world complexity를 줄이는 대신 causal variable을 통제한다. 실제 deployment accuracy를 대표하기보다 failure mechanism을 찾는 도구다.

### 3) Representation probe는 layer choice에 민감하다

Last-token hidden state와 selected layer를 사용한다. Different pooling, token, layer에서 metric이 달라질 수 있다.

논문은 plateau criterion과 cross-layer stability를 제공하지만, new model에 적용할 때 layer cherry-picking을 피해야 한다.

### 4) Correlation이 intervention을 대신하지는 않는다

Coh_D가 robustness와 correlate해도 coherence를 직접 높이면 counter accuracy가 올라간다는 causal result는 아니다.

다음 단계는 representation regularizer, RGB-D auxiliary loss, balanced counter curriculum이 Coh_D and robustness를 함께 바꾸는지 실험하는 것이다.

# 6. Limitations

1. SpatialTunnel은 simplified synthetic world다.
   - Real scene의 occlusion, texture, complex camera, dynamic object를 충분히 포함하지 않는다.
   - Synthetic robustness가 real embodied reasoning으로 그대로 이어지지 않을 수 있다.

2. Probe가 linear hidden geometry에 의존한다.
   - Information이 nonlinear manifold에 존재하거나 distributed token에 분산되어 있으면 delta cosine이 놓칠 수 있다.

3. Final-token representation이 전체 spatial computation을 대표한다고 가정한다.
   - Visual token, intermediate cross-attention, answer token의 역할이 다를 수 있다.

4. Layer selection이 metric에 영향을 준다.
   - Model마다 selected layer가 다르다.
   - Cross-model comparison에서 equivalent computational stage인지 완전히 보장하기 어렵다.

5. Main model family 수가 제한적이다.
   - Three fine-tuned families와 two reference models가 모든 VLM architecture를 대표하지 않는다.

6. Training recipe가 reference model 사이에서 통제되지 않는다.
   - RoboRefer and Qwen3-VL의 data, architecture, scale이 동시에 다르다.
   - Clean representation의 원인을 one factor로 귀속하기 어렵다.

7. Benchmark skew는 real-world prior일 수도 있다.
   - Natural image에서 above-far correlation은 실제로 유용하다.
   - Shortcut을 완전히 제거하기보다 uncertainty와 conflicting cue handling을 개선해야 할 수 있다.

8. Relative depth는 full 3D understanding의 일부다.
   - Metric distance, camera geometry, object permanence, action consequence, multi-view consistency는 별도 capability다.

9. Correlation is not causation.
   - Coh_D and counter accuracy가 함께 좋아지는 hidden factor가 있을 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 좋은 점은 benchmark audit, synthetic intervention, representation probe를 하나의 story로 묶은 데 있다.

Shortcut paper는 흔히 counterexample 몇 개를 보여주거나 average gap만 보고 끝난다. 이 논문은 왜 gap이 생기는지 hidden geometry까지 연결하고, 그 geometry가 다른 benchmark robustness를 예측하는지 확인한다.

Document VLM에도 같은 접근을 적용할 수 있다.

- Value가 오른쪽에 있으면 answer라는 layout shortcut
- Bold text면 key라는 style shortcut
- Table top row를 header로 고정하는 shortcut
- OCR confidence와 semantic relevance의 entanglement
- Long address에서 first bounding box만 선택하는 shortcut

Average extraction accuracy가 올라가도 counter-layout case에서 무너질 수 있다. 따라서 relation-consistent and counter split을 만들고 hidden representation의 axis separation을 볼 수 있다.

## 7-2. Reuse potential

### 1) Counter-heuristic split

Dataset에서 common prior와 반대되는 case를 별도 tag한다.

- Above but close
- Below but far
- Small but close
- Large but far
- Occluded but foreground

Training and evaluation에서 counter subset을 반드시 보고한다.

### 2) Minimal contrastive pair

Object identity and image를 유지하고 question relation만 invert한다. Hidden delta를 사용하면 relation feature를 더 clean하게 볼 수 있다.

이 방법은 spatial relation 외에도 적용 가능하다.

- Before / after
- Cause / effect
- Key / value
- Header / body
- Evidence / distractor

### 3) Representation-aware checkpoint selection

Validation accuracy만 보지 않고 coherence and entanglement metric을 함께 본다. Same score라면 counter robustness가 높은 checkpoint를 선택한다.

### 4) Balanced curriculum

Consistent and counter sample ratio를 통제한다. More data보다 correlation-breaking data가 representation을 더 잘 바꿀 수 있다.

### 5) Auxiliary geometry supervision

RGB-D, camera pose, metric depth, scene graph를 auxiliary target으로 사용해 distance axis를 직접 구조화할 수 있다.

### 6) Intervention-based validation

Synthetic renderer에서 one factor만 바꾼다.

- Vertical position fixed, depth varied
- Depth fixed, vertical position varied
- Apparent size fixed, depth varied
- Background fixed, camera pose varied

Behavior and representation가 같이 바뀌는지 본다.

## 7-3. Follow-up papers

- EmbSpatial-Bench
- CV-Bench: A Benchmark for 2D and 3D Spatial Understanding in Vision-Language Models
- RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models
- SpatialRGPT and 3D-aware VLM studies
- Probing Classifiers: Promises, Shortcomings, and Advances
- Shortcut Learning in Deep Neural Networks
- Counterfactual evaluation for vision-language reasoning

# 8. Summary

- VLM은 natural image의 above-far correlation을 genuine distance reasoning 대신 사용할 수 있다.
- Existing benchmark는 perspective-consistent sample이 많고 counter sample이 적어 average score가 shortcut을 숨긴다.
- SpatialTunnel은 vertical placement와 actual depth를 분리해 bias를 controlled하게 측정한다.
- Contrastive hidden-state delta로 horizontal, vertical, distance axis coherence와 VD-EI를 계산한다.
- Data scaling은 overall score를 올려도 consistent-counter gap과 vertical-distance entanglement를 자동으로 제거하지 않는다.
- Distance coherence는 counter accuracy와 강하게 상관하며, representation structure가 robustness marker가 될 수 있다.
