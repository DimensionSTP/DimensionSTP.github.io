---
layout: single
title: "From Activation to Causality: Discovery of Causal Visual Representations in the Human Brain Review"
categories: Study-concept
tag: [BrainCause, NeuroAI, fMRI, Causality, Vision, Representation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2605.23895)

[Project page](https://yuvalgol123.github.io/BrainCause/)

[Data link](https://huggingface.co/datasets/BrainCause/Concept_Targeted_Causal_Images)

From Activation to Causality는 visual neuroscience 논문이지만, AI researcher나 engineer 입장에서도 꽤 읽을 가치가 큰 논문이다. 이 논문의 진짜 흥미로운 지점은 fMRI로 새로운 brain map을 만들었다는 데만 있지 않다. 더 중요한 포인트는, "activation이 크다"와 "그 concept 때문에 activation이 생긴다"를 분리해서 보려 한다는 점이다.

기존의 많은 brain representation 분석은 특정 category image를 넣었을 때 어떤 voxel이나 region이 강하게 반응하는지를 본다. 예를 들어 face image에 강하게 반응하면 face-selective region이라고 해석하는 식이다. 하지만 strong activation만으로는 충분하지 않다. Face image에는 face만 있는 것이 아니라 skin color, pose, background, gaze, human body, social context 같은 correlated cue가 같이 들어간다. Region이 face 자체에 반응한 것인지, 아니면 face와 자주 같이 나타나는 cue에 반응한 것인지는 별도로 검증해야 한다.

BrainCause는 이 간극을 generative model, VLM verifier, image-to-fMRI encoder를 조합해 메우려 한다. Target concept이 주어지면 positive image, semantic negative image, counterfactual edit image를 자동으로 만들고, 이 controlled stimulus set에 대해 brain response를 예측하거나 측정해서 candidate region의 causal specificity를 평가한다.

> 한 줄 요약: BrainCause는 visual concept에 강하게 반응하는 brain region을 찾는 데서 멈추지 않고, semantic negative와 counterfactual edit을 통해 그 반응이 target concept 자체에 specific한지 검증하는 automated causal discovery framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- activation maximization이나 concept localization을 그대로 representation evidence로 읽는 관행에 꽤 강한 경고를 준다.
- generative model을 "pretty image generator"가 아니라 controlled experiment generator로 쓰는 좋은 예시다.
- VLM, image editing, image-to-fMRI encoder, retrieval, statistical testing을 하나의 closed-loop discovery pipeline으로 묶는다.
- AI model interpretability에서도 그대로 재사용할 수 있는 평가 철학을 준다. 잘 켜지는 feature와 causal하게 필요한 feature는 다르다.
- 실험 결과에서 activation-based discovery의 false positive 문제가 숫자로 꽤 선명하게 드러난다.

이 논문의 핵심 메시지는 단순하다. 어떤 region이 concept image에 강하게 반응한다고 해서, 그 region이 그 concept을 represent한다고 바로 말하면 안 된다. Representation claim을 하려면 최소한 "target concept이 있을 때 켜지고, target과 헷갈리는 대안이나 target만 제거한 counterfactual에서는 내려가는가"를 같이 봐야 한다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 human brain에서 visual concept representation을 더 신뢰성 있게 찾는 것이다. 여기서 visual concept은 face, place, body, word 같은 broad category일 수도 있고, human hand, animal face, handwritten text, logo, social interaction 같은 더 fine-grained concept일 수도 있다.

문제는 다음 질문으로 정리할 수 있다.

- 어떤 brain voxel 또는 region이 target concept에 강하게 반응하는가.
- 그 반응이 target concept 자체 때문인가.
- 아니면 target concept과 함께 나타나는 background, color, pose, object co-occurrence, semantic context 때문인가.
- 기존 measured fMRI data만으로 이 질문에 답할 수 있는가.
- 부족하다면 어떤 follow-up stimulus를 새로 찍어야 하는가.

이 논문은 단순 localization이 아니라 validation까지 포함한다. 즉 output은 "여기가 concept region이다"에서 끝나지 않는다. BrainCause는 candidate region, causal score, confidence, measured-data coverage, follow-up fMRI stimulus recommendation까지 같이 내놓는다.

## 1-2. Why previous approaches are insufficient

기존 접근은 대체로 activation-based contrast에 의존한다. Target concept image를 넣었을 때 response가 평균보다 높으면 candidate region으로 본다. Classical neuroscience에서는 face, body, place, word 같은 broad category를 carefully controlled stimulus로 비교해 큰 성과를 냈다. 최근에는 image-to-fMRI encoder가 등장하면서 measured image에 없는 image에 대해서도 predicted fMRI response를 만들 수 있게 되었고, large image pool retrieval이나 activation maximization으로 더 다양한 concept을 탐색할 수 있게 되었다.

하지만 이 방식에는 핵심 약점이 있다.

- High activation은 specificity를 보장하지 않는다.
- Target concept과 correlated cue를 분리하지 못하면 false positive가 생긴다.
- Retrieval 기반 방법은 retrieved image가 정말 target concept만을 isolate하는지 보장하기 어렵다.
- Generative activation maximization도 region이 좋아하는 visual artifact를 만들 수는 있지만, 그 artifact가 concept 자체인지 correlated feature인지는 별도 검증이 필요하다.
- Measured fMRI dataset의 coverage가 concept마다 다르기 때문에, measured validation score의 의미도 concept별로 달라진다.

예를 들어 surfing concept을 찾는다고 하자. Region이 surfing image에 강하게 반응하더라도, 그것이 surfing action 때문인지, water 때문인지, human body pose 때문인지, beach scene 때문인지는 분리해야 한다. Activation만 보면 이 region을 surfing representation으로 부를 수 있지만, counterfactual에서 surfboard나 person action을 제거했는데도 activation이 유지된다면 이야기가 달라진다.

이 논문의 문제 설정은 neuroscience 안에만 머물지 않는다. ML interpretability에서도 같은 문제가 반복된다. 특정 neuron이나 feature가 "dog feature"처럼 보인다고 해도, 실제로는 fur texture, grass background, image style, camera angle에 반응하는 경우가 많다. BrainCause는 이 문제를 fMRI setting에서 매우 체계적으로 재정의한다.

# 2. Core Idea

## 2-1. Main contribution

BrainCause의 핵심 기여는 target concept에 대해 controlled causal stimulus set을 자동으로 만들고, 그 stimulus set으로 candidate brain region의 specificity를 평가하는 것이다.

구성은 크게 세 단계다.

1. Concept-targeted causal dataset generation

   Target concept에 대해 positive image, semantic negative image, counterfactual negative image를 만든다. Positive는 target concept을 포함한다. Semantic negative는 target과 visual 또는 semantic하게 관련 있지만 target 자체는 포함하지 않는다. Counterfactual negative는 positive image에서 target concept만 제거하거나 대체하고 나머지 content는 최대한 유지한다.

2. Concept-selective representation search

   각 voxel에 대해 activation score와 causal specificity score를 계산한다. Positive에 강하게 반응하는지만 보지 않고, semantic negative와 counterfactual edit보다 positive에 더 강하게 반응하는지를 본다. Candidate representation은 causal score가 높은 voxel set으로 정의된다.

3. Final verdict and follow-up experiment design

   Generated validation data와 measured fMRI data에서 causal evidence를 다시 확인한다. Measured dataset에 target positive와 negative가 충분히 있는지도 coverage 관점에서 평가한다. Evidence가 충분하면 high-confidence discovery로 보고, evidence가 부족하면 필요한 follow-up stimulus를 제안한다.

이 논문이 activation-based discovery와 가장 크게 갈리는 지점은 score definition이다. Activation만 보던 ranking을 causal specificity ranking으로 바꾼다. 즉 "얼마나 세게 켜지는가"보다 "target을 지웠거나 헷갈리는 대안을 넣었을 때 얼마나 내려가는가"를 같이 본다.

## 2-2. Design intuition

BrainCause의 설계 직관은 실험 설계 관점에서 보면 꽤 자연스럽다. 어떤 region이 target concept을 represent한다고 말하려면 최소한 세 가지 evidence가 필요하다.

- Positive evidence: target concept이 있는 image에서 response가 높아야 한다.
- Semantic control: target과 관련 있지만 target 자체는 없는 image에서는 response가 낮아야 한다.
- Counterfactual control: 같은 image context에서 target만 제거했을 때 response가 낮아야 한다.

이 세 가지 중 하나라도 빠지면 claim이 약해진다. Positive evidence만 있으면 region이 그냥 target image distribution에 반응한 것일 수 있다. Semantic negative가 없으면 co-occurring object나 scene에 속을 수 있다. Counterfactual edit이 없으면 background, composition, color, pose 같은 nuisance factor를 통제하기 어렵다.

BrainCause가 generative model을 쓰는 이유도 여기에 있다. 기존 measured fMRI dataset은 이미 찍힌 image set이라서 원하는 counterfactual pair가 충분하지 않다. 반면 text-to-image와 image editing model을 쓰면 target concept을 isolate하는 새로운 stimulus를 대량으로 만들 수 있다. 이 generated stimulus에 image-to-fMRI encoder를 붙이면, 실제 scanner를 바로 돌리기 전에 candidate region을 탐색하고 검증할 수 있다.

이 논문은 "causal"이라는 말을 brain stimulation이나 lesion 같은 direct intervention 의미로 쓰지는 않는다. 여기서 causality는 stimulus-level intervention에 가깝다. Target concept을 image 안에서 제거하거나 관련 대안으로 바꾸었을 때 predicted 또는 measured brain response가 어떻게 바뀌는지를 본다. 이 차이를 정확히 이해해야 한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | visual concept에 대한 causally specific brain representation discovery |
| Method name | BrainCause |
| Input | target visual concept and subject-specific image-fMRI data |
| Core data | positive images, semantic negatives, counterfactual negatives |
| Main model components | LLM prompt generator, text-to-image model, image editing model, VLM verifier, image-to-fMRI encoder |
| Search target | voxel set or region with high activation and high causal specificity |
| Validation | generated held-out stimuli and measured fMRI retrieval data |
| Output | candidate region, causal score, confidence, coverage analysis, follow-up stimuli |
| Main baseline | Max Activation, MindSimulator, MindSimulator+VLM |
| Main claim | activation alone produces many false positives, while causal ranking finds more faithful concept representations |

## 3-2. Module breakdown

### 1) Concept-targeted causal dataset generation

BrainCause는 target concept 하나마다 dedicated dataset을 만든다. 논문은 260 visual concepts를 다루며, 각 concept에 대해 세 종류의 stimulus를 구성한다.

첫 번째는 positive image다. LLM이 target concept을 다양하게 표현하는 prompts를 만들고, text-to-image model이 prompt당 image를 생성한다. 논문 기준으로 concept마다 training positive image 200개, validation positive image 100개를 생성한다.

두 번째는 semantic negative image다. LLM은 target과 헷갈릴 수 있는 counter concept 10개를 제안한다. 예를 들어 human hands에 대해 human body, human legs, robot hands 같은 대안이 나올 수 있다. 각 counter concept마다 prompts를 만들고 image를 생성한 뒤, VLM verifier가 target concept이 실제로 없는지 확인한다. Filtering 뒤에는 training과 validation에 각각 대략 80개에서 100개의 semantic negative가 남는다.

세 번째는 counterfactual negative image다. Positive image에서 target concept만 제거하거나 대체하되, 나머지 content는 최대한 유지하도록 image editing을 수행한다. 논문은 training positive 50개와 validation positive 20개에 대해 edit prompts를 만들고, 대략 400개에서 500개의 counterfactual negative를 얻는다.

이 pipeline의 핵심은 negative를 그냥 random negative로 만들지 않는다는 점이다. BrainCause의 negative는 target과 가까운 semantic alternative이거나, 같은 image에서 target만 제거한 counterfactual이다. 그래서 false positive를 훨씬 더 세게 때린다.

### 2) Image-to-fMRI response prediction

Generated image는 실제 fMRI scanner에서 측정된 것이 아니다. BrainCause는 image-to-fMRI encoder를 사용해 subject-specific predicted fMRI response를 얻는다. 즉 image 하나가 들어오면 각 subject에 대한 voxel-wise response vector가 나온다.

논문은 NSD preprocessed fMRI response를 사용한다. NSD는 7T fMRI dataset이고, 8명의 subject가 natural image를 본 데이터로 구성된다. 실험에서는 모든 session을 완료한 4명의 subject를 보고한다. 각 subject는 대략 10000개 natural image를 보았고, preprocessing 뒤 각 image는 voxel response vector와 연결된다.

Measured data도 버리지 않는다. BrainCause는 generated data로만 평가하지 않고, NSD measured image pool에서 target positive와 semantic negative를 CLIP retrieval로 찾은 뒤 VLM verification을 거쳐 measured validation에도 사용한다. 이 점이 중요하다. Generated stimulus만으로 high score가 나오는 것과, 실제 measured fMRI dataset에서도 같은 방향의 evidence가 나오는 것은 다른 수준의 claim이다.

### 3) Activation and causal specificity scoring

BrainCause는 각 voxel $v$에 대해 세 종류의 score를 계산한다. 표기는 이해를 위한 개념식이다. $P$는 positive image set, $N$은 semantic negative image set, $E(x)$는 positive image $x$에 대응되는 counterfactual edits, $r_v(x)$는 image $x$에 대한 voxel $v$의 response라고 두자.

Positive activation은 다음처럼 볼 수 있다.

$$
A(v) = \operatorname{mean}_{x \in P} r_v(x)
$$

Semantic negative specificity는 positive response가 가장 헷갈리는 negative response보다 얼마나 높은지를 본다.

$$
S_{sem}(v) = A(v) - \operatorname{mean}_{x \in TopK(N, v)} r_v(x)
$$

여기서 $TopK(N, v)$는 voxel $v$를 가장 강하게 activate하는 semantic negative 10개다. 즉 easy negative가 아니라 hardest negative를 본다.

Counterfactual specificity는 target을 제거한 edited image에서도 response가 유지되는지 본다.

$$
S_{cf}(v) = \operatorname{mean}_{x \in P} (r_v(x) - \max_{e \in E(x)} r_v(e))
$$

최종 causal score는 semantic negative score와 counterfactual score를 합친다.

$$
S_{causal}(v) = \frac{S_{sem}(v) + S_{cf}(v)}{2}
$$

이 score가 positive이면, 적어도 현재 만들어진 semantic negative와 counterfactual edit 기준에서는 target concept에 더 specific하게 반응한다고 볼 수 있다. Candidate representation은 causal score가 양수인 voxel set이나, ranking 상위 voxel set으로 정의된다. Main quantitative evaluation에서는 concept마다 top 100 voxels region을 주로 사용한다.

### 4) Final verdict and measured-data coverage

BrainCause는 candidate region을 만든 뒤 바로 결론을 내리지 않는다. 두 가지를 같이 본다.

첫째는 causal evidence다. Generated validation data와 measured fMRI data에서 activation, semantic negative specificity, counterfactual specificity가 target concept에 대해 유의미한지 본다. 논문은 concept-specific baseline과 one-sided empirical p-value를 사용해 significance를 평가한다.

둘째는 measured-data coverage다. 어떤 concept은 NSD 안에 positive image가 충분히 있을 수 있고, 어떤 concept은 거의 없을 수 있다. 이 coverage가 낮으면 measured validation score의 해석력이 약해진다. BrainCause는 target positive와 semantic negative가 measured dataset에서 얼마나 잘 retrieved and verified되는지 분석하고, coverage가 낮은 concept에 대해서는 follow-up fMRI experiment가 필요하다고 표시한다.

이 설계가 좋은 이유는 negative result 해석을 조심스럽게 만든다는 점이다. Measured-data score가 낮다고 해서 region이 없다고 바로 말하지 않는다. 만약 measured image pool에 해당 concept이나 좋은 semantic negative가 충분하지 않았다면, evidence 부족으로 남긴다. 이건 실제 실험 설계에서 매우 중요한 태도다.

### 5) Follow-up experiment recommendation

BrainCause의 output에는 follow-up stimulus recommendation도 포함된다. 어떤 concept은 generated validation에서는 promising하지만 measured coverage가 낮을 수 있다. 이때 BrainCause는 추가로 찍으면 좋은 positive example, semantic negative, counterfactual edit을 제안한다.

이 부분은 논문의 엔지니어링적 가치가 크다. 단순히 "automatic discovery"에서 끝나는 것이 아니라, 다음 실험에서 어떤 image를 scanner에 넣어야 evidence가 강해질지까지 이어진다. ML 관점으로 보면 active data collection에 가깝고, neuroscience 관점으로 보면 hypothesis generation과 experiment design을 연결한 형태다.

# 4. Training / Data / Recipe

## 4-1. Data

가장 중요한 measured dataset은 NSD다. 논문은 preprocessed NSD fMRI response를 사용하며, response는 image-level beta value로 정리된다. 각 voxel은 measured image 전체에서 mean 0, standard deviation 1이 되도록 normalize된다. 따라서 response가 0보다 크면 해당 voxel의 평균 response보다 높은 activation으로 해석할 수 있다.

BrainCause는 measured data 외에도 large predicted image-fMRI pool을 만든다. COCO에서 120K unlabeled images를 가져와 image-to-fMRI encoder로 response를 예측하고, retrieval과 MindSimulator baseline 구현에 사용한다.

Data 구성은 다음처럼 정리할 수 있다.

| Data component | Role | Note |
| --- | --- | --- |
| NSD measured fMRI | measured validation and retrieval | 8 subjects, approximately 10000 viewed images per subject |
| Reported subjects | quantitative evaluation | 4 subjects who completed all sessions |
| Generated positives | target concept evidence | 200 train and 100 validation images per concept |
| Semantic negatives | correlated alternative control | 10 counter concepts, approximately 80 to 100 images after filtering |
| Counterfactual negatives | target-removal control | approximately 400 to 500 edited negatives |
| COCO predicted pool | retrieval and baseline support | 120K images with predicted fMRI responses |
| Public HF dataset | released controlled images | viewer exposes concept-targeted generated images and metadata |

여기서 주의할 점은 paper-level concept count와 public dataset viewer count를 혼동하지 않는 것이다. 논문은 260 visual concepts를 study한다고 설명한다. 반면 Hugging Face viewer는 접근 시점에 235 subsets로 보였다. Public release scope, viewer indexing, filtering 상태가 다를 수 있으므로, 발행 전 dataset count를 논문 claim처럼 쓰지는 않는 편이 안전하다.

## 4-2. Model components

BrainCause는 하나의 end-to-end model을 새로 학습하는 논문이라기보다, 여러 foundation model을 실험 설계 pipeline으로 조합하는 논문에 가깝다.

논문 기준 주요 component는 아래와 같다.

| Component | Model or method | Role |
| --- | --- | --- |
| Prompt and counter-concept generation | Gemma-3-27B-IT | target positives, semantic negatives, edit instructions 생성 |
| Image generation and editing | FLUX.2-Klein-4B | positive and counterfactual image synthesis |
| Image-concept verification | Qwen3-VL-8B-Instruct | generated or retrieved image가 target을 포함하는지 확인 |
| Image retrieval | CLIP-based retrieval | measured and external image pool에서 positive or negative 후보 검색 |
| Image-to-fMRI encoder | Beliy et al. encoder | generated or retrieved image의 subject-specific fMRI response 예측 |
| Statistical testing | empirical one-sided p-value | discovered region의 target specificity 평가 |

이 조합에서 VLM verifier가 꽤 중요하다. Prompt가 "target을 제외하라"고 해도 generated image가 실제로 target을 포함할 수 있다. 특히 sky, reflection, lighting contrast 같은 broad visual property는 negative generation이 더 어렵다. 논문도 남은 false positive 중 일부가 semantic-negative generation failure와 연결된다고 분석한다.

## 4-3. Recipe and engineering notes

BrainCause의 전체 recipe는 아래처럼 볼 수 있다.

1. Target visual concept을 정한다.
2. LLM으로 positive prompts를 만든다.
3. Text-to-image model로 positive images를 생성한다.
4. LLM으로 semantically related counter concepts를 만든다.
5. Text-to-image model로 semantic negatives를 생성한다.
6. VLM verifier로 target absence와 negative concept presence를 확인한다.
7. Positive image에 대해 target-removal edit instruction을 만든다.
8. Image editing model로 counterfactual negatives를 만든다.
9. 모든 generated image를 image-to-fMRI encoder에 넣어 predicted response를 얻는다.
10. Voxel별 activation score와 causal specificity score를 계산한다.
11. Causal ranking으로 candidate voxel region을 만든다.
12. Generated held-out data와 measured fMRI retrieval data로 validate한다.
13. Coverage와 statistical evidence를 합쳐 final verdict를 낸다.
14. 필요하면 follow-up fMRI stimuli를 추천한다.

계산 비용도 작지는 않다. 논문은 target concept 하나에 대해 positive and negative image generation, 약 1000개 image 생성, score computation, region proposal까지 single H200 GPU에서 약 2시간이 걸린다고 설명한다. 모든 model은 open source model을 사용했다고 한다.

실무적으로 이 recipe의 가치는 "fully automatic brain discovery"라는 hype보다, controlled evaluation data를 어떻게 자동으로 만들고 검증할지에 있다. 특정 concept에 대한 hard negative와 counterfactual pair를 자동으로 만들고, verifier로 quality control을 건 뒤, score를 분해해서 region claim을 보수적으로 내는 방식은 다른 multimodal interpretability에도 꽤 재사용 가능하다.

# 5. Evaluation

## 5-1. Main results

논문은 Max Activation, MindSimulator, MindSimulator+VLM, BrainCause를 비교한다. Main quantitative table은 top 50 concepts와 4 subjects 평균이며, discovered region size는 100 voxels다.

| Method | Activation Gen | Activation Meas | Causal Gen | Causal Meas | Causal Edits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Max Activation | 2.76 | 0.70 | 0.08 | 0.18 | 0.44 |
| MindSimulator | 1.89 | 1.02 | -0.44 | 0.27 | 0.23 |
| MindSimulator+VLM | 2.13 | 1.12 | -0.26 | 0.41 | 0.38 |
| BrainCause | 2.05 | 1.08 | 0.62 | 0.71 | 0.98 |

이 표의 핵심은 BrainCause가 activation을 가장 크게 만드는 방법이 아니라는 점이다. Max Activation은 generated positive activation이 2.76으로 가장 높다. 하지만 causal score는 낮다. BrainCause는 activation이 2.05로 조금 낮아도, generated semantic negative, measured semantic negative, counterfactual edit 기준 causal score가 모두 훨씬 높다.

즉 이 논문이 말하는 것은 "더 잘 켜지는 region"이 아니라 "더 specific하게 켜지는 region"이다. Representation discovery에서는 이 차이가 중요하다.

Known functional region alignment도 확인한다.

| Region | Top 100 | Top 200 | Top 500 |
| --- | ---: | ---: | ---: |
| Bodies | 99% | 99% | 97% |
| Faces | 90% | 87% | 84% |
| Places | 74% | 75% | 74% |
| Words | 99% | 98% | 97% |

이 결과는 BrainCause가 completely arbitrary한 map을 만드는 것이 아니라, broad category에 대해서는 기존에 알려진 functional organization과 잘 맞는다는 sanity check에 가깝다. 그 위에서 human hand, human leg, animal face, handwritten text, logo, social interaction 같은 fine-grained concept을 더 살펴본다.

## 5-2. What really matters in the experiments

### 1) Activation-based discovery의 false positive가 매우 크다

논문에서 가장 강한 메시지는 activation-based discovery의 false positive 문제다. 저자들은 activation-based method가 high activation region을 잘 찾지만, causal evaluation을 통과하지 못하는 경우가 많다고 보인다. Generated stimuli 기준으로는 causal validation 없이 localization하면 많은 candidate가 false positive가 된다.

특히 Fig. 4 분석에서는 causal ranking을 쓰면 false positive rate가 73.4%에서 23%로 줄고, true positive rate는 26.6%에서 38.7%로 오른다고 보고한다. 이 결과는 단순히 conservative해져서 discovery를 덜 하는 것이 아니라, 더 faithful한 region을 찾는 데도 도움이 된다는 주장으로 이어진다.

### 2) Semantic negative와 counterfactual edit은 역할이 다르다

Semantic negative는 target과 가까운 alternative concept을 때린다. 예를 들어 text region을 찾고 싶다면 sign, sketch, symbol처럼 비슷하지만 target과 다른 image를 넣어야 한다. 이것은 semantic confound를 줄인다.

Counterfactual edit은 같은 image context에서 target만 제거한다. 이것은 background, color, object location, composition 같은 low-level and mid-level confound를 줄인다.

둘 중 하나만으로는 부족하다. Semantic negative만 있으면 같은 image context 통제가 약하고, counterfactual만 있으면 target과 co-occurring semantic alternative를 충분히 때리지 못할 수 있다. BrainCause가 둘을 합치는 이유가 여기에 있다.

### 3) Measured-data coverage는 score 해석의 일부다

논문에서 좋았던 지점은 coverage를 단순 부록으로 넘기지 않는다는 점이다. Measured dataset에 target positive가 거의 없으면 measured validation score가 낮아도 해석이 어렵다. 반대로 target positive와 hard semantic negative가 충분히 있으면 measured validation이 훨씬 강한 evidence가 된다.

이 관점은 benchmark 해석과도 닮았다. Dataset에 어떤 example이 얼마나 있는지 모르면, score가 낮은 이유를 model failure와 data coverage failure로 분해하기 어렵다. BrainCause는 이것을 brain representation discovery에서도 명시적으로 처리한다.

### 4) Causal map은 더 sparse하고 localized하게 나온다

Project page와 paper figure를 보면 activation-based localization은 더 broad한 high-response pattern을 만들 수 있다. 반면 causal scoring은 correlated cue response를 누르기 때문에 더 selective한 map을 만든다. Tools가 EBA 근처 body-part and action-related region에, animal faces가 FFA and OFA 같은 face-selective area에 가까이 localize되는 식의 결과도 보고된다.

이건 interpretability에서 많이 보는 현상과 비슷하다. Activation maximization은 "무엇이든 세게 켜는 input"을 찾기 쉽지만, specificity filter를 걸면 feature description이 더 좁아진다. BrainCause는 이 filtering을 brain response setting에서 formalize한 셈이다.

### 5) Subject consistency를 보되, individual variability도 남긴다

Appendix는 Animal, Human Interaction, Hands in Action 같은 concept에 대해 subject 1과 subject 2의 causal maps를 비교한다. 완전히 같은 map은 아니지만, 주요 representation region은 high-level visual cortex에서 correspondence를 보인다. 논문은 이것을 discovered representation이 individual variability에도 불구하고 어느 정도 robust한 visual organization을 포착한다는 evidence로 해석한다.

다만 이 부분도 과장하면 안 된다. fMRI subject variability는 본질적으로 크고, image-to-fMRI encoder noise도 있다. 그래서 subject consistency는 중요한 sanity check지만, 모든 fine-grained concept map을 universal map처럼 읽으면 안 된다.

# 6. Limitations

1. "Causal"은 stimulus-level causality다.

   BrainCause는 target concept을 image에서 제거하거나 related alternative로 바꾸는 stimulus intervention을 사용한다. 이것은 brain region을 직접 stimulation하거나 lesion하는 causal neuroscience와는 다르다. 따라서 결과를 "이 region이 해당 concept에 causal necessity를 가진다"로 읽으면 과하다. 더 안전한 표현은 "이 region의 response가 현재 설계한 visual intervention에서 target concept에 specific하게 변한다"에 가깝다.

2. Generated negative 품질에 크게 의존한다.

   Semantic negative와 counterfactual edit이 target을 제대로 제거하지 못하면 causal test가 흔들린다. 논문도 sky, reflection, lighting contrast 같은 broad property에서 semantic-negative generation failure가 남는다고 분석한다. VLM verifier가 들어가도 모든 confound를 막을 수는 없다.

3. Image-to-fMRI encoder noise가 있다.

   Generated image에 대한 response는 실제 measured fMRI가 아니라 encoder prediction이다. 논문은 measured data validation도 같이 쓰지만, generated candidate search와 evaluation 상당 부분은 encoding model 품질에 의존한다. Encoder가 특정 visual feature나 subject-specific response를 잘못 예측하면 causal score도 영향을 받는다.

4. Counterfactual set이 모든 confound를 덮지는 못한다.

   어떤 target concept은 매우 넓거나 추상적이다. Social interaction, lighting contrast, reflection 같은 concept은 target을 제거하면서 나머지 image를 그대로 유지하기 어렵다. 이 경우 "current negative set 기준 causal"과 "all possible confounds 기준 causal" 사이에는 차이가 남는다.

5. Measured data coverage가 concept마다 다르다.

   NSD는 매우 큰 dataset이지만 arbitrary visual concept과 semantic negative를 모두 충분히 포함하지는 않는다. BrainCause는 coverage를 추정하고 follow-up을 제안하지만, high-confidence discovery로 인정될 수 있는 concept은 measured coverage에 영향을 받는다.

6. Compute cost와 pipeline complexity가 작지 않다.

   Concept 하나당 약 1000개 image를 생성하고, verifier와 image-to-fMRI encoder를 돌리고, score를 계산한다. Single H200 기준 약 2시간이라는 설명은 연구 pipeline으로는 가능하지만, concept search를 매우 크게 확장할 때는 비용이 부담될 수 있다.

7. LLM-generated concept list와 negative list의 bias가 남는다.

   Concept set과 counter concept proposal은 LLM에 의존한다. LLM이 자주 떠올리는 concept, Western visual dataset에서 흔한 object, COCO-style scene bias가 stimulus design에 들어갈 수 있다. 이 bias는 discovered representation의 coverage와 interpretation에도 영향을 줄 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 NeuroAI 논문이지만, 실제로는 representation evaluation paper로 읽는 편이 더 생산적이다. 핵심 메시지는 아래 한 줄이다.

- Activation evidence와 causal specificity evidence를 분리해서 봐야 한다.

이건 LLM/VLM interpretability에도 그대로 적용된다. 예를 들어 특정 attention head, MLP neuron, SAE feature가 "OCR feature"처럼 보인다고 하자. OCR image에서 activation이 높다는 사실만으로는 부족하다. Text가 제거된 같은 background, text와 비슷한 line pattern, logo, traffic sign, handwritten text, printed word 같은 hard negatives를 넣었을 때 어떻게 변하는지 봐야 한다.

BrainCause가 좋은 이유는 이 평가 관점을 pipeline으로 만든다는 점이다. Concept을 정의하고, positive를 만들고, semantic negative를 만들고, counterfactual edit을 만들고, verifier로 품질을 확인하고, activation과 causal score를 분리한다. 이 방식은 brain mapping보다 넓게, concept probe benchmark나 model behavior audit에도 쓸 수 있다.

## 7-2. Reuse potential

재사용하고 싶은 포인트는 아래 5가지다.

1. Counterfactual-first evaluation

   Feature나 region을 설명할 때 positive examples만 보지 않는다. Target만 제거한 counterfactual pair를 같이 만든다.

2. Hard semantic negatives

   Random negative 대신 target과 가장 헷갈리는 semantic alternative를 사용한다. 실제 model이나 brain region이 무엇에 속는지를 보려면 easy negative는 거의 의미가 없다.

3. Coverage-aware claims

   Dataset에 target과 hard negatives가 충분히 없으면, negative evidence를 약하게 해석한다. 이 태도는 benchmark reporting에서도 중요하다.

4. Discovery and validation split

   Candidate region을 찾은 data와 최종 validation data를 분리한다. Interpretability feature naming에서도 같은 원칙이 필요하다.

5. Follow-up data recommendation

   분석 결과가 부족하면 "불확실하다"에서 끝내지 않고, 어떤 stimulus를 추가하면 uncertainty가 줄어드는지 제안한다. 실험 설계와 model evaluation이 연결되는 지점이다.

## 7-3. Follow-up papers

후속으로 같이 보면 좋을 논문과 자료는 아래와 같다.

- MindSimulator
- BrainExplore
- Natural Scenes Dataset, NSD
- Brain Diffusion for visual exploration
- CLIP and concept-based retrieval literature
- Category-selective region modeling papers
- Causal representation learning and counterfactual data generation papers

특히 ML 쪽 독자라면 MindSimulator와 BrainExplore를 같이 보는 편이 좋다. BrainCause가 어느 지점을 보완하려는지 더 잘 보인다. Neuroscience 쪽 배경이 약하다면 NSD와 classical category-selective region literature를 먼저 훑는 것도 도움이 된다.

# 8. Summary

- BrainCause는 visual concept에 대한 brain representation을 activation이 아니라 causal specificity 기준으로 검증하려는 framework다.
- 핵심은 positive image, semantic negative, counterfactual edit을 자동 생성하고 image-to-fMRI encoder로 response를 예측해 candidate region을 평가하는 것이다.
- Main result에서 BrainCause는 activation은 약간 낮을 수 있지만 generated, measured, edit 기준 causal score를 크게 개선한다.
- Activation-based discovery는 false positive가 많으며, causal ranking은 false positive를 줄이면서 true positive도 개선하는 방향을 보인다.
- 이 논문은 NeuroAI뿐 아니라 model interpretability, feature localization, concept probe design에서도 재사용 가능한 평가 철학을 준다.
