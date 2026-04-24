---
layout: single
title: "OpenVLThinkerV2 Review"
categories: Study-concept
tag: [VLM, RL, MultimodalReasoning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.08539)

[Project page](https://gordonhu608.github.io/openvlthinkerv2.github.io/)

[Code](https://github.com/uclanlp/OpenVLThinker)

OpenVLThinkerV2는 "Qwen3-VL에 RL을 더 얹은 후속작" 정도로 읽으면 아쉬운 논문이다. 이 논문의 진짜 문제의식은 generalist multimodal model을 multi-task RL로 키울 때, task마다 reward topology가 너무 다르다는 점이다. math VQA는 sparse binary reward에 가깝고, grounding은 dense IoU 계열 reward에 가깝다. 이런 상태에서 standard GRPO를 그대로 쓰면 task마다 gradient scale과 distribution shape가 크게 달라져 학습이 쉽게 흔들린다.

또 하나 중요한 축은 perception과 reasoning의 trade-off다. visual grounding이나 OCR처럼 짧고 직접적인 답이 유리한 task가 있는 반면, math VQA나 science VQA처럼 길고 structured한 chain of thought가 필요한 task도 있다. OpenVLThinkerV2는 이 두 문제를 "더 좋은 reward model"이나 "더 많은 cold-start data"가 아니라, advantage normalization과 task-level shaping 문제로 재정의한다.

> 한 줄 요약: OpenVLThinkerV2는 multi-domain multimodal RL에서 reward topology mismatch를 G2RPO로 완화하고, task-level response length shaping과 entropy shaping으로 perception과 reasoning을 함께 유지하려는 generalist VLM post-training 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- 최근 multimodal RL의 핵심 병목이 data quantity보다 **multi-task optimization stability** 일 수 있다는 점을 보여준다.
- OCR, document understanding, grounding 같은 vision-centric task와 math, science, chart reasoning 같은 reasoning-centric task를 하나의 8B open model 안에서 같이 다루려 한다.
- Qwen3-VL-8B-Instruct를 그대로 RL base로 쓰고, training code와 example data까지 공개해 recipe 관점에서 읽을 가치가 있다.

이 논문의 핵심 메시지는 분명하다. multimodal RL에서 중요한 것은 "더 강한 verifier" 하나가 아니라, 서로 다른 task가 만들어내는 reward distribution과 output style을 어떻게 하나의 안정된 optimization loop 안으로 집어넣느냐이다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 **multi-domain multimodal RL의 불안정성** 이다.
- 같은 VLM이라도 task에 따라 reward topology가 매우 다르다. 예를 들어 MathVista 같은 math VQA는 정답형 sparse reward에 가깝고, RefCOCO 계열 grounding은 연속적인 localization score를 갖는다.
- 이런 상태에서 모든 task를 한 policy 안에 넣어 RL을 돌리면, 어떤 task는 outlier reward 때문에 gradient를 과하게 만들고, 어떤 task는 low variance 때문에 update가 거의 묻혀버린다.
- 동시에 task가 요구하는 출력 형태도 다르다. reasoning-heavy task는 길고 단계적인 답이 유리하지만, OCR나 grounding은 짧고 직접적인 답이 더 낫다.
- 결국 이 논문의 문제 설정은 Qwen3-VL을 더 똑똑하게 만들 수 있는가가 아니라, **reward scale과 response style이 다른 visual tasks를 하나의 RL loop로 안정화할 수 있는가** 에 가깝다.

## 1-2. Why previous approaches are insufficient

- standard GRPO는 prompt group 내부 reward를 mean and std로 정규화한다. single-task에서는 잘 작동할 수 있지만, multi-task에서는 reward shape 차이를 그대로 남겨 heavy-tail outlier나 bimodal reward에 취약하다.
- Dr.GRPO는 intra-task variance 문제를 완화하지만, 반대로 high variance task가 update를 과도하게 지배할 수 있다.
- EMA-GRPO는 task-wise moving average로 variance를 맞추지만, 결국 선형 정규화라 distribution shape 자체를 바꾸지 못한다.
- perception을 살리기 위해 visual anchor나 auxiliary objective를 추가하는 접근도 있지만, annotation cost나 extra module cost가 커서 generalist multi-task setting으로 넓히기 어렵다.
- 즉 기존 접근의 한계는 algorithm 이름보다 **linear normalization과 auxiliary-heavy recipe가 multi-domain setting에서 충분히 scalable하지 않다** 는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

OpenVLThinkerV2의 핵심 기여는 크게 3가지다.

1. **G2RPO**
   - task별 empirical reward distribution을 standard normal 분포로 보내는 non-linear distributional matching 기반 advantage normalization을 제안한다.
   - 논문이 강조하는 포인트는 mean and variance만 맞추는 것이 아니라, reward distribution의 higher-order shape까지 안정된 topology로 바꾸는 데 있다.

2. **Task-level response length shaping**
   - reasoning-centric task에는 더 긴 chain을 유도하고, vision-centric task에는 짧고 직접적인 출력을 유도한다.
   - 핵심은 "길면 좋다"가 아니라 task별 optimal length band를 따로 둔다는 점이다.

3. **Task-level entropy shaping**
   - reasoning task에서 생기는 entropy explosion과 vision-centric task에서 생기는 entropy collapse를 함께 막는다.
   - 즉 exploration 자체를 task별로 적정 구간 안에 묶어두는 regularization으로 읽을 수 있다.

개념적으로 G2RPO의 핵심은 아래처럼 정리할 수 있다.

$$
A_i^{G2RPO} = \Phi^{-1}(F_n(r_i))
$$

여기서 $F_n$은 task 내부 empirical CDF, $\Phi^{-1}$는 standard normal의 inverse CDF다. 실제 구현에서는 reward rank, tie handling, original ordering 복원이 추가된다.

## 2-2. Design intuition

설계 직관은 명확하다.

- multi-task RL에서 reward normalization을 선형 스케일링으로만 처리하면, outlier와 bimodal reward가 그대로 남는다.
- 그렇다면 reward를 표준화하는 것이 아니라, 아예 **well-behaved target topology** 로 transport하는 편이 낫다.
- 동시에 perception과 reasoning의 균형은 auxiliary branch를 더 붙여서 푸는 대신, task별 response length와 entropy dynamics를 직접 shaping하는 쪽이 더 단순하고 scalable하다.

이 논문이 좋은 이유는 이 두 축이 서로 이어지기 때문이다. G2RPO가 task 간 gradient mismatch를 줄이고, length shaping과 entropy shaping이 task별 output behavior를 잡아주면서, 결국 하나의 8B policy가 broad visual tasks를 더 균형 있게 다루게 된다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | multi-domain multimodal RL에서 reward topology mismatch와 perception-reasoning trade-off를 함께 안정화 |
| Base model | Qwen3-VL-8B-Instruct |
| Key module | G2RPO, task-level response length shaping, task-level entropy shaping |
| Difference from prior work | linear variance scaling 대신 non-linear distributional matching을 쓰고, auxiliary visual module 없이 task-level optimization으로 perception과 reasoning을 함께 다룸 |

## 3-2. Module breakdown

### 1) G2RPO

- standard GRPO는 prompt group 안에서 reward를 z-score처럼 정규화한다.
- G2RPO는 이 방식을 버리고, task 내부 reward의 relative rank를 standard normal quantile로 보낸다.
- 논문 기준 절차는 다음과 같다.
  1. reward를 sort하고 rank를 구한다.
  2. rank를 uniform probability로 바꾼다.
  3. inverse normal CDF로 quantile mapping을 수행한다.
  4. 같은 reward 값에는 같은 target quantile을 주기 위해 tie-breaking average를 쓴다.
- 이 방식의 장점은 heavy-tail outlier를 수학적으로 cap하고, binary reward처럼 bimodal한 경우도 symmetric Gaussian tail로 바꿔준다는 점이다.

### 2) Task-level response length shaping

- 논문은 reasoning-centric task와 vision-centric task가 학습 중 response length trajectory가 다르게 움직인다고 본다.
- reasoning task는 초기에 길이가 줄었다가, 이후 더 긴 reasoning chain으로 수렴하는 경향이 있다.
- 반대로 vision-centric task는 길이가 계속 줄어드는 편이 더 낫다. 불필요한 장문 reasoning이 hallucination과 visual grounding 오류를 늘릴 수 있기 때문이다.
- 그래서 각 task마다 최소 길이, optimal plateau 시작점, optimal plateau 끝점, 최대 길이 같은 threshold를 두고 trapezoidal reward envelope를 적용한다.
- 이 부분이 꽤 실무적이다. chain of thought를 무조건 길게 만드는 것이 아니라, task별로 "어느 정도 길이가 유리한가"를 reward shaping으로 명시한다.

### 3) Task-level entropy shaping

- multi-task RL에서는 task마다 exploration pattern도 다르다.
- reasoning-heavy task는 entropy가 과도하게 커져 incoherent token을 탐색하는 entropy explosion이 생기기 쉽다.
- vision-centric task는 반대로 너무 빨리 high-probability token만 고집해 entropy collapse가 생길 수 있다.
- OpenVLThinkerV2는 task별 최소 entropy threshold와 최대 entropy threshold를 두고, 그 구간을 벗어날 때만 margin-based penalty를 적용한다.
- 즉 이 방법은 entropy를 낮추거나 높이라는 지시가 아니라, **task별 optimal exploration band를 유지하라** 에 가깝다.

### 4) Final objective와 training behavior

- G2RPO는 advantage normalization 쪽을 바꾸고,
- response length shaping은 reward side에서 behavior target을 잡고,
- entropy shaping은 final objective에 추가 regularization으로 들어간다.
- 그래서 구조를 단순화하면 아래처럼 볼 수 있다.

$$
L = L_{G2RPO} + \lambda_{ent} L_{ent}
$$

- 여기서 length shaping은 reward computation 단계에 반영되고, entropy shaping은 explicit regularizer로 더해진다.
- exact threshold와 coefficient는 HTML에서 일부 symbol이 누락되어 있으므로, 최종 발행 전 PDF 재확인이 필요하다.

# 4. Training / Data / Recipe

## 4-1. Data

- 학습 초기화는 Qwen3-VL-8B-Instruct에서 시작한다.
- RL training data는 **filtered subset of OneThinker-600k** 를 사용한다.
- 이 점이 중요한 이유는 새 architecture를 만든 논문이 아니라, 이미 강한 generalist base model 위에서 **post-training optimization recipe** 를 검증하는 논문이기 때문이다.
- GitHub 기준으로는 example training and validation data와 training/evaluation code가 공개되어 있다.
- 다만 checkpoint는 아직 "coming" 상태이며, README에는 현재 수치가 cold-start SFT 없이 direct RL from Qwen3-VL-8B로도 얻어진다고 적혀 있다.

## 4-2. Training strategy

| Item | Setting |
| --- | --- |
| Initialization | Qwen3-VL-8B-Instruct |
| Training data | filtered subset of OneThinker-600k |
| Optimizer | AdamW |
| Epoch | 1 |
| Batch size | 128 |
| Max generation length | 4096 |
| KL regularization | disabled |
| Extra filtering | uniformly correct or uniformly incorrect rollout discard |
| Hardware | AWS Trainium, Trn1.32xlarge |
| Training time | about 3 days |

- learning rate는 arXiv HTML에서 symbol이 누락되어 있어 PDF 기준 재확인이 필요하다.
- uniformly correct or uniformly incorrect rollout을 버리는 dynamic data filtering은 gradient signal quality를 높이기 위한 engineering으로 읽으면 된다.
- 이 논문은 reward function 하나보다 **training stability stack** 을 같이 설계한 케이스다.

## 4-3. Engineering notes

1. **Reward normalization을 task-aware하게 다뤄야 한다**
   - multimodal task는 reward topology가 task마다 달라서, single-task RL에서 잘 되던 recipe가 그대로 안 맞는다.

2. **Length와 entropy는 monitor가 아니라 control variable이다**
   - 이 논문은 response length와 entropy를 단순 로그 지표로 남기지 않고, 실제 optimization target으로 끌어들인다.

3. **Dynamic filtering이 중요하다**
   - 이미 전부 맞거나 전부 틀린 rollout은 gradient 정보량이 낮다.
   - 이런 샘플을 버리는 것은 reward modeling보다 훨씬 저렴한데, training stability에는 꽤 직접적으로 영향을 준다.

4. **코드 공개 상태는 좋지만 model release는 아직 완전하지 않다**
   - code, example data, evaluation path는 공개됐지만, GitHub README 기준 checkpoint는 아직 coming 상태다.
   - 따라서 지금 당장은 재현 실험과 method inspection 중심으로 읽는 편이 낫다.

# 5. Evaluation

## 5-1. Main results

논문은 18개 benchmark를 6개 task category로 묶어 평가한다. broad category는 general science knowledge, mathematics, chart understanding, document understanding, spatial reasoning, visual grounding이다.

가장 먼저 볼 만한 표는 Qwen3-VL-8B-Instruct baseline 대비 visual reasoning 계열 개선이다.

| Benchmark | Qwen3-VL-8B-Instruct | OpenVLThinkerV2 | 같이 볼 비교점 |
| --- | ---: | ---: | --- |
| MMMU | 60.2 | 71.6 | GPT-4o 70.7 |
| MMBench | 85.1 | 88.2 | GPT-4o 84.3 |
| MMStar | 68.5 | 73.8 | GPT-4o 65.1 |
| MathVista | 74.2 | 79.5 | GPT-4o 63.8 |
| ChartQA | 82.8 | 87.4 | Gemini 2.5 Pro 83.3 |
| CharXiv(RQ) | 44.5 | 53.0 | GPT-4o 47.1 |

- visual reasoning 표에서 이 논문의 장점은 baseline 대비 improvement가 broad하게 퍼져 있다는 점이다.
- 특히 MMMU, MathVista, ChartQA, CharXiv(RQ)처럼 perception과 reasoning이 같이 필요한 영역에서 improvement가 크다.
- project page 기준으로도 relative improvement figure가 general VQA, math VQA, chart VQA, spatial reasoning, document understanding, grounding 전 영역에 걸쳐 제시된다.

document understanding과 spatial reasoning, grounding은 따로 보는 편이 낫다.

| Task | OpenVLThinkerV2 | 인상적인 비교점 |
| --- | ---: | --- |
| DocVQA | 96.7 | GPT-5 91.5, Gemini 2.5 Pro 92.6 |
| OCRBench | 911 | DeepEyesV2 882, Gemini 2.5 Pro 866 |
| InfoVQA | 86.4 | GPT-5 79.0, Gemini 2.5 Pro 84.2 |
| EMbSpatial | 83.1 | GPT-5 82.9, Gemini 2.5 Pro 79.1 |
| RefSpatial | 44.6 | Qwen3-VL-8B-Instruct 43.9, RoboRefer-SFT 48.4 |
| RoboSpatial | 63.2 | GPT-5 53.5, SpatialRGPT-8B 66.7 |
| RefCOCO | 93.4 | Grounding DINO 90.6 |
| RefCOCO+ | 88.2 | Grounding DINO 88.2 |
| RefCOCOg | 90.4 | Grounding DINO 86.1 |

여기서 중요한 해석은 두 가지다.

1. document understanding 쪽은 상당히 설득력 있다.
   - OCRBench 911, DocVQA 96.7은 generalist 8B model 기준으로 꽤 강한 숫자다.

2. spatial reasoning은 "완전 압도"로 읽으면 안 된다.
   - EMbSpatial에서는 강하지만, RefSpatial에서는 specialized expert가 더 높고, RoboSpatial에서도 SpatialRGPT-8B가 66.7로 OpenVLThinkerV2 63.2보다 높다.
   - 즉 generalist model이 broad average를 끌어올린 것이지, 모든 spatial subset에서 expert를 넘어섰다고 보기 어렵다.

## 5-2. What really matters in the experiments

이 논문의 실험에서 진짜 중요한 부분은 절대 점수보다 **ablation의 방향성** 이다.

| Variant | General VQA | Math VQA | Chart VQA | Grounding | Document Understanding | Spatial Reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B-Instruct | 71.3 | 59.2 | 69.9 | 87.1 | 86.8 | 60.9 |
| + G2RPO | 76.9 | 64.8 | 74.5 | 90.2 | 90.6 | 62.3 |
| + task-level entropy loss | 77.0 | 65.1 | 75.3 | 90.4 | 90.8 | 62.8 |
| + task-level length reward | 77.4 | 65.7 | 75.4 | 90.5 | 91.1 | 63.2 |
| OpenVLThinkerV2 | 77.9 | 66.2 | 76.0 | 90.7 | 91.4 | 63.6 |

이 표가 말해주는 것은 꽤 선명하다.

1. **초기 improvement의 대부분은 G2RPO에서 온다**
   - baseline에서 G2RPO만 넣어도 6개 category 전부가 오른다.
   - 즉 이 논문의 중심 기여는 여전히 reward normalization에 있다.

2. **length shaping이 entropy shaping보다 더 넓게 듣는다**
   - 논문도 length reward가 broader improvement를 준다고 해석한다.
   - multimodal RL에서 chain length control이 생각보다 강한 regularizer라는 뜻이다.

3. **두 shaping은 서로 대체 관계가 아니라 complementary하다**
   - full model이 항상 가장 높다.
   - 결국 G2RPO로 gradient topology를 안정화하고, length와 entropy shaping으로 task behavior를 잡는 구조가 함께 먹힌다.

이 논문의 포인트는 G2RPO가 무조건 최고다보다, **generalist multimodal RL에서는 normalization, response style, exploration band를 같이 설계해야 한다** 는 데 있다.

# 6. Limitations

1. learning rate와 length/entropy threshold 일부는 arXiv HTML에서 symbol이 누락되어 있다.
   - 최종 발행 전 PDF 기준 재확인이 필요하다.

2. evaluation claim을 과장하면 안 된다.
   - document understanding은 매우 강하지만, spatial reasoning은 일부 subset에서 specialized expert가 여전히 더 높다.

3. task-level shaping threshold는 여전히 heuristic이다.
   - 논문도 automated search는 future work로 남긴다.
   - 즉 threshold-free 이론적 해법이라기보다, empirical trend를 반영한 안정화 recipe로 읽는 편이 맞다.

4. 현재 공개 상태는 code 중심이다.
   - GitHub README 기준으로 checkpoint는 아직 release 예정 상태다.
   - 따라서 지금 당장은 fully reproducible open weights paper라기보다, method and code release paper에 가깝다.

5. training data filtering과 reproduced baseline에 대한 해석도 주의가 필요하다.
   - 같은 setting에서 Qwen3-VL-8B-Instruct를 reproduce했다고 밝히지만, 실제 reproduction gap은 evaluation protocol과 filtering detail에 민감할 수 있다.

6. generalist strength가 expert dominance를 항상 의미하지는 않는다.
   - grounding과 document task는 매우 설득력 있지만, spatial reasoning 일부 subset처럼 specialist가 더 강한 영역도 남아 있다.

# 7. My Take

## 7-1. Why this matters for my work

- 이 논문이 중요한 이유는 multimodal RL을 reward model 개선 문제가 아니라 **optimization topology를 어떻게 안정화할 것인가** 로 본다는 점이다.
- document AI, OCR, chart QA, grounding을 같이 다루는 generalist VLM을 만들려면, 각 task가 요구하는 답변 길이와 reward shape가 다르다.
- 이때 architecture를 자꾸 바꾸기보다, advantage normalization과 shaping policy를 먼저 만지는 것이 더 재사용 가능성이 높다.
- 특히 Qwen3-VL 같은 strong base 위에서 broad category를 같이 올리려는 상황이라면, 이 논문의 framing은 꽤 직접적으로 참고할 만하다.

## 7-2. Reuse potential

1. **Rank-based advantage normalization**
   - reward outlier가 자주 생기는 task mixture라면, z-score보다 quantile mapping 방식이 더 안정적일 수 있다.

2. **Task-type별 response policy 분리**
   - reasoning task와 perception task를 같은 decoding style로 몰아가지 말고, target response length band를 다르게 두는 발상은 실제 서비스에서도 유용하다.

3. **Entropy band control**
   - multimodal RL에서 entropy는 단순 regularization coefficient 하나로 끝내기보다, task별로 lower and upper band를 두는 편이 더 해석 가능하다.

4. **Dynamic filtering**
   - uniformly correct or uniformly incorrect rollout discard는 구현 난이도 대비 얻는 이득이 꽤 커 보인다.

5. **Generalist VLM eval dashboard**
   - broad category를 같이 운영할 때는 overall score보다, category별 response length and entropy trajectory를 같이 로그로 봐야 한다는 점도 재사용 가치가 있다.

## 7-3. Follow-up papers

- OneThinker
- OpenVLThinker
- VisionZero
- Qwen3-VL technical report
- multimodal RL에서 entropy shaping이나 staged RL을 다루는 후속 논문들

# 8. Summary

- OpenVLThinkerV2는 multi-domain multimodal RL에서 reward topology mismatch를 G2RPO로 다루는 논문이다.
- 핵심은 reward를 선형 정규화하지 않고, task별 empirical reward distribution을 standard normal topology로 보내는 데 있다.
- 여기에 task-level response length shaping과 entropy shaping을 더해 perception-heavy task와 reasoning-heavy task를 함께 안정화한다.
- 결과는 Qwen3-VL-8B-Instruct baseline 대비 broad benchmark improvement로 나타나며, 특히 document understanding과 grounding에서 강하다.
- 다만 spatial reasoning 일부 subset에서는 expert가 여전히 더 강하고, 일부 hyperparameter와 checkpoint release 상태는 최종 재확인이 필요하다.
