---
layout: single
title: "RationalRewards: Reasoning Rewards Scale Visual Generation Both Training and Test Time Review"
categories: Study-concept
tag: [RewardModel, DiffusionRL, ImageGeneration, Multimodal, TestTimeOptimization]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.11626)

[Project page](https://tiger-ai-lab.github.io/RationalRewards/)

[Code link](https://github.com/TIGER-AI-Lab/RationalRewards)

RationalRewards는 "reward model이 점수만 내면 되는가"를 정면으로 묻는 논문이다. 요즘 visual generation에서는 generator 자체만큼이나 reward model이 병목이 되는 경우가 많다. 그런데 기존 reward model 다수는 text faithfulness, visual quality, physical plausibility, text rendering 같은 서로 다른 판단 축을 하나의 scalar로 눌러 버린다. 그러면 RL에서는 reward hacking이 쉬워지고, test time에서는 무엇을 고쳐야 하는지 generator가 알기 어렵다.

이 논문이 흥미로운 이유는 reward model을 더 잘 맞추는 분류기 정도로 다루지 않는다는 점이다. 저자들은 reward model을 "평가기"에서 "최적화 인터페이스"로 바꾸려 한다. 즉, train time에는 RL용 structured reward를 주고, test time에는 생성 결과를 보고 바로 prompt를 다시 써주는 Generate-Critique-Refine loop까지 연결한다.

> 한 줄 요약: RationalRewards는 scalar reward model을 critique-first, score-later 구조로 바꾸고, PARROT라는 teacher-student rationalization pipeline으로 pairwise preference data만으로 reasoning reward model을 학습해, RL과 test-time prompt refinement를 모두 가능하게 만든 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- visual generation reward model을 black-box score가 아니라 **reasoning interface** 로 재정의한다.
- 같은 reward model을 **parameter space** 와 **prompt space** 에서 모두 쓰는 구성이 꽤 설득력 있다.
- 8B open model로도 strong preference prediction과 downstream optimization을 동시에 보여줘서, 실무 재사용성이 높다.

이 논문의 핵심 메시지는 단순하다. 생성 모델을 더 학습시키는 문제와 prompt를 더 잘 쓰는 문제를 따로 보지 말고, 둘 다 사람이 왜 선호하는지 설명할 수 있는 reward model 쪽에서 통합해보자는 것이다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 visual generation reward model이 너무 자주 "하나의 점수"로 모든 판단을 압축한다는 점이다.
- text-to-image와 image editing에서는 보통 instruction following, text rendering, composition, perceptual quality, image faithfulness 같은 여러 축이 동시에 중요하다.
- 그런데 scalar reward는 이 축들을 하나의 수치로 눌러 버리기 때문에, generator 입장에서는 무엇이 좋아졌고 무엇이 나빠졌는지 분해된 피드백을 받기 어렵다.
- 그 결과 train time RL에서는 shortcut exploitation과 reward hacking 위험이 커지고, test time refinement에서는 실제 실패 원인을 prompt에 다시 반영하기가 어렵다.
- 결국 이 논문의 문제 설정은 "더 좋은 judge를 만들자"라기보다, "생성 품질을 실제로 끌어올릴 수 있는 reward interface를 만들자"에 가깝다.

## 1-2. Why previous approaches are insufficient

- 기존 scalar reward model은 설명 없이 점수만 출력한다. 그러면 RL이 reward를 올리더라도, 그 reward가 인간 선호와 계속 정렬되는지 확인하기 어렵다.
- generic VLM judge를 그대로 reward model처럼 쓰는 것도 충분하지 않다. 논문은 generic judge가 reasoning을 할 수는 있어도, fine-grained quality discrimination에 필요한 **low-variance calibration** 이 부족하다고 본다.
- human rationale annotation은 가장 직접적인 해결책처럼 보이지만, visual preference 데이터에서 rationale까지 수집하는 비용은 너무 크다.
- test-time prompt optimization 쪽도 한계가 있다. blind prompt rewriter는 생성 결과를 보기 전에 prompt를 다시 쓰므로, 실제 실패 사례를 보고 고치는 post-hoc correction과는 다르다.
- 결국 기존 방식의 한계는 score만 있고 explanation이 없거나, reasoning은 있어도 preference calibration이 없거나, calibration은 있어도 rationales를 싸게 만들 수 없다는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- RationalRewards의 첫 번째 기여는 reward model이 **structured rationale** 을 먼저 만들고 그 뒤에 점수를 내도록 바꾼 것이다.
- 두 번째 기여는 **PARROT** 이다. pairwise preference data만 있는 상황에서 rationale을 latent variable처럼 다루고, teacher VLM이 anchored rationale을 만든 뒤 consistency filtering과 distillation으로 student를 학습시킨다.
- 세 번째 기여는 **dual-space optimization** 이다. 같은 reward model을 train time RL reward와 test-time prompt refinement 양쪽에 모두 사용한다.
- 네 번째 기여는 practical scale이다. 논문은 8B reward model이 open-source reward model 중 strong preference prediction을 보이고, 일부 setting에서는 test-time prompt tuning이 RL fine-tuning과 맞먹거나 더 좋을 수 있다고 주장한다.

## 2-2. Design intuition

RationalRewards의 설계 직관은 꽤 선명하다. 사람이 이미지를 선호하는 이유는 보통 하나의 scalar가 아니다. "instruction은 맞지만 text rendering이 깨졌다", "구도는 좋지만 physical plausibility가 이상하다", "editing instruction은 따랐지만 source image faithfulness가 떨어진다" 같은 식으로, 실패 원인은 보통 분해 가능하다. 그렇다면 reward model도 이 구조를 따라야 한다는 것이다.

또 하나 중요한 직관은 **reason-before-score** 가 implicit regularizer로 작동한다는 점이다. 단순 scalar scorer는 특정 shortcut에 점수를 높게 주기 쉽지만, rationale을 먼저 써야 하는 모델은 각 점수에 대한 근거를 같이 만들어야 한다. 논문은 이 구조가 reward hacking을 줄이고 reward trajectory를 더 안정적으로 만든다고 해석한다.

마지막으로, test time prompt refinement의 설계 이유도 분명하다. 생성 모델에 latent capability가 있어도, prompt가 그 capability를 잘 끌어내지 못하면 결과는 제한적일 수 있다. RationalRewards는 blind rewriting이 아니라 실제 생성 결과를 본 뒤에 critique를 만들고 prompt를 다시 쓰기 때문에, test-time compute를 더 "문제 지향적"으로 쓸 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | visual generation reward model을 scalar scorer가 아니라 reasoning-based optimizer로 바꾸는 것 |
| Backbone | Qwen3-VL-Instruct-8B 기반 8B student reward model, teacher는 Qwen3-VL-32B-Instruct |
| Key module | rationale generation, consistency filtering, student distillation, dual-space optimization |
| Output | multi-dimensional critique + dimension-wise scores + overall preference signal |
| Optimization spaces | parameter space RL, prompt space Generate-Critique-Refine |
| Difference from prior work | generic VLM judge나 scalar reward에 비해 preference calibration과 explicit rationale를 함께 학습 |

## 3-2. Module breakdown

### 1) Rationalized reward model

- RationalRewards는 먼저 critique를 만들고, 그 다음 점수를 낸다.
- text-to-image에서는 text faithfulness, physical or visual quality, text rendering 같은 축으로 이미지를 평가한다.
- image editing에서는 여기에 source image faithfulness 축이 추가된다.
- 즉 이 모델은 "좋다 / 나쁘다"를 하나로 뭉개지 않고, 어떤 축에서 왜 좋아 보이거나 나빠 보이는지 서술한 뒤 점수를 붙인다.
- 이 구조 덕분에 reward output 자체가 RL reward이면서 동시에 디버깅 로그가 된다.

### 2) PARROT pipeline

- PARROT은 Preference-Anchored Rationalization의 약자다.
- 핵심은 rationale annotation이 없는 pairwise preference data에서 rationale supervision을 복원하는 것이다.
- 논문이 제안하는 practical pipeline은 3단계다.

1. **Anchored generation**  
   teacher VLM이 정답 preference label에 맞춰 rationale candidate를 생성한다.

2. **Consistency filtering**  
   rationale에서 label hint를 제거한 뒤에도 preference를 예측할 수 있는지 확인해서, hallucinated rationale을 걸러낸다.

3. **Distillation**  
   student model이 이제 label을 보지 않고도 critique와 preference를 예측하도록 학습한다.

- 이 과정에서 generated rationale의 약 72 percent가 consistency filtering을 통과한다.
- 이 모듈의 진짜 포인트는 teacher를 크게 쓰는 데 있지 않다. 오히려 **preference label -> rationale -> student** 라는 supervision 변환기를 만든 데 있다.

### 3) Parameter space optimization

- train time에서는 RationalRewards를 RL reward로 사용한다.
- 논문은 DiffusionNFT 기반 RL setting을 사용하고, scalar reward 대신 dimension-wise structured reward를 넣는다.
- 비교 baseline은 크게 두 가지다.
  - scalar reward model인 MultiReward or EditReward
  - generic reasoning judge인 Qwen3-VL-32B-Instruct
- 저자들의 주장은 단순히 reasoning model이 좋다는 것이 아니다. **preference-trained reasoning reward** 가 generic reasoning judge보다 RL에 더 안정적인 reward를 준다는 쪽이다.
- 이 부분이 중요하다. RL에서는 judge accuracy만큼이나 reward variance와 calibration이 중요하기 때문이다.

### 4) Prompt space optimization

- test time에서는 Generate-Critique-Refine loop를 쓴다.
- generator가 먼저 이미지를 만든다.
- RationalRewards가 결과를 보고 multi-dimensional critique와 refinement suggestion을 만든다.
- 특정 축의 점수가 threshold보다 낮으면, refined prompt를 다시 generator에 넣어 재생성한다.
- 논문 설정에서는 이 single iteration loop가 이미지당 약 0.4초 정도의 추가 VLM inference overhead를 만든다고 보고한다.
- 중요한 차이는 blind rewriting이 아니라 **post-hoc, preference-aware refinement** 라는 점이다. 즉 실제 실패를 본 뒤에 prompt를 고친다.

### 5) Why generic judges are not enough

- 이 논문에서 꽤 설득력 있는 포인트는 "큰 generic VLM judge를 그대로 쓰면 되지 않는가?"에 대한 반박이다.
- 논문은 generic judge가 reasoning text를 만들 수는 있어도, semantically similar image들 사이에서 score variance가 커서 RL reward noise를 만들기 쉽다고 본다.
- 반대로 PARROT로 preference calibration을 거친 RationalRewards는 더 **low-variance, preference-aligned score** 를 내기 때문에 optimization interface로서 더 쓸 만하다는 주장이다.
- 즉 이 논문은 "reasoning ability"와 "reward usability"를 같은 것으로 보지 않는다.

# 4. Training / Data / Recipe

## 4-1. Data

- image editing용 training data는 EditReward에서 가져온 30K query-preference pairs다.
- text-to-image용 training data는 HPDv3와 RapidData에서 가져온 50K preference pairs다.
- 즉 총 80K raw preference pairs에서 시작한다.
- 이 데이터에는 rationale이 없고, binary or ranked preference label만 있다.
- teacher는 Qwen3-VL-32B-Instruct를 사용해 preference-anchored rationale을 생성한다.
- consistency filtering 이후 약 57.6K pair가 남고, 저자들은 이 데이터 규모가 기존 scalar reward baseline보다 10x to 20x 정도 작다고 설명한다.
- 이 데이터 recipe의 포인트는 quantity보다 **teacher prior를 rationale supervision으로 바꾸는 과정** 이다.

## 4-2. Training strategy

| Stage | What happens | Why it matters |
| --- | --- | --- |
| Phase 1 | teacher가 label anchored rationale 생성 | rationale annotation 비용 없이 explanation supervision 확보 |
| Phase 2 | predictive consistency filtering | hallucinated rationale 제거, preference calibration 강화 |
| Phase 3 | 8B student distillation | deployable reward model 획득 |
| RL stage | DiffusionNFT 기반 RL에 RationalRewards 사용 | structured reward로 generator fine-tuning |
| Test-time stage | Generate-Critique-Refine | parameter update 없이 prompt space optimization |

- 본문 기준 메인 student는 8B 모델이며, 표에서는 Qwen3-VL-8B 기반 variant와 Qwen2.5-VL-7B 기반 variant가 같이 보고된다.
- RL 실험에서는 Flux.1-dev, SD-3.5-Medium, Qwen-Image 같은 text-to-image generator와 Flux.1 Kontext [dev], Qwen-Image-Edit 같은 editing generator를 사용한다.
- appendix 기준 RL training prompt는 initial generation reward가 평균 이하인 사례를 중심으로 골라, 개선 여지가 큰 샘플에 학습 budget을 집중한다.
- test-time prompt tuning에서는 어떤 dimension score가 3.0 아래로 떨어지면 refined prompt를 사용한다.

## 4-3. Engineering notes

1. **reasoning reward는 설명 가능한 logger이기도 하다**  
   reward model output이 바로 왜 점수가 그렇게 나왔는지를 담고 있기 때문에, 단순 scalar보다 훨씬 디버깅 친화적이다.

2. **generic judge보다 calibration이 더 중요할 수 있다**  
   RL에서는 mean quality보다 variance control이 더 중요할 때가 많다. 논문은 이 지점을 preference-trained reward의 장점으로 민다.

3. **post-hoc refinement라서 실패 원인에 직접 대응할 수 있다**  
   blind prompt enhancement는 생성 전에 추상적으로 prompt를 확장하지만, RationalRewards는 결과를 본 뒤 실제 실패 축을 보고 prompt를 수정한다.

4. **serving cost가 꽤 현실적이다**  
   vLLM prefix caching과 paged attention을 켠 상태에서 single critique-and-refine pass가 이미지당 약 0.4초 추가라고 보고한다. 완전히 공짜는 아니지만, RL fine-tuning에 비하면 훨씬 가볍다.

# 5. Evaluation

## 5-1. Main results

### 1) Preference prediction 성능

아래 표는 논문 Table 1의 핵심 비교만 뽑은 것이다.

| Judge | MMRB2 T2I | MMRB2 Edit | EditReward Bench | GenAI T2I | GenAI Edit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-32B | 64.1 | 67.3 | 64.2 | 66.9 | 76.3 |
| RationalRewards (Qwen3-VL-8B) | 64.2 | 70.3 | 66.2 | 69.8 | 80.1 |
| Gemini 2.5 Pro | 70.5 | 71.3 | 71.3 | 66.2 | 78.9 |
| Gemini 3 Pro | 74.4 | 74.9 | 72.2 | 73.1 | 80.5 |

- open-source 기준으로는 RationalRewards 8B가 꽤 강하다.
- 특히 generic Qwen3-VL-32B judge보다 editing과 GenAI-Bench 쪽에서 더 높다.
- commercial top model을 전부 넘는 것은 아니지만, 8B open model이 이 정도 preference prediction을 보인다는 점은 practical하다.
- direct distillation baseline인 Qwen3-VL-32B-Instruct Distillation이 MMRB2 T2I 57.4, GenAI Edit 62.8에 머문다는 점도 중요하다. teacher scale보다 PARROT pipeline 자체가 더 중요하다는 근거가 된다.

### 2) Text-to-image RL 결과

대표적인 UniGen overall 결과는 아래처럼 읽을 수 있다.

| Model | Base | Scalar reward | Generic judge | RationalRewards |
| --- | ---: | ---: | ---: | ---: |
| FLUX.1-dev | 60.97 | 60.12 | 66.53 | 70.34 |
| Qwen-Image | 78.36 | 75.61 | 80.17 | 82.60 |

- FLUX.1-dev에서는 RationalRewards RL이 60.97 -> 70.34로 크게 오른다.
- 같은 setting에서 scalar MultiReward는 60.12라서 거의 개선이 없거나 오히려 악화된다.
- generic Qwen3-VL-32B judge도 66.53까지 올리지만, RationalRewards가 더 높다.
- Qwen-Image에서도 base 78.36이 RationalRewards RL로 82.60까지 오른다.

여기서 중요한 건 8B reward가 32B judge를 이긴다는 문장 자체가 아니다. **preference-calibrated reasoning reward가 optimization signal로 더 쓸 만하다** 는 점이 더 중요하다.

### 3) Editing task에서 RL과 test-time tuning 비교

아래 표는 editing task에서 대표 수치만 뽑은 것이다.

| Setting | Base | Scalar or other | RationalRewards RL | RationalRewards PT |
| --- | ---: | ---: | ---: | ---: |
| Flux.1 Kontext [dev], ImgEdit Overall | 3.52 | 3.66 with EditReward RL | 3.84 | 4.01 |
| Qwen-Image-Edit, ImgEdit Overall | 4.27 | 4.25 with EditReward RL | 4.38 | 4.43 |
| Flux.1 Kontext [dev], PICA overall | 41.07 | 45.28 with PromptEnhance | 44.25 | 48.12 |
| Qwen-Image-Edit, PICA overall | 49.71 | 50.97 with PromptEnhance | 54.11 | 55.65 |

- ImgEdit-Bench에서는 Flux.1 Kontext가 base 3.52에서 RR RL로 3.84, RR PT로 4.01까지 오른다.
- Qwen-Image-Edit도 4.27에서 4.38, 그리고 PT까지 쓰면 4.43이 된다.
- PICA representative aspects에서도 prompt tuning이 RL alone보다 더 좋게 나온다.
- 즉 이 논문의 재미있는 포인트는 RL이 항상 끝판왕이 아니라는 점이다. **test-time critique loop만으로도 충분히 강한 generator의 latent capability를 끌어낼 수 있다** 는 주장과 맞물린다.

## 5-2. What really matters in the experiments

### 1) 이 논문은 "better judge" paper가 아니라 "better optimizer interface" paper에 가깝다

겉으로 보면 preference prediction 결과가 headline처럼 보인다. 하지만 진짜 중요한 건 Tables 2 and 3이다. reward model이 정확하기만 한 게 아니라, 실제로 generator optimization에 쓰였을 때 scalar reward와 generic judge보다 일관되게 낫다는 점이 핵심이다.

### 2) reward hacking 분석이 중요하다

논문은 Fig. 3과 Fig. 11, 12에서 scalar reward model을 쓰면 reward는 올라가는데 visual quality가 무너지는 reward hacking 사례를 보여준다. 반대로 RationalRewards는 held-out reward curve와 실제 benchmark score가 더 잘 맞물리고, reward standard deviation도 점차 안정화된다고 해석한다. 이건 단순 leaderboard 결과보다 훨씬 중요한 메시지다.

### 3) prompt space optimization이 생각보다 강하다

이 논문이 정말 흥미로운 이유는 PT가 RL을 완전히 대체한다는 주장이 아니라, **어떤 generator에서는 RL과 거의 맞먹거나 더 나을 수 있다** 는 결과를 보여준다는 점이다. 특히 Qwen-Image-Edit에서는 RL 위에 PT를 더했을 때 최고 점수가 나온다. 즉 parameter tuning과 prompt tuning은 경쟁 관계라기보다 서로 다른 레버일 수 있다.

### 4) 8B vs 32B 비교가 남기는 메시지

generic Qwen3-VL-32B judge보다 RationalRewards 8B가 optimization에서 더 잘 작동하는 결과는, raw model scale보다 preference calibration이 더 중요할 수 있다는 신호다. 이건 실무적으로도 중요하다. 큰 모델을 judge로 두는 것만으로는 끝나지 않는다. **어떤 objective로 calibration됐는가** 가 reward usability를 바꾼다.

# 6. Limitations

1. preference data의 bias inheritance를 피하지 못한다. 논문도 EditReward, HPDv3, RapidData의 annotator preference와 teacher VLM pretraining bias가 visual style, demographic, content type 선호를 주입할 수 있다고 인정한다.

2. test-time tuning의 강한 결과는 분명 흥미롭지만, 저자들도 latent capability hypothesis는 아직 representation-level validation이 더 필요하다고 말한다. 즉 "generator는 원래 다 할 수 있었고 prompt만 문제였다"를 단정하면 과하다.

3. RL 성능 ceiling에는 method 외적인 제약도 있다. 논문은 LoRA-based fine-tuning의 update capacity 한계와 RL query distribution coverage 부족을 같이 언급한다. 즉 RL이 약해서가 아니라, 현재 실험 recipe가 완전한 full-capacity RL은 아닐 수 있다.

4. visual generation 쪽 preference benchmark는 여전히 특정 스타일 선호를 포함할 수 있다. 따라서 benchmark alignment와 실제 user alignment가 항상 같다고 보면 안 된다.

5. reward model이 rationale을 생성한다고 해서 explanation이 항상 faithful하다고 보장되는 것은 아니다. consistency filtering이 hallucination을 줄이지만, explanation faithfulness 자체를 완전히 증명한 것은 아니다.

6. 추가로 주의할 점은 domain transfer다. 이 논문은 text-to-image와 editing에서 강하지만, video generation이나 multi-image editing, controllable 3D generation까지 그대로 일반화된다고 보긴 어렵다.

# 7. My Take

## 7-1. Why this matters for my work

RationalRewards의 가장 큰 가치는 reward model을 마지막 judge가 아니라 **학습과 추론 둘 다를 연결하는 control surface** 로 본다는 데 있다. 이 관점은 visual generation에만 묶이지 않는다.

예를 들어 multimodal OCR correction, document parsing, chart QA, UI grounding 같은 응용에서도 final score 하나보다 "어디가 틀렸는지"가 더 중요하다. 실제 시스템을 고치려면 scalar보다 structured critique가 훨씬 유용하다. RationalRewards는 그 방향을 visual generation domain에서 꽤 설득력 있게 보여준다.

## 7-2. Reuse potential

1. **reason-before-score evaluator**  
   OCR, document AI, multimodal QA에서도 black-box judge보다 rationale-first judge를 두는 편이 훨씬 디버깅 친화적일 수 있다.

2. **post-hoc critique loop**  
   generator나 parser를 다시 학습시키지 못하는 상황에서도, 출력 결과를 보고 prompt나 instruction을 refinement하는 loop는 바로 응용 가능하다.

3. **preference-calibrated small judge**  
   큰 generic model을 judge로 쓰는 대신, 작은 모델을 preference data로 calibration해서 low-variance optimizer로 쓰는 전략은 cost 대비 효율이 좋다.

4. **data curation interface**  
   논문 Fig. 8이 보여주듯, explicit rationale reward는 training data filtering과 quality control에도 바로 연결할 수 있다.

## 7-3. Follow-up papers

- EditReward
- UnifiedReward
- DiffusionNFT
- PromptEnhance
- RM-R1: Reward Modeling as Reasoning

# 8. Summary

- RationalRewards는 scalar reward model을 critique-first reward model로 바꿔, visual generation optimization을 더 구조적으로 만들려는 논문이다.
- PARROT는 pairwise preference data만으로 rationale supervision을 복원해 8B reward model을 학습시킨다.
- 이 reward model은 RL reward로도, test-time Generate-Critique-Refine optimizer로도 사용된다.
- 핵심 실험은 preference prediction보다 downstream optimization이다. RationalRewards는 scalar reward와 generic 32B judge보다 더 좋은 optimization signal을 보인다.
- 이 논문의 진짜 메시지는 reward model을 score function이 아니라 optimization interface로 보자는 데 있다.
