---
layout: single
title: "CoVEBench: Can Video Editing Models Handle Complex Instructions? Review"
categories: Study-concept
tag: [VideoEditing, Benchmark, MultimodalAI, Evaluation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.08415)

[Project page](https://nju-link.github.io/CoVEBench)

[Code](https://github.com/NJU-LINK/CoVEBench)

[Dataset](https://huggingface.co/datasets/NJU-LINK/CoVEBench)

CoVEBench는 video editing model을 볼 때 질문을 바꾸는 논문이다. 지금까지 많은 video editing 데모는 "강아지를 고양이로 바꿔줘", "배경을 눈 오는 풍경으로 바꿔줘" 같은 단일 edit 중심으로 인상적인 결과를 보여줬다. 하지만 실제 creator workflow는 보통 그렇게 단순하지 않다. 한 prompt 안에 subject 변경, action 변경, camera movement, style, position, background preservation이 동시에 들어간다.

이 논문이 보는 핵심 문제는 바로 이 compositional editing이다. 사용자는 여러 edit을 한 번에 요구하면서도, 요구하지 않은 region과 temporal structure는 그대로 보존되길 기대한다. 모델 입장에서는 단순히 더 많이 바꾸는 것이 답이 아니다. target은 충분히 바꾸고, non-target은 건드리지 않아야 한다.

CoVEBench는 이 문제를 416개 curated source video, 626개 multi-point instruction, 9,990개 fine-grained checklist item으로 benchmark화한다. 중요한 점은 단일 global score가 아니라, edit execution, visual realism, source fidelity를 분리해 보는 evaluation apparatus를 만든다는 것이다.

> 한 줄 요약: CoVEBench는 복합 video editing instruction을 여러 atomic edit point와 checklist로 분해하고, instruction compliance, video quality, source fidelity를 별도로 평가해 현재 video editing model의 compositional failure를 진단하는 benchmark다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Text-guided video editing이 demo quality에서 workflow quality로 넘어갈 때 필요한 evaluation protocol을 제시한다.
- Single edit 성공 여부가 아니라 multi-point instruction 안에서 무엇을 수행했고 무엇을 보존했는지 분해해서 본다.
- UAS, VQR, SEM 같은 primary metric을 두어 instruction following, quality, fidelity를 하나의 scalar로 뭉개지 않는다.
- Closed-source model도 복합 instruction에서는 UAS 기준으로 아직 낮은 영역에 있다는 점을 보여준다.
- Code와 Hugging Face dataset이 공개되어 있어, video editing model evaluation pipeline을 실제로 재사용할 수 있다.

이 논문의 진짜 메시지는 "video editing benchmark가 하나 더 나왔다"가 아니다. 더 정확히는, real-world video editing을 평가하려면 prompt-level success가 아니라 edit point별 execution, physical plausibility, preservation failure를 모두 추적해야 한다는 주장이다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 compositional instruction-guided video editing evaluation이다.

문제를 조금 더 풀면 다음과 같다.

1. 입력으로 source video와 natural language editing instruction이 주어진다.
2. Instruction은 하나의 단순 요청이 아니라 여러 edit point를 포함한다.
3. 모델은 target edit을 수행해야 한다.
4. 동시에 instruction에서 바꾸라고 하지 않은 subject, background, motion, structure는 보존해야 한다.
5. 출력 video는 visual quality와 physical plausibility도 유지해야 한다.

예를 들어 instruction이 다음과 같다고 생각해보자.

- 사람의 action을 sitting에서 standing으로 바꾼다.
- subject position을 frame left로 옮긴다.
- laptop을 제거한다.
- camera를 static하게 만든다.
- 그 외 balcony와 background object는 유지한다.

이 경우 모델은 단일 edit 성공만으로는 충분하지 않다. action만 바꾸고 laptop을 남기면 실패다. laptop을 제거했지만 사람의 identity나 background를 망가뜨려도 실패다. camera를 static하게 만들었지만 motion ghosting이 생겨도 실패다. 그래서 이 task는 edit execution과 preservation 사이의 trade-off를 강하게 드러낸다.

## 1-2. Why previous approaches are insufficient

기존 video editing benchmark가 부족한 이유는 세 가지로 정리할 수 있다.

첫째, prompt가 너무 단순하다. 기존 benchmark는 object replacement, style transfer, background change처럼 isolated edit 중심인 경우가 많다. 이런 benchmark에서는 최신 model이 꽤 잘해 보일 수 있지만, 실제 사용자가 여러 조건을 동시에 넣었을 때의 failure는 잘 드러나지 않는다.

둘째, operation coverage가 좁다. Real workflow에는 subject edit뿐 아니라 camera control, motion edit, positional relationship, special effect, background preservation이 같이 들어간다. 특히 camera나 motion은 video editing의 temporal structure와 직접 연결되기 때문에 image editing benchmark식 평가로는 보기 어렵다.

셋째, metric이 너무 coarse하다. CLIP 계열 global score나 overall preference score는 결과가 대략 prompt와 맞는지 보여줄 수는 있지만, 어떤 edit point가 빠졌는지, 어떤 non-target region이 오염됐는지, physical realism이 깨졌는지 분리해주지 못한다.

CoVEBench는 이 세 가지 한계를 동시에 다룬다. Prompt를 multi-point로 만들고, taxonomy를 넓히고, checklist 기반 evaluation으로 failure location을 더 잘 보이게 만든다.

# 2. Core Idea

## 2-1. Main contribution

CoVEBench의 핵심 기여는 크게 세 가지다.

1. Compositional video editing benchmark
   - 416개 curated source video와 626개 complex editing instruction을 만든다.
   - Instruction 하나는 평균적으로 약 3개 atomic edit operation을 포함한다.
   - Subject, background, camera, style, motion, position, special effects까지 포함하는 7개 practical editing dimension을 다룬다.

2. Fine-grained checklist evaluation
   - Complex instruction을 9,990개 verifiable checklist item으로 분해한다.
   - Checklist는 execution accuracy, physical logic, semantic preservation을 분리해서 본다.
   - MLLM judge를 사용하되, MCQ와 yes/no question을 통해 edit point별 판단을 구조화한다.

3. Three-axis metric system
   - Instruction Compliance: edit이 수행되었는지와 realism이 유지되는지를 본다.
   - Video Quality: aesthetic, motion smoothness, technical quality, comprehensive quality를 본다.
   - Video Fidelity: source에서 바꾸지 말아야 할 semantic, structure, motion, static region이 유지되는지 본다.

이 구조 덕분에 CoVEBench는 leaderboard보다 diagnostic benchmark에 가깝다. 어떤 모델이 몇 점을 받았는지보다, 왜 그 점수가 나왔는지, instruction following과 preservation 중 어디서 깨지는지를 보는 것이 핵심이다.

## 2-2. Design intuition

이 논문의 설계 직관은 꽤 현실적이다.

첫째, real user prompt는 compositional하다. Video creator는 단일 effect만 요청하지 않는다. 한 clip 안에서 subject, action, camera, object, style, background를 동시에 조정하려 한다. Benchmark도 이 workflow를 따라가야 한다.

둘째, video editing에는 target edit과 non-target preservation이 동시에 필요하다. 모델이 적극적으로 바꾸는 능력만 강하면 source fidelity가 깨진다. 반대로 보존만 잘하면 edit이 안 된다. 좋은 metric은 이 둘을 분리해서 보여줘야 한다.

셋째, complex instruction은 checklist로 쪼개야 한다. Prompt 하나에 대해 overall score 하나를 주면, 부분 성공과 부분 실패가 모두 가려진다. CoVEBench는 instruction을 edit point와 verification question으로 바꾸어, failure를 localized signal로 만든다.

CoVEBench의 가장 큰 가치는 "video editing model이 좋은가"를 묻기보다, "모델이 어떤 종류의 edit interaction에서 깨지는가"를 묻게 만든다는 점이다. 특히 multi-point edit에서 target 변경이 non-target 영역을 오염시키는 문제는 실제 서비스 품질과 바로 연결된다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Complex video editing instruction을 fine-grained checklist로 평가 |
| Source videos | Paper 기준 416 curated videos |
| Instructions | 626 multi-point editing instructions |
| Checklist items | 9,990 fine-grained checklist items |
| Editing dimensions | Subject, Background, Camera, Style, Motion, Position, Special Effects |
| Primary metrics | UAS, VQR, SEM |
| Evaluated models | 10 video editing models |
| Key difference | Single aggregate score 대신 instruction compliance, quality, fidelity를 분리 |

## 3-2. Module breakdown

### 1) Source video collection and filtering

CoVEBench는 diverse source video pool을 만들기 위해 stock platform과 academic dataset을 함께 사용한다.

- Stock platform: Pexels, Mixkit
- Academic dataset: Vript, UltraVideo, ViDiC, LMArena

이후 resolution 480p, duration 3 to 21 seconds, visual quality, duplicate removal, editability를 기준으로 filter를 적용한다. 마지막에는 human review를 통해 overall quality와 editability를 확인하고 416개 source video를 선택한다.

이 단계가 중요한 이유는 benchmark가 source video quality에 민감하기 때문이다. Source video가 너무 짧거나, 너무 낮은 품질이거나, 이미 artifact가 많으면 editing model의 failure와 source 자체의 noise를 구분하기 어렵다.

### 2) Editing instruction generation

Instruction은 7개 editing dimension을 기반으로 만든다.

| Dimension | Example role |
| --- | --- |
| Subject | subject addition, removal, replacement, attribute change |
| Background | scene or background modification |
| Camera | camera motion, viewpoint, framing change |
| Style | visual style transfer |
| Motion | action or movement change |
| Position | spatial relationship and object placement |
| Special Effects | effect insertion or transformation |

논문은 이 taxonomy를 기반으로 83개 category combination을 manually formulate한다. 그 다음 여러 MLLM이 source video에 적합한 combination을 선택하고, tailored editing instruction을 생성한다. 생성된 instruction은 manual review를 거쳐 부적절하거나 반복적인 output을 제거한다.

여기서 중요한 것은 instruction length 자체가 아니라 조합 구조다. 단순히 긴 prompt를 만드는 것이 아니라, 서로 간섭할 수 있는 edit operation을 한 prompt 안에 넣는 것이 목적이다.

### 3) Checklist generation and refinement

CoVEBench는 complex instruction을 그대로 하나의 score로 평가하지 않는다. LLM이 editing instruction과 source video description을 보고 distinct editing points를 추출한 뒤, fine-grained verifiable checklist question으로 바꾼다.

Checklist question은 크게 세 관점으로 읽을 수 있다.

- Execution Accuracy: 요청한 edit이 실제로 수행되었는가.
- Physical Logic: edited content가 visually natural하고 physical constraint를 크게 어기지 않는가.
- Semantic Preservation: 바꾸지 말아야 할 source semantics가 유지되었는가.

초기 checklist output은 manual filtering을 거치며, 논문은 initial output 중 약 67.2%를 retain했다고 보고한다. 이 단계는 benchmark 품질을 좌우한다. Checklist가 너무 모호하면 MLLM judge가 흔들리고, 너무 쉬우면 compositional failure를 못 잡는다.

### 4) Evaluation matrix

CoVEBench의 metric은 크게 3개 dimension으로 구성된다.

| Dimension | Metric | Meaning |
| --- | --- | --- |
| Instruction Compliance | UAS | instruction following과 realism question이 모두 맞아야 성공 |
| Instruction Compliance | IFS | edit action이 수행되었는지만 평가 |
| Instruction Compliance | VRS | edited content의 realism과 visual naturalness 평가 |
| Video Quality | VQR | VisualQuality-R1 기반 comprehensive quality |
| Video Quality | AES | frame과 keyframe aesthetic score |
| Video Quality | MSM | optical-flow 기반 motion smoothness |
| Video Quality | TQ | DOVER++ technical quality |
| Video Fidelity | SEM | unchanged semantics 보존 여부 |
| Video Fidelity | SSIM | pixel-level structure preservation |
| Video Fidelity | MF | CoTracker 기반 motion trajectory fidelity |
| Video Fidelity | SRC | SAM2와 DINOv2 기반 static region consistency |

이 중 논문과 project page는 UAS, VQR, SEM을 primary holistic indicator로 둔다. UAS는 "요청한 edit을 했는가"와 "그 edit이 그럴듯한가"를 동시에 만족해야 하므로 가장 strict한 instruction compliance metric이다. 그래서 IFS가 높아도 UAS가 낮으면, edit은 시도했지만 완성도나 realism까지 함께 만족하지 못했다는 뜻으로 읽을 수 있다.

### 5) Judge design

Instruction compliance와 semantic fidelity는 Qwen3.5-122B-A10B를 MLLM judge로 사용한다. Evaluation format은 single-video와 dual-video를 동적으로 사용한다.

- Single-video format: edited video만 보고 판단 가능한 question에 사용한다.
- Dual-video format: source video와 edited video를 비교해야 하는 question에 사용한다.

이 설계는 중요하다. Preservation은 edited video만 봐서는 알기 어렵다. 어떤 object가 원래 있었는지, 어떤 background가 유지되어야 하는지는 source와 비교해야 한다. CoVEBench는 이 차이를 evaluation prompt level에서 반영한다.

# 4. Training / Data / Recipe

## 4-1. Data

CoVEBench는 training dataset이라기보다 evaluation dataset이다. 핵심 파일은 Hugging Face dataset과 GitHub evaluation code로 공개되어 있다.

Hugging Face repository 기준 주요 파일 구조는 다음과 같다.

| File or directory | Role |
| --- | --- |
| `checklist.json` | 626 checklist items와 source video path, editing instruction, evaluation group 포함 |
| `data/` | released source video files와 `metadata.jsonl` 포함 |
| `docs/assets/figures/` | README와 project page figure asset |
| `docs/assets/tables/` | README와 project page table asset |

`checklist.json`의 각 entry는 task id, source video path, category, original description, editing instruction, target video description, evaluation groups를 포함한다. 즉 단순히 video와 prompt만 주는 것이 아니라, evaluation을 위해 필요한 structured checklist가 같이 제공된다.

주의할 점도 있다. Paper와 project page는 416 curated source videos와 626 instructions를 설명하고, Hugging Face dataset viewer는 626 rows의 test split을 보여준다. 실제 사용 시에는 paper count보다 checklist id와 source video path mapping이 맞는지를 우선 확인하는 것이 좋다.

## 4-2. Evaluation recipe

CoVEBench의 reproduction recipe는 크게 두 부분이다.

1. Objective metric evaluation
   - Source video와 edited video를 task id 기준으로 맞춘다.
   - AES, TQ, MSM, SSIM, MF, VQR 같은 objective metric을 실행한다.
   - `scripts/run_model.py`가 source와 edited video pair를 내부 작업 디렉토리에 맞춰 구성한다.

2. Subjective MLLM-checklist evaluation
   - Qwen3.5-122B-A10B judge를 사용한다.
   - `scripts/run_subjective.py`가 checklist와 source/edited video pair를 받아 UAS, IFS, VRS, SEM 계열 점수를 계산한다.
   - Edited video는 checklist task id에 맞춰 저장해야 한다.

GitHub README 기준으로 SRC evaluation에 필요한 source-mask data는 아직 제공되지 않았고, 추후 release 예정이라고 설명된다. 따라서 현재 바로 재현할 수 있는 metric과 추후 추가될 metric을 구분해서 보는 것이 필요하다.

## 4-3. Engineering notes

실무 관점에서 중요한 engineering note는 세 가지다.

첫째, output naming convention이 benchmark 재현성의 일부다. Edited videos는 `data/my_model/{id}.mp4`처럼 checklist task id와 맞춰 저장해야 한다. Source video는 `checklist.json`의 `videoA_path`를 기준으로 resolve된다.

둘째, objective metric과 MLLM checklist metric은 서로 다른 dependency를 갖는다. Objective metric은 visual quality model, optical flow, SSIM, CoTracker 같은 toolchain이 필요하고, subjective metric은 large MLLM inference 환경이 필요하다. 그래서 CoVEBench는 evaluation benchmark이지만 실행 비용이 가볍지는 않다.

셋째, benchmark score를 운영 지표로 쓰려면 metric별 failure sample을 같이 봐야 한다. 예를 들어 UAS가 낮은 모델이 IFS는 높을 수 있다. 이 경우 모델은 edit을 시도하지만 visual naturalness나 realism에서 깨지는 것이다. 반대로 SEM이 낮으면 target edit을 위해 non-target content를 과하게 바꿨을 가능성이 있다.

# 5. Evaluation

## 5-1. Main results

논문은 closed-source 2개와 open-source 8개, 총 10개 video editing model을 평가한다. 아래 표는 원문 Table 3에서 주요 지표만 발췌한 것이다. 모든 metric은 higher-is-better다.

| Model | Source | UAS | IFS | VRS | VQR | SEM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Wan2.7 | Closed | 56.89 | 82.02 | 79.97 | 4.407 | 87.90 |
| HappyHorse1.0 | Closed | 55.18 | 76.54 | 84.52 | 4.388 | 92.73 |
| OmniWeaving | Open | 30.14 | 57.18 | 61.75 | 3.660 | 85.05 |
| Kiwi | Open | 29.03 | 53.90 | 56.13 | 3.670 | 79.51 |
| Ditto | Open | 26.50 | 49.45 | 60.69 | 3.921 | 58.02 |
| Lucy | Open | 26.01 | 50.85 | 58.68 | 3.688 | 86.13 |
| ICVE | Open | 25.83 | 53.14 | 54.00 | 3.277 | 71.02 |
| ReCo | Open | 24.35 | 54.16 | 47.42 | 3.146 | 70.03 |
| InsV2V | Open | 14.61 | 37.18 | 47.36 | 3.307 | 77.85 |
| VACE | Open | 9.69 | 22.92 | 41.35 | 3.718 | 81.73 |

이 결과에서 가장 중요한 포인트는 closed-source model이 앞서지만, 가장 높은 Wan2.7도 UAS 56.89에 머문다는 점이다. IFS 82.02와 비교하면 gap이 크다. 즉 개별 edit action은 어느 정도 수행하지만, 모든 edit point와 realism 조건을 동시에 만족하는 것은 아직 어렵다.

두 번째 포인트는 open-source model의 gap이다. 상위 open-source인 OmniWeaving은 UAS 30.14, Kiwi는 29.03이다. 반면 SEM은 상당히 높은 모델도 있다. 이는 보존을 잘하는 것과 복합 instruction을 정확히 실행하는 것이 같은 능력이 아니라는 의미다.

세 번째 포인트는 edit strength와 preservation trade-off다. 논문은 Ditto가 execution-related score는 비교적 경쟁력이 있지만 SEM이 낮다고 해석한다. 더 강하게 바꾸는 모델이 non-target semantics를 손상할 수 있다는 것이다.

## 5-2. What really matters in the experiments

### 1) UAS가 이 benchmark의 핵심이다

CoVEBench를 읽을 때 가장 먼저 봐야 할 metric은 UAS다. IFS는 requested edit을 했는지를 보고, VRS는 edit result가 natural한지를 본다. UAS는 이 둘이 동시에 만족되어야 올라간다.

그래서 UAS가 낮고 IFS가 높은 모델은 "수행은 했지만 완성도가 부족한 모델"로 볼 수 있다. 반대로 fidelity metric이 높은데 UAS가 낮은 모델은 "source를 보존하지만 edit execution이 약한 모델"일 수 있다. 이 분해가 CoVEBench의 장점이다.

### 2) Joint editing이 stepwise decomposition보다 낫다

복잡한 edit을 순서대로 쪼개면 더 쉬워 보일 수 있다. 하지만 논문은 joint editing과 sequential decomposition을 비교했을 때, joint editing이 UAS 30.63 vs 23.70, IFS 56.56 vs 48.68, VRS 56.48 vs 51.62로 더 높다고 보고한다.

이 결과는 직관적으로도 중요하다. Stepwise editing은 중간 결과가 다음 input이 되기 때문에 artifact가 누적된다. 또 뒤 edit이 앞 edit을 overwrite할 수 있다. Multi-point prompt를 단순히 single-edit chain으로 바꾸는 것은 생각보다 좋은 baseline이 아니다.

### 3) Metric validity를 따로 검증한 점이 좋다

CoVEBench는 MLLM judge에 크게 의존한다. 이 경우 가장 먼저 의심해야 할 것은 judge reliability다. 논문은 OmniWeaving과 Kiwi에 대해 repeated checklist evaluation을 수행했고 score variation이 0.5 point 이내라고 보고한다. 또한 Wan과 Kiwi에서 sampled 100 cases를 human evaluator와 Qwen3.5-122B-A10B가 함께 평가했을 때, objective question agreement가 93%를 넘었다고 보고한다.

Pairwise preference consistency에서도 SEM, TQ, AES, MSM, VQR, MF, SSIM, SRC가 모두 85% 이상 human preference agreement를 보인다. 물론 이것이 judge가 완벽하다는 뜻은 아니지만, benchmark paper로서 최소한의 sanity check를 제공한다는 점은 긍정적이다.

### 4) Complexity가 올라가면 성능이 내려간다

Further analysis에서 InsV2V, Kiwi, Lucy를 대상으로 generated frame count와 source video duration을 늘려 temporal scalability를 본다. 전반적으로 longer temporal span이 task difficulty를 키우며 성능을 낮춘다.

또한 edit point 수와 instruction length가 늘어날수록 performance가 하락한다. 이 결과는 CoVEBench의 문제 설정과 직접 연결된다. 현재 video editing model은 단일 visual transformation보다, 여러 spatiotemporal constraint를 한 번에 묶어서 처리하는 능력이 약하다.

### 5) Error analysis가 실제 개선 방향을 잘 보여준다

논문은 5개 model, Wan2.7, OmniWeaving, Kiwi, Ditto, InsV2V에 대해 100개 identical sample을 random selection해서 error analysis를 수행한다. Failure는 크게 네 가지로 분류된다.

1. Execution inadequacies: instruction following failure 또는 text rendering failure.
2. Spatial entanglement: non-target region이 같이 바뀌는 문제.
3. Lack of physical grounding: unnatural motion 또는 physical logic violation.
4. Visual degradation: photorealism loss나 generative artifact.

가장 중요한 병목은 inadequate instruction following으로 보고된다. Closed-source model은 상대적으로 instruction adherence가 강하지만, complex instruction에서는 completion rate가 여전히 크게 떨어지고, physical violation이나 unnatural blending도 나타난다.

이 분석이 좋은 이유는 benchmark가 단순히 "누가 이겼다"로 끝나지 않기 때문이다. Model developer 입장에서는 어느 failure type을 줄여야 하는지, evaluation sample을 열어 직접 볼 수 있다.

# 6. Limitations

1. Text-guided instruction으로 scope가 제한된다.
   - 실제 video editing workflow에는 reference image, mask, bounding box, audio cue, keyframe control 같은 multimodal control signal이 자주 들어간다.
   - CoVEBench는 이 부분을 현재 framework에 포함하지 않는다.

2. Benchmark이지 solution paper가 아니다.
   - 논문은 complex compositional editing의 failure를 잘 보여주지만, 이를 해결하는 new editing model이나 agent를 제안하지 않는다.
   - 따라서 성능 개선 recipe를 직접 얻기보다는, evaluation target과 failure taxonomy를 얻는 논문으로 읽는 편이 맞다.

3. Paired large-scale training dataset은 제공하지 않는다.
   - CoVEBench는 evaluation dataset과 checklist를 제공하지만, compositional editing model을 학습시킬 paired source/target large-scale corpus는 제공하지 않는다.
   - Benchmark score를 개선하려면 별도 data generation이나 training recipe가 필요하다.

4. MLLM judge 의존성이 있다.
   - UAS, IFS, VRS, SEM은 checklist 기반 MLLM judge에 크게 의존한다.
   - 논문은 human agreement를 검증하지만, judge model version, decoding setting, prompt sensitivity에 따라 future reproduction 결과가 달라질 수 있다.

5. Closed-source model coverage는 제한적이다.
   - Closed-source는 Wan2.7과 HappyHorse1.0 두 개만 포함된다.
   - Proprietary model API는 시간이 지나면 버전이 바뀔 수 있으므로, leaderboard 수치는 evaluation date와 model version context를 함께 봐야 한다.

6. Released artifact snapshot 확인이 필요하다.
   - Paper와 project page는 416 curated videos와 626 multi-point instructions를 설명한다.
   - Hugging Face dataset viewer는 626 rows의 test split을 보여준다.
   - 실제 evaluation 재현 시에는 checklist id 기준 source video path mapping을 우선으로 확인하는 것이 안전하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 video editing model benchmark지만, 더 넓게 보면 multimodal generation evaluation paper다. 특히 OCR, document AI, VLM agent, video understanding에서도 같은 문제가 반복된다. User instruction은 복합적이고, target change와 non-target preservation이 동시에 필요하며, global score만으로는 failure를 설명할 수 없다.

CoVEBench의 핵심은 checklist 자체보다 "preservation을 evaluation의 first-class dimension으로 올린다"는 점이다. Generation model 평가에서는 보통 얼마나 잘 바꿨는가에 집중하기 쉽다. 하지만 editing은 본질적으로 selective transformation이다. 잘 바꾸는 것만큼, 건드리지 말아야 할 것을 유지하는 능력이 중요하다.

이 관점은 실제 서비스에도 바로 연결된다. 예를 들어 product video editing, ad creative generation, short-form content editing에서 고객이 싫어하는 failure는 "요청한 색이 덜 바뀜"보다 "브랜드 로고나 제품 형태가 같이 망가짐"인 경우가 많다. CoVEBench의 SEM, SRC 같은 metric은 이런 문제를 benchmark 수준으로 끌어올린다.

## 7-2. Reuse potential

재사용할 만한 포인트는 다음과 같다.

1. Checklist-based evaluation
   - 하나의 prompt를 atomic requirement로 분해하고, requirement별 yes/no 또는 MCQ로 평가한다.
   - Video editing뿐 아니라 document editing, UI generation, image editing, workflow agent evaluation에도 적용 가능하다.

2. Primary metric separation
   - UAS, VQR, SEM처럼 execution, quality, preservation을 분리한다.
   - 단일 score를 만들더라도, dashboard에서는 세 축을 따로 보여주는 것이 좋다.

3. Source-to-output comparison
   - Preservation은 output만 봐서는 평가하기 어렵다.
   - Source와 output을 함께 보는 dual-input judge가 필요하다.

4. Failure taxonomy
   - Execution inadequacy, spatial entanglement, physical grounding, visual degradation 같은 taxonomy는 model debugging에 바로 쓸 수 있다.
   - 특히 generation model QA에서는 failure type이 training data 개선 방향과 직결된다.

5. Joint vs sequential editing baseline
   - Complex instruction을 단일 edit chain으로 쪼개는 baseline은 항상 안전하지 않다.
   - Intermediate artifact accumulation과 overwrite 문제가 생길 수 있으므로, pipeline design에서 별도 검증이 필요하다.

## 7-3. Follow-up papers

- IVEBench: modern benchmark suite for instruction-guided video editing assessment.
- VEditBench: video editing benchmark with broader video editing evaluation setup.
- InstructPix2Pix: instruction-guided image editing의 초기 대표 접근.
- TokenFlow: video editing에서 temporal consistency를 다루는 대표적인 diffusion feature 기반 접근.
- UniVBench: unified video capability benchmark로 CoVEBench와 비교해 읽기 좋다.

# 8. Summary

- CoVEBench는 complex video editing prompt를 626개 instruction과 9,990개 checklist item으로 benchmark화한다.
- 핵심은 single global score가 아니라 instruction compliance, video quality, video fidelity를 분리해 보는 것이다.
- Closed-source model이 앞서지만, 가장 높은 Wan2.7도 UAS 56.89로 complex compositional editing을 완전히 해결하지 못한다.
- Joint editing은 sequential decomposition보다 나았고, edit point 수와 instruction length가 늘어날수록 성능이 떨어진다.
- 이 논문은 새 video editing model보다, real workflow를 평가하기 위한 diagnostic evaluation system으로 읽는 것이 가장 유용하다.
