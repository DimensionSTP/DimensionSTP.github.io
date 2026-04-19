---
layout: single
title: "Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models Review"
categories: Study-concept
tag: [Molmo, PixMo, VLM]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2409.17146)

Molmo and PixMo는 “open VLM도 꽤 강해졌다” 정도로 보면 아까운 논문이다. 이 논문의 진짜 흥미로운 지점은 **closed VLM distillation 없이도 competitive한 open-data / open-weight VLM을 어떻게 만들 수 있는가**를 꽤 구체적으로 보여준다는 데 있다.

최근 open VLM 생태계를 보면, weights는 공개되어도 실제 성능을 만든 데이터는 GPT-4V류의 proprietary VLM outputs인 경우가 많다. 그러면 결과는 공개되지만, **community는 결국 처음부터 강한 VLM을 어떻게 만드는지 배우지 못한다**. Molmo는 바로 그 지점을 겨냥한다. 저자들은 단순히 모델 체크포인트만 내놓지 않고, **PixMo라는 데이터 스위트, training recipe, evaluation methodology**까지 함께 묶어서 제시한다.

특히 이 논문은 architecture novelty를 전면에 내세우지 않는다. backbone은 오히려 상당히 정석적인 편이다. 대신 승부처를 **dense caption data의 품질, free-form QA 수집 방식, pointing supervision, overlapping multi-crop design, benchmark-style control, human evaluation**에 둔다. 그래서 이 논문은 “새로운 VLM 블록” 논문이라기보다 **data-centric open VLM construction manual**에 가깝다.

> 한 줄 요약: Molmo는 standard한 VLM 아키텍처에 PixMo라는 고품질 open multimodal data suite와 overlapping multi-crop / attention pooling / point-based supervision을 결합해, proprietary VLM synthetic data 없이도 강한 open-data / open-weight VLM family를 만든 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- open VLM이 정말로 **open한 방법론** 위에 설 수 있는지 묻는 논문이라서, 결과보다 **과정의 공개성**이 중요하다.
- OCR, document-like image understanding, grounding, counting, user preference까지 이어져서 **실서비스형 VLM 파이프라인** 관점에서 재사용 가치가 높다.
- “좋은 VLM은 아키텍처보다 데이터와 인터페이스 설계가 좌우한다”는 점을 매우 설득력 있게 보여준다.

내가 보기엔 이 논문의 핵심 메시지는 단순하다. **좋은 open VLM을 만드는 문제는 모델 블록 하나를 교체하는 문제라기보다, 어떤 supervision을 어떤 인터페이스로 모으고, 그것을 어떤 training loop로 연결할 것인가의 문제**다. Molmo는 그 연결 고리를 꽤 정직하게 드러낸다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 **강한 VLM을 공개 가능한 방식으로 만드는 것**이다.
- 여기서 공개 가능성은 단순히 weights 공개만 의미하지 않는다.
  - training data
  - data collection method
  - training code
  - evaluation setup
  - human preference evaluation 방식
  까지 최대한 따라갈 수 있어야 한다.
- 동시에 모델은 단순한 captioner가 아니라, 아래 능력을 함께 가져야 한다.
  - **natural image understanding**
  - **fine-grained OCR / document understanding**
  - **grounding**
  - **counting**
  - **user-facing QA quality**
- 즉 문제 설정은 “open model 하나 만들기”가 아니라, **연구 커뮤니티가 재현하고 확장할 수 있는 수준으로 경쟁력 있는 VLM recipe를 만드는 것**에 가깝다.

## 1-2. Why previous approaches are insufficient

- 기존 proprietary VLM들은 강하지만, weights와 data, recipe가 닫혀 있어서 **어떻게 만들었는지 학습할 수 없다**.
- 반대로 최근 강한 open-weight VLM들 중 다수는 사실상 **proprietary VLM distillation**에 크게 의존한다. 즉 모델은 열려 있어도, 성능을 만든 핵심 supervision은 닫혀 있는 경우가 많다.
- 일반적인 short caption 데이터는 세밀한 시각 이해에는 한계가 있다. 특히 OCR, counting, dense perception처럼 **정말 이미지를 자세히 봐야 하는 작업**에는 부족하다.
- 또 standard ViT는 고정 해상도의 정사각형 입력에 묶이기 쉬워서, 문서나 표, 인포그래픽처럼 **aspect ratio가 길고 텍스트가 많은 이미지**를 다루기 어렵다.
- grounding 쪽도 마찬가지다. 기존 referring expression 데이터는 category coverage가 제한적이거나, multi-instance / not-present / explanation 같은 실제 상호작용에 중요한 경우를 충분히 담지 못한다.
- 마지막으로 evaluation도 문제다. 프롬프트나 preprocessing detail이 조금만 달라도 숫자가 크게 흔들릴 수 있는데, 많은 모델 리포트는 그 부분을 충분히 드러내지 않는다.
- 결국 기존 접근의 한계는 개별 요소 하나가 아니라, **data openness / visual supervision quality / spatial interface / evaluation hygiene가 하나의 시스템으로 설계되지 않았다**는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- Molmo의 가장 큰 기여는 하나의 fancy한 architecture가 아니라 **open VLM full-stack recipe의 설계**다.
- 첫째, **PixMo**라는 데이터 스위트를 만든다. 여기에는 dense caption용 PixMo-Cap, free-form QA용 PixMo-AskModelAnything, grounding / counting / explanation용 PixMo-Points가 포함된다.
- 둘째, synthetic data도 목적별로 제한적으로 설계한다. PixMo-CapQA, PixMo-Docs, PixMo-Clocks, PixMo-Count는 각각 **caption-derived QA, document-heavy understanding, clock reading, counting**을 보강한다.
- 셋째, 모델 아키텍처는 일부러 standard한 형태를 유지한다. 이 덕분에 성능 향상을 **data와 recipe의 효과**로 더 깔끔하게 읽을 수 있다.
- 넷째, dense caption pre-training부터 pointing-aware fine-tuning까지를 하나의 일관된 interface로 연결한다.
- 다섯째, academic benchmark뿐 아니라 대규모 human preference evaluation까지 수행해서 **숫자와 사용자 선호를 함께 본다**.

## 2-2. Design intuition

- 이 논문의 설계 직관은 매우 실무적이다.
- **좋은 dense perception은 짧은 캡션으로는 잘 안 생긴다.** 그래서 PixMo-Cap은 annotator가 최소 60~90초 동안 이미지를 말로 설명하게 만든다.
- **좋은 user-facing QA는 benchmark conversion만으로는 안 생긴다.** 그래서 PixMo-AskModelAnything은 사람이 실제로 궁금한 질문을 쓰고, language-only LLM이 OCR + caption 정보를 바탕으로 답변 초안을 만들고, 사람이 그것을 reject / revise하는 루프를 둔다.
- **grounding과 counting은 따로 놀지 않는다.** 그래서 PixMo-Points는 point를 통해 grounding을 학습시키면서, 같은 point sequence를 counting의 chain-of-thought처럼 쓰게 만든다.
- **가능한 한 standard backbone을 유지해야 무엇이 실제로 효과가 있었는지 보인다.** 그래서 저자들은 backbone보다는 data, crop design, pooling, training recipe에 에너지를 쓴다.
- **full benchmark 실행은 비싸다.** 그래서 pre-training 단계에서는 dense caption metric인 `cap`을 만들어 빠르게 iterate하고, 이것이 downstream 평균과 얼마나 연결되는지도 나중에 확인한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | proprietary VLM synthetic data 없이 경쟁력 있는 open-data / open-weight VLM family를 만드는 것 |
| Key module | PixMo data suite + overlapping multi-crop preprocessor + attention-pooled connector + point-based fine-tuning |
| Core design principle | dense perception은 long caption으로, grounding / counting은 points로, user-facing QA는 human-in-the-loop QA data로 해결 |
| Difference from prior work | architecture novelty보다 data quality, spatial interface, evaluation openness를 전면에 둠 |

## 3-2. Module breakdown

### 1) Standard backbone을 유지하고 data effect를 드러낸다

- Molmo의 기본 구조는 비교적 전형적이다.
  - image pre-processor
  - ViT image encoder
  - vision-language connector
  - decoder-only LLM
- 논문은 이 네 조각을 크게 뒤집지 않는다.
- 주로 사용하는 vision encoder는 **CLIP ViT-L/14 336px**이고, 비교 대상으로 SigLIP과 MetaCLIP도 실험한다.
- LLM 쪽은 openness 수준과 scale을 달리하면서
  - OLMo-7B-1024-preview
  - OLMoE-1B-7B
  - Qwen2 7B
  - Qwen2 72B
  를 사용한다.
- 이 선택이 중요한 이유는, 결과를 “새 backbone이 좋아서”가 아니라 **recipe가 좋아서**라고 해석하기 쉬워지기 때문이다.

### 2) Overlapping multi-crop은 OCR과 fine-grained perception의 핵심이다

- fixed-resolution ViT는 문서나 작은 텍스트를 읽기에 해상도가 부족하다.
- 그래서 Molmo는 **저해상도 full image overview + 여러 개의 high-resolution crop**을 함께 사용한다.
- 여기서 포인트는 단순 crop이 아니라 **overlap**이다.
- crop 경계에 걸린 텍스트나 작은 물체는 주변 문맥이 없으면 인코더가 해석하기 어렵다. 저자들은 crop을 겹치게 만들어 border patch도 이웃 문맥을 보게 한다.
- 중요한 건 overlap 영역의 patch feature를 그대로 다 넘기지 않는다는 점이다. **겹쳐서 인코딩하되, 최종적으로 LLM에 넘기는 patch들은 정확히 high-res image를 tile하도록 정리**한다.
- 이 구조는 OCR / detailed captioning / fine-grained recognition에서 꽤 큰 차이를 만든다.

### 3) Connector는 화려하지 않지만 잘 다듬어져 있다

- crop이 ViT를 통과하면, 저자들은 **ViT의 두 개 intermediate layer feature**를 합쳐 patch feature를 만든다.
- 이후 각 2×2 patch window를 하나의 벡터로 줄이는데, 여기서 simple stacking이 아니라 **multi-head attention pooling**을 쓴다.
- query는 patch 평균을 사용하고, pooled feature는 다시 MLP를 거쳐 LLM embedding space로 projection된다.
- 이 connector는 논문 전체의 tone을 잘 보여준다. 새로운 대형 모듈을 만들기보다, **표현 손실을 줄이는 작은 선택들을 하나씩 쌓는 방식**이다.

### 4) Token arrangement와 text-only dropout도 중요한 디테일이다

- vision token은 low-res overview를 먼저 넣고, 뒤이어 high-res crop patch들을 row-major order로 이어 붙인다.
- low-res / high-res sequence 시작과 끝, 그리고 row transition을 나타내는 special token도 둔다.
- 이런 배치는 단순 formatting처럼 보이지만, 실제로는 **공간 구조를 언어 모델에게 어떻게 전달할 것인가**의 문제다.
- pre-training에서는 residual dropout을 LLM에만 적용하되, **text token에만 dropout**을 넣는다.
- 이 선택은 모델이 language prior만으로 답을 찍는 대신, **이미지 쪽 representation을 더 적극적으로 쓰게 만드는 장치**로 이해할 수 있다.

### 5) Multi-annotated images와 point interface는 실용성이 높다

- VQA처럼 이미지 하나에 여러 QA가 달린 데이터는 image encoding이 반복되기 쉽다.
- Molmo는 동일 이미지의 여러 annotation token을 한 시퀀스로 묶고, attention mask를 조절해 **annotation 간 leakage 없이 image encoding 재사용**을 한다.
- 이 방식은 processed image 수를 약 2/3 줄이고, training time도 절반 이상 줄이는 실용적인 최적화다.
- 또 point는 Molmo의 중요한 interface다.
  - grounding
  - counting
  - explanation
  을 하나의 supervision 형식으로 연결한다.
- point 좌표를 normalized plain-text로 내보내고, 여러 object를 가리킬 때는 **top-down, left-to-right** 순서로 출력한다.
- 이 설계는 “모델이 어디를 보고 셌는지”를 드러내는 장점도 있다.

# 4. Training / Data / Recipe

## 4-1. Data

PixMo는 **3개의 human-annotated dataset + 4개의 synthetic dataset**으로 구성된다. 이 중 핵심은 human data이고, synthetic data는 목적별로 skill gap을 메우는 식으로 들어간다.

| Dataset | Type | Scale | Main role |
| --- | --- | --- | --- |
| PixMo-Cap | Human | 712k images, 1.3M transcripts/captions | dense caption pre-training |
| PixMo-AskModelAnything | Human | 162k QA pairs, 73k images | real-world free-form visual QA |
| PixMo-Points | Human | 2.3M question-points pairs, 223k images + 79k point-explanation annotations | grounding, counting, explanation |
| PixMo-CapQA | Synthetic | 214k QA pairs, 165k images | caption-derived QA augmentation |
| PixMo-Docs | Synthetic | 255k document-like images + 2.3M QA pairs | charts / documents / tables / diagrams 이해 |
| PixMo-Clocks | Synthetic | 826k examples | clock reading |
| PixMo-Count | Semi-synthetic | 36k train images + manually verified val/test | counting |

조금 더 해석해보면 아래가 중요하다.

- **PixMo-Cap**이 사실상 프로젝트의 중심축이다.
  - 저자들은 70개 topic에서 web image를 모은다.
  - annotator는 이미지를 최소 60초, 이후 단계에서는 90초 이상 말로 설명한다.
  - 음성 transcript를 얻은 뒤 language-only LLM으로 정리해 최종 caption을 만든다.
  - 최종 caption 평균 길이는 **196 words**로, COCO caption의 11 words와 비교하면 supervision density가 전혀 다르다.
- **PixMo-AskModelAnything**는 사람이 질문을 쓰고, OCR 결과와 PixMo-Cap 기반 dense description을 바탕으로 language-only LLM이 답안을 만들고, 사람이 accept / reject / revise하는 구조다.
  - 즉, 단순 synthetic QA가 아니라 **human intention이 먼저 들어가고, LLM은 answer drafting assistant로 쓰인다**.
- **PixMo-Points**는 point annotation을 단순 localization dataset으로 쓰지 않는다.
  - arbitrary referring expression grounding
  - absent target handling
  - count-by-pointing
  - point-based explanation
  을 하나의 supervision으로 묶는다.
- **PixMo-Docs**도 재미있다.
  - charts, tables, diagrams, documents 같은 이미지에 대해, 직접 이미지를 읽어 QA를 만드는 대신 **code generation + code-aware QA** 파이프라인을 사용한다.
  - 즉, image QA를 만들기 위해 external VLM을 부르지 않는다.
- **PixMo-Clocks**와 **PixMo-Count**는 targeted synthetic data다.
  - 이 논문은 synthetic을 무조건 크게 넣는 방식이 아니라, **현재 VLM이 놓치고 있는 skill을 특정해서 보강하는 방식**으로 쓴다.

## 4-2. Training strategy

### Pre-training

- pre-training은 **모든 파라미터를 PixMo-Cap에 대해 학습**한다.
- 목표는 주어진 이미지에 대해 caption 또는 transcript style text를 생성하는 것이다.
- 프롬프트는 어떤 style을 생성할지 지정하고, **90% 확률로 length hint**를 넣어 출력 길이를 유도한다.
- 이 length conditioning은 단순 prompt trick이 아니라, 실제로 caption quality와 downstream에 도움이 되는 설계로 제시된다.
- 흥미로운 점은, 많은 prior work가 쓰던 **connector-only 초기 적응 단계**를 별도로 두지 않는다는 점이다.
- 대신 connector에는 더 높은 learning rate와 짧은 warmup을 주고, 아예 end-to-end pre-training으로 밀어붙인다.
- 논문 기준 주요 hyperparameter는 다음과 같다.
  - optimizer: AdamW
  - schedule: cosine decay to 10% of peak
  - pre-training epochs: 4
  - learning rate: connector 2e-4 / ViT 6e-6 / LM 2e-5
  - warmup: connector 200 steps / ViT·LM 2000 steps
  - gradient clipping: LM, image encoder, connector를 따로 적용

### Fine-tuning

- fine-tuning은 PixMo 데이터와 기존 open academic datasets의 mixture 위에서 진행된다.
- 포함되는 대표 데이터는 다음과 같다.
  - VQA v2.0
  - TextVQA
  - OK-VQA
  - ChartQA
  - DocVQA
  - InfographicVQA
  - AI2D
  - A-OKVQA
  - AndroidControl
  - ScienceQA
  - TabMWP
  - ST-VQA
  - TallyQA
  - DVQA
  - FigureQA
  - PlotQA
- 샘플링은 기본적으로 **dataset size의 square root에 비례**하게 하되, 너무 큰 synthetic dataset은 수동으로 down-weight한다.
- 반대로 **pointing data는 QA보다 학습이 느리기 때문에 강하게 up-weight**한다.
- benchmark 데이터는 대개 매우 짧고 특이한 answer style을 요구한다. 그래서 Molmo는 dataset-specific style tag를 둔다.
  - 예: `vqa2:`
- 이건 꽤 중요하다. benchmark에서 높은 점수를 내기 위한 style을 배우게 하되, 그것이 **사용자 대화 스타일 전체를 오염시키지 않게 하는 장치**이기 때문이다.
- point output은 0~100 범위의 normalized plain-text coordinate로 출력하고, count는 point sequence 뒤에 total count를 주는 식으로 구성된다.

## 4-3. Engineering notes

이 논문에서 가장 실용적인 인사이트는 architecture보다 engineering detail에서 많이 나온다.

### 1) `cap` metric은 꽤 쓸만한 개발 proxy다

- 저자들은 대부분의 개발 과정에서 full downstream benchmark를 계속 돌리지 않았다.
- 대신 dense caption의 precision / recall을 보는 **`cap` metric**을 개발 proxy로 썼다.
- 나중에 ablation을 모아보니 `cap`과 11-benchmark 평균(`11-avg`) 사이에 **상당한 상관**이 보인다.
- 이건 실무적으로도 중요하다. full SFT + full eval이 비싸다면, **dense perception을 잘 반영하는 싼 proxy를 먼저 세우는 것**이 iteration 속도를 크게 바꾼다.

### 2) Multi-crop과 overlap은 optional이 아니다

- model ablation에서 single low-res input은 `11-avg`가 **62.8**이고,
- multi-crop인데 overlap이 없으면 **75.7**,
- multi-crop + overlap이면 **76.9**까지 올라간다.
- 즉 고해상도 OCR / fine-grained understanding을 원한다면, **crop 설계는 전처리 detail이 아니라 사실상 모델 설계의 일부**다.

### 3) Fully-open vision stack도 충분히 가능하다

- vision encoder ablation에서는 OpenAI CLIP, MetaCLIP, SigLIP가 거의 비슷하게 나온다.
- 특히 **MetaCLIP 336px가 CLIP과 비슷하거나 약간 더 낫게** 나와서, vision encoder 쪽은 fully-open 경로가 꽤 현실적임을 보여준다.
- 저자들도 MetaCLIP + OLMo 조합이라면 **모델 구성요소와 데이터 모두 open**이라고 강조한다.

### 4) Data quality가 benchmark와 human preference를 같이 움직인다

- data ablation이 아주 설득력 있다.
- academic-only fine-tuning은 `11-avg`가 **72.5**이고,
- 여기에 PixMo-Docs만 더해도 **74.0**,
- full PixMo + academic mixture는 **76.9**다.
- 특히 pointing task를 빼면 **76.2**로 내려간다.
- human evaluation에서도 **PixMo-Cap과 PixMo-AskModelAnything이 사용자 선호를 끌어올리는 핵심 데이터**로 나온다.
- 즉 benchmark용 dataset만 모아서 fine-tune한다고 user-facing VLM이 되지는 않는다는 뜻이다.

### 5) Counting은 “point then count”가 가장 강하다

- counting ablation은 꽤 인상적이다.
- 단순 count-only보다, **먼저 point sequence를 생성하고 그 다음 count를 말하는 전략**이 CountBenchQA와 PixMo-Count 양쪽에서 가장 좋다.
- 또 point의 출력 순서는 random이 아니라 **top-down / left-to-right** 같은 예측 가능한 순서가 중요하다.
- 좌표 표현도 special token보다 **plain-text coordinate**가 더 낫다.
- 이건 결국, counting을 answer generation이 아니라 **grounded reasoning procedure**로 다뤘을 때 성능이 좋아진다는 의미다.

# 5. Evaluation

## 5-1. Main results

이 논문의 결과는 단순히 “open model 중 하나가 좋다” 수준이 아니라, **어떤 openness class에서 어떤 capability profile을 보였는가**로 읽는 편이 좋다.

### 1) Family-level result는 꽤 강하다

- 논문 기준에서 Molmo family는 **open weights, open data, open training code, open evaluations**를 함께 제시한다.
- 대표적으로,
  - **MolmoE-1B**는 11-benchmark average가 **68.6**, Elo가 **1032**다.
  - **Molmo-7B-O**는 **74.6 / 1051**,
  - **Molmo-7B-D**는 **77.3 / 1056**,
  - **Molmo-72B**는 **81.2 / 1077**을 기록한다.
- 비교용으로 논문 내 표를 보면,
  - **GPT-4V**는 **71.1 / 1041**,
  - **GPT-4o-0513**은 **78.5 / 1079**다.
- 즉 Molmo-72B는 논문 기준 **highest academic score + Elo 2위**를 찍고, Molmo-7B-D도 이미 GPT-4V를 넘는 구간이 나온다.
- 다만 중요한 nuance가 있다. strongest result인 Molmo-72B는 **Qwen2 72B 기반의 open-weight model**이다. 이 논문은 openness를 binary로 보지 않고, **각 openness class 안에서 얼마나 강한가**를 보여주는 방식으로 읽는 편이 정확하다.

### 2) 어디에서 특히 강한가

- Molmo는 **natural image QA**에서 강하다.
  - 논문 저자들은 RealWorldQA와 VQA v2.0에서 특히 강한 profile을 강조한다.
- **OCR-centric benchmark**에서도 강하다.
  - ChartQA, DocVQA, InfoQA, TextVQA에서 open model들을 넘고 일부 proprietary 모델도 앞선다.
  - 다만 이 구간에서는 **Qwen2-VL이 약간 더 강한 benchmark**도 있다.
- **counting**은 Molmo의 확실한 강점이다.
  - CountBenchQA와 PixMo-Count에서 leading result를 보여주는데, 핵심 원인은 PixMo-Points와 point-then-count interface다.
- 반면 **MMMU, MathVista 같은 reasoning-heavy task**에서는 상대적으로 약하다.
  - 논문도 이를 training mix가 advanced reasoning에 더 최적화되어 있지 않기 때문이라고 해석한다.

### 3) Skill-specific evaluation도 흥미롭다

- **clock reading**에서는 Molmo가 매우 강하다.
- 특히 일반-purpose VLM들 대부분이 clock reading에서 크게 고전하는데, Molmo는 all scale에서 큰 gap을 보인다.
- 다만 specialized single-task clock reading model보다는 여전히 아래다.
- **AndroidControl**에서도 Molmo-72B는 준수한 step-wise accuracy를 보여, 단순 QA를 넘는 action grounding 잠재력도 확인한다.

### 4) Human evaluation을 함께 봐야 한다

- 이 논문은 15k image-text prompts, 870 annotators, 325k+ ratings를 이용해 Bradley-Terry 기반 Elo를 계산한다.
- 이 부분이 중요하다. academic benchmark만 보면 놓치기 쉬운 **response usefulness와 naturalness**를 추가로 본다.
- 전반적으로 academic benchmark와 human eval은 비슷한 경향을 보이지만, **Qwen2-VL처럼 benchmark는 강한데 human preference는 상대적으로 덜 나오는 예외**도 있다.
- 즉 user-facing VLM에서는 숫자만큼이나 **사람이 실제로 선호하는 응답을 만드는 데이터**가 중요하다는 뜻이다.

## 5-2. What really matters in the experiments

### 1) 이 논문의 진짜 성과는 data attribution이다

- headline만 보면 “Molmo-72B가 강하다”로 읽힐 수 있다.
- 하지만 내가 보기엔 더 중요한 건 **왜 강한지를 상당 부분 data와 interface 수준에서 설명할 수 있다는 점**이다.
- PixMo-Cap scaling, PixMo-Docs 추가, pointing 제거 ablation, human eval ablation이 모두 같은 방향을 가리킨다.
- 결국 Molmo의 성능은 “모델이 좋아서”라기보다, **dense caption + real QA + points + targeted synthetic docs/clocks/count**의 조합이 잘 설계되었기 때문이라고 읽힌다.

### 2) Evaluation openness 자체도 contribution이다

- 저자들은 모델 비교 시 prompt와 preprocessing detail 때문에 점수가 **10% 가까이 흔들릴 수 있다**고 직접 적는다.
- 이건 굉장히 중요한 고백이다.
- 많은 leaderboard는 숫자만 있고 조건이 없다. Molmo는 적어도 **비교 숫자가 얼마나 implementation-sensitive한지**를 드러낸다.
- 특히 benchmark별 style tag 사용 여부, crop 수, human eval에서는 point를 가리지 않고 text output만 보여주는 방식 등, **evaluation protocol을 꽤 구체적으로 남긴다**.

### 3) PixMo-Cap이 프로젝트의 실질적 중심축이다

- 프로젝트 전체를 보면 PixMo-Cap이 거의 모든 것의 출발점이다.
- pre-training target이고,
- AskModelAnything answer drafting의 기반이며,
- `cap` metric의 기반이고,
- human preference ablation에서도 중요하게 나온다.
- 심지어 GPT-4o로 PixMo images를 caption한 variant도 강하게 나오는데, 이건 “역시 GPT-4o가 최고다”라기보다 **이미지 풀 자체가 좋고 dense caption supervision이 중요하다**는 해석이 더 맞다.
- 저자들도 distillation이 효과적일 수는 있지만, community가 **competitive VLM을 distillation 없이 이해하고 만들 수 있어야 한다**는 점을 분명히 한다.

### 4) Chatbot Arena 결과와의 차이도 의미가 있다

- 논문 자체 human eval에서는 Molmo-72B가 GPT-4o 바로 뒤까지 올라가지만,
- independent Chatbot Arena snapshot에서는 여전히 여러 proprietary 모델 아래에 있다.
- 이 차이는 질문 분포 차이에서 왔을 가능성이 크다.
- 저자들도 Molmo 데이터가 counting / image-description에 강하다고 해석한다.
- 즉 **평가셋의 성격이 모델 순위를 크게 바꾼다**는 당연하지만 자주 잊히는 사실을 다시 보여준다.

# 6. Limitations

1. **현재 data pipeline은 완전히 open하다고 보기 어렵다.**  
   PixMo를 만들 때 external VLM은 쓰지 않았지만, 일부 데이터 생성 / 정제 과정에는 closed text-only LLM이 들어간다. 논문도 이 점을 명시한다. 즉 circular VLM distillation은 피했지만, 완전한 의미의 end-to-end openness는 아직 아니다.

2. **가장 강한 결과가 곧 fully-open을 의미하지는 않는다.**  
   Molmo family는 openness level이 다층적이다. strongest model인 Molmo-72B는 Qwen2 72B 기반 open-weight 모델이다. 반면 MetaCLIP + OLMo 경로는 fully-open에 더 가깝다. 따라서 이 논문은 “완전히 열린 하나의 최고 모델”보다 **openness class별 frontier**를 제시하는 논문으로 보는 편이 정확하다.

3. **능력 profile이 perception-heavy 쪽으로 치우쳐 있다.**  
   Molmo는 natural image understanding, OCR, counting, clock reading 쪽에서는 매우 강하지만, MMMU / MathVista 같은 reasoning-heavy benchmark에서는 상대적으로 약하다. training mix가 그런 방향으로 설계된 결과다.

4. **spatial interface는 crop mismatch에 민감하다.**  
   논문은 일반 academic evaluation에서는 36 crops를 쓰지만, pointing은 train / test crop 수가 달라지면 성능이 크게 흔들릴 수 있다고 보여준다. 즉 spatial coordinate interface는 단순 decoding 포맷이 아니라, **training-time image geometry와 강하게 묶인 설계**다.

5. **multimodal fine-tuning은 text-only capability를 일부 깎을 수 있다.**  
   appendix의 text-only benchmark를 보면 component LLM 대비 손실이 나타난다. 저자들은 소량의 text-only post-training으로 일부를 복구할 수 있다고 보이지만, 결국 multimodal specialization과 text-only generality 사이에는 trade-off가 있다.

6. **open recipe라고 해서 가볍게 재현 가능한 것은 아니다.**  
   72만 장 dense caption 수집, 2.3M pointing pair, large-scale human eval까지 포함된 pipeline은 상당히 무겁다. 이 논문은 “재현 가능한 철학”을 공개한 것이지, **누구나 바로 따라할 수 있는 저비용 recipe**를 준 것은 아니다.

# 7. My Take

## 7-1. Why this matters for my work

- 내가 이 논문을 높게 보는 이유는, VLM 성능 향상을 **backbone 경쟁**이 아니라 **supervision interface 설계**로 보게 만든다는 점 때문이다.
- 특히 아래 관심 축과 직접 연결된다.
  - OCR / Document AI
  - grounding-based multimodal assistant
  - point-based explanation
  - 실제 서비스용 visual QA pipeline
- Molmo를 보고 나면, “좋은 VLM을 만들려면 더 큰 encoder가 필요한가?”보다
  - 더 긴 caption supervision이 필요한가?
  - human-written QA가 필요한가?
  - point annotation이 counting과 grounding을 같이 해결할 수 있는가?
  같은 질문을 하게 된다.
- 이 framing은 실무적으로도 훨씬 유용하다.

## 7-2. Reuse potential

- **speech-to-caption annotation**  
  문서, UI, 웹페이지, 산업 장비 화면처럼 dense perception이 중요한 도메인에서는 짧은 typed caption보다 spoken description 기반 dense caption 수집이 훨씬 나을 수 있다.

- **overview + overlapping crop**  
  OCR, chart, infographic, UI understanding에서는 거의 바로 가져다 쓸 수 있는 설계다. 특히 border context 문제를 overlap으로 푸는 아이디어는 단순하지만 효과가 크다.

- **point supervision**  
  segmentation mask보다 저렴하면서도 grounding / counting / explanation을 한 번에 밀 수 있다. 실제 annotation budget이 한정된 팀에서 특히 매력적이다.

- **multi-annotation packing**  
  image encoding 비용이 큰 MLLM / VLM에서는 매우 현실적인 최적화다. inference보다 training cost가 문제인 팀이라면 작은 논문 trick이 아니라 바로 ROI가 나오는 설계다.

- **cheap proxy metric 설정**  
  full downstream benchmark를 계속 돌릴 수 없는 환경이라면, Molmo처럼 dense perception을 반영하는 개발 지표를 하나 세우는 게 중요하다.

- **document-like synthetic data generation**  
  PixMo-Docs처럼 image를 직접 읽게 하지 않고 code / structure를 활용해 QA를 생성하는 접근은 chart / table / diagram data 구축에 꽤 재사용 가치가 있다.

## 7-3. Follow-up papers

- **MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning**  
  Molmo가 open VLM data pipeline 쪽에 강하다면, MM1.5는 capability-balanced MLLM recipe와 OCR / grounding / multi-image trade-off 분석 쪽에서 좋은 비교점이다.

- **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution**  
  Molmo가 OCR-centric benchmark에서 자주 비교되는 모델이고, dynamic resolution / high-resolution perception 관점에서 같이 읽을 가치가 있다.

# 8. Summary

- Molmo는 “open VLM도 강할 수 있다”가 아니라, **강한 open VLM을 어떻게 설계할 것인가**를 보여주는 논문이다.
- 핵심은 fancy architecture보다 **PixMo라는 데이터 스위트와 point-based supervision, overlapping crop design**에 있다.
- dense caption, free-form QA, grounding / counting용 points가 서로 연결되면서 user-facing quality까지 끌어올린다.
- strongest model은 open-weight class에서 매우 강하지만, fully-open과 open-weight의 구분은 계속 신경써서 읽어야 한다.
- 실무적으로 가장 남는 건, **좋은 VLM은 결국 supervision interface와 data recipe의 문제**라는 사실이다.
