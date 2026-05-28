---
layout: single
title: "LocateAnything Review"
categories: Study-concept
tag: [MultimodalAI, VLM, VisualGrounding]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf)

[arXiv link](https://arxiv.org/abs/2605.27365)

[Project page](https://research.nvidia.com/labs/lpr/locate-anything)

[GitHub link](https://github.com/NVlabs/Eagle/tree/main/Embodied)

[Model link](https://huggingface.co/nvidia/LocateAnything-3B)

LocateAnything은 VLM 기반 visual grounding과 detection을 "coordinate-token generation" 문제가 아니라 box 단위의 structured decoding 문제로 다시 정의하는 논문이다. 기존 generative grounding model은 box를 여러 coordinate token으로 직렬화하고, $x_1$, $y_1$, $x_2$, $y_2$를 순서대로 생성한다. 이 방식은 LM decoder와 잘 맞지만, box라는 geometry object의 내부 결합 구조와는 잘 맞지 않는다.

**이 글의 핵심은 PBD가 단순한 decoding trick이 아니라 output contract 자체를 바꾸는 설계라는 점이다.**

논문은 Parallel Box Decoding, 줄여서 PBD를 제안한다. PBD는 bounding box나 point를 하나의 atomic unit으로 보고, box 내부 coordinate를 한 번의 parallel step에서 예측한다. 그래서 autoregressive coordinate decoding의 latency를 줄이고, 동시에 box 내부 coordinate가 서로 독립적으로 흔들리는 문제도 완화하려고 한다.

> 한 줄 요약: LocateAnything은 bounding box를 여러 token의 나열이 아니라 box-aligned atomic unit으로 디코딩하는 PBD와 12M images, 138M+ queries, 785M boxes 규모의 LocateAnything-Data를 결합해, fast and precise visual grounding을 목표로 한 unified VLM framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- VLM grounding에서 병목이 backbone 능력만이 아니라 coordinate output format에도 있다는 점을 명확하게 보여준다.
- document understanding, GUI grounding, OCR localization, dense detection, referring comprehension을 하나의 generative localization interface로 묶는다.
- Fast Mode, Slow Mode, Hybrid Mode를 통해 serving latency와 localization robustness 사이의 실전 trade-off를 다룬다.

LocateAnything의 핵심 메시지는 단순하다. Visual grounding을 잘하려면 image encoder를 더 키우는 것만으로는 부족하다. 모델이 공간 정보를 어떤 단위로 말하게 할지, 그리고 그 단위를 serving에서 어떻게 병렬화할지까지 같이 설계해야 한다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 VLM이 image와 text query를 받아 object, text region, GUI element, document layout, point 등을 정확하고 빠르게 localization하는 것이다.

구체적으로는 다음 입력과 출력을 다룬다.

- Input image는 native resolution에 가깝게 들어가며, text query는 category list, referring phrase, OCR request, GUI instruction, pointing request 등이 될 수 있다.
- Output은 semantic label과 coordinate structure를 포함하는 text sequence다.
- Box output은 보통 $(x_1, y_1, x_2, y_2)$ 형태의 2D coordinate로 해석된다.
- Point output은 $(x, y)$ 형태로 해석된다.

문제는 이 coordinate를 어떻게 생성하느냐에 있다. 기존 VLM grounding은 box 하나를 여러 coordinate token으로 나누어 autoregressive하게 생성한다. 예를 들어 box 하나가 여러 digit 또는 quantized coordinate token으로 serialize되면, decoder는 앞 token을 만든 뒤 다음 token을 만든다.

**논문이 보는 핵심 병목은 box geometry와 token generation order가 어긋난다는 점이다.**

이 병목은 두 방향에서 나타난다.

- Efficiency bottleneck: box가 많아질수록 coordinate token을 순차 생성해야 해서 latency가 커진다.
- Geometry bottleneck: 같은 box 안의 coordinate가 서로 강하게 결합되어야 하는데, token-level generation은 이 결합을 output layer에서 직접 보장하지 않는다.

## 1-2. Why previous approaches are insufficient

기존 접근은 대략 세 부류로 볼 수 있다.

첫째, detector 기반 grounding이다. Faster R-CNN, DETR, Grounding DINO 계열처럼 detection head나 box head를 갖고 region을 예측한다. 이 방식은 detection에는 강하지만, text instruction, GUI phrase, OCR phrase, document layout query처럼 다양한 natural language localization task를 하나의 conversational interface로 처리하기 어렵다.

둘째, generative VLM 기반 grounding이다. 이 방식은 box를 text-like token sequence로 출력하므로 interface가 단순하다. 하지만 coordinate가 token sequence로 serialize되면서 box 내부 구조가 약해지고, 여러 object를 dense하게 출력할 때 decoding이 느려질 수 있다.

셋째, multi-token prediction 기반 acceleration이다. 여러 token을 한 번에 예측하면 속도는 올라갈 수 있다. 하지만 structure-agnostic하게 chunk를 자르면 box boundary나 semantic boundary와 맞지 않을 수 있다. LocateAnything은 이 지점을 중요하게 본다.

"빠르게 여러 token을 뽑는다" 와 "box라는 단위를 한 번에 뽑는다" 는 같은 말이 아니다.

LocateAnything은 후자를 선택한다. 즉 MTP를 쓰더라도 token chunk가 아니라 box-aligned block을 예측하게 만들어야 한다는 주장이다.

# 2. Core Idea

## 2-1. Main contribution

LocateAnything의 핵심 기여는 세 가지로 정리할 수 있다.

**첫째, Parallel Box Decoding으로 box를 atomic unit으로 예측한다.**

PBD는 bounding box 또는 point를 fixed-length block으로 다룬다. 모델은 box 내부 coordinate를 순차적으로 한 token씩 만드는 대신, box block 전체를 병렬 예측한다. 이렇게 하면 box 내부 coordinate의 geometry consistency를 유지하면서 decoding throughput을 높일 수 있다.

**둘째, Fast, Slow, Hybrid inference mode를 같은 model formulation 안에 둔다.**

Fast Mode는 MTP 기반 parallel decoding을 사용한다. Slow Mode는 NTP 또는 autoregressive decoding을 사용한다. Hybrid Mode는 기본적으로 Fast Mode로 가다가, format irregularity나 spatial ambiguity가 감지되면 해당 block을 버리고 verified prefix로 되돌아가 Slow Mode로 다시 생성한다.

**셋째, LocateAnything-Data를 통해 task diversity를 크게 늘린다.**

논문과 모델 카드 기준으로 LocateAnything-Data는 12M unique images, 138M+ language queries, 785M bounding boxes 규모다. 여기에는 general object detection, GUI grounding, referring comprehension, OCR localization, layout grounding, point-based localization이 포함된다.

이 세 가지를 묶으면 LocateAnything의 방향이 분명해진다. 이 논문은 localization model을 더 큰 VLM으로 만드는 논문이라기보다, VLM의 spatial output contract, training data mixture, inference mode를 같이 설계하는 systems paper에 가깝다.

## 2-2. Design intuition

LocateAnything의 설계 직관은 box의 구조를 모델 output에 직접 반영하자는 것이다.

기존 coordinate-token generation에서는 box 하나가 긴 sequence의 일부가 된다. 이 경우 decoder는 coordinate token을 차례로 생성하므로, box 내부 좌표가 같이 움직여야 한다는 inductive bias가 약하다. 또한 dense detection처럼 box가 많을 때 token length가 빠르게 늘어난다.

PBD에서는 box를 하나의 block으로 둔다. 이 block은 semantic token, box token, negative block, end block 같은 structured output 단위를 포함한다. 특히 box block은 fixed length 구조를 갖고, unused position은 null token으로 padding된다.

**이 논문의 중요한 설계 포인트는 model architecture보다 decoding grammar다.**

VLM은 결국 text를 생성한다. 그러면 visual grounding에서 중요한 질문은 "어떤 text grammar가 geometry를 잘 표현하는가" 이다. LocateAnything은 이 grammar를 natural language sentence가 아니라 box-aligned block sequence로 잡는다.

이 설계는 GUI agent나 robotics perception에서도 의미가 있다. Agent가 화면의 버튼, 문서의 표, 장면의 물체를 다룰 때 필요한 것은 길고 그럴듯한 caption이 아니라 stable coordinate API다. LocateAnything은 VLM output을 API처럼 쓰기 위해 box 단위의 안정성을 높이려는 시도라고 볼 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | 빠르고 정확한 visual grounding and detection |
| Core method | Parallel Box Decoding |
| Base architecture | MoonViT vision encoder, Qwen2.5-3B-Instruct language decoder, MLP projector |
| Output unit | box-aligned fixed-length block |
| Inference modes | Fast Mode, Slow Mode, Hybrid Mode |
| Training data | 12M images, 138M+ queries, 785M boxes |
| Main metric | F1 at IoU thresholds, mean IoU, BPS |
| Default eval mode | Hybrid Mode |

논문 구조를 크게 보면 image encoder가 visual token을 만들고, projector가 language decoder로 연결한다. 이후 decoder는 normal text token만 생성하는 것이 아니라, visual grounding에 맞춘 structured coordinate block을 생성한다.

여기서 핵심은 PBD가 별도 detector head처럼 동작하는 것이 아니라, generative VLM output formulation 안에서 작동한다는 점이다. 즉 모델은 여전히 text sequence를 생성하지만, 그 sequence 안의 box 관련 부분은 block-wise parallel prediction으로 처리된다.

## 3-2. Module breakdown

### 1) Parallel Box Decoding

PBD는 각 bounding box 또는 point를 atomic unit으로 본다. 논문 설명 기준으로 box는 $(x_1, y_1, x_2, y_2)$ coordinate set을 가진다. 이 coordinate set은 하나의 geometric object이므로, token-by-token으로 독립적으로 만들어서는 안 된다는 것이 PBD의 출발점이다.

PBD는 complete coordinate set을 single parallel step에서 예측해 다음 두 가지 효과를 노린다.

- Intra-box geometric coherence를 높인다.
- Dense output에서 autoregressive decoding step 수를 줄인다.

수식으로 단순화하면 기존 방식은 다음처럼 볼 수 있다.

$$
p(box) = p(x_1) p(y_1 | x_1) p(x_2 | x_1, y_1) p(y_2 | x_1, y_1, x_2)
$$

PBD가 의도하는 관점은 다음에 더 가깝다.

$$
p(box) = p(x_1, y_1, x_2, y_2 | context)
$$

실제 구현은 language decoder와 block-wise prediction을 결합하지만, intuition은 이 차이에 있다. Box 내부 coordinate를 serial decision으로 보는지, coupled geometric unit으로 보는지의 차이다.

### 2) Fast Mode

Fast Mode는 MTP 기반이다. 최대 throughput을 목표로 box-aligned block을 병렬로 예측한다. 논문과 모델 카드 설명 기준으로 Fast Mode는 simple scene이나 latency-sensitive setting에 적합하다.

예를 들어 GUI agent가 화면에서 버튼 위치를 빠르게 찾아야 하거나, robotics stack에서 perception result를 빠르게 넘겨야 할 때 Fast Mode는 유리할 수 있다. 다만 dense clutter나 spatial ambiguity가 큰 경우에는 parallel output이 불안정해질 수 있다.

### 3) Slow Mode

Slow Mode는 NTP 또는 autoregressive decoding을 사용한다. 가장 느리지만 가장 robust한 mode로 설명된다. High-precision labeling, offline evaluation, dataset curation처럼 latency보다 안정성이 중요한 경우에 적합하다.

**Slow Mode는 PBD의 반대편 baseline이 아니라 same model 안에 남겨둔 safety path에 가깝다.**

이 점이 중요하다. LocateAnything은 parallel decoding만 강제하지 않는다. 빠른 path가 흔들릴 수 있는 순간을 인정하고, 느리지만 안정적인 path를 fallback으로 남긴다.

### 4) Hybrid Mode and corrected NTP re-decoding

Hybrid Mode는 LocateAnything에서 가장 실용적인 inference mode다. 기본적으로 Fast Mode를 쓰고, 문제가 생기면 Slow Mode로 전환한다. 홈페이지 설명 기준으로 문제 상황은 format irregularity와 spatial ambiguity로 나뉜다.

- Format irregularity: category boundary 주변에서 malformed syntax가 생기는 경우
- Spatial ambiguity: densely arranged objects 사이에서 intermediate coordinate가 나오는 경우

이때 model은 compromised block을 버리고, last verified prefix로 되돌아간다. 이후 NTP가 해당 block을 autoregressive하게 생성하고, box boundary가 다시 안정화되면 MTP로 돌아간다.

이 부분은 논문의 가장 engineering-heavy한 지점이다. PBD 자체는 명확한 아이디어지만, 실제 serving에서 중요한 것은 실패를 어떻게 감지하고 복구하는지다. Hybrid Mode는 "parallel decoding은 항상 맞다" 라고 가정하지 않는다. 대신 빠른 path와 안전한 path를 runtime에서 섞는다.

### 5) Block output contract

모델 카드 기준으로 LocateAnything output은 fixed-length block으로 구성된다. Box block은 length 6 구조를 갖고, coordinate와 structural token을 포함한다. unused position은 null token으로 채운다.

이 구조가 중요한 이유는 output parser가 예측 결과를 안정적으로 읽을 수 있게 만들기 때문이다. VLM이 free-form text만 생성하면 downstream system은 coordinate parsing, syntax recovery, duplicate handling을 별도로 해야 한다. 반대로 block contract가 명확하면 GUI agent, annotation tool, robotics system에서 결과를 API output처럼 다루기 쉽다.

### 6) Backbone and integration

모델 카드 기준 LocateAnything-3B는 Transformer-based VLM이다. 주요 구성은 다음과 같다.

- Vision encoder: MoonViT
- Language model: Qwen2.5-3B-Instruct
- Multimodal projector: MLP projector
- Output formulation: block-based visual grounding

또한 모델 카드는 production image resolution up to 2.5K, prompt length up to 24K tokens, training detection and grounding max sequence length 25,600 tokens, inference max new tokens 8,192를 언급한다.

이 숫자들은 단순 spec이 아니라 task design과 직접 연결된다. Visual grounding은 object가 많아질수록 output sequence가 길어지고, document나 GUI는 high-resolution visual input을 요구한다. 따라서 LocateAnything은 decoding speed만 보는 것이 아니라 high-resolution input, long prompt, dense output을 함께 고려해야 한다.

# 4. Training / Data / Recipe

## 4-1. Data

LocateAnything-Data는 이 논문의 또 다른 축이다. 홈페이지와 모델 카드 기준 규모는 다음과 같다.

| Data item | Value |
| --- | --- |
| Unique images | 12M |
| Language queries | 138M+ |
| Bounding boxes | 785M |
| Modalities | image and text |
| Main domains | OD, GUI, referring, OCR, layout, pointing |

데이터 분포도 중요하다.

| Query type | Share |
| --- | --- |
| General object detection | 66.9% |
| GUI element grounding | 16.5% |
| Referring comprehension | 7.3% |
| Text localization | 3.6% |
| Layout grounding | 3.5% |
| Point-based localization | 2.2% |

이 분포를 보면 general object detection이 가장 크다. 하지만 LocateAnything이 흥미로운 이유는 detection만 키우지 않고 GUI, document, OCR, referring, pointing을 함께 넣었다는 점이다.

**핵심은 visual grounding을 하나의 task가 아니라 spatial instruction following의 집합으로 본다는 점이다.**

모델 카드에는 labeling이 human, synthetic, automated를 섞는다고 되어 있으며, Qwen3-VL, Molmo, SAM 3, Rex-Omni 등을 활용한 model-assisted and synthetic annotation generation, automated post-verification도 언급된다.

## 4-2. Training strategy

모델 카드 기준 training은 four-stage pipeline으로 설명된다. 첫 단계는 captioning, VQA, OCR 등으로 initial multimodal knowledge adaptation을 수행하고, 이후 grounding과 dense-scene localization fine-tuning으로 넘어간다.

논문 관점에서 중요한 것은 NTP와 MTP를 따로 떼어놓지 않는다는 점이다. LocateAnything은 Slow Mode와 Fast Mode를 모두 지원해야 하므로, training에서도 autoregressive path와 parallel block prediction path가 충돌하지 않도록 설계해야 한다.

Ablation에서는 joint dual-formulation training이 Slow Mode upper bound를 50.1에서 52.1 F1로 올리고, Hybrid Mode가 13.2 BPS에서 51.6 F1을 유지한다고 설명된다. 이 결과는 PBD가 단순히 decoding shortcut만이 아니라 training formulation과 같이 맞물려야 한다는 점을 보여준다.

## 4-3. Engineering notes

실제로 이 모델을 쓰려면 논문 방법보다 serving detail이 더 중요할 수 있다.

- 모델 카드는 `generation_mode="hybrid"`를 default로 제안한다.
- `max_new_tokens=8192`를 제안해 dense output truncation을 피하라고 설명한다.
- Runtime은 Transformers 기반이며, TensorRT, TensorRT-LLM, Triton은 아직 지원되지 않는다고 적혀 있다.
- Test hardware는 H100으로 제시된다.
- Model card 기준 LocateAnything-3B는 research and development only이며, NVIDIA License는 non-commercial use를 전제로 한다.

여기서 deployment 관점의 주의점이 나온다. 논문 수치의 BPS는 single H100 batch size 1 조건이다. 따라서 edge device, robotics onboard compute, multi-tenant serving에서는 실제 throughput과 tail latency를 다시 측정해야 한다.

또 하나의 체크 포인트는 모델 크기 표기다. 모델 카드 본문은 3B-parameter research model이라고 설명하지만, Hugging Face UI 하단 metadata에는 4B params로 보이는 부분이 있다. 최종 배포 글에서는 checkpoint config 기준으로 이 부분을 다시 확인하는 편이 안전하다.

# 5. Evaluation

## 5-1. Main results

LocateAnything의 main evaluation은 accuracy와 throughput을 같이 본다. 홈페이지 설명 기준 default는 Hybrid Mode이고, throughput은 BPS, 즉 boxes per second로 측정된다. H100 single GPU 기준 LocateAnything은 12.7 BPS를 보고한다. 이는 textual-based Qwen3-VL의 1.1 BPS보다 10x 이상 빠르고, quantized-based Rex-Omni의 5.0 BPS보다 2.5x 빠른 수치로 제시된다.

주요 결과를 정리하면 다음과 같다.

| Area | Reported result |
| --- | --- |
| Throughput | 12.7 BPS on H100 in Hybrid Mode |
| LVIS and COCO | Rex-Omni same size 대비 mean F1 +3.8 and +1.8 |
| LVIS high IoU | IoU=0.95에서 31.1 vs 20.7 |
| Dense200 and VisDrone | 58.7 and 39.9 mean F1 |
| ScreenSpot-Pro | 60.3 mean F1 |
| DocLayNet and M6Doc | 76.8 and 70.1 mean F1 |
| TotalText OCR | 43.3 mean F1 |
| HumanRef | 78.7 mean F1 |

이 표에서 중요한 것은 LocateAnything이 speed만 주장하지 않는다는 점이다. 논문은 high IoU localization quality도 강조한다. 특히 LVIS IoU=0.95에서의 31.1 vs 20.7 비교는 box boundary precision을 보는 데 중요하다.

**Visual grounding에서는 mAP나 F1 평균보다 high-IoU behavior가 실사용 품질에 더 직접적으로 연결될 때가 많다.**

예를 들어 OCR localization이나 GUI grounding에서는 box가 대충 맞는 것과 정확히 클릭 가능한 region을 잡는 것이 다르다. Layout grounding에서도 table이나 paragraph boundary가 몇 pixel씩 흔들리면 downstream crop, OCR, interaction이 망가질 수 있다.

## 5-2. What really matters in the experiments

### 1) PBD는 speed와 geometry를 같이 노린다

PBD의 장점은 speedup만이 아니다. Box를 block으로 예측하면 coordinate가 하나의 object로 취급된다. 그래서 high-IoU metric에서 차이가 날 수 있다. 논문 결과가 맞다면, PBD는 decoding acceleration과 localization quality를 동시에 건드리는 rare case다.

### 2) Hybrid Mode가 실제 default다

Fast Mode만 보면 throughput이 좋아 보일 수 있지만, 실전에서는 malformed syntax나 ambiguous coordinate가 반드시 나온다. LocateAnything은 이 문제를 감추지 않고 Hybrid Mode로 해결한다. 그래서 이 논문의 serving claim을 읽을 때는 Fast Mode peak speed보다 Hybrid Mode의 average speed와 failure recovery를 봐야 한다.

### 3) 데이터 규모와 task mixture가 method만큼 중요하다

12M images, 138M+ queries, 785M boxes는 단순 부록 수치가 아니다. PBD가 좋은 output grammar라 하더라도, 다양한 domain의 spatial supervision이 없으면 model은 narrow detector에 머물 수 있다. LocateAnything은 PBD와 data engine을 같이 밀어붙인다.

### 4) GUI와 document는 좋은 stress test다

GUI grounding과 document layout은 natural image detection보다 output precision과 text-visual alignment가 까다롭다. ScreenSpot-Pro, DocLayNet, M6Doc, TotalText 결과를 같이 보는 이유가 여기에 있다. 이 task들은 VLM grounding이 실제 agent와 document intelligence pipeline에서 쓸 수 있는지 확인하는 데 더 가깝다.

# 6. Limitations

1. PDF 본문 표와 홈페이지 summary 수치가 완전히 같은지 최종 확인이 필요하다. 이번 초안은 arXiv abstract, project page, model card의 공개 텍스트를 기준으로 정리했다.

2. Hybrid Mode의 tail latency는 별도 확인이 필요하다. Average BPS가 좋아도 fallback이 자주 발생하는 scene에서는 p95 latency가 달라질 수 있다.

3. Model card 기준 language content는 primarily English task-oriented queries로 설명된다. 한국어 query, multilingual OCR, Korean GUI text에서는 별도 evaluation이 필요하다.

4. TensorRT, TensorRT-LLM, Triton은 모델 카드 기준 아직 지원되지 않는다. Production serving에서 NVIDIA stack을 기대하는 경우에는 현재 runtime path를 확인해야 한다.

5. NVIDIA License 기준 non-commercial use가 전제된다. 연구용 검토와 상업 제품 적용은 license path가 다르다.

6. Data mixture가 크고 synthetic annotation을 포함한다. Synthetic box나 model-assisted label이 high-IoU boundary quality에 어떤 bias를 만드는지는 task별로 다시 봐야 한다.

7. LocateAnything-3B라는 이름과 HF UI metadata의 parameter count 표기가 서로 다르게 보이는 부분이 있다. 최종 글 발행 전 checkpoint config를 확인하는 것이 좋다.

# 7. My Take

## 7-1. Why this matters for my work

LocateAnything의 가장 중요한 지점은 "VLM이 공간을 말하는 방식" 을 바꾼다는 점이다. Multimodal model을 agent에 붙일 때 가장 자주 필요한 output은 장문의 설명이 아니라 coordinate다. 화면에서 어떤 버튼을 누를지, 문서에서 어느 영역을 crop할지, robot이 어느 물체를 집을지 결정하려면 coordinate가 안정적이어야 한다.

LocateAnything은 이 문제를 detector head로만 풀지 않는다. Generative VLM의 flexible interface를 유지하면서, coordinate output에는 block structure를 강하게 넣는다. 이 hybrid design은 agentic VLM에서 꽤 현실적인 방향이다.

**VLM grounding의 다음 경쟁 포인트는 model size보다 output grammar와 serving path일 가능성이 크다.**

## 7-2. Reuse potential

이 논문에서 재사용할 만한 아이디어는 다음과 같다.

- Box-aligned block output: coordinate를 free-form text가 아니라 fixed-length structured block으로 다루기
- Dual decoding path: fast parallel path와 safe autoregressive path를 동시에 유지하기
- Runtime correction: format irregularity와 spatial ambiguity를 감지해 verified prefix로 rollback하기
- Data mixture design: OD, GUI, OCR, layout, referring, pointing을 하나의 grounding corpus로 묶기
- Evaluation design: throughput뿐 아니라 high-IoU, GUI, document, OCR stress test를 함께 보기

특히 document AI나 GUI agent 쪽에서는 PBD 자체보다 output schema가 더 유용할 수 있다. 많은 시스템에서 VLM output parsing이 병목이 되는데, LocateAnything처럼 처음부터 block grammar를 만들면 downstream engineering이 단순해질 수 있다.

## 7-3. Follow-up papers

- Pix2Seq: A Language Modeling Framework for Object Detection
- Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection
- ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use
- Detect Anything via Next Point Prediction
- Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models
- Qwen2.5-VL and Qwen3-VL technical reports

# 8. Summary

- LocateAnything은 visual grounding을 coordinate-token generation이 아니라 box-aligned block decoding 문제로 재정의한다.
- PBD는 box를 atomic unit으로 예측해 parallelism과 geometry consistency를 동시에 노린다.
- Hybrid Mode는 Fast Mode를 기본으로 쓰되, 문제가 생기면 Slow Mode로 fallback해 robustness를 확보한다.
- LocateAnything-Data는 12M images, 138M+ queries, 785M boxes 규모의 multi-domain grounding corpus다.
- 실전 관점에서는 BPS average보다 fallback frequency, high-IoU quality, license, runtime support를 같이 봐야 한다.
