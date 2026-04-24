---
layout: single
title: "HunyuanVideo: A Systematic Framework For Large Video Generative Models Review"
categories: Study-concept
tag: [HunyuanVideo, VideoGeneration, DiffusionTransformer]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2412.03603)

HunyuanVideo의 강점은 sample quality 자체도 있지만, **open-source video foundation model을 실제로 어떻게 만들었는가**를 꽤 넓은 범위에서 드러낸다는 데 있다. 데이터 수집과 필터링, structured captioning, 3D VAE, diffusion backbone, text encoder, scaling law, curriculum training, prompt rewrite, inference acceleration, distributed infrastructure까지를 한 문서 안에 묶어서 보여준다.

특히 video generation 쪽은 image generation보다 훨씬 시스템 의존적이다. 단순히 더 좋은 DiT block 하나 넣는다고 끝나지 않고, **긴 시퀀스 비용**, **고해상도 / 긴 길이에서의 수렴 문제**, **데이터 품질 편차**, **text-video alignment**, **실사용 inference latency**가 동시에 얽힌다. HunyuanVideo는 그걸 architecture paper라기보다 **video foundation model 운영 보고서**처럼 정리한다.

또 하나 흥미로운 점은, 이 논문이 "13B model을 만들었다"는 headline보다 **왜 13B를 선택했는가**를 scaling law로 설명하려 한다는 점이다. 즉 무작정 모델을 키운 게 아니라, text-to-image와 text-to-video 각각의 scaling property를 따로 보고 최종 model size와 training configuration을 정했다는 메시지가 강하다.

> 한 줄 요약: HunyuanVideo는 hierarchical data curation, Causal 3D VAE, full-attention dual-stream -> single-stream DiT, decoder-only MLLM text encoder, scaling-law 기반 13B 선택, progressive image-video joint training, prompt rewrite와 acceleration/infrastructure를 한 시스템으로 묶은 **open video foundation model technical report**다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- video generation에서 실제 성능 차이는 이제 backbone 한 줄보다 **data factory + training curriculum + inference system**에서 크게 갈리는 경우가 많다.
- 이 논문은 model architecture만이 아니라 **structured captioning, camera movement annotation, prompt rewriting, step reduction, CFG distillation**까지 포함한 전반적 설계를 드러낸다.
- open-source video model을 서비스나 연구에 재사용하려는 입장에서는, sample demo보다 **어떤 구성 요소가 system-level로 묶여야 하는가**를 보는 쪽이 훨씬 유익하다.

내가 보기엔 HunyuanVideo는 "open-source SOTA claim"보다 **video generation full-stack 설계 문서**로 읽는 편이 더 좋다. 특히 data, text interface, scaling, acceleration을 한 논문 안에서 같이 다룬다는 점이 인상적이다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 **open-source video generation이 closed-source 계열과 비교해 성능, 규모, 재현 가능한 시스템 설계 측면에서 모두 뒤처져 있다**는 점이다.
- video generation은 image generation보다 훨씬 많은 token과 compute를 요구한다. 프레임 수와 해상도가 동시에 늘어나므로, transformer 비용과 VAE 품질 문제가 바로 병목이 된다.
- 여기에 video data는 image data보다 품질 편차가 크고, 장면 전환, motion blur, subtitle, watermark, low-motion clip 같은 노이즈도 훨씬 많다.
- 결국 문제는 "좋은 text-to-video backbone 하나 찾기"가 아니라, **데이터 전처리 -> caption interface -> latent compression -> multimodal conditioning -> curriculum training -> inference acceleration**을 하나의 시스템으로 맞추는 것이다.
- 즉 HunyuanVideo의 문제 설정은 "강한 video model 하나 만들기"가 아니라, **open community에서도 쓸 수 있는 대규모 video foundation model pipeline을 구축하는 것**에 가깝다.

## 1-2. Why previous approaches are insufficient

- 기존 open video generation 계열은 일부 구성 요소에서는 강했지만, 대체로 **closed-source 상위권 모델과의 품질 격차**가 컸다.
- 많은 접근이 architecture 자체에는 집중하지만, 실제 성능을 좌우하는 **data filtering, captioning, resolution-duration curriculum, prompt interface, inference tricks**는 충분히 드러나지 않는 경우가 많다.
- text encoder도 주로 CLIP이나 T5-XXL을 쓰는 흐름이 많았는데, 이 경우 복잡한 서술이나 instruction-like prompt 해석에서 한계가 생길 수 있다.
- 또 video training은 image model을 warm start로 쓰더라도, 그대로 고해상도/장시간 video generation으로 가면 **수렴 난이도와 비용**이 급격히 커진다.
- 결국 기존 접근의 한계는 단일 모듈의 부족이 아니라, **video foundation model을 end-to-end 시스템으로 설계하지 못했다**는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- HunyuanVideo의 핵심 기여는 단일 블록보다 **full-stack system recipe**다.
- 첫째, hierarchical filtering과 structured captioning을 통해 **video data factory**를 만든다.
- 둘째, Causal 3D VAE로 images와 videos를 함께 다루는 compact latent space를 만들고, 여기에 **full-attention DiT backbone**을 얹는다.
- 셋째, text encoder로 CLIP/T5 대신 **decoder-only MLLM + bidirectional token refiner**를 사용해 더 세밀한 semantic guidance를 노린다.
- 넷째, scaling law를 먼저 text-to-image에서 세우고, 그 위에서 text-to-video scaling을 정해 **13B model size**를 선택한다.
- 다섯째, progressive image pretraining, video-image joint training, prompt rewrite, step reduction, CFG distillation, 5D parallel training까지 묶어서 **model보다 system이 중요한 보고서**로 만든다.

## 2-2. Design intuition

- 이 논문의 설계 직관은 꽤 분명하다. **좋은 video model은 좋은 diffusion block 하나로 완성되지 않는다.**
- 먼저 data quality가 낮으면 model size를 늘려도 효과가 제한적이므로, filtering과 recaptioning이 먼저 필요하다.
- text conditioning도 단순 sentence embedding보다, **instruction-following이 가능한 richer text representation**이 유리하다고 본다.
- image/video joint training은 단순 데이터 보충이 아니라, video에서 부족한 세계 지식과 정적 장면 표현을 image에서 보완하고, 동시에 video training이 image semantics를 잊어버리는 걸 막는 장치다.
- inference 단계에서도 quality만 볼 게 아니라, **낮은 step 수에서 얼마나 무너지지 않는가**, **CFG 비용을 얼마나 줄일 수 있는가**가 중요하다.
- 그래서 HunyuanVideo를 architecture novelty paper로 보기보다, **video foundation model을 어디서부터 어디까지 최적화해야 하는지 정리한 설계 문서**로 읽는 편이 맞다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | open-source video generation과 closed-source 계열 사이의 격차를 줄일 수 있는 대규모 foundation video model을 구축하는 것 |
| Key module | hierarchical data factory + Causal 3D VAE + full-attention DiT + decoder-only MLLM text encoder + scaling-law 기반 training recipe |
| Core design principle | architecture만이 아니라 data, text interface, curriculum, acceleration, infrastructure를 하나의 시스템으로 설계 |
| Difference from prior work | video backbone 단일 개선보다, **full-stack recipe와 운영 체계**를 통합적으로 제시 |

## 3-2. Module breakdown

### 1) Causal 3D VAE

- HunyuanVideo는 pixel space에서 바로 diffusion하지 않고, **Causal 3D VAE**로 spatial-temporal latent space를 만든다.
- repo 설명 기준으로 video length / spatial / channel 압축 비율을 각각 **4 / 8 / 16**으로 둔다. 핵심은 긴 video를 transformer가 직접 보지 않도록 token budget을 줄이는 것이다.
- 중요한 건 이 VAE가 단순한 보조 모듈이 아니라는 점이다. 저자들은 pre-trained image VAE 초기화에 의존하지 않고, **video와 image reconstruction을 함께 고려해 scratch부터 학습**한다.
- loss도 reconstruction loss와 KL loss만 쓰지 않고 perceptual loss와 GAN adversarial loss를 넣어, text, small face, texture 같은 디테일을 살리려 한다.
- 또 inference 시 OOM 문제를 피하기 위해 **spatial-temporal tiling**을 도입하고, train/inference mismatch로 생기는 artifact를 줄이기 위해 tiling을 랜덤하게 켜고 끄는 추가 finetuning을 수행한다.
- 내가 보기엔 이 부분이 꽤 중요하다. 즉 "좋은 video DiT"보다 먼저 **쓸 만한 video latent interface**를 만드는 데 공을 들였다.

### 2) Unified Full-Attention Diffusion Backbone

- backbone은 unified Full Attention을 쓰는 DiT 계열이다.
- 저자들은 full attention을 택한 이유로 세 가지를 든다. **분리된 spatiotemporal attention보다 성능이 좋고**, **image/video unified generation을 지원하며**, **LLM 쪽 가속 기술을 더 잘 활용할 수 있다**는 것이다.
- 구조적으로는 **dual-stream -> single-stream** 하이브리드 설계를 쓴다.
- dual-stream 단계에서는 video token과 text token을 독립적으로 처리해 modality별 modulation을 안정화하고,
- single-stream 단계에서는 둘을 concat해 본격적으로 multimodal fusion을 수행한다.
- paper의 13B foundation model hyperparameter table 기준으로, **dual-stream blocks 20개 / single-stream blocks 40개 / model dim 3072 / FFN dim 12288 / 24 heads / head dim 128**을 사용한다.
- positional encoding은 3D RoPE를 써서 time / height / width 좌표를 분리해 처리한다. 이건 multi-resolution, multi-aspect ratio, varying-duration generation을 지원하기 위한 선택이다.
- 즉 backbone 쪽 메시지는 "새 attention 하나 제안"이 아니라, **video용 full-attention DiT를 안정적으로 크게 키우는 구조적 정리**에 가깝다.

### 3) MLLM Text Encoder + Bidirectional Token Refiner + CLIP Global Summary

- text encoder는 이 논문의 인상적인 포인트 중 하나다.
- 기존 text-to-video 모델들이 많이 쓰던 CLIP 또는 T5-XXL 대신, HunyuanVideo는 **decoder-only 구조의 MLLM**을 text encoder로 사용한다.
- 논문의 논리는 명확하다.
  - T5보다 image-text alignment가 더 낫고,
  - CLIP보다 image detail description과 complex reasoning이 강하며,
  - system instruction을 prepend해 zero-shot style control도 할 수 있다는 것이다.
- 다만 causal attention 기반 MLLM만으로는 diffusion conditioning에 필요한 bidirectional text context가 부족할 수 있으므로, **extra bidirectional token refiner**를 붙여 text feature를 보강한다.
- 여기에 CLIP-Large의 마지막 non-padded token을 global summary로 써서 timestep embedding 쪽에 더한다.
- 내가 보기엔 이 조합은 단순히 "LLM을 text encoder로 썼다"보다, **long and stylistically messy한 human prompt를 diffusion-friendly한 semantic signal로 다시 정리하는 interface**에 가깝다.

### 4) Scaling-law-driven Model Sizing

- HunyuanVideo는 model size를 무작정 늘리지 않는다.
- 먼저 **DiT-T2X(I)** family를 92M~6.6B 규모에서 실험해 text-to-image scaling law를 얻고,
- 그다음 이 결과를 이용해 **text-to-video scaling law**를 구축한다.
- 그리고 training consumption과 inference cost를 함께 고려한 뒤, 최종 foundation model 크기를 **13B**로 정한다.
- 이 선택은 매우 실무적이다. video model은 단순히 학습만 되는 게 아니라, inference cost까지 감당 가능한 범위여야 하기 때문이다.
- 그래서 이 논문의 scaling section은 단순 장식이 아니라, **왜 13B가 되었는가**를 설명하는 핵심 축이다.

### 5) Prompt Rewrite as a System Interface

- prompt rewrite는 paper 전체에서 상대적으로 덜 주목받지만, 서비스 관점에서는 꽤 실용적인 포인트다.
- HunyuanVideo는 **Hunyuan-Large**를 이용해 user prompt를 model-preferred prompt로 바꾼다.
- 기능은 크게 세 가지다.
  - multilingual input adaptation
  - standardized prompt structure로의 정규화
  - 복잡한 표현의 단순화
- 여기에 self-revision도 넣어 원문과 rewrite 결과를 비교하며 다듬고, 실사용을 위해 LoRA 버전까지 따로 만든다.
- 내 해석은 이렇다. video generation에서 prompt rewrite는 부가기능이 아니라, **training caption world와 real user prompt world 사이를 연결하는 번역기**다.

# 4. Training / Data / Recipe

## 4-1. Data

- 이 논문은 data를 단순 원천 수집이 아니라 **계층적 정제 파이프라인**으로 다룬다.
- video data 쪽에서는 PySceneDetect로 raw video를 single-shot clip으로 자르고, Laplacian 기반 clear frame을 찾고, internal VideoCLIP embedding으로 **deduplication + 10K concept centroid 기반 resampling/balancing**을 수행한다.
- 이후 aesthetics, clarity, motion, scene boundary, OCR, watermark/border/logo 제거까지 여러 필터를 붙인다.
- 특히 OCR model로 **과도한 텍스트가 들어간 clip을 제거하고 subtitle 위치를 찾아 crop**하는 부분이 흥미롭다. document-like noise를 줄이는 것이 text-to-video 학습에도 꽤 중요하다는 뜻이다.
- 이 필터들은 progressively stricter threshold를 적용해 **256p / 360p / 540p / 720p** training dataset을 만들고, 마지막 fine-tuning 단계에서는 **1M human-annotated sample**을 따로 구축한다.
- 이 1M dataset은 색 조화, 조명, object emphasis, spatial layout 같은 aesthetic decomposition과 motion speed, action integrity, motion blur 같은 motion decomposition을 기준으로 수작업 선별된다.

- annotation 쪽도 꽤 공들였다.
- 저자들은 in-house VLM으로 JSON-formatted structured captions를 만든다.
- 이 caption은 short description, dense description, background, style, shot type, lighting, atmosphere 등을 포함하며, dropout / permutation / combination을 통해 **길이와 패턴이 다양한 caption**을 합성한다.
- 또 14종 camera movement classifier를 따로 학습해, zoom / pan / tilt / static / handheld 같은 카메라 동작을 structured caption에 넣는다.
- 결국 data 부분의 핵심은 "많은 video를 모았다"가 아니라, **모델이 배우기 좋은 형태로 장면, 스타일, 조명, 카메라, motion을 구조화했다**는 점이다.

- image data도 video 보조재 수준이 아니다.
- motion-related filter를 제외한 유사한 pipeline으로 image pool을 정제하고,
- 첫 번째 image dataset은 billions of samples 규모,
- 두 번째 image dataset은 hundreds of millions 규모로 구성한다.
- 이건 뒤의 image pretraining stage와 직접 연결된다.

## 4-2. Training strategy

- foundation model 학습 objective는 **Flow Matching**이다.
- inference에서는 Euler ODE solver를 사용한다.
- 학습 흐름은 크게 네 부분으로 읽을 수 있다.

### 1) 3D VAE from scratch

- VAE는 pre-trained image VAE 초기화 없이 scratch부터 학습된다.
- video와 image reconstruction quality를 균형 있게 맞추기 위해 두 데이터를 섞고,
- low-resolution short video에서 high-resolution long video로 가는 curriculum을 사용한다.
- high-motion video reconstruction을 위해 sampling interval을 랜덤하게 뽑아 frame을 고르게 선택하는 전략도 쓴다.

### 2) Two-stage image pretraining

- 저자들은 **잘 학습된 image model이 video training convergence를 크게 가속한다**고 본다.
- 그래서 먼저 text-to-image pretraining을 두 단계로 수행한다.
- stage 1은 **256px multi-aspect image pretraining**이다.
- stage 2는 **256px + 512px mix-scale training**이다. 여기서 중요한 건 512px 단독 fine-tuning을 하면 256px 능력이 망가져 이후 video pretraining에 악영향을 줄 수 있다는 관찰이다.
- 이를 막기 위해 두 scale을 하나의 global batch 안에 같이 넣고, micro batch size를 scale별로 다르게 두어 GPU utilization을 최대화한다.

### 3) Progressive video-image joint training

- video-image joint training은 이 논문의 backbone만큼 중요한 축이다.
- 비디오는 duration bucket과 aspect ratio bucket으로 나누고, bucket별 token 수에 맞춰 max batch size를 조절한다.
- 각 rank가 서로 다른 bucket을 랜덤하게 pre-fetch하도록 해서, 단일 size에 과적합되지 않게 한다.
- curriculum은 세 단계로 요약된다.
  - low-resolution, short video stage
  - low-resolution, long video stage
  - high-resolution, long video stage
- 각 stage에서 image를 varying proportion으로 계속 섞는다. 저자들의 설명대로, 이건 단순 데이터 보강이 아니라 **video data scarcity 보완 + broader world knowledge 학습 + image semantics forgetting 방지** 역할을 한다.

### 4) High-performance fine-tuning

- pretraining dataset은 크지만 quality variance가 크기 때문에, 마지막에는 **네 개의 특화 subset**을 골라 추가 fine-tuning을 수행한다.
- paper 본문은 이 subset의 정체를 아주 세밀하게 풀어주진 않지만, 목적은 분명하다. **high-quality, dynamic video**, **continuous motion control**, **character animation** 같은 practical capability를 밀어주는 것이다.
- 이 단계는 자동 필터링 뒤 manual review가 붙는다.

## 4-3. Engineering notes

- 이 논문에서 실무적으로 바로 참고할 만한 포인트는 architecture보다 engineering notes 쪽에도 많다.

### 1) Time-step shifting for low-step inference

- inference step을 줄이면 video quality가 쉽게 무너지는데, HunyuanVideo는 **time-step shifting**으로 이를 보완한다.
- 논문은 50-step일 때 shifting factor를 **7**, 20-step 미만일 때는 **17**로 키운다고 적는다.
- 즉 step 수가 줄수록 더 early timestep 쪽을 강조해 low-step quality를 끌어올린다.
- 이건 "step을 줄였다"보다 **낮은 step 분포에 맞춰 scheduler를 다시 설계했다**는 데 의미가 있다.

### 2) Text-guidance distillation

- CFG는 quality에는 좋지만 conditional + unconditional forward를 모두 해야 해서 비용이 크다.
- HunyuanVideo는 conditional/unconditional 결합 출력을 student에 distill하는 방식으로 **text-guidance distillation**을 수행한다.
- 논문은 이 방식이 **약 1.9x acceleration**을 가져온다고 보고한다.
- 중요한 건 quality를 유지하면서 inference overhead를 줄이는 practical trade-off라는 점이다.

### 3) Efficient and scalable training infrastructure

- training infrastructure도 paper의 주요 기여다.
- AngelPTM 위에서 학습하고, Tencent XingMai network를 통해 inter-server communication을 최적화한다.
- parallelism은 **TP + SP + CP + DP + ZeroCache**까지 포함하는 5D 전략이다.
- CP에는 Ring Attention을 활용해 long sequence training을 처리하고,
- FusedAttention, recomputation, activation offload, automatic fault tolerance까지 붙인다.
- 특히 자동 fault tolerance로 training stability **99.5%**를 보고하는데, 이건 model paper라기보다 infra report의 톤에 가깝다.

# 5. Evaluation

## 5-1. Main results

- 이 논문의 실험은 크게 두 축으로 읽는 게 좋다.
  1. **VAE reconstruction quality**
  2. **foundation model의 human evaluation**

먼저 VAE reconstruction 비교부터 보면, Table 1은 HunyuanVideo의 3D VAE가 open-source baselines 대비 꽤 강하다는 걸 보여준다.

| Model | Downsample Factor | ImageNet 256x256 PSNR | MCL-JCV PSNR |
| --- | --- | --- | --- |
| FLUX-VAE | 16 | 32.70 | - |
| OpenSora-1.2 | 4 | 28.11 | 30.15 |
| CogVideoX-1.5 | 16 | 31.73 | 33.22 |
| Cosmos-VAE | 16 | 30.07 | 32.76 |
| HunyuanVideo VAE | 16 | 33.14 | 35.39 |

- 특히 paper는 text, small faces, complex textures 쪽에서 자사 VAE가 유리하다고 해석한다.
- 이건 foundation model의 final sample quality 이전에, **latent interface 자체가 강하다**는 주장에 해당한다.

foundation model 비교는 human evaluation 중심이다.

- baseline은 **폐쇄형 video generation 모델 5개**다.
- 총 **1,533 prompts**를 사용했고,
- HunyuanVideo는 각 prompt당 한 번만 생성하며, cherry-picking을 피하려고 했다고 밝힌다.
- 평가자는 **60 professional evaluators**이고,
- 기준은 text alignment, motion quality, visual quality, overall ranking이다.

Table 3의 핵심 결과를 요약하면 아래와 같다.

| Model | Duration | Text Alignment | Motion Quality | Visual Quality | Overall Ranking |
| --- | --- | --- | --- | --- | --- |
| HunyuanVideo | 5s | 61.8% | 66.5% | 95.7% | 41.3% |
| CNTopA | 5s | 62.6% | 61.7% | 95.6% | 37.7% |
| CNTopB | 5s | 60.1% | 62.9% | 97.7% | 37.5% |
| Gen-3 alpha | 6s | 47.7% | 54.7% | 97.5% | 27.4% |
| Luma 1.6 | 5s | 57.6% | 44.2% | 94.1% | 24.8% |
| CNTopC | 5s | 48.4% | 47.2% | 96.3% | 24.6% |

- 여기서 중요한 건 HunyuanVideo가 **모든 축에서 무조건 최고는 아니라는 점**이다.
- text alignment는 CNTopA가 약간 높고,
- visual quality는 CNTopB와 Gen-3 alpha가 더 높다.
- 그런데 HunyuanVideo는 **motion quality에서 가장 강하고**, 종합 ranking에서 1위를 차지한다.
- 즉 이 논문의 메시지는 "모든 축을 압도했다"가 아니라, **균형 잡힌 overall performance, 특히 motion dynamics에서 강했다**는 쪽이다.

- qualitative 쪽에서도 text alignment, high-quality detail, high-motion dynamics, concept generalization, automatic scene cut, character understanding and writing 같은 사례를 제시한다.
- 나는 이 qualitative set이 단순 demo보다, **structured captioning과 prompt rewrite가 어떤 타입의 control을 겨냥하는지** 보여준다고 느꼈다.

## 5-2. What really matters in the experiments

- 내가 보기엔 이 논문에서 진짜 중요한 건 "41.3% overall" 같은 절대 숫자보다, **어떤 설계가 어떤 능력과 연결되는가**다.
- VAE reconstruction quality가 높다는 건 단순 PSNR bragging이 아니라, 뒤의 text rendering, facial detail, texture fidelity와 연결된다.
- MLLM text encoder + bidirectional refiner + structured captioning은 complex prompt following과 camera/style control 쪽으로 이어진다.
- progressive video-image curriculum은 long-duration / high-resolution 수렴성과 world knowledge 보존 문제를 동시에 다룬다.
- time-step shifting과 text-guidance distillation은 high-quality version만이 아니라 **실제 낮은 step inference**까지 시야에 넣고 있다는 신호다.

- 또 한 가지 포인트는 evaluation이 motion에 상당히 민감하다는 점이다.
- HunyuanVideo는 visual quality만 놓고 보면 일부 baseline이 더 높지만, overall rank와 motion quality에서 강하다.
- 이건 video generation에서 결국 중요한 건 single frame sharpness보다 **움직임의 설득력과 시간적 coherence**라는 사실을 다시 보여준다.

# 6. Limitations

1. **완전한 recipe reproducibility는 아니다.**
   - code와 weights는 공개됐지만, 데이터는 licensed material과 internal filtering/captioning stack에 크게 의존한다.
   - internal VideoCLIP, OCR, VLM captioner, Hunyuan-Large rewrite model, 여러 MLLM 설정, Tencent internal infra가 섞여 있어 외부에서 동일 recipe를 재현하긴 쉽지 않다.

2. **평가가 human survey 중심이고 비교 조건이 완전히 통제되었다고 보긴 어렵다.**
   - 1,533 prompt와 60 professional evaluator는 인상적이지만, baseline이 closed API/Web models라 설정 parity를 완전히 맞추기 어렵다.
   - default settings를 유지했다는 점은 공정성 확보 시도이지만, 동시에 각 모델의 최적 operating point와는 어긋날 수 있다.

3. **보고서의 초점이 넓은 대신 숫자 레벨 recipe transparency는 일부 부족하다.**
   - scaling law와 stage 구분은 잘 설명하지만, 전체 dataset 규모와 stage별 compute, fine-tuning subset 구성 같은 부분은 모두 세밀하게 공개되진 않는다.
   - 그래서 이 논문은 "어떤 철학으로 만들었는가"는 잘 드러나지만, **그대로 다시 만드는 매뉴얼**은 아니다.

4. **public release와 evaluation 버전을 구분해서 읽을 필요가 있다.**
   - GitHub README는 main comparison이 high-quality version 기준이며, 현재 공개된 fast version과는 다르다고 적고 있다.
   - 따라서 evaluation headline을 곧바로 "지금 공개 체크포인트가 같은 품질을 같은 비용으로 낸다"로 읽으면 과장이 될 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

- 내 관점에서 HunyuanVideo의 가장 큰 가치는 **video foundation model을 system engineering 문제로 재정의했다**는 데 있다.
- 보통 video generation 논문을 보면 attention variant, sampler, latent compressor 중 하나에 초점이 가는 경우가 많은데, 이 논문은 오히려 "좋은 open video model을 만들려면 어떤 부품들이 같이 굴러야 하는가"를 보여준다.
- 특히 structured captioning, camera movement annotation, prompt rewrite, video-image joint training은 실제 서비스에서 **user prompt -> internal prompt -> latent generation** 흐름을 설계할 때 바로 참고할 수 있다.
- 또 scaling law를 이용해 13B를 선택한 부분은, 연구에서도 무작정 "더 큰 모델"로 가지 않고 **어디까지가 계산 대비 이득인지**를 따져야 한다는 점을 잘 보여준다.

## 7-2. Reuse potential

- 재사용 가치가 높은 요소를 꼽으면 아래가 먼저 보인다.
  - **structured captioning + camera tag**: video generation뿐 아니라 video understanding / retrieval용 annotation interface로도 응용 가치가 높다.
  - **decoder-only MLLM text encoder + bidirectional refiner**: generative model conditioning에 LLM/MLLM을 붙일 때 꽤 좋은 설계 패턴이다.
  - **dual-stream -> single-stream hybrid fusion**: image/video뿐 아니라 다른 multimodal diffusion 계열에도 이식 가능하다.
  - **progressive image-video curriculum**: data scarcity가 있는 modality에서 image/video 공동 학습을 설계할 때 유용하다.
  - **time-step shifting + CFG distillation**: 품질을 크게 해치지 않으면서 inference를 줄이는 practical trick으로 재사용성이 높다.
- 반대로 바로 재현하기 어려운 부분은 internal data stack과 infra다. 따라서 이 논문은 "전체를 복제"하기보다, **각 레버를 분리해 차용**하는 편이 맞다.

## 7-3. Follow-up papers

- Wan: Open and Advanced Large-Scale Video Generative Models
- LTX-Video: Realtime Video Latent Diffusion

# 8. Summary

- HunyuanVideo는 video generation 모델 하나 소개라기보다 **open video foundation model full-stack report**에 가깝다.
- 핵심 축은 hierarchical data curation, structured captioning, Causal 3D VAE, full-attention DiT, MLLM text encoder, scaling-law 기반 13B 선택이다.
- training recipe는 image warmup -> progressive video-image joint training -> high-performance fine-tuning으로 정리할 수 있다.
- evaluation은 모든 축을 압도했다기보다, **motion quality와 overall balance**에서 강점을 보인다는 점이 중요하다.
- 논문의 진짜 메시지는 "좋은 video model은 architecture 하나가 아니라 data, text interface, curriculum, acceleration, infrastructure의 합"이라는 데 있다.
