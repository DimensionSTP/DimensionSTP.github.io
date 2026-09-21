---
layout: single
title: "LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes Review"
categories: Study-concept
tag: [LLaDAImage, ImageGeneration, DiffusionModel]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/pdf/2609.03796)

[Code link](https://github.com/inclusionAI/LLaDA-Image)

[Model collection](https://huggingface.co/collections/inclusionAI/llada-image)

강한 open image generator를 만드는 문제는 architecture 하나로 끝나지 않는다. 어떤 data를 어느 stage에서 넣는지, visual prior와 language alignment를 언제 결합하는지, understanding backbone과 generator를 어떻게 연결하는지, editing reference를 어느 경로로 보존하는지, 마지막으로 수십 step의 diffusion trajectory를 어떻게 few-step model로 줄이는지가 모두 연결되어 있다.

LLaDA-Image는 이 전체 경로를 하나의 training recipe로 정리한다. 6B Diffusion Transformer, 이하 DiT를 처음부터 학습하되, 초기부터 대규모 image-text pair에 의존하지 않는다. 먼저 real image 중심의 image-only pre-training and mid-training으로 visual prior를 만들고, 이후 paired language alignment, high-resolution refinement, joint generation-editing, TwinFlow distillation을 순차적으로 적용한다.

이 논문을 단순히 Qwen-Image-Bench score가 높은 새 image model로만 읽으면 아쉽다. 더 중요한 질문은 아래와 같다.

> Caption supervision이 가장 비싼 초기 학습 구간에서 정말 필요한가. 그리고 understanding model, image generator, editing path, few-step distillation을 하나의 재현 가능한 stack으로 어떻게 연결할 것인가.

> 한 줄 요약: LLaDA-Image는 image-only visual-prior learning, frozen dLLM-based VLM, Residual Query Adapter, single-stream DiT, dual-path editing condition, TwinFlow distillation을 결합해 generation과 editing을 하나의 6B model family로 묶은 open training recipe다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Visual prior learning과 language alignment를 분리해 paired caption data의 역할을 stage별로 다시 본다.
- Frozen understanding backbone을 RQA and Transformer connector로 generator에 연결한다.
- Editing reference를 semantic feature와 clean VAE latent의 두 경로로 넣어 instruction following과 preservation을 분리한다.
- 220M-scale data mixture, progressive resolution schedule, Muon, parameter-free RMSNorm, distillation까지 system-level recipe를 공개한다.
- Base, Turbo, BF16, FP8 checkpoint와 inference code가 공개되어 있어 결과를 직접 점검할 수 있다.

# 1. Problem Setting

## 1-1. Strong image generator의 병목은 한 군데가 아니다

Text-to-image generation은 prompt만 잘 이해하면 끝나는 문제가 아니다. Model은 동시에 다음 성질을 가져야 한다.

- Prompt의 object, relation, count, style, text를 정확히 해석한다.
- Real image distribution의 texture, lighting, material, composition을 학습한다.
- Reference image editing에서 바꿀 부분과 보존할 부분을 구분한다.
- Chinese and English text를 안정적으로 그린다.
- High-resolution output을 만들면서도 training이 collapse하지 않는다.
- Deployment에서는 긴 sampling trajectory를 줄인다.

이 요구사항은 서로 독립적이지 않다. Synthetic caption pair를 많이 쓰면 early alignment는 빨라질 수 있지만, generator가 synthetic artifact를 visual prior로 학습할 수 있다. 반대로 image-only data만 쓰면 appearance prior는 좋아져도 text instruction과의 연결이 약하다. Editing에서는 semantic instruction만 강하게 넣으면 reference의 texture와 identity가 깨질 수 있고, pixel evidence만 강하게 넣으면 원하는 변화가 약해질 수 있다.

LLaDA-Image는 이 문제를 하나의 end-to-end loss로 해결하려 하지 않는다. Capability가 필요한 순서에 따라 training stage와 condition path를 분리한다.

## 1-2. Caption-first training의 비용

많은 image generator는 초기부터 image-text pair로 학습한다. 이 방식은 condition interface가 단순하지만 세 가지 문제가 있다.

### 1) Caption은 비싸고 손실이 있다

Image의 모든 texture, geometry, small object, typography를 caption 한 문장에 담기 어렵다. Caption model이 놓친 정보는 supervision에서 사라진다.

### 2) Low-resolution stage와 caption detail이 어긋날 수 있다

초기 pre-training이 $256^2$ crop에서 진행되면 caption에 언급된 작은 text나 object가 crop or resize 과정에서 보이지 않을 수 있다. Model은 condition과 target이 맞지 않는 sample을 학습하게 된다.

### 3) Synthetic-heavy mixture는 early convergence와 final realism을 바꿀 수 있다

Synthetic data는 caption alignment와 controllability를 빠르게 올릴 수 있지만, artifact와 limited visual distribution도 함께 전달한다. LLaDA-Image는 initial benchmark speed보다 long-run realism ceiling을 우선해 real-data-dominant recipe를 선택한다.

## 1-3. Unified generation and editing의 interface 문제

Generation은 text condition에서 새 image를 만들고, editing은 reference image를 보존하면서 instruction에 맞게 일부를 바꾼다. 두 task를 같은 DiT에 넣으려면 아래 정보가 함께 필요하다.

- 상위 수준의 instruction semantics
- Reference object의 identity
- Spatial layout 정보
- 미세 texture와 background evidence
- Noise가 추가된 target latent

Reference image를 VLM 안에만 넣으면 semantic abstraction은 얻지만 low-level pixel evidence가 약해질 수 있다. 반대로 clean image latent만 넣으면 instruction과 semantic relation을 해석하기 어렵다. LLaDA-Image는 두 종류의 reference condition을 분리한다.

# 2. Core Idea

## 2-1. Image-only visual prior를 먼저 만든다

LLaDA-Image의 가장 큰 recipe choice는 초기 generation training의 대부분을 image-only data로 구성하는 것이다. 220M generation-training samples 중 약 98%가 real image이고, 전체의 90% 이상이 image-only stage에 사용된다.

Image-only stage에서도 generator는 condition 없이 학습되는 것이 아니다. Frozen VLM이 같은 image crop에서 visual token을 읽고 condition representation을 만든다. Generator는 자신이 복원해야 할 region과 맞는 visual condition을 받기 때문에 caption mismatch 없이 semantic signal을 사용할 수 있다.

이 구조를 간단히 쓰면 다음과 같다.

$$
h_{cond}
=
C_{\phi}
\left(
G
\left(
\operatorname{concat}(c, Q_{\psi}(q_0,c))
\right)
\right)
$$

- $c$: text or image-only condition token
- $Q_{\psi}$: Residual Query Adapter, RQA
- $G$: frozen LLaDA 2.0 Mini VLM
- $C_{\phi}$: VLM feature를 DiT condition space로 옮기는 connector

Generator는 이 condition을 사용해 flow field를 예측한다.

$$
\hat{v}_t
=
F_{\theta}(x_t,t,h_{cond})
$$

초기 stage에서 language alignment를 완성하려는 것이 아니라, 먼저 visual distribution 자체를 충분히 학습하고 later SFT에서 text control을 붙이는 순서다.

## 2-2. RQA와 connector가 understanding-to-generation bridge를 만든다

Frozen VLM hidden state는 image generation에 바로 최적화되어 있지 않다. LLaDA-Image는 전체 VLM을 fine-tune하지 않고 두 개의 lightweight bridge를 둔다.

### 1) Residual Query Adapter

Learnable query token이 input condition에 cross-attention한다. 얻어진 residual query를 original token sequence 뒤에 붙여 VLM prefill을 실행한다.

이 query는 VLM의 general understanding을 바꾸기보다, generation에 필요한 feature를 더 잘 노출시키는 역할을 한다.

### 2) Transformer connector

VLM hidden state와 DiT condition space는 dimension and geometry가 다르다. Shallow Transformer connector가 VLM representation을 DiT가 사용하기 좋은 token space로 바꾼다.

RQA는 무엇을 꺼낼지 결정하고, connector는 꺼낸 feature를 어느 공간으로 옮길지 결정한다. 두 module을 분리한 점이 중요하다.

## 2-3. Editing reference는 VLM을 우회한다

Image editing에서 reference image는 VLM input으로 들어가지 않는다. Instruction text는 common VLM path를 사용하지만, reference image는 DiT로 직접 들어간다.

두 reference path가 있다.

1. Semantic path
   - SigLIP-VQ feature를 dedicated reference branch로 projection한다.
   - Object identity, high-level content, semantic correspondence를 제공한다.

2. Pixel path
   - Clean reference image를 FLUX.2 VAE latent로 변환한다.
   - Noise가 추가된 target latent와 concatenate한다.
   - Texture, background, shape, unedited region의 low-level evidence를 보존한다.

개념적으로 editing input은 다음처럼 볼 수 있다.

$$
x_t^{edit}
=
\operatorname{concat}(x_t,\operatorname{VAE}(y_{ref}))
$$

$$
h_{joint}
=
\operatorname{concat}(h_{cond},h_{ref})
$$

$$
\hat{v}_t^{edit}
=
F_{\theta}(x_t^{edit},t,h_{joint})
$$

Editing을 별도 backbone으로 만들지 않고, same DiT 안에서 condition path만 확장한다.

## 2-4. TwinFlow로 deployment model을 분리한다

Base model은 high-fidelity multi-step generation을 담당한다. Turbo model은 TwinFlow distillation으로 2 to 4 sampling steps에 맞춘다.

이 구분은 하나의 checkpoint가 quality and latency를 모두 만족한다고 주장하는 방식보다 현실적이다.

- Base: 50-step 권장 경로로 quality를 우선
- Turbo: 2 to 4-step 경로로 latency를 우선
- FP8 variants: memory와 serving cost를 우선

Model family 안에서 deployment profile을 명시적으로 나눈다.

# 3. Architecture / Method

## 3-1. Overview

| Component | Role |
| --- | --- |
| LLaDA 2.0 Mini VLM | Text, visual token, structured reasoning을 해석하는 frozen understanding backbone |
| RQA | Generation-relevant information을 query token으로 끌어냄 |
| Transformer connector | VLM hidden state를 DiT condition space로 projection |
| 6B single-stream DiT | Condition token과 image latent를 같은 Transformer stack에서 처리 |
| SigLIP-VQ reference branch | Editing reference의 semantic content 제공 |
| Clean VAE latent path | Editing reference의 pixel-level evidence 보존 |
| TwinFlow | Base trajectory를 few-step Turbo model로 distill |

## 3-2. Single-stream DiT

LLaDA-Image DiT는 condition stream과 image stream을 별도 tower로 유지하지 않는다. Condition token, image latent token, timestep representation을 하나의 sequence로 처리한다.

이 구조의 장점은 모든 layer에서 semantic condition과 visual token이 직접 상호작용한다는 점이다. Cross-attention module을 별도로 유지할 필요가 없고, generation and editing에서 같은 Transformer block을 재사용하기 쉽다.

단점도 있다. Sequence가 길어지고 condition type이 늘수록 self-attention cost와 interference가 커질 수 있다. 따라서 condition compression quality와 token budget이 중요하다.

## 3-3. Parameter-free RMSNorm와 Muon

논문은 DiT 전체에 parameter-free RMSNorm을 사용하고, generation training에는 Muon optimizer를 적용한다.

이 조합을 headline novelty로 볼 필요는 없지만, large DiT를 scratch에서 안정적으로 학습하기 위한 recipe element로 중요하다. Image model report에서 architecture diagram만 공개하고 optimizer, normalization, data stage를 생략하는 경우가 많은데, LLaDA-Image는 이 요소를 full stack의 일부로 다룬다.

## 3-4. CoT SFT와 frozen understanding backbone

Generator training 전에 understanding backbone을 CoT SFT로 준비한다. Reported configuration은 약 2.6M packed sequences, maximum length 16,384 tokens, generation-understanding-text mixture 9:9:2를 사용한다.

이 stage의 목적은 VLM을 image decoder로 바꾸는 것이 아니다. Prompt, image token, structured reasoning trace를 잘 해석하는 backbone을 만든 뒤, generation stage에서는 이를 frozen condition provider로 사용한다.

이 설계는 generator training 중 understanding drift를 줄이지만, final generation quality가 frozen backbone의 representation ceiling에 묶일 수 있다.

# 4. Training / Data / Recipe

## 4-1. Data composition

Generation pipeline은 약 220M samples를 처리한다.

| Data property | Reported value |
| --- | ---: |
| Real-image share | 약 98% |
| Image-only share | 90% 초과 |
| Real-image share during paired SFT | 70% 초과 |
| Main stages | Image-only PT, image-only MT, paired alignment, refinement, joint generation-editing |

중요한 점은 single static mixture가 아니라 stage별 role이다.

- Pre-training and mid-training: real image-only data로 visual prior 형성
- Paired alignment: language condition과 generation 연결
- Refinement: typography, portrait, high-value subset 강화
- Joint generation-editing: text-to-image and reference editing 통합

## 4-2. Data filtering and captioning

Paper는 metadata, aesthetics, quality의 세 filter를 사용한다.

- Total pixel count가 $1024^2$보다 큰 image를 유지한다.
- Bytes per pixel threshold로 과도한 compression image를 제거한다.
- ArtiMuse and DeQA score로 aesthetics and quality를 거른다.
- Caption은 large language and vision-language models로 생성한다.
- 별도 consistency check로 hallucinated object, wrong relation, text transcription error, private information, watermark signal을 제거한다.

이 pipeline은 caption generation보다 caption rejection이 중요하다는 점을 보여준다. Synthetic caption을 많이 만드는 것만으로는 충분하지 않고, image-caption agreement를 다시 검사해야 한다.

## 4-3. Progressive resolution schedule

Training progression은 다음과 같다.

1. $256^2$ image-only pre-training
   - Random square crop
   - Visual prior를 low cost로 학습

2. Aspect-ratio-bucketed image-only mid-training
   - 다양한 composition and aspect ratio로 확장

3. Paired alignment at $512^2$ and $1024^2$
   - Language control과 high-resolution detail 연결

4. Targeted refinement
   - Text-rich and portrait data를 강화

5. Joint generation-editing training
   - Generation and editing을 one checkpoint에 통합

6. TwinFlow distillation
   - Base model을 Turbo variants로 변환

이 순서가 논문의 핵심이다. Dataset size보다 언제 어떤 supervision을 넣는지가 더 중요한 claim이다.

## 4-4. Engineering notes

### 1) Image-only condition path를 명시적으로 재현해야 한다

Image-only stage는 unconditional pre-training이 아니다. Same image region에서 visual token condition을 만들고 target crop과 compatibility를 유지한다. 이 detail을 빼면 recipe의 의미가 바뀐다.

### 2) Frozen VLM version을 pin해야 한다

RQA and connector는 특정 VLM hidden geometry에 맞춰 학습된다. VLM checkpoint, tokenizer, image tokenizer, query count, connector depth를 함께 versioning해야 한다.

### 3) Editing path는 semantic and pixel condition을 따로 ablate해야 한다

Semantic reference branch만 제거했을 때 instruction following이 어떻게 바뀌는지, clean VAE latent를 제거했을 때 identity preservation이 어떻게 바뀌는지 분리해야 한다.

### 4) Base and Turbo를 같은 setting으로 비교하면 안 된다

Turbo는 sampling step budget이 다르다. Quality comparison뿐 아니라 latency, VRAM, throughput, prompt complexity별 degradation을 같이 봐야 한다.

### 5) Data openness와 executable release를 구분해야 한다

Paper는 weights, training code, inference code, recipe release를 명시한다. 다만 2026-09-21 기준 official repository의 release checklist는 inference code and weights는 공개, training code는 coming soon으로 표시되어 있다. Report의 open-recipe claim과 현재 다운로드 가능한 artifact 범위를 구분해야 한다.

# 5. Evaluation

## 5-1. Main results

LLaDA-Image는 Qwen-Image-Bench에서 다음 overall score를 보고한다.

| Track | Score |
| --- | ---: |
| English | 53.53 |
| Chinese | 53.38 |

논문은 open-source models 중 두 track 모두 가장 높은 overall score라고 보고한다. 또한 LongText-Bench, CVTG-2K, GEdit-Bench를 통해 long text rendering, bilingual text generation, instruction-guided editing을 평가한다.

Base와 Turbo는 서로 다른 deployment point다.

- LLaDA-Image Base: recommended 50 sampling steps
- LLaDA-Image Turbo: recommended 4 steps, report는 2 to 4 step variants를 설명

Headline score만 보는 것보다 Base and Turbo 사이의 trade-off를 보는 편이 중요하다.

## 5-2. What really matters in the experiments

### 1) Image-only stage가 단순 data scale 효과인지 확인해야 한다

Image-only pre-training, real-image ratio, progressive resolution, Muon, RMSNorm이 동시에 바뀐다. Final score만으로 각 ingredient의 독립 효과를 알기 어렵다. Ablation에서 stage removal과 matched-compute comparison을 봐야 한다.

### 2) Open-source SOTA는 benchmark and protocol에 종속된다

Qwen-Image-Bench overall score는 중요한 evidence지만, 모든 generation axis를 대표하지 않는다. Counting, dense text layout, rare language, exact identity preservation, adversarial editing 같은 failure mode를 별도로 봐야 한다.

### 3) Editing은 prompt compliance와 preservation을 동시에 봐야 한다

Instruction을 잘 따라도 background and identity가 무너지면 좋은 edit가 아니다. 반대로 reference를 그대로 복사하면 instruction-following score가 낮아진다. GEdit-style aggregate score의 submetric을 분리해 읽어야 한다.

### 4) Qualitative grid는 controlled human study가 아니다

Paper의 photorealism showcase는 model capability를 이해하는 데 유용하지만, blind preference study를 대체하지 않는다. 저자도 reader challenge를 controlled perceptual study가 아니라고 명시한다.

# 6. Limitations

1. **Training code release 상태가 report의 표현과 완전히 일치하지 않는다.**
   - Paper는 training code release를 명시하지만, 2026-09-21 official repository에서는 training code가 coming soon이다.
   - Weights and inference path는 확인 가능하지만 full recipe reproduction은 아직 제한될 수 있다.

2. **Ingredient-level causal evidence가 제한적이다.**
   - Image-only ratio, real-data ratio, Muon, RMSNorm, RQA, connector, progressive resolution이 함께 움직인다.
   - 각 choice의 contribution을 matched compute로 분리하기 어렵다.

3. **220M sample pipeline은 moderate라고 해도 소규모 연구팀 기준으로 크다.**
   - Data collection, captioning, filtering, deduplication, high-resolution training cost가 상당하다.
   - Open recipe와 accessible reproduction은 같은 의미가 아니다.

4. **Text rendering and counting의 tail failure는 남는다.**
   - Benchmark average가 높아도 dense typography, exact repeated object count, long structured layout은 별도 stress test가 필요하다.

5. **Unified model이 모든 task에서 specialized model을 이긴다는 증거는 아니다.**
   - Generation and editing을 한 checkpoint에 넣는 operational value는 크지만, task-specific backbone과 동일 budget에서의 비교가 더 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

LLaDA-Image의 가장 재사용 가치가 큰 부분은 model 이름보다 stage ordering이다. Visual prior, language alignment, editing preservation, latency optimization을 한 번에 해결하지 않고, 각 capability가 필요한 시점에 supervision을 추가한다.

VLM or document generation system에서도 비슷한 원리를 적용할 수 있다.

- Appearance or layout prior를 먼저 학습한다.
- Language alignment는 paired data가 충분히 reliable해진 뒤 넣는다.
- Preservation-sensitive task는 semantic and low-level condition을 분리한다.
- Deployment model은 original training objective와 별도로 distill한다.

## 7-2. Reuse potential

### 1) Frozen understanding backbone plus lightweight bridge

Large VLM을 end-to-end fine-tune하지 않고, RQA and connector만 generation-facing interface로 학습하는 방식은 compute-limited setting에서 유용하다.

### 2) Image-only pre-training for document and design generation

Caption이 layout detail을 충분히 담지 못하는 document image, poster, UI generation에서도 image-only prior stage가 도움이 될 수 있다.

### 3) Dual reference path for editing

Semantic feature와 clean latent를 분리하는 구조는 document correction, localized style transfer, identity-preserving editing에 직접 재사용 가능하다.

### 4) Release-state-aware reproduction

Paper recipe를 읽는 것과 code를 바로 실행하는 것은 다르다. Checkpoint, inference pipeline, training stage, data artifact가 실제로 어느 수준까지 공개되었는지 release manifest를 따로 관리해야 한다.

## 7-3. Follow-up papers

- LLaDA 2.0-Uni: dLLM 기반 multimodal understanding and generation backbone
- IOMM: image-only pre-training과 Residual Query Adapter 설계
- TwinFlow: few-step self-adversarial flow distillation 방법
- Qwen-Image and Qwen-Image-Edit: generation과 editing baseline family
- FLUX.2: VAE와 large-scale rectified-flow image generation stack

# 8. Summary

- LLaDA-Image는 6B single-stream DiT와 frozen dLLM-based VLM을 연결한 generation-editing model family다.
- 220M sample pipeline에서 image-only and real-image data를 early stage의 중심에 둔다.
- RQA and Transformer connector가 understanding feature를 generator condition으로 바꾼다.
- Editing reference는 semantic branch와 clean VAE latent path를 통해 DiT로 직접 들어간다.
- TwinFlow distillation으로 Base 50-step model과 Turbo 2 to 4-step deployment model을 분리한다.
- Headline benchmark score보다 visual prior, alignment, preservation, deployment를 나눈 progressive recipe가 핵심 기여다.
