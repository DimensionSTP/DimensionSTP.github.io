---
layout: single
title: "DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation Review"
categories: Study-concept
tag: [DomainShuttle, S2V, TextToVideo, VideoGeneration, Personalization]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[논문 링크](https://arxiv.org/abs/2606.26058)

[프로젝트 페이지](https://cn-makers.github.io/DomainShuttle/)

[코드 링크](https://github.com/HKUST-C4G/DomainShuttle)

DomainShuttle은 subject-driven text-to-video, 이하 S2V, 를 단순히 "reference image와 비슷한 subject를 video 안에 유지하는 문제"로 보지 않는다. 이 논문이 겨냥하는 핵심은 **subject fidelity와 generation flexibility를 동시에 만족해야 하는 open-domain personalization**이다.

기존 S2V 방법은 대체로 in-domain fidelity에 강하게 최적화되어 있다. 예를 들어 사람 reference를 주면 같은 사람처럼 보이게 만들고, object reference를 주면 같은 object처럼 보이게 만드는 식이다. 이 자체는 중요하다. 하지만 실제 사용에서는 더 복잡한 요청이 많다. Real-world person을 fantasy character처럼 만들거나, fantasy subject를 real-world figurine처럼 만들거나, real-world subject and fantasy subject가 같은 scene에서 상호작용하게 만들고 싶을 수 있다. 이 경우 reference image의 모든 feature를 그대로 붙잡으면 cross-domain editability가 오히려 약해진다.

DomainShuttle은 이 tension을 feature decomposition 문제로 본다. Reference image에서 보존해야 하는 것은 intrinsic subject features다. Shape, identity, clothing, texture, characteristic appearance는 유지되어야 한다. 반대로 lighting, style, domain attributes, background composition 같은 subject-irrelevant features는 prompt에 맞게 바뀌어야 한다. 즉 reference feature를 그대로 복사하는 것이 아니라, subject의 핵심만 다른 domain으로 shuttle해야 한다.

> 한 줄 요약: DomainShuttle은 open-domain S2V에서 in-domain fidelity and cross-domain flexibility를 동시에 만족하기 위해 Domain-MoT, Video-Reference DualRoPE, Cross-Pair Consistent Loss를 결합한 Wan 기반 subject-driven video personalization framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- S2V personalization을 in-domain identity preservation에서 open-domain subject transfer로 확장한다.
- Reference feature injection이 base video model의 text controllability and domain flexibility를 망가뜨리는 문제를 정면으로 다룬다.
- Domain-MoT로 video branch and reference branch를 분리하고, domain-aware AdaLN으로 reference domain attribute를 다룬다.
- Video-Reference DualRoPE로 video token and reference token을 다른 RoPE space에 두어 subject-level spatial relation을 제어한다.
- Cross-Pair Consistent Loss로 redundant reference attribute가 아니라 intrinsic subject feature를 추출하려 한다.
- Wan2.1-14B and Wan2.2-14B 위에서 구현해 modern DiT video foundation model에 붙는 personalization recipe를 제공한다.

# 1. Problem Setting

## 1-1. Problem definition

Subject-driven text-to-video generation은 reference images $R$ and text prompt $p$가 주어졌을 때, subject identity를 보존하면서 video $v$를 생성하는 task다.

$$
v \sim G_{\theta}(R,p)
$$

Open-domain S2V에서는 두 시나리오를 동시에 만족해야 한다.

| 시나리오 | 요구사항 |
| --- | --- |
| In-domain | Reference subject의 appearance를 최대한 보존해야 한다. |
| Cross-domain | Intrinsic subject feature는 보존하되, domain attribute는 prompt에 따라 바뀌어야 한다. |

In-domain만 보면 reference fidelity가 가장 중요하다. 하지만 cross-domain에서는 fidelity의 의미가 바뀐다. 모든 pixel-level attribute를 유지하는 것이 아니라, subject의 invariant feature만 유지해야 한다.

이를 reference feature의 분해로 쓰면 다음과 같다.

$$
z_R = (z_{\mathrm{intrinsic}}, z_{\mathrm{domain}})
$$

Open-domain S2V가 원하는 것은 다음이다.

$$
z_{\mathrm{intrinsic}} \rightarrow \mathrm{preserve}
$$

$$
z_{\mathrm{domain}} \rightarrow \mathrm{adapt\ according\ to\ prompt}
$$

기존 reference injection이 이 둘을 섞어 넣으면, subject consistency는 올라갈 수 있지만 prompt controllability and cross-domain flexibility는 떨어질 수 있다.

## 1-2. Why previous approaches are insufficient

기존 접근이 부족한 이유는 네 가지로 정리할 수 있다.

1. **Reference feature copy-paste 문제**
   - I2V prior나 reference injection을 강하게 쓰면 identity preservation은 좋아진다.
   - 하지만 reference image의 lighting, style, pose, background, domain attribute까지 같이 따라오면서 prompt가 요구하는 transformation을 막을 수 있다.

2. **Shared attention path 문제**
   - Video tokens and reference tokens를 같은 attention path에서 섞으면 base video model의 prior와 reference feature가 entangled된다.
   - Reference branch가 너무 강해지면 video branch가 갖고 있던 open-domain text controllability가 약해질 수 있다.

3. **Reference image를 video frame처럼 다루는 문제**
   - Reference image는 generated video의 시간축 frame이 아니다.
   - 같은 subject를 설명하는 multiple reference일 수도 있고, 서로 다른 subject reference일 수도 있다.
   - 단순 temporal RoPE만으로는 이런 subject-level relation을 표현하기 어렵다.

4. **Single-reference supervision 문제**
   - Single reference set만으로 학습하면 viewpoint, occlusion, lighting, crop, composition 같은 redundant feature에 과적합할 수 있다.
   - Cross-domain에서는 이런 redundant feature보다 invariant subject feature가 더 중요하다.

# 2. Core Idea

## 2-1. Main contribution

DomainShuttle의 contribution은 세 가지다.

| 구성요소 | 역할 |
| --- | --- |
| Domain-MoT | Video latents and reference image features를 independent processing path로 분리한다. |
| Video-Reference DualRoPE | Video token and reference token을 서로 다른 RoPE space에 배치한다. |
| Cross-Pair Consistent Loss | 서로 다른 reference set에서 공유되는 intrinsic subject feature를 학습하게 한다. |

이 세 가지는 모두 같은 목표를 향한다. Reference에서 subject identity는 보존하되, reference image가 가진 domain-specific residue는 prompt에 맞게 release하는 것이다.

## 2-2. Design intuition

DomainShuttle의 design intuition은 다음 문장으로 요약할 수 있다.

```text
Preserve what defines the subject.
Release what defines the domain.
```

이를 architecture choice로 옮기면 다음과 같다.

| 문제 | 설계 선택 |
| --- | --- |
| Reference feature가 video generation prior와 entangle됨 | Video and reference feature path를 분리 |
| Reference domain attribute가 그대로 복사됨 | Domain-aware AdaLN 도입 |
| Multiple reference의 subject relation이 모호함 | Video-Reference DualRoPE 적용 |
| Single reference가 redundant details에 과적합함 | Cross-Pair Consistent Loss 사용 |
| Text controllability가 손상됨 | Textual cross-attention freezing |

이 논문의 중요한 점은 cross-domain flexibility를 단순 prompt-following loss로 해결하려 하지 않는다는 것이다. Feature path, position encoding, reference pair consistency, text attention preservation을 함께 설계한다.

# 3. Architecture / Method

## 3-1. Overview

| 항목 | 설명 |
| --- | --- |
| 목표 | Open-domain subject-driven text-to-video generation |
| Base model | Wan2.1-14B-T2V and Wan2.2-14B-T2V |
| 핵심 모듈 | Domain-MoT, VR-DualRoPE, CCL |
| 핵심 trade-off | In-domain fidelity vs cross-domain flexibility |
| Reference modeling | Separate reference branch with domain-aware AdaLN |
| Position encoding | Video token and reference token in separate RoPE spaces |
| Training data | 200K image personalization and 750K video personalization |
| 주요 결과 | Prior SOTA 대비 CD-Score 18.7% improvement 보고 |

## 3-2. Module breakdown

### 1) Domain-MoT

Domain-MoT는 Mixture-of-Transformers 구조로, video latents and reference image features를 independent processing path로 나눈다. Self-attention 안에서 video branch and reference branch는 separate QKV projection and separate RoPE를 사용한다.

단순화하면 다음과 같은 구조다.

$$
h_v' = \mathrm{Attn}_v(h_v,h_R)
$$

$$
h_R' = \mathrm{Attn}_R(h_R,h_v)
$$

여기서 $h_v$는 video latent tokens, $h_R$는 reference image tokens다. 실제 구현은 in-context self-attention 안에서 branch별 projection을 두는 방식에 가깝다. 중요한 점은 reference feature extraction이 video generation prior를 직접 오염시키지 않도록 path를 나눈다는 것이다.

### 2) Domain-aware AdaLN

Domain-aware AdaLN은 reference branch에 subject domain attribute를 주입한다. 논문은 다음 네 가지 domain attribute를 사용한다.

| Domain attribute | 의미 |
| --- | --- |
| real-world human | Real-world human subject |
| real-world object | Real-world object subject |
| background | Background subject |
| fantasy subject | Fantasy or non-real subject |

여기서 중요한 점은 domain attribute가 reference image의 domain만 뜻하지 않는다는 것이다. Generated video에서 subject가 어떤 domain으로 나타나야 하는지를 conditioning할 수 있다. 예를 들어 real person reference를 fantasy domain으로 보내거나, fantasy subject를 real-world object처럼 만들 수 있다.

### 3) Textual cross-attention freezing

DomainShuttle은 training 중 textual cross-attention을 freeze한다. 이는 prior preservation choice다. Reference personalization training이 text cross-attention까지 크게 바꾸면, base video model의 instruction following and prompt controllability가 손상될 수 있다.

즉 text control and reference control을 분리한다. Text branch는 base model의 언어 제어 능력을 최대한 유지하고, reference branch가 subject feature extraction을 담당한다.

### 4) Video-Reference DualRoPE

Video-Reference DualRoPE, 이하 VR-DualRoPE, 는 reference image tokens를 video tokens와 다른 RoPE space에 둔다. Reference image는 generated video의 temporal frame이 아니라 subject evidence다. 따라서 video frame처럼 같은 temporal position space에 넣으면 subject relation을 표현하기 어렵다.

VR-DualRoPE의 의도는 다음과 같다.

- Same subject를 설명하는 multiple reference images는 더 가까운 positional relation을 갖는다.
- Different subjects는 더 분리된 relation을 갖는다.
- Video token and reference token은 서로 다른 positional semantics를 갖는다.

이 설계는 multi-reference and multi-subject S2V에서 특히 중요하다.

### 5) Cross-Pair Consistent Loss

Cross-Pair Consistent Loss, 이하 CCL, 은 같은 video에 대응하는 서로 다른 reference set pair를 샘플링한다.

$$
R^{(1)},R^{(2)} \sim \mathcal{P}(v)
$$

같은 noise timestep에서 한 branch는 frozen target으로 두고, 다른 branch는 trainable path로 둔다. 단순화하면 다음과 같은 consistency loss로 볼 수 있다.

$$
\mathcal{L}_{CCL}
=
\left\|
f_{\mathrm{train}}(R^{(1)},t)
-
\mathrm{stopgrad}(f_{\mathrm{frozen}}(R^{(2)},t))
\right\|_2^2
$$

서로 다른 reference set은 viewpoint, occlusion, motion blur, illumination이 다를 수 있다. 그럼에도 같은 subject를 설명한다면 공유되는 정보가 있다. CCL은 그 공유되는 intrinsic subject feature를 학습하도록 유도한다.

# 4. Training / Data / Recipe

## 4-1. Data

Training data는 image personalization data and video personalization data로 구성된다.

| 구성 | 규모 | 출처 또는 설명 |
| --- | ---: | --- |
| Image personalization | 200K | UNO, Nano-Consistent-150K, Echo-4o, MUSAR 기반 |
| Video personalization | 750K | Phantom-Data, OpenS2V, Ditto-1M subset 기반 |
| Phantom-Data | 400K | Single and multi subject |
| OpenS2V | 300K | Multi subject |
| Ditto-1M subset | 50K | Single and multi subject augmentation |

논문은 training data가 open-source dataset 기반이라고 설명한다. Video personalization set은 "multiple reference set -> single video" and "single reference set -> multiple videos" 형태를 모두 지원하도록 구성된다. 이는 CCL의 cross-pair 학습에 중요하다.

## 4-2. Training strategy

Training은 두 단계다.

| 단계 | 데이터 | Step | Batch size | Update 대상 |
| --- | --- | ---: | ---: | --- |
| Stage 1 | 200K image personalization | 2,000 | 96 | patch embedding and self-attention |
| Stage 2 | 750K video personalization | 12,000 | 64 | video personalization modules, text cross-attention frozen |

Total training cost는 about 30,000 GPU-hours로 보고된다. Optimizer는 Adam 계열로 설명된다. HTML rendering에서 일부 LR and CCL coefficient 값이 명확히 보이지 않으므로 최종 반영 전 PDF 원문 재확인이 필요하다.

## 4-3. Engineering notes

1. **Reference fidelity를 무조건 최대화하지 말 것**
   - Cross-domain prompt controllability가 필요하면 reference의 subject-irrelevant details를 버릴 수 있어야 한다.

2. **Text control and reference control을 분리할 것**
   - Textual cross-attention freezing은 base model의 prompt-following prior를 보존하는 practical choice다.

3. **Reference image를 video frame처럼 다루지 말 것**
   - Reference는 temporal frame이 아니라 subject evidence다.
   - VR-DualRoPE는 이 차이를 positional encoding에서 반영한다.

4. **Multiple reference pair를 적극 활용할 것**
   - 같은 subject를 설명하는 서로 다른 reference set이 invariant feature supervision을 만든다.

5. **In-domain and cross-domain을 따로 평가할 것**
   - In-domain similarity가 높아도 cross-domain transformation이 실패할 수 있다.

# 5. Evaluation

## 5-1. Main results

논문은 video quality, text controllability, in-domain subject consistency, cross-domain subject consistency를 함께 평가한다.

| Method | AES | MS | GMEScore | NANO-CLIP | Qwen-CLIP | CD-Score | Qwen-Score | DINO-I | CLIP-I |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kling 1.6 | 0.515 | 0.965 | 0.596 | 0.621 | 0.640 | 0.725 | 0.771 | 0.401 | 0.672 |
| VACE-Wan2.2-14B | 0.480 | 0.974 | 0.685 | 0.606 | 0.622 | 0.546 | 0.679 | 0.303 | 0.679 |
| Ours, Wan2.1-14B | 0.510 | 0.977 | 0.689 | 0.627 | 0.647 | 0.787 | 0.781 | 0.405 | 0.703 |
| Ours, Wan2.2-14B | 0.516 | 0.987 | 0.705 | 0.636 | 0.658 | 0.861 | 0.829 | 0.400 | 0.690 |

가장 중요한 신호는 cross-domain consistency다. 논문은 prior SOTA 대비 CD-Score 18.7% improvement를 보고한다. 반면 in-domain similarity metrics는 모든 지표에서 단조롭게 좋아지는 것은 아니다. 즉 cross-domain flexibility와 exact reference similarity 사이에는 여전히 trade-off가 있다.

## 5-2. Evaluation data

Evaluation set은 in-domain and cross-domain으로 나뉜다.

| Split | 구성 |
| --- | --- |
| In-domain | 110 samples, including 90 OpenS2V-Eval and 20 human-object interaction cases |
| Cross-domain | 40 real-to-fantasy, 40 fantasy-to-real, 30 real-fantasy interaction samples |

Subject similarity metrics는 generated video에서 16 frames를 uniform sampling한 뒤 frame-level score를 average하는 방식으로 계산된다.

## 5-3. Human preference evaluation

Human preference evaluation은 각 방법에서 open-domain videos를 sampling해 전체 video quality, text controllability, open-domain subject consistency를 ranking한다. 이는 중요하다. Automatic subject similarity metric은 cross-domain transformation이 semantic하게 자연스러운지 충분히 잡지 못할 수 있다.

## 5-4. Ablation

Ditto-1M augmentation 관련 ablation은 다음과 같다.

| Variant | NANO-CLIP | CD-Score | DINO-I | CLIP-I |
| --- | ---: | ---: | ---: | ---: |
| w/o Ditto-1M | 0.631 | 0.823 | 0.432 | 0.701 |
| w Ditto-1M | 0.636 | 0.861 | 0.400 | 0.690 |

논문은 Ditto-1M 없이도 cross-domain SOTA 수준의 성능을 달성한다고 설명한다. Ditto-1M을 추가하면 CD-Score는 더 좋아지지만 일부 in-domain similarity metric은 내려간다. 이는 cross-domain flexibility and exact in-domain matching 사이 trade-off를 보여주는 좋은 사례다.

## 5-5. What really matters in the experiments

### 1) Cross-domain metric이 핵심이다

기존 S2V는 in-domain subject preservation에 이미 강하다. DomainShuttle의 novelty는 subject identity를 다른 domain으로 옮길 때 intrinsic feature를 얼마나 보존하느냐에 있다.

### 2) Text controllability와 subject consistency를 같이 봐야 한다

Subject를 보존하려고 prompt를 무시하면 personalization model로서 실용성이 낮다. 반대로 prompt를 잘 따르지만 subject identity가 사라지면 S2V가 아니다. DomainShuttle의 claim은 이 둘의 joint improvement에 있다.

### 3) Metric이 model-based라는 점을 주의해야 한다

CD-Score는 GPT-5.2 기반, Qwen-Score는 Qwen3-VL-8B-Instruct 기반이다. MLLM judge bias and prompt sensitivity를 고려해야 한다.

### 4) Trade-off는 아직 남아 있다

Cross-domain score가 좋아졌다고 해서 모든 in-domain consistency metric이 좋아지는 것은 아니다. 실제 product에서는 mode별 control이 필요하다.

# 6. Limitations

1. **Metric dependency**
   - Cross-domain consistency가 MLLM-based judge and proxy edited-reference similarity에 의존한다.
   - Judge bias or prompt sensitivity가 결과 해석에 영향을 줄 수 있다.

2. **Training cost**
   - About 30,000 GPU-hours는 가볍지 않다.
   - Reproduction cost가 높다.

3. **Base model scope**
   - Wan2.1 and Wan2.2 14B variants에서 검증되었다.
   - 다른 DiT video backbone에 얼마나 일반화되는지는 추가 확인이 필요하다.

4. **Domain label dependency**
   - Domain-aware AdaLN은 subject domain attribute를 필요로 한다.
   - 이 domain annotation이 잘못되면 conditioning이 흔들릴 수 있다.

5. **Evaluation set size**
   - Cross-domain evaluation set은 유용하지만 아직 충분히 크다고 보기는 어렵다.
   - Community benchmark로 자리 잡는지 봐야 한다.

6. **Edited-video data risk**
   - Ditto-1M subset이 augmentation으로 들어간다.
   - 논문은 main supervision이 아니라고 설명하지만, data leakage and evaluation overlap은 확인이 필요하다.

7. **Identity preservation ambiguity**
   - Fantasy or stylized domain에서는 "같은 subject"의 정의가 애매할 수 있다.

8. **Safety issue**
   - Human subject를 다른 domain으로 옮기는 capability는 identity misuse and deepfake risk를 키울 수 있다.

9. **Latency analysis 부족**
   - Quality metric 중심이고 serving cost, memory overhead, inference latency는 상대적으로 덜 다뤄진다.

10. **Mode control 필요**
    - In-domain and cross-domain mode를 user가 명시적으로 조절할 수 있어야 실사용에서 혼선이 줄어든다.

# 7. My Take

## 7-1. Why this matters for my work

DomainShuttle의 핵심은 "subject consistency가 좋아졌다"가 아니다. 더 중요한 점은 **reference conditioning을 flexible control signal로 재설계했다는 것**이다.

대부분의 personalization method는 reference를 보존 대상으로 본다. DomainShuttle은 reference를 분해 대상으로 본다.

```text
subject identity: keep
domain attribute: route
prompt style: follow
redundant reference details: suppress
```

이 decomposition은 image/video editing, subject-driven generation, multimodal personalization 전반에 재사용 가능하다.

## 7-2. Reuse potential

### Video personalization

실제 personalized video system을 만들려면 in-domain and cross-domain을 따로 평가해야 한다. In-domain identity preservation만으로는 부족하다.

### Image personalization

같은 설계 원리는 image generation에도 적용 가능하다. Content identity와 domain attribute를 분리하고, reference pair consistency로 invariant feature를 학습할 수 있다.

### Agentic creative tools

사용자가 "이 캐릭터를 장난감처럼 만들어줘" 또는 "이 object를 fantasy style로 바꿔줘"라고 요청할 때, system은 intrinsic identity를 유지하면서 domain을 바꿔야 한다. DomainShuttle은 이 문제에 model-level recipe를 준다.

### Dataset construction

Cross-pair reference pool은 강력한 data design이다. 같은 subject에 대한 여러 reference set은 single-reference training보다 훨씬 좋은 invariance supervision을 만든다.

## 7-3. Production considerations

- User가 in-domain mode and cross-domain mode를 명시적으로 선택할 수 있게 해야 한다.
- Human subject reference는 consent and safety review가 필요하다.
- Domain attribute가 잘못 붙었을 때 output이 어떻게 흔들리는지 audit해야 한다.
- Automatic metric만 보지 말고 human preference evaluation을 같이 둬야 한다.
- Prompt-following degradation을 별도 metric으로 모니터링해야 한다.

## 7-4. Follow-up papers

- Phantom
- VACE
- FFGO
- HuMo
- BindWeave
- OpenS2V
- Ditto-1M
- Wan2.1 and Wan2.2
- Qwen-Image-Agent
- Subject-driven image and video personalization papers

# 8. Summary

- DomainShuttle은 open-domain subject-driven text-to-video generation을 다룬다.
- 핵심 문제는 in-domain fidelity and cross-domain flexibility를 동시에 만족하는 것이다.
- Domain-MoT는 video and reference feature path를 분리하고 domain-aware AdaLN을 넣는다.
- VR-DualRoPE는 video and reference token의 positional semantics를 분리한다.
- CCL은 reference set 간 consistency로 intrinsic subject feature를 학습하게 한다.
