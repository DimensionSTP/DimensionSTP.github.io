---
layout: single
title: "Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models Review"
categories: Study-concept
tag: [Wan-Streamer, StreamingModel, MultimodalAI, FullDuplex, VideoGeneration]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.25041)

[Website](https://wan-streamer.com/)

Wan-Streamer v0.1은 "realtime avatar generation"이나 "omni-modal chat model"로만 보면 작게 읽힌다. 이 논문이 실제로 겨냥하는 문제는 훨씬 구조적이다. **text, audio, video를 input과 output 양쪽에서 동시에 다루는 full-duplex interaction을 하나의 causal model과 streaming serving contract로 만들 수 있는가**다.

현재 real-time multimodal assistant는 대부분 cascade다. VAD, ASR, LLM, TTS, audio-driven animation, video rendering이 module chain으로 붙는다. 이 구조는 구현하기 쉽지만, module boundary마다 latency가 생긴다. Speech recognition error가 language reasoning에 누적되고, TTS와 video animation이 나중에 맞춰지며, user가 중간에 interrupt해도 upstream/downstream state가 한꺼번에 업데이트되지 않는다. Visual listening behavior, turn management, response timing, cross-modal synchronization을 하나의 behavior로 학습하기도 어렵다.

Wan-Streamer는 이 문제를 native-streaming end-to-end model로 푼다. Language, audio, video가 모두 input token과 output token으로 같은 causal stream 안에 들어간다. Single Transformer가 interleaved multimodal tokens를 block-causal attention으로 처리하고, audio/video output은 continuous latent flow matching으로 생성된다. 모든 component는 causal encoders, causal decoders, block-causal attention, low-latency token scheduling을 전제로 설계된다.

> 한 줄 요약: Wan-Streamer는 text/audio/video를 input과 output 양쪽에서 interleaved causal sequence로 모델링하는 native-streaming interactive foundation model이며, causal multimodal encoders/decoders, block-causal attention, thinker-performer inference, flow-matching latent generation을 통해 160 ms streaming unit, 약 200 ms model-side latency, 약 550 ms total interaction latency를 보고한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Multimodal interaction을 understanding과 generation의 합이 아니라 full-duplex causal stream으로 재정의한다.
- VAD, ASR, LLM, TTS, animation, video generation cascade를 하나의 model state로 합치려 한다.
- Audio와 video output을 post-hoc synchronization하지 않고 coupled latent generation으로 다룬다.
- 160 ms unit, 25 FPS, 200 ms model-side latency 같은 real-time constraint를 architecture level에서 반영한다.
- Thinker-performer pipeline으로 unified model semantics와 hardware overlap 사이의 trade-off를 설계한다.
- Interactive foundation model을 latency, streamability, turn management, interruption handling 관점에서 평가하게 만든다.

이 글에서는 Wan-Streamer를 "빠른 avatar model"보다, **full-duplex audio-visual interaction을 causal model architecture와 serving schedule로 동시에 푼 paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Real-time interaction은 turn-based text generation과 다르다. User는 말하고, 움직이고, interrupt하고, agent는 들으면서 반응하고, 말하면서도 다시 user feedback을 받아야 한다.

Streaming unit $k$에서 user observation은 다음과 같다.

$$
u_k
=
(u_k^t,u_k^a,u_k^v)
$$

Agent response는 다음과 같다.

$$
y_k
=
(y_k^t,y_k^a,y_k^v)
$$

Wan-Streamer는 streaming unit 위에서 response를 autoregressive하게 모델링한다.

$$
p_{\theta}(y_{1:K}\mid u_{1:K})
=
\prod_{k=1}^{K}
p_{\theta}
\left(
y_k^t,y_k^a,y_k^v
\mid
u_{\leq k}^t,u_{\leq k}^a,u_{\leq k}^v,
y_{<k}^t,y_{<k}^a,y_{<k}^v
\right)
$$

핵심 constraint는 causality다. Model은 current unit까지의 user observation과 prior agent output은 사용할 수 있지만 future input은 사용할 수 없다.

이것이 full-duplex다. User stream과 agent stream은 overlap된다. "User가 말을 끝내고, system이 생각하고, system이 말한다"는 turn-taking 구조가 아니다.

## 1-2. Why previous approaches are insufficient

### 1) Cascaded real-time systems

Cascade 구조는 perception과 generation을 분리한다.

```text
VAD -> ASR -> LLM -> TTS -> avatar renderer
```

각 module은 latency와 error를 만든다. 더 중요한 것은 response timing과 cross-modal alignment가 end-to-end로 학습되지 않는다는 점이다.

### 2) Speech-only full-duplex models

Speech-only model은 language/audio latency를 줄일 수 있지만 synchronized visual agent behavior를 생성하지는 않는다. Digital human이나 embodied video interaction에서는 visual response도 communication의 일부다.

### 3) Offline video generation models

Video generation model은 보통 bidirectional 또는 chunk-level offline generation을 사용한다. Quality는 높을 수 있지만 live interaction에 필요한 causality와 low latency를 만족하지 못할 수 있다.

### 4) Hidden text intermediate

일부 multimodal system은 module 사이의 intermediate로 text를 사용한다. 이는 audio-visual timing을 깨뜨리고 non-verbal cue를 잃게 만들 수 있다.

Wan-Streamer는 perception, reasoning, generation, response timing, turn management, synchronization을 jointly train해 hidden intermediate module boundary를 제거하려 한다.

# 2. Core Idea

## 2-1. Main contribution

Wan-Streamer의 핵심 아이디어는 세 가지다.

1. **Single Transformer for multimodal in/out**
   - Language, audio, video를 interleaved input/output token으로 표현한다.
   - 하나의 causal state가 perception, reasoning, generation, committed history를 처리한다.

2. **Fully causal stack**
   - Strictly causal audio/video VAE를 사용한다.
   - Causal audio-visual encoder/decoder를 사용한다.
   - Block-causal multimodal attention을 사용한다.
   - Full-history autoregressive streaming을 수행한다.

3. **Thinker-performer inference**
   - Thinker는 encoding, Transformer state update, KV cache, decoding을 담당한다.
   - Performer는 expensive audio-video latent generation을 담당한다.
   - KV slice와 latent를 교환해 current perception, previous output decoding, next latent denoising을 overlap한다.

## 2-2. Design intuition

설계 직관은 다음과 같다.

```text
Streamability는 serving optimization이 아니다.
Modeling constraint다.
```

Training이 offline bidirectional component를 사용하면 deployment에서 natural streaming behavior를 완전히 회복하기 어렵다. Model은 generated unit이 즉시 history에 commit되고, 새로운 user unit이 future response에 영향을 줄 수 있음을 학습해야 한다.

두 번째 intuition은 audio와 video가 decoding 전에 coupling되어야 한다는 것이다. Speech와 face/video를 별도 module로 생성한 뒤 post-hoc synchronize하면 timing과 expression mismatch가 생기기 쉽다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Real-time full-duplex text/audio/video interaction |
| Architecture | Single Transformer over interleaved multimodal tokens |
| Attention | Block-causal attention |
| Audio/video generation | Conditional flow matching in continuous latent space |
| Encoders/decoders | Strictly causal audio/video VAE와 causal AV encoder/decoder |
| Streaming unit | 25 FPS에서 최소 160 ms |
| Inference | Thinker-performer streaming pipeline |
| Model-side latency | About 200 ms |
| Total latency | About 550 ms with 350 ms bidirectional network latency |

## 3-2. Module breakdown

### 1) Multimodal causal sequence

Text, audio, video의 input/output은 하나의 sequence 안에 interleave된다.

각 streaming unit은 user side token과 agent side token을 포함한다. 한 번 생성된 agent output token은 이후 context의 일부가 된다.

이 구조는 별도의 hidden module state를 두는 문제를 피한다.

### 2) Block-causal attention

Block-causal attention은 incremental streaming을 조율한다. Model은 허용된 past block에는 attention할 수 있지만 future block에서 정보가 새는 것은 막는다. 이를 통해 causality를 보존하면서 full-history memory를 사용할 수 있다.

### 3) Audio와 video latent generation

Text response는 discrete하므로 cross-entropy로 학습한다. Audio와 video response는 continuous latent이므로 flow matching으로 학습한다.

Modality $m \in \{a,v\}$에 대해 noisy latent는 다음과 같다.

$$
z_{\tau}^{m}
=
(1-\tau)z_0^m+\tau\epsilon^m
$$

Target velocity는 다음과 같다.

$$
\frac{\partial z_{\tau}^{m}}{\partial\tau}
=
\epsilon^m-z_0^m
$$

Flow-matching loss는 다음과 같다.

$$
\mathcal{L}_{FM}^{m}
=
\mathbb{E}
\left[
\left\|
f_{\theta}
\left(
z_{\tau}^{a},z_{\tau}^{v},c_k,\tau
\right)
-
\frac{\partial z_{\tau}^{m}}{\partial\tau}
\right\|_2^2
\right]
$$

같은 clean causal context가 audio와 video를 함께 condition하므로, speech, motion, scene evolution을 coupled response로 optimize할 수 있다.

### 4) Training stages

Training은 세 stage로 진행된다.

| Stage | Purpose |
| --- | --- |
| Independent-task pretraining | 하나의 sequence model 안에 multimodal understanding과 generation을 구축 |
| End-to-end interaction training | Duplex text/audio/video interaction에 적응 |
| Low-latency distillation | CFG와 multi-step teacher를 효율적인 student에 흡수 |

세 번째 stage는 distillation과 rolling/self-forcing style training을 사용해 long-horizon train-test mismatch를 줄인다.

### 5) Thinker-performer serving

Model은 unified하지만, hardware overlap을 위해 inference는 두 부분으로 나뉜다.

| Component | Role |
| --- | --- |
| Thinker | Current observation을 encode하고, KV cache를 update하며, previous latent를 decode |
| Performer | 다음 audio/video latent를 위한 flow-matching solver를 실행 |

Unit $k$에서 thinker는 current user input을 처리하고 previous response latent를 decode한다. Performer는 current KV slice를 받아 다음 clean audio/video latent를 생성한다. 이를 통해 인접한 streaming unit들이 overlap된다.

# 4. Training / Data / Recipe

## 4-1. Data

Wan-Streamer는 넓은 data mixture를 사용한다.

| Data type | Purpose |
| --- | --- |
| Image/audio/video understanding | Multimodal perception |
| Text dialogue | Dialogue competence |
| ASR, TTS, audio dialogue | Speech interface와 language-audio alignment |
| Image/audio/video generation | Output modality generation |
| Joint audio-visual generation | Cross-modal synchronization |
| End-to-end duplex interaction | Target full-duplex behavior |

이 data mixture가 중요한 이유는 model이 multimodal understanding model이면서 동시에 multimodal generator여야 하기 때문이다.

## 4-2. Training strategy

Three-stage recipe는 다음과 같다.

1. **Independent-task pretraining**
   - Language model에서 unified Transformer를 initialize한다.
   - Audio/video encoder, generation latent, multimodal interface를 학습한다.

2. **End-to-end interaction training**
   - Interleaved user input과 agent output을 가진 duplex interaction data로 학습한다.

3. **Distillation for low-latency streaming**
   - CFG와 더 많은 solver step을 쓰는 강한 teacher를 효율적인 student로 distill한다.
   - Long-horizon degradation을 줄이기 위해 rolling distillation을 사용한다.

## 4-3. Engineering notes

1. **모든 것이 causal해야 한다**
   - Encoder, decoder, attention, scheduling이 모두 streamability를 보존해야 한다.

2. **Latency boundary를 명시해야 한다**
   - Model-side latency, first-packet latency, API TTFB, endpointing, total user-visible latency는 서로 다르다.

3. **Decoding 전에 synchronize한다**
   - Audio와 video는 decoding 전에 coupled latent로 생성되어야 한다.

4. **Model state를 깨지 않는 serving split을 사용한다**
   - Thinker-performer split은 cache exchange를 통해 unified KV state를 유지한다.

5. **Streaming unit duration은 hard budget이다**
   - Performer 실행 시간과 통신 시간이 160 ms unit 안에 들어와야 한다.

# 5. Evaluation

## 5-1. Latency

Wan-Streamer는 다음을 보고한다.

| Metric | Value |
| --- | ---: |
| Streaming unit | 160 ms |
| Video frame rate | 25 FPS |
| Model-side signal-to-signal latency | About 200 ms |
| Bidirectional network latency assumption | 350 ms |
| Total interaction latency | About 550 ms |

이 숫자는 latency boundary별로 읽어야 한다. 논문은 speech system과 omni-modal system이 model response, first token, first packet, endpointing, API TTFB, user-visible response처럼 서로 다른 latency metric을 보고한다는 점을 명시적으로 경고한다.

## 5-2. Comparison scope

논문은 speech system 및 omni-modal system과 비교하지만, raw latency 숫자가 그대로 서로 교환 가능하다고 보지는 않는다. 어떤 system은 speech-only이고, 어떤 system은 video input을 받지만 synchronized visual output을 생성하지 않는다. 어떤 system은 web-app latency를 보고하고, 어떤 system은 model-only latency를 보고한다.

중요한 비교점은 Wan-Streamer가 text/audio/video input과 output을 하나의 causal stream에 포함하고, model-side latency와 total interaction latency를 함께 보고한다는 점이다.

## 5-3. What really matters in the experiments

### 1) End-to-end boundary가 중요하다

빠른 TTS나 빠른 renderer를 가진 system이라도 upstream module이 cascade되어 있으면 total interaction latency가 여전히 클 수 있다.

### 2) Full-duplex behavior가 핵심이다

Model은 response를 생성하는 동안 perception을 수행하고, perception을 수행하는 동안 response를 생성해야 한다. 이는 일반적인 turn-based multimodal generation을 넘어선다.

### 3) Visual output이 중요하다

Omni input에 speech output을 붙이는 것은 synchronized audio-visual agent response와 같지 않다.

### 4) Serving schedule도 architecture의 일부다

Thinker-performer split은 단순한 engineering이 아니다. Real-time hardware constraint 아래에서 unified model semantics를 가능하게 하는 구조다.

# 6. Limitations

1. **v0.1 report**
   - 논문은 초기 technical report다. 장기 안정성과 공개 재현성은 추가 확인이 필요하다.

2. **Data detail이 high-level이다**
   - 전체 dataset composition과 licensing은 추가 확인이 필요하다.

3. **Two-GPU serving assumption이 있다**
   - 보고된 thinker-performer path는 hardware와 optimized kernel에 의존한다.

4. **Latency comparison은 heterogeneous하다**
   - 다른 system들은 서로 다른 latency boundary를 보고한다.

5. **Abstract에서는 quality metric보다 latency가 중심이다**
   - Abstract에서는 quality metric보다 latency가 더 중심에 놓인다. Human preference와 audiovisual quality analysis는 full paper에서 더 자세히 확인해야 한다.

6. **Safety와 misuse 문제가 있다**
   - Real-time visual agent generation은 impersonation이나 deceptive interaction에 악용될 수 있다.

7. **Identity와 long-session consistency가 어렵다**
   - 논문은 causal history를 통한 persistent state를 주장하지만, long-horizon degradation은 여전히 핵심 challenge다.

8. **Network latency assumption이 있다**
   - 350 ms bidirectional network budget은 deployment 환경에 따라 달라질 수 있다.

9. **End-to-end model은 debugging이 어렵다**
   - Module을 제거하면 boundary error는 줄지만 modular interpretability도 줄어든다.

10. **Public release status를 확인해야 한다**
    - 재사용 전 model weight, demo, implementation availability를 확인해야 한다.

# 7. My Take

## 7-1. Why this matters for my work

Wan-Streamer의 핵심은 "faster avatar"가 아니라, **interactive multimodal AI를 full-duplex causal stream으로 모델링한 것**이다.

많은 system은 module이 빠르기 때문에 interactive해 보인다. 그러나 perception, reasoning, speech, video가 여전히 분리되어 있다면 timing과 state는 흩어진다. Wan-Streamer는 streamability를 architectural constraint로 둔다는 것이 무엇인지 보여준다.

## 7-2. Reuse potential

### Realtime assistants

Future voice/video agent는 이 모델을 causal encoder, unified state, coupled output latent, streaming schedule을 갖춘 reference architecture로 사용할 수 있다.

### Digital humans

Avatar system은 흔히 TTS와 animation에 의존한다. Wan-Streamer는 speech와 visual response를 jointly generate하는 경로를 제시한다.

### Interactive world models

Real-time game, simulation, embodied agent는 continuous bidirectional perception과 action이 필요할 수 있다. Causal streaming contract는 avatar를 넘어 일반화된다.

### Serving system design

Thinker-performer decomposition은 state update와 expensive latent generation을 overlap할 수 있는 모든 model에 재사용 가능하다.

## 7-3. Production considerations

- System을 비교하기 전에 latency boundary를 정의해야 한다.
- Model-side latency, network latency, endpointing, user-visible latency를 분리해야 한다.
- Long session에서 synchronization과 identity consistency를 audit해야 한다.
- Real-time visual identity generation에 대한 명시적 abuse prevention을 추가해야 한다.
- Noisy network에서 interruption handling과 turn management를 test해야 한다.
- Rolling history에서 quality degradation을 monitor해야 한다.

## 7-4. Follow-up papers

- Qwen2.5-Omni and Qwen3-Omni
- Moshi
- GPT-4o Realtime API papers and documentation
- MiniCPM-o
- X-Streamer
- minWM
- Self-Forcing
- Diffusion Forcing
- Wan video generation papers

# 8. Summary

- Wan-Streamer는 full-duplex text/audio/video interaction을 위한 end-to-end native-streaming model이다.
- Input과 output modality를 하나의 Transformer가 처리하는 interleaved causal sequence로 표현한다.
- Audio와 video는 flow matching을 통해 coupled continuous latent로 생성된다.
- Thinker-performer inference는 perception, state update, decoding, latent denoising을 overlap한다.
- 핵심 contribution은 streamability를 post-hoc serving optimization이 아니라 modeling constraint로 취급한 점이다.
