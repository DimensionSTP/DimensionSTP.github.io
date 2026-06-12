---
layout: single
title: "Audio Interaction Model Review"
categories: Study-concept
tag: [Audio, Streaming, LALM, MultimodalAI]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.05121)

[Project page](https://xzf-thu.github.io/Audio-Interaction)

[Code link](https://github.com/xzf-thu/Audio-Interaction)

Audio Interaction Model은 "audio model이 clip을 다 들은 뒤 답하는 방식" 자체를 다시 묻는 논문이다. 요즘 audio LLM은 ASR, speech QA, audio understanding, voice chat 쪽에서 빠르게 좋아지고 있지만, 대부분은 여전히 offline clip input-output 형식에 가깝다. 사용자가 파일을 주면 모델이 답하고, streaming이라고 해도 보통 ASR나 voice chat처럼 task 하나에 맞춘 별도 system으로 설계된다.

이 논문은 그 가정에서 출발하지 않는다. Audio는 본질적으로 연속적이고, 모델은 매 순간 들어오는 소리를 해석하면서 "지금 말해야 하는가, 아니면 계속 들어야 하는가"를 결정해야 한다고 본다. 그래서 저자들은 Audio Interaction Model이라는 regime을 정의하고, 이를 Audio-Interaction이라는 3B급 streaming model과 SoundFlow라는 training/deployment framework로 구현한다.

> 한 줄 요약: Audio Interaction Model은 audio LLM을 offline clip QA model이 아니라, chunk마다 perceive-decide-respond loop를 돌며 silence와 response를 고르는 always-on streaming interaction model로 바꾸려는 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- audio LLM의 다음 병목이 recognition accuracy만이 아니라 interaction timing일 수 있음을 정면으로 다룬다.
- offline LALM, streaming ASR, full-duplex voice chat을 task별 system으로 쪼개지 않고 하나의 streaming instruction-following model로 묶으려 한다.
- StreamAudio-2M, Proactive-Sound-Bench, SoundFlow를 함께 제안해 data, training objective, inference scheduling을 하나의 system recipe로 본다.
- 실험이 단순 score 경쟁보다 "언제 침묵해야 하는가"와 "언제 먼저 개입해야 하는가"를 따로 평가한다는 점이 흥미롭다.

이 논문의 핵심 메시지는 단순하다. 좋은 audio assistant는 들은 내용을 잘 맞히는 모델만으로는 부족하다. 실제 interaction에서는 모델이 계속 듣고 있어야 하며, 대부분의 순간에는 침묵해야 하고, 정말 필요한 순간에만 응답해야 한다. 이 논문은 그 decision layer를 audio LLM의 부가 기능이 아니라 model training의 중심 objective로 끌어올린다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 real-time audio interaction이다. 여기서 모델은 완성된 audio clip 하나를 입력으로 받는 것이 아니라, 시간 순서대로 들어오는 audio stream을 chunk 단위로 받는다. 그리고 각 chunk마다 두 가지를 동시에 결정해야 한다.

1. 지금까지 들은 내용을 어떻게 이해할 것인가.
2. 지금 응답해야 하는가, 아니면 계속 침묵해야 하는가.

기존 LALM setting을 아주 단순하게 쓰면 다음과 같다.

$$
y = f(x, q)
$$

여기서 $x$는 완성된 audio clip이고, $q$는 instruction이다. 모델은 $x$ 전체를 본 뒤 $y$를 생성한다. 하지만 streaming setting에서는 입력이 $x_1, x_2, ..., x_t$처럼 들어오고, 각 시점 $t$에서 decision $d_t$가 필요하다.

$$
d_t, y_t = f(x_t, h_{t-1}, q)
$$

여기서 $h_{t-1}$은 지금까지의 streaming context다. $d_t$가 silent이면 모델은 아무 말도 하지 않고 다음 chunk를 기다린다. $d_t$가 response이면 모델은 현재까지의 context를 바탕으로 답변을 생성한다.

이 문제는 단순한 low latency inference 문제가 아니다. 핵심은 semantic triggering이다. 즉 소리가 났다는 이유만으로 답하면 안 되고, 의미적으로 개입할 가치가 있을 때만 답해야 한다.

## 1-2. Why previous approaches are insufficient

기존 접근은 크게 세 부류로 볼 수 있다.

1. Offline LALM
   - 전체 audio clip을 받은 뒤 답한다.
   - speech QA, audio QA, music understanding 같은 task에서는 강할 수 있다.
   - 하지만 stream 중간의 hesitation, cough, alarm, background music 같은 event에 즉시 반응하기 어렵다.

2. Task-specific streaming model
   - streaming ASR, simultaneous translation, full-duplex dialogue처럼 task 하나에 맞춰 설계된다.
   - latency와 task accuracy는 좋아질 수 있다.
   - 하지만 audio understanding, speech translation, voice chat, proactive response를 하나의 model behavior로 통합하기 어렵다.

3. Voice chat 중심 model
   - turn-taking과 말 끊기 같은 conversation UX에 집중한다.
   - 그러나 non-speech sound나 environment event를 의미 있는 interaction signal로 다루지 못하는 경우가 많다.

이 논문이 지적하는 핵심 한계는 두 가지다.

- Comprehension-grounded response triggering: 모델은 acoustic energy나 pause만 보고 반응하는 것이 아니라, 현재 stream의 의미를 이해하고 응답 여부를 골라야 한다.
- Real-time context continuity: audio를 chunk로 쪼개면 long-range context와 acoustic continuity가 깨지기 쉬운데, streaming latency를 지키면서 이 context를 유지해야 한다.

결국 이 논문의 문제 설정은 "더 좋은 audio encoder를 만들자"보다, "audio model을 어떻게 always-on interaction policy로 바꿀 것인가"에 가깝다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 아래 네 가지로 정리할 수 있다.

1. Audio Interaction Model이라는 regime 정의
   - audio stream을 chunk 단위로 처리한다.
   - 매 chunk마다 silent 또는 response decision을 내린다.
   - ASR, translation, audio understanding, voice chat, proactive help를 하나의 streaming instruction-following setting으로 묶는다.

2. Audio-Interaction model
   - Qwen2.5-Omni-3B를 초기화 모델로 사용한다.
   - 기존 offline audio capability를 유지하면서 streaming interaction behavior를 추가한다.
   - 400 ms chunk 단위에서 response triggering을 학습한다.

3. SoundFlow framework
   - streaming-native data construction, comprehension-aware training, asynchronous inference를 하나로 묶은 framework다.
   - training data를 long-form multi-turn stream으로 만들고, special streaming control token을 통해 언제 말할지 학습한다.

4. StreamAudio-2M과 Proactive-Sound-Bench
   - StreamAudio-2M은 7개 core capability와 28개 sub-task를 포함하는 streaming instruction-following corpus다.
   - Proactive-Sound-Bench는 모델이 explicit instruction 없이도 위험하거나 도움이 필요한 acoustic event에 적절히 개입하는지 본다.

이 논문에서 중요한 점은 model architecture 하나만 제안하지 않는다는 것이다. Data composition, silence supervision, trigger objective, inference scheduling, benchmark가 모두 같은 문제를 향한다. 즉 "audio model을 streaming interaction model로 바꾸려면 무엇이 필요한가"를 system stack 전체로 답한다.

## 2-2. Design intuition

설계 직관은 꽤 명확하다.

첫째, audio interaction에서는 대부분의 시간이 response가 아니라 silence다. 항상 듣고 있는 assistant가 every sound에 반응하면 실제 product로는 사용할 수 없다. 따라서 silent decision은 단순 negative class가 아니라 핵심 capability다.

둘째, response timing은 local acoustic cue만으로 정하기 어렵다. 짧은 cough 하나는 무시할 수도 있지만, prolonged coughing fit이나 smoke alarm은 proactive intervention이 필요할 수 있다. 이 차이는 audio event 자체와 주변 context를 함께 봐야 한다.

셋째, streaming 학습 데이터는 자연스럽게 생기지 않는다. 기존 audio dataset은 대부분 짧은 clip과 label, 또는 clip과 QA pair로 되어 있다. 그러나 Audio-Interaction은 3-15 turn 정도의 heterogeneous stream과 sparse response cue가 필요하다. 그래서 저자들은 short clip을 그냥 random stitch하지 않고, scenario planning과 event refinement를 통해 더 그럴듯한 long-form stream을 만든다.

넷째, real-time deployment에서는 encoder와 decoder scheduling이 method의 일부다. Audio chunk는 계속 들어오고, decoder는 때로는 silence token만 내고, 때로는 긴 response를 생성한다. 이 둘을 잘못 묶으면 stall이 생긴다. SoundFlow의 FIFO scheduling은 이 deployment path까지 논문 안으로 가져온다.

이 논문의 좋은 점은 audio understanding paper라기보다 audio interaction system paper로 읽힌다는 데 있다. Model score보다도 "무엇을 data로 만들고, 무엇을 objective로 두며, 무엇을 latency path에서 분리해야 하는가"가 더 오래 남는다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Offline LALM을 always-on streaming audio interaction model로 변환 |
| Main model | Audio-Interaction |
| Base model | Qwen2.5-Omni-3B |
| Core loop | Perceive, decide, respond |
| Chunk size | 400 ms main setting |
| Key decision | 매 chunk마다 silent 또는 response를 예측 |
| Dataset | StreamAudio-2M |
| Framework | SoundFlow |
| Benchmark | MMAU, spoken dialogue, LibriSpeech, CoVoST2, Proactive-Sound-Bench 등 |
| Deployment idea | FIFO-scheduled asynchronous inference |

기존 offline audio LLM은 보통 audio encoder, adapter, LLM backbone으로 구성되고, complete clip을 넣은 뒤 response를 생성한다. Audio-Interaction은 이 구조를 완전히 버리기보다, Qwen2.5-Omni-3B를 기반으로 streaming behavior를 추가한다.

핵심 변화는 아래 한 줄이다.

$$
input = audio\ chunk, output = silent\ token\ or\ response
$$

즉 모델은 매 chunk에서 반드시 full response를 생성하는 것이 아니라, 대개는 silent control token을 생성하고 다음 chunk를 기다린다. 충분한 semantic evidence가 쌓이거나 proactive intervention이 필요하면 response generation으로 전환한다.

## 3-2. Module breakdown

### 1) Streaming formulation

Audio-Interaction은 stream을 고정 길이 chunk로 나눈다. 논문 main setting에서는 chunk가 400 ms다. 각 step에서 모델은 현재 chunk와 이전 context를 보고 special control token을 예측한다.

- silent이면 다음 chunk를 계속 듣는다.
- response이면 autoregressive generation으로 넘어간다.
- response가 끝나면 다시 listening loop로 돌아간다.

이 설계는 ASR와 translation에도 그대로 적용된다. 예를 들어 실시간 ASR에서는 partial transcript를 chunk마다 낼 수 있고, simultaneous translation에서는 source speech가 완전히 끝나기 전에 translation을 interleaved하게 생성할 수 있다. Proactive response에서는 user instruction이 없어도 acoustic event만으로 개입할 수 있다.

중요한 것은 이 모든 task가 별도 model이 아니라 같은 decision format 안에 들어간다는 점이다. 이 논문의 language로 쓰면, recognition, translation, dialogue가 모두 streaming instruction inside one always-on loop가 된다.

### 2) TFJP preprocessing

Streaming data를 만들 때 short audio clip을 이어 붙이면 boundary artifact가 쉽게 생긴다. 앞 clip의 silence, background noise, spectral mismatch가 다음 clip과 어긋나면 모델은 실제 semantic transition이 아니라 편집 흔적을 trigger cue로 학습할 수 있다.

이를 줄이기 위해 논문은 TFJP, 즉 time-frequency joint preprocessing을 쓴다. 구성은 대략 아래처럼 읽을 수 있다.

- silence_cut: 과도한 내부 silence를 줄인다.
- noise_profile: low-energy region에서 noise profile을 추정한다.
- denoise: frequency domain에서 noise를 줄인다.
- core_locate: informative span을 찾는다.
- boundary_norm과 spec_smooth: boundary를 half-chunk alignment와 spectral smoothing으로 다듬는다.

이 부분은 model architecture보다 data engineering에 가깝지만, streaming interaction에서는 꽤 중요하다. Model이 언제 말해야 하는지를 배울 때, audio boundary artifact가 label shortcut으로 작동하면 실제 환경에서 over-triggering이 발생할 수 있기 때문이다.

### 3) Hierarchical event selection

Random concatenation은 streaming data를 만들 때 가장 쉬운 방식이다. 하지만 이 논문은 random stitching이 context consistency를 깨뜨릴 수 있다고 본다. 예를 들어 speech가 이어지는 중간에 전혀 관계없는 car horn이나 music clip이 갑자기 섞이면, 모델은 coherent scene을 학습하기 어렵다.

그래서 SoundFlow는 hierarchical event curation을 사용한다.

1. Scenario planning
   - LLM이 high-level scenario를 만든다.
   - 여러 topic 또는 sub-event가 포함된다.

2. Event refinement
   - scenario 안의 topic을 concrete audio event sequence로 바꾼다.

3. Clip grounding
   - audio clip database에서 top candidate를 검색한다.
   - 적절한 clip이 없으면 audio generation model로 event를 합성한다.
   - 이후 suitability verification을 거친다.

이 설계는 synthetic data를 많이 쓰더라도, 단순 augmentation보다 "interaction episode"를 만들려는 쪽에 가깝다. 특히 proactive response 학습에서는 event의 종류뿐 아니라 앞뒤 context가 중요하기 때문에 이런 composition strategy가 method의 일부가 된다.

### 4) Streaming objective and silence training

Audio-Interaction은 standard language modeling objective와 streaming control objective를 같이 쓴다. 개념적으로는 아래처럼 볼 수 있다.

$$
L = L_text + lambda * L_stream
$$

여기서 $L_text$는 일반 response token에 대한 loss이고, $L_stream$은 streaming control token에 대한 loss다. $lambda$는 streaming objective의 상대적 weight다.

논문에서 흥미로운 부분은 false triggering을 별도 failure mode로 본다는 점이다. Audio assistant가 irrelevant sound에 자꾸 반응하면 usability가 크게 나빠진다. 그래서 저자들은 Proactive-Sound-Bench의 agent verification을 통해 response가 필요 없는 silent audio를 많이 넣고, 모델이 침묵해야 하는 상황을 적극적으로 학습시킨다.

또 하나는 history review training이다. Long stream에서는 초반 context를 잊기 쉽기 때문에, 뒤쪽 위치에서 앞선 내용을 묻는 질문을 삽입한다. 이 방식은 streaming model이 단순히 최근 audio만 보는 것이 아니라, 앞쪽 context를 retrieval해야 하는 상황을 만들어 준다.

### 5) Four-stage conversion recipe

Training recipe는 네 단계로 구성된다.

| Stage | Role |
| --- | --- |
| Format training | Offline data로 target sequence format과 streaming control token 사용법 학습 |
| Adapter training | Chunk-wise acoustic representation을 LLM space로 mapping |
| Large-scale streaming SFT | Audio understanding, ASR, spoken dialogue 등 core capability 학습 |
| Instruction-following fine-tuning | Continuous assistance, comprehension-aware intervention, proactive response 학습 |

이 recipe는 기존 offline model을 streaming interaction model로 conversion하는 관점에서 중요하다. 처음부터 audio interaction model을 pretrain하는 것이 아니라, 강한 omni model을 기반으로 streaming decision layer와 data format을 추가한다.

### 6) FIFO-scheduled asynchronous inference

Streaming deployment에서는 audio encoder와 LLM decoder가 서로 다른 cadence로 돈다. Encoder는 incoming chunk를 계속 처리해야 하고, decoder는 silent token을 빠르게 내거나 response를 길게 생성할 수 있다. 이 둘이 동기적으로 묶이면 response generation 중 listening이 stall될 수 있다.

SoundFlow는 FIFO queue를 둔다.

- Encoder는 audio chunk feature를 시간 순서대로 queue에 append한다.
- Decoder가 말하고 있지 않을 때 queued feature를 소비한다.
- Decoder가 response를 생성하는 동안에도 encoder는 새 chunk를 받아 queue에 쌓을 수 있다.

Ablation table 기준으로 Audio-Interaction setting은 average first-chunk latency 392 ms와 stall 0.0 percent를 보고한다. 비교 setting은 831 ms와 stall 5.2 percent로 나타난다. 표의 row label은 최종 발행 전 PDF 기준으로 다시 확인하는 편이 좋지만, 핵심 메시지는 encoder-decoder decoupling이 real-time interaction에서 단순 optimization이 아니라 안정성 조건이라는 점이다.

# 4. Training / Data / Recipe

## 4-1. Data

StreamAudio-2M은 이 논문에서 model만큼 중요한 구성 요소다. 기존 audio dataset이 짧은 clip 중심이라면, StreamAudio-2M은 streaming instruction-following을 위해 long-form interaction sequence를 만든다.

논문 본문 overview에서는 StreamAudio-2M을 2.6M items, 302K hours로 소개하고, 7 major categories와 28 sub-tasks를 포함한다고 설명한다. 다만 상세 통계 figure와 repository 설명에서는 item count와 hour count가 다르게 읽히는 부분이 있어, 최종 발행 전 원문 PDF의 figure/table 기준으로 숫자를 다시 확인할 필요가 있다.

논문에서 제시하는 category는 다음처럼 정리할 수 있다.

| Capability | Main role |
| --- | --- |
| Voice chatting | Speech-based multi-turn interaction |
| Streaming instruction following | Audio stream 위에서 instruction 수행 |
| Streaming audio understanding | Sound, music, speech context 이해 |
| Streaming translation | Speech translation 및 simultaneous interpretation |
| Real-time ASR | Chunk-level transcription |
| Proactive response | Explicit query 없이 필요한 순간 개입 |
| Environment-aware audio agent | 생활 환경과 주변 event 기반 응답 |

Source 측면에서는 CommonVoice, GigaSpeech, LibriSpeech, VoxPopuli, CoVoST2, AISHELL, FMA, AudioSet, MOSS, MUSAN, WHAM!, DNS-Challenge 등 여러 공개 corpus를 조합한다. Text-only instruction source는 TTS로 speech화하고, ASR checking과 rewriting으로 spoken-form supervision을 정리한다.

이 data pipeline의 핵심은 단순히 양을 키우는 것이 아니다. StreamAudio-2M은 model이 언제 대답하지 않아야 하는지도 학습하도록 silent turn과 no-response cue를 명시적으로 포함한다. Repository sample format에서도 response가 필요 없는 turn은 assistant field에 no-response marker를 넣는 구조를 보여준다.

## 4-2. Training strategy

Training strategy는 offline capability 보존과 streaming behavior 획득 사이의 trade-off를 관리하는 쪽에 가깝다.

- Base는 Qwen2.5-Omni-3B다.
- Streaming chunk size는 main setting에서 400 ms다.
- Control token prediction을 별도 objective로 둔다.
- Large-scale streaming SFT로 basic audio capability를 유지한다.
- 마지막 stage에서 proactive response와 comprehension-aware intervention을 강화한다.

특히 dual-loss weight ablation이 중요하다. Streaming control token의 weight를 높이면 trigger accuracy는 좋아질 수 있지만, 너무 높이면 general comprehension 성능이 떨어질 수 있다. 이 논문은 streaming interaction을 추가해도 기존 LALM capability를 크게 잃지 않는 지점을 찾으려 한다.

## 4-3. Engineering notes

실무적으로 재사용할 때 볼 지점은 네 가지다.

1. Chunk size는 latency만의 문제가 아니다.
   - 0.2 s chunk는 latency는 낮지만 semantic context가 부족해 성능이 크게 떨어진다.
   - 0.6 s와 0.8 s는 accuracy를 회복하지만 latency가 커진다.
   - 0.4 s는 논문 ablation에서 accuracy-latency trade-off가 가장 균형 잡힌 operating point로 제시된다.

2. Silent data가 production quality를 좌우한다.
   - Voice assistant에서 false positive는 UX를 바로 망친다.
   - 따라서 positive trigger event만 많이 넣는 것보다, 침묵해야 하는 ambiguous sound를 잘 구성하는 것이 중요하다.

3. Data stitching은 model shortcut을 만들 수 있다.
   - Boundary artifact를 제거하지 않으면 model이 실제 semantic event가 아니라 편집 흔적을 trigger signal로 볼 수 있다.
   - TFJP와 hierarchical event selection은 이 shortcut을 줄이기 위한 장치로 읽힌다.

4. Inference scheduling은 architecture만큼 중요하다.
   - Streaming audio에서는 encoder가 계속 입력을 받고 decoder가 response를 생성한다.
   - 이 path를 decouple하지 않으면 real-time listening continuity가 깨질 수 있다.

# 5. Evaluation

## 5-1. Main results

논문은 8개 benchmark 축에서 Audio-Interaction을 평가한다. 크게 보면 mainstream offline audio capability를 유지하는지, 그리고 offline LALM이 표현하기 어려운 streaming capability를 얻는지 두 방향으로 본다.

### MMAU

MMAU table에서 Audio-Interaction 3B는 audio instruction setting 평균 58.15를 기록한다. 같은 표에서 Qwen2.5-Omni-3B의 audio instruction 평균은 42.51, Qwen2.5-Omni-7B는 49.58이다. 이 결과는 이 논문의 주장을 잘 보여준다. Text instruction이 아니라 spoken/audio instruction이 들어왔을 때도 model이 크게 무너지지 않는다는 점이 중요하다.

다만 text instruction setting에서는 Qwen2.5-Omni-7B가 더 높고, Audio-Interaction이 모든 metric에서 SOTA라고 읽으면 안 된다. 이 논문의 강점은 offline audio benchmark 하나를 압도하는 것이 아니라, streaming behavior를 추가하면서도 mainstream audio understanding을 크게 잃지 않는 데 있다.

### Spoken dialogue

Spoken dialogue benchmark에서는 Audio-Interaction 3B가 Llama Questions 67.31, Web Questions 54.34, AlpacaEval 4.28, SD-QA 52.14를 기록한다. Qwen2.5-Omni-7B 같은 larger baseline이 일부 score에서는 더 강하다. 따라서 여기서 메시지는 "dialogue SOTA"가 아니라, streaming conversion 이후에도 spoken QA와 voicebench류 ability가 유지된다는 쪽에 가깝다.

### ASR and speech translation

LibriSpeech WER에서는 Audio-Interaction이 clean 3.17, other 6.04로 specialized ASR model보다 약하다. Canary 계열이나 Qwen2-Audio 계열이 ASR 자체는 더 강하다. 반면 CoVoST2 speech-to-text translation에서는 en-zh 55.22, zh-en 35.21로 Qwen2.5-Omni-3B initialization보다 크게 오른다.

이 결과는 trade-off를 잘 보여준다. Chunk-wise streaming decoder로 가면 pure ASR head처럼 최적화된 model보다는 WER가 나빠질 수 있다. 대신 streaming translation과 audio instruction robustness 같은 interactive capability가 생긴다.

### Proactive-Sound-Bench

Proactive-Sound-Bench는 이 논문에서 가장 중요한 평가라고 볼 수 있다. 이 benchmark는 644개 human-designed acoustic event로 구성되고, 모델이 trigger해야 하는지 abstain해야 하는지, trigger했다면 어떤 response를 내야 하는지 본다.

Audio-Interaction은 average Single 61.2, Multi 62.8을 기록한다. 비교군 중 MiniCPM-o-4.5는 58.9와 58.9, Qwen2.5-Omni-7B는 58.2와 32.1로 나타난다. 특히 Multi tier에서 Audio-Interaction이 상대적으로 안정적인 점이 중요하다. Multi tier는 같은 category의 event를 이어 붙이고 distractor를 포함하므로, 단순 event classification보다 지속적인 intervention policy를 더 요구한다.

## 5-2. What really matters in the experiments

### 1) Audio instruction robustness가 핵심이다

MMAU에서 text instruction과 audio instruction gap을 같이 봐야 한다. 기존 audio LLM은 audio clip은 잘 이해해도, instruction 자체가 speech/audio form으로 들어오면 성능이 크게 흔들릴 수 있다. Audio-Interaction은 streaming training을 통해 audio instruction setting에서 강한 결과를 보인다.

이건 실제 voice assistant 관점에서 중요하다. 사용자는 text prompt를 입력하지 않는다. 대부분의 instruction은 speech로 들어오고, 동시에 background sound도 섞인다. 따라서 audio instruction robustness는 demo 성능이 아니라 product assumption에 가깝다.

### 2) Proactive response는 별도 능력이다

Proactive response는 audio captioning이나 sound event detection과 다르다.

- Sound event detection은 "무슨 소리가 났는가"를 본다.
- Audio captioning은 "무슨 상황인가"를 말한다.
- Proactive response는 "지금 말을 걸어야 하는가"와 "말한다면 무엇을 말해야 하는가"를 본다.

이 차이가 이 논문의 좋은 문제 설정이다. 예를 들어 smoke alarm을 들었을 때, 모델은 "beeping sound"라고 captioning하는 데서 끝나면 안 된다. 위험을 인식하고 사용자가 silent 상태여도 경고해야 한다. 반대로 일반 생활 소음에는 계속 침묵해야 한다.

### 3) Chunk size ablation은 system design issue다

Table 7의 chunk size ablation은 꽤 실용적이다.

| Chunk size | Alpaca | MMAU | Latency |
| --- | ---: | ---: | ---: |
| 0.2 s | 3.41 | 49.74 | 258 ms |
| 0.4 s | 4.28 | 58.15 | 392 ms |
| 0.6 s | 4.27 | 58.46 | 674 ms |
| 0.8 s | 4.30 | 59.13 | 786 ms |

0.2 s는 빠르지만 context가 부족해 성능이 무너진다. 0.6 s와 0.8 s는 성능은 올라가지만 latency가 커진다. 0.4 s는 성능과 latency 사이에서 꽤 좋은 compromise다. 실제 서비스에서도 이 type의 knob는 model quality와 UX latency를 동시에 움직인다.

### 4) Streaming behavior는 한두 module이 아니라 data, objective, inference가 같이 만든다

Ablation에서 V2 streaming SFT만 넣어도 trigger accuracy가 92.42 percent까지 올라간다. 하지만 TFJP preprocessing을 제거하면 85.35 percent, event selection을 제거하면 88.51 percent로 떨어진다. Full Audio-Interaction은 96.77 percent를 기록한다.

이 결과는 streaming trigger가 그냥 special token 하나 추가한다고 생기는 능력이 아니라는 뜻이다. Boundary smoothing, coherent event composition, silent supervision, control token objective가 함께 작동한다.

### 5) Internal analysis도 흥미롭다

논문은 streaming model이 chunk 간 continuity를 어디에서 복원하는지도 분석한다. Encoder output 단계에서는 chunk boundary가 분절되어 있지만, GPT Layer 0에서 continuity ratio가 크게 올라간다고 보고한다. 또 silent와 response control token decision에 대해 single attention head가 강하게 관여한다는 분석도 제시한다.

이 부분은 아직 mechanism claim으로 과하게 읽기보다는, 후속 interpretability study의 출발점으로 보는 편이 좋다. 다만 streaming objective가 model 내부에 task-independent decision pathway를 만들 수 있다는 관찰은 꽤 흥미롭다.

# 6. Limitations

1. Proactive intervention policy는 안전과 UX가 걸린 문제다.

   - 언제 먼저 말해야 하는지는 benchmark label로 완전히 고정하기 어렵다.
   - Smoke alarm, prolonged coughing, glass shattering 같은 event는 개입이 필요할 수 있지만, context에 따라 false alarm도 생길 수 있다.
   - 실제 product에서는 false positive와 false negative cost가 category별로 다르기 때문에 thresholding과 policy layer가 추가로 필요하다.

2. Dataset composition의 synthetic gap을 봐야 한다.

   - StreamAudio-2M은 real dataset, TTS, generated sound effect, synthetic event composition을 함께 사용한다.
   - 이런 방식은 scale과 coverage에 유리하지만, 실제 microphone, room impulse, device noise, overlapping speech distribution과는 차이가 날 수 있다.
   - 특히 proactive response는 rare event가 많기 때문에 synthetic rare-event clip의 realism이 중요하다.

3. ASR 최적 model은 아니다.

   - LibriSpeech WER에서는 specialized ASR baseline이 더 강하다.
   - Audio-Interaction은 one-model streaming interaction을 얻는 대신, pure ASR만 최적화한 model과 비교하면 손해가 있다.
   - 따라서 ASR production system을 대체한다기보다, ASR, audio understanding, proactive interaction을 하나로 묶는 방향으로 읽는 편이 맞다.

4. Benchmark가 실제 deployment를 완전히 대변하지는 않는다.

   - Proactive-Sound-Bench는 중요한 시작점이지만, real household, car cabin, factory floor, call center 같은 deployment domain은 noise와 privacy condition이 다르다.
   - Multi-user environment에서 누구에게 말해야 하는지, 어떤 device action을 해야 하는지는 별도 문제다.

5. Reproducibility는 공개 범위를 확인해야 한다.

   - Repository는 inference code, model weights, dataset 관련 resource를 공개하고 있다.
   - 다만 README 기준으로 full dataset과 data curation pipeline은 coming 항목으로 남아 있다.
   - 논문 수치 재현을 위해서는 정확한 dataset version, filtering rule, training config 공개 상태를 확인해야 한다.

6. Trigger decision이 좁은 pathway에 의존할 수 있다.

   - 논문 내부 분석은 silent-response decision에 single important head가 관여한다는 관찰을 제시한다.
   - 이것은 interpretability 관점에서는 흥미롭지만, robustness 관점에서는 특정 pathway perturbation에 민감할 가능성도 뜻한다.
   - Safety-critical audio assistant라면 이 decision path의 calibration과 adversarial robustness를 별도로 봐야 한다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 multimodal model을 볼 때 중요한 기준을 하나 더 추가한다. 지금까지 audio LLM은 주로 "무엇을 들었는가"를 중심으로 평가했다. 그런데 실제 assistant나 agent에서는 "언제 말할 것인가"가 거의 같은 수준으로 중요하다.

이 관점은 audio뿐 아니라 video stream, UI agent, robot perception에도 연결된다. Long-running agent는 계속 관찰하지만, 모든 frame이나 event에 반응하면 안 된다. 대부분은 state를 업데이트하고 침묵해야 하며, 특정 조건에서만 action이나 response를 내야 한다. Audio-Interaction은 이 pattern을 audio domain에서 비교적 선명하게 보여준다.

특히 실무적으로는 아래 질문이 중요해진다.

- 모델이 observation을 받는 cadence와 response를 내는 cadence가 같은가.
- Silent state를 explicit하게 학습하고 평가하는가.
- Positive event만 학습하는 것이 아니라, no-response event를 충분히 넣었는가.
- Runtime에서 perception path와 generation path가 서로 stall을 만들지 않는가.

이 논문은 이 네 질문을 모두 건드린다.

## 7-2. Reuse potential

재사용하고 싶은 포인트는 아래 다섯 가지다.

1. Silent token을 first-class target으로 두기

   - Interactive agent에서는 no-op이 중요한 action이다.
   - Audio든 video든 UI든, 대부분의 time step에서 해야 하는 일은 "기다리기"다.

2. Proactive benchmark 설계

   - 기존 recognition benchmark와 별개로, intervention decision을 평가하는 benchmark가 필요하다.
   - Trigger accuracy뿐 아니라 response quality와 abstain quality를 같이 봐야 한다.

3. Long-form stream synthesis

   - Short clip dataset을 그대로 쓰는 대신, scenario-level planning으로 coherent stream을 만들 수 있다.
   - 이 idea는 document workflow나 UI automation trace에도 응용 가능하다.

4. History review training

   - Long stream에서 앞쪽 context를 잊는 문제를 explicit QA로 찌르는 방식은 꽤 범용적이다.
   - Streaming video, meeting assistant, browsing agent에도 비슷하게 쓸 수 있다.

5. Encoder-decoder scheduling을 method로 보기

   - Real-time system에서는 model architecture와 serving schedule이 분리되지 않는다.
   - FIFO queue나 asynchronous path는 논문 부록의 engineering detail이 아니라 actual capability를 만드는 조건이다.

## 7-3. Follow-up papers

후속으로 같이 보면 좋은 논문과 system은 아래 정도다.

- Qwen2.5-Omni technical report
- Qwen2-Audio technical report
- Moshi
- Freeze-Omni
- LLaMA-Omni2
- VoiceBench
- Seamless streaming speech translation
- Audio-Reasoner

특히 Moshi나 full-duplex voice chat 계열과 비교하면, Audio-Interaction이 어떤 점에서 voice conversation system이고 어떤 점에서 broader audio interaction model인지 더 명확해진다. 또 Qwen2.5-Omni를 같이 보면, base omni model 위에 streaming interaction behavior를 얹는 conversion recipe가 더 잘 보인다.

# 8. Summary

- Audio Interaction Model은 audio LLM을 offline clip QA model이 아니라 always-on streaming interaction model로 바꾸려는 논문이다.
- Audio-Interaction은 400 ms chunk마다 silent 또는 response를 결정하고, ASR, translation, audio understanding, voice chat, proactive response를 하나의 loop로 묶는다.
- SoundFlow의 핵심은 streaming-native data construction, comprehension-aware silence training, FIFO-scheduled asynchronous inference다.
- 실험에서는 MMAU audio instruction 58.15, Proactive-Sound-Bench Single 61.2와 Multi 62.8 등으로 mainstream audio capability를 유지하면서 proactive interaction capability를 보여준다.
- 이 논문은 audio recognition paper라기보다, real-time multimodal assistant를 만들 때 no-op, trigger, latency, data composition을 어떻게 설계할지 보여주는 system paper에 가깝다.
