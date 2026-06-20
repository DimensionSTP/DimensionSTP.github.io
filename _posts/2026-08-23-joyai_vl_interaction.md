---
layout: single
title: "JoyAI-VL-Interaction: Real-Time Vision-Language Interaction Intelligence Review"
categories: Study-concept
tag: [JoyAI-VL-Interaction, VLM, StreamingVideo, MultimodalAI, AIAgent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.14777)

[Project page](https://joyai-vl-video-future-academy-jd.github.io/JoyAI-VL-Interaction/)

[Code link](https://github.com/jd-opensource/JoyAI-VL-Interaction)

[Model link](https://huggingface.co/jdopensource/JoyAI-VL-Interaction-Preview)

[Dataset link](https://huggingface.co/datasets/jdopensource/JoyAI-VL-Interaction)

JoyAI-VL-Interaction을 "웹캠 영상을 실시간으로 보는 8B VLM" 정도로 읽으면 핵심을 놓치기 쉽다. 이 논문이 실제로 바꾸려는 것은 vision encoder나 language backbone보다 **interaction contract**다. 기존 multimodal assistant는 사용자가 질문해야 눈을 뜨고, 질문 하나에 답한 뒤 다시 기다린다. 반면 이 논문은 모델이 지속적으로 장면을 보고, 매 시점마다 지금 말할지, 계속 침묵할지, 더 무거운 background model에 일을 넘길지를 스스로 결정하게 만든다.

이 차이는 단순 latency 개선이 아니다. Turn-based model은 응답 생성이 아무리 빨라도 user turn이 없으면 행동을 시작하지 않는다. Monitoring, live translation, counting, step-by-step guidance처럼 사건이 먼저 발생하고 사용자는 제때 질문하기 어려운 환경에서는, 병목이 "얼마나 빨리 답하는가"가 아니라 **무엇을 보고 언제 개입할 것인가**로 이동한다.

JoyAI-VL-Interaction은 이 문제를 model, data, training, memory, serving을 함께 설계하는 방식으로 푼다. Model은 1초 단위로 `</silence>`, `</response>`, `</delegate>` 중 하나를 선택한다. Data는 4M개가 넘는 time-aligned interaction clip으로 구성된다. AdaCodec은 변화가 적은 frame의 token cost를 낮춘다. Hierarchical memory와 vLLM prefix reuse는 긴 stream을 계속 처리하게 한다. 복잡한 문제는 asynchronous background agent로 넘기되, foreground model은 현재 장면을 계속 본다.

> 한 줄 요약: JoyAI-VL-Interaction은 vision stream을 지속적으로 관찰하며 매초 silence, response, delegation을 선택하도록 학습한 8B interaction model이고, time-aligned data, predictive video codec, hierarchical memory, asynchronous background loop, vLLM serving을 결합해 turn-based VLM을 always-on visual assistant로 바꾸려는 full-stack system이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Real-time multimodal AI의 병목을 response latency가 아니라 **response timing policy**로 재정의한다.
- Silence를 output 부재가 아니라 학습해야 하는 first-class action으로 다룬다.
- Model이 빠른 interaction을 맡고, background model이 느린 reasoning을 맡는 two-loop architecture를 구체화한다.
- Video tokenization, memory consolidation, KV cache reuse를 하나의 serving path로 연결한다.
- Model weight, data, training recipe, deployment code를 함께 공개해 interaction model을 재현 가능한 engineering problem으로 만든다.

이 글에서는 이 논문을 범용 VLM leaderboard paper라기보다, **event-driven visual interaction을 위한 model policy와 runtime contract를 함께 정의한 system paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

일반적인 turn-based multimodal assistant의 실행 흐름은 다음과 같다.

1. 사용자가 질문한다.
2. System이 그 시점의 image, video clip, audio를 수집한다.
3. Model이 답변을 생성한다.
4. 다음 user turn까지 멈춘다.

이 구조는 static image QA나 video chat에는 자연스럽다. 하지만 현실의 중요한 event는 user turn과 맞춰 발생하지 않는다.

- 사람의 낙상은 질문을 기다리지 않는다.
- 화재나 침입 징후는 monitoring prompt가 들어오는 순간에만 나타나지 않는다.
- Live subtitle이나 game event는 지나간 뒤 질문하면 timing value가 사라진다.
- Cooking guidance는 장면의 상태 변화에 맞춰 다음 지시를 내야 한다.
- Long stream에서는 대부분의 시점에 말하지 않는 것이 맞다.

따라서 interaction model의 핵심 문제는 content generation 하나가 아니다. 시점 $t$에서 누적된 visual stream, user context, memory를 보고 action을 선택해야 한다.

$$
a_t = \pi_{\theta}(v_{\le t}, q_{\le t}, m_t)
$$

여기서 action space는 다음과 같다.

$$
a_t \in \{\text{silence}, \text{response}, \text{delegate}\}
$$

- `silence`: 지금은 말할 이유가 없으므로 계속 관찰한다.
- `response`: 현재 event에 맞춰 즉시 답하거나 먼저 말을 건다.
- `delegate`: 실시간 model이 처리하기 어려운 subtask를 background model이나 agent에 넘긴다.

이 setting에서는 정답 text만 맞히는 것으로 충분하지 않다. 같은 문장을 5초 늦게 말하면 monitoring task에서는 실패할 수 있다. 반대로 맞는 답을 너무 자주 말하면 false alarm과 interruption cost가 커진다. Interaction quality는 대략 content quality와 timing quality의 결합으로 봐야 한다.

## 1-2. Why previous approaches are insufficient

### 1) Faster turn-based model

Speech-to-speech model이나 omni model은 user utterance 이후의 latency를 크게 줄일 수 있다. 그러나 user turn을 기다리는 interaction contract가 그대로라면, visual event가 먼저 발생하는 proactive setting은 해결되지 않는다.

즉 다음 두 문장은 다르다.

- "질문을 받은 뒤 빨리 답한다."
- "질문이 없어도 지금 답할 가치가 있는지 판단한다."

JoyAI-VL-Interaction이 겨냥하는 것은 두 번째다.

### 2) External polling

기존 video-call product에 periodic trigger를 붙이면 몇 초마다 scene을 점검할 수 있다. 구현은 간단하지만 event detection latency가 polling interval 아래로 내려가기 어렵다. 더 중요한 문제는 "왜 지금 말해야 하는가"가 model policy가 아니라 external timer와 heuristic에 의해 결정된다는 점이다.

Polling interval을 줄이면 더 자주 model을 호출하므로 cost와 false response가 늘어난다. Interval을 늘리면 중요한 event를 늦게 잡는다. 이는 단순 serving issue가 아니라 decision boundary를 어디에 둘 것인가의 문제다.

### 3) Offline video understanding

많은 streaming video 연구는 frame이 순차적으로 들어온다는 조건을 사용하지만, 평가는 최종 QA accuracy나 offline benchmark에 머무르는 경우가 많다. 이 경우 다음 능력이 충분히 측정되지 않는다.

- Event가 발생한 정확한 시점에 반응하는가.
- 아무 일도 없을 때 침묵하는가.
- 수십 분 전의 scene을 기억하는가.
- Background task가 진행되는 동안 foreground stream을 계속 처리하는가.
- 실제 serving stack에서 지속 가능한 latency를 유지하는가.

### 4) One-model-does-everything design

실시간 model 하나에 broad knowledge, long reasoning, speech generation, tool use, memory를 모두 넣으면 runtime이 무거워진다. 반대로 작은 model만 사용하면 open-ended answer quality가 약해질 수 있다.

이 논문은 이 trade-off를 foreground interaction policy와 background intelligence의 분리로 푼다. 중요한 것은 작은 model이 모든 질문에 답하는 것이 아니라, **지금 직접 답할 문제와 넘겨야 할 문제를 구분하는 것**이다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 네 가지로 정리할 수 있다.

1. **Interaction decision을 model 내부로 이동**
   - Model은 매초 silence, response, delegation을 직접 선택한다.
   - External voice activity detector나 periodic trigger가 response timing을 결정하지 않는다.

2. **Time-aligned interaction data construction**
   - 단순 video-caption pair가 아니라, 어느 초에 말하고 어느 초에 침묵해야 하는지 label을 만든다.
   - Silence를 negative space가 아니라 explicit supervision으로 다룬다.

3. **Real-time foreground와 asynchronous background의 결합**
   - Foreground model은 장면을 계속 보면서 immediate interaction을 유지한다.
   - 어려운 reasoning이나 artifact generation은 background model, API, agent로 넘긴다.

4. **Long stream을 위한 deployable system**
   - AdaCodec으로 predictable frame의 token cost를 줄인다.
   - Short-term visual context, mid-term summary, long-term memory를 계층적으로 관리한다.
   - Stable text prefix를 만들어 vLLM KV cache reuse와 연결한다.

## 2-2. Design intuition

이 논문의 설계 직관은 "interactivity 자체를 별도 capability로 scale해야 한다"는 주장에 가깝다.

기존 model scaling은 주로 다음 축에 집중해 왔다.

- 더 많은 parameter
- 더 많은 pre-training data
- 더 긴 context
- 더 많은 test-time compute
- 더 강한 tool use

JoyAI-VL-Interaction은 여기에 다른 축을 추가한다.

- 적절한 순간을 감지하는 능력
- 대부분의 순간에 침묵하는 능력
- 시간이 흐르고 있다는 사실을 행동에 반영하는 능력
- 현재 interaction을 멈추지 않고 복잡한 task를 비동기로 넘기는 능력

이 관점에서 silence는 실패가 아니다. 오히려 always-on assistant에서는 silence가 가장 빈번하고 중요한 action이다. Proactive model이 유용하려면 "말할 수 있음"보다 "말하지 않아야 할 때를 앎"이 먼저 안정되어야 한다.

또 하나의 직관은 intelligence와 interaction latency를 같은 model size로 해결하지 않는다는 점이다. Foreground model은 빠른 perception과 timing policy에 집중한다. Background model은 필요할 때만 호출되어 깊은 reasoning을 수행한다. 이를 간단히 쓰면 다음과 같다.

$$
\text{fast presence} + \text{selective delegation} \rightarrow \text{responsive system}
$$

여기서 delegation은 fallback이 아니라 architecture의 정상 경로다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Continuous visual stream에서 적절한 순간에 자율적으로 반응 |
| Base model | Qwen3-8B 기반 language model, Qwen3-VL ViT 기반 visual encoder |
| Core action | `silence`, `response`, `delegate` |
| Video encoding | AdaCodec 기반 reference frame plus compact predictive token |
| Interaction cadence | 기본적으로 1초 단위 decision |
| Memory | Short-term visual context, mid-term text summary, long-term compressed memory |
| Background path | External model, API, agent로 asynchronous delegation |
| Serving | OpenAI-compatible request path와 vLLM prefix reuse |
| System boundary | Model이 timing을 결정하고 ASR, TTS, UI, memory service는 교체 가능 |

## 3-2. Module breakdown

### 1) JoyAI-VL 1.0 backbone

JoyAI-VL-Interaction은 JoyAI-VL 1.0에서 시작한다.

- Language model은 Qwen3-8B로 초기화한다.
- Visual encoder는 Qwen3-VL ViT를 사용한다.
- Vision-language projection layer는 처음부터 학습한다.
- Base model은 representation alignment, vision-language pre-training, OPD와 RL을 포함한 post-training을 거친다.
- 이후 interaction-specific continue training과 RL로 per-second behavior를 학습한다.

이 구분이 중요하다. Interaction capability는 backbone architecture를 완전히 새로 만드는 데서 나오기보다, **time-aligned data와 control-token policy를 post-training에 주입하는 방식**으로 형성된다.

### 2) Native interaction action space

매 step에서 model은 세 가지 control path 중 하나를 선택한다.

| Control token | Runtime behavior | Main risk |
| --- | --- | --- |
| `</silence>` | 아무 말 없이 stream 관찰을 계속함 | 중요한 event를 놓침 |
| `</response>` | Text reply를 즉시 생성함 | false alarm, 과도한 개입 |
| `</delegate>` | Background task를 비동기로 시작함 | 불필요한 cost, tool risk |

Control token을 반드시 backtick 안에 쓰는 이유는 Jekyll이 angle bracket을 HTML tag처럼 해석하지 않게 하기 위해서다.

Model policy의 진짜 난점은 response quality보다 class imbalance다. 긴 stream에서 silence step이 압도적으로 많다. Naive SFT를 적용하면 model이 계속 침묵하는 쪽으로 쉽게 무너질 수 있다. 반대로 response를 과도하게 강화하면 always-respond policy가 된다. 논문은 weighted objective와 RL을 함께 사용해 이 균형을 맞춘다.

### 3) AdaCodec

Continuous video를 모든 frame마다 full ViT token으로 변환하면 token budget이 frame count에 거의 비례해 늘어난다. AdaCodec은 video codec의 predictive coding과 비슷한 방향을 사용한다.

- Scene의 기준이 되는 reference frame에는 full visual token을 사용한다.
- 변화가 예측 가능한 frame에는 motion과 residual을 담은 compact P-token을 사용한다.
- Prediction cost가 커지면 새로운 reference frame을 연다.
- 논문은 predictable frame당 약 16개 token을 사용한다고 보고한다.

이 설계의 핵심은 고정 frame rate compression이 아니라 **scene change에 따라 visual token budget을 배분한다는 것**이다. 정적인 감시 화면은 싸게 처리하고, 변화가 큰 순간에는 더 많은 visual detail을 사용한다.

AdaCodec이 interaction policy와 잘 맞는 이유도 여기에 있다. Model이 반응해야 하는 시점은 대개 scene change나 event onset과 연결된다. Codec도 변화가 커지는 시점에 reference frame을 갱신하므로, token allocation과 decision difficulty가 어느 정도 같은 방향을 가진다.

### 4) Two concurrent loops

System은 두 개의 loop를 동시에 실행한다.

#### Foreground loop

- Camera 또는 RTSP stream을 수신한다.
- 기본 1 Hz로 frame을 sampling한다.
- Current frame, active query, recent interaction, memory를 model input으로 만든다.
- Model은 silence, response, delegate를 선택한다.
- Response는 text로 표시되거나 TTS로 합성된다.

#### Background loop

- Foreground model이 hard task를 발견하면 tagged request를 만든다.
- Bridge는 task ID, question, foreground note, bounded frame snapshot을 정규화한다.
- External model이나 agent를 timeout이 있는 isolated task로 실행한다.
- Result는 started, ready, error event로 돌아온다.
- Full artifact 전체를 context에 넣지 않고 bounded digest만 foreground history에 다시 주입한다.

핵심은 background result를 기다리는 동안 foreground가 멈추지 않는다는 점이다. Model은 계속 scene을 보고, 새 질문에 답하거나 침묵할 수 있다. 이 구조는 "작은 model plus 큰 model" 조합보다 **latency class가 다른 두 policy를 병렬로 운영하는 runtime**에 가깝다.

### 5) Pluggable speech I/O

JoyAI-VL-Interaction은 vision-first model이고 speech를 backbone 안에 통합하지 않는다.

- ASR은 audio를 text query로 변환한다.
- TTS는 model response를 speech로 변환한다.
- Interaction decision은 visual-language model이 담당한다.
- ASR과 TTS model 또는 API는 deployment 환경에 맞게 교체할 수 있다.

이 선택은 end-to-end omni model보다 modality integration은 약할 수 있지만, system modularity가 높다. Language, voice, latency requirement가 다른 deployment에서 ASR과 TTS를 독립적으로 바꿀 수 있다.

논문은 이전 utterance가 재생 중일 때 다음 response를 text-only로 내보내 audio queue가 쌓이지 않게 한다. Live commentary처럼 output frequency가 높은 setting에서는 TTS를 끄는 것을 권장한다. 이는 실시간 assistant에서 generation throughput뿐 아니라 **speech playback time 자체가 bottleneck**이라는 현실을 반영한다.

### 6) Hierarchical memory

긴 video stream을 raw token으로 계속 쌓을 수 없으므로 memory를 세 층으로 나눈다.

1. Short-term memory
   - 최근 $T_s$초의 raw visual token을 유지한다.
   - 현재 event와 precise timing을 담당한다.

2. Mid-term memory
   - Short-term chunk를 text summary로 압축한다.
   - 최대 $M$개 summary를 유지한다.

3. Long-term memory
   - 여러 mid-term summary를 다시 compressed block으로 합친다.
   - 최대 $L$개 block을 유지한다.

각 horizon은 다음처럼 볼 수 있다.

$$
T_m = M T_s
$$

$$
T_l = L M T_s
$$

Evaluation configuration은 $T_s = 100$ seconds, $M = 5$, $L = 15$를 사용한다. 단순 계산으로 long-term range는 약 7,500초이고, 논문은 이를 약 2시간 규모의 continuous context로 설명한다.

Visual memory와 dialogue memory를 분리하고 consolidation을 asynchronous하게 수행한다는 점도 중요하다. Summary text는 chunk 안에서 stable prefix가 되므로, 매 step마다 전체 history를 다시 prefill하지 않고 vLLM의 KV cache를 재사용할 수 있다.

### 7) vLLM-native serving

Streaming context management는 model architecture만큼 serving engine과의 compatibility가 중요하다.

- Full history recomputation은 context와 prefill cost를 빠르게 늘린다.
- Sliding window는 context를 제한하지만 step마다 prefix가 달라져 cache reuse를 깨뜨릴 수 있다.
- Aggressive token pruning도 stable prefix를 보장하지 않는다.

JoyAI-VL-Interaction은 memory summary를 chunk 단위의 stable text prefix로 만들고, current frame과 직전 response만 incremental input으로 추가한다. AdaCodec은 predictable frame의 visual token을 줄인다. 논문은 이 조합으로 standard vLLM에서 2시간이 넘는 video stream을 sub-second end-to-end latency로 처리했다고 보고한다.

다만 이 latency는 paper가 보고한 system-level result다. Hardware configuration, concurrent session 수, percentile latency, background task load에 따른 변화는 별도로 확인해야 한다.

# 4. Training / Data / Recipe

## 4-1. Data

Interaction model의 핵심 asset은 4M개가 넘는 time-aligned clip이다. 모든 data family를 같은 per-second action format으로 변환한다.

| Data family | 학습하려는 behavior | Timing supervision의 핵심 |
| --- | --- | --- |
| Alerting and anomaly detection | Event onset에 즉시 경고 | 최초 발생 frame을 찾음 |
| Time-aligned QA | Evidence가 나타날 때 답함 | backward, present, forward timing |
| Counting and perception over time | Event를 누적하고 시점별로 갱신 | 반복 등장과 누적 상태 추적 |
| Live commentary and narration | 장면 변화에 맞춰 자연스럽게 말함 | 실제 commentary cadence와 pause |
| Multi-turn casual chat | 긴 stream 위에서 대화를 유지함 | 현재 frame에 grounded된 turn 배치 |
| Delegation episodes | Hard task를 background로 넘김 | trigger, pending delay, result return |

### Backward, present, forward QA

Offline video QA를 streaming task로 바꾸기 위해 question timing을 세 유형으로 나눈다.

- Backward: evidence가 이미 나온 뒤 question이 들어오며 바로 답한다.
- Present: question과 evidence가 비슷한 시점에 나타나며 그때 답한다.
- Forward: question이 먼저 들어오고, model은 evidence가 나타날 때까지 침묵한 뒤 답한다.

Forward type이 특히 중요하다. 일반 QA model은 question을 받으면 즉시 답하려는 경향이 있지만, interaction model은 아직 evidence가 없다는 이유로 기다릴 수 있어야 한다.

### Silence as a label

Silence는 annotation이 없는 상태가 아니다. 각 초마다 `</silence>`를 명시적으로 부여한다. 이 덕분에 model은 다음 두 종류의 error를 모두 학습할 수 있다.

- False negative: 말해야 할 때 침묵
- False positive: 아무 일도 없는데 말함

Always-on system에서는 false positive가 사용자 피로와 신뢰 하락을 빠르게 만든다. 따라서 silence quality는 response quality만큼 중요하다.

### Multi-level verification

Time-aligned data는 content와 timing을 동시에 맞춰야 한다.

- Global verifier는 전체 frame sequence와 annotation의 일관성을 본다.
- Local verifier는 annotated timestamp 주변 frame과 response를 본다.
- 두 검사를 모두 통과한 example만 corpus에 넣는다.
- Alerting data는 candidate onset 이전의 1 fps frame을 다시 검사해 event가 더 일찍 나타났는지 확인한다.

이 pipeline은 synthetic data scale보다 temporal label precision을 우선한다. Interaction model에서 한두 초의 label noise는 일반 caption error보다 더 직접적으로 policy를 흐릴 수 있기 때문이다.

## 4-2. Supervised interaction training

JoyAI-VL 1.0에 time-aligned interaction data와 conventional turn-based data를 섞어 continue training을 수행한다.

문제는 class imbalance다. 대부분의 step이 silence이므로 standard cross-entropy를 그대로 쓰면 repeated silence token이 gradient를 지배한다. 논문은 assistant token의 역할에 따라 weight를 다르게 둔다.

$$
L(\theta) = -\frac{1}{|A|}\sum_{j \in A} w_j \log p_{\theta}(y_j \mid y_{<j})
$$

논문에 제시된 주요 weight는 다음과 같다.

- Silence run의 첫 `</silence>`: 1.0
- 반복되는 `</silence>`: 0.4
- `</response>` onset: 1.5
- 나머지 assistant token: 1.0

Delegation은 response 안에서 발생하므로 별도 weight를 두지 않는다. Conventional turn-based data에는 standard SFT loss를 사용한다.

이 objective가 해결하려는 것은 단순 imbalance가 아니다. Silence continuation은 같은 행동이 길게 반복되므로 token 수가 많고, response onset은 행동 전환점이지만 token 하나뿐이다. Weighting은 **행동 지속 token과 행동 전환 token의 credit 밀도 차이**를 보정한다.

## 4-3. Reinforcement learning

SFT만으로는 exact timing과 delegation boundary를 정밀하게 맞추기 어렵다. 논문은 EasyVideoR1 위에서 GRPO stage를 추가한다.

Reward가 다루는 항목은 다음과 같다.

- Correct response를 적절한 timing window 안에 냈는가.
- 말할 이유가 없을 때 침묵했는가.
- Easy task를 불필요하게 delegate하지 않았는가.
- Hard task를 무리하게 foreground에서 답하지 않고 delegate했는가.
- Background result가 돌아왔을 때 이를 올바르게 사용했는가.
- Pending 상태에서도 foreground interaction을 유지했는가.
- False alarm, mistimed response, always-respond behavior가 나타나지 않았는가.
- Response content가 task-specific rubric에 맞는가.

긴 stream을 그대로 rollout하면 대부분 silence step이 되어 learning signal이 희석된다. 이를 줄이기 위해 answer-centered window sampling을 사용한다.

- Gold response 주변의 causal turn만 유지한다.
- 수백 turn의 stream을 timing decision이 있는 짧은 trajectory로 압축한다.
- Response onset과 주변 silence에 credit을 집중한다.

이 recipe는 long-horizon policy optimization에서 모든 timestamp를 동일하게 학습하지 않고, **decision boundary 주변을 우선 sampling한다는 점**에서 실용적이다.

## 4-4. Delegation episode training

Delegation data는 단순 tool-call example보다 복잡하다.

1. Visual stream 안에서 hard subtask trigger가 나타난다.
2. Foreground model은 사용자에게 짧은 holding response를 준다.
3. Hidden delegate request를 background로 보낸다.
4. Random delay 동안 stream을 계속 관찰한다.
5. Background result가 돌아오면 현재 context 안에 다시 통합한다.
6. Foreground model이 natural response를 생성한다.

Random delay는 중요한 training device다. Delay가 없으면 model은 delegate 직후 바로 answer가 들어오는 synchronous tool-use pattern만 배울 수 있다. Variable delay를 주면 pending task가 있는 상태에서 새 scene과 user turn을 계속 처리하는 behavior를 학습한다.

## 4-5. Engineering notes

실제 구현에서 재사용할 만한 지점은 다음과 같다.

- Event timing label을 content label과 별도로 검증한다.
- Silence ratio를 단순 downsampling하기보다 onset token에 다른 weight를 준다.
- Foreground와 background context를 같은 raw transcript로 합치지 않고 bounded digest contract를 둔다.
- Background artifact는 외부 storage에 두고 model context에는 summary와 status만 넣는다.
- Memory consolidation은 inference critical path와 분리한다.
- Stable summary prefix를 만들어 KV cache reuse가 가능하게 한다.
- High-frequency response에서는 TTS playback queue를 별도 제어한다.
- Session reset, timeout, cancellation, stale frame dropping을 model quality와 같은 수준의 requirement로 본다.

# 5. Evaluation

## 5-1. Benchmark setup

논문은 offline video QA benchmark보다 실제 video-call product와의 head-to-head comparison을 선택한다.

총 58개 case를 여섯 scenario로 구성한다.

| Scenario | Number of cases |
| --- | ---: |
| Monitoring and alerting | 10 |
| Real-time counting | 10 |
| Real-time translation | 10 |
| Time awareness | 10 |
| Live commentary and guidance | 9 |
| Long-horizon memory | 9 |
| Total | 58 |

Baseline은 Doubao와 Gemini의 in-app video-call assistant다. 저자들은 2026년 5월 말부터 6월 초 사이의 product version을 사용했다고 설명한다.

Protocol은 다음과 같다.

- 같은 video input과 matched timestamp를 사용한다.
- Baseline이 system API를 제공하지 않으므로 실제 app UI를 통해 실행한다.
- 5명의 LLM researcher가 pairwise result를 평가한다.
- Quality와 timing을 각각 good, fair, poor로 평가한다.
- 두 축을 같은 weight로 합친다.
- System identity를 가리고 presentation order를 randomize한다.
- 최종적으로 JoyAI-VL-Interaction 기준 win, tie, loss를 계산한다.

이 평가에서 timing은 부가 metric이 아니다. 아무 일도 없을 때 침묵했는지, 너무 일찍 또는 늦게 반응하지 않았는지를 content quality와 같은 weight로 둔다.

## 5-2. Main results

### Overall pairwise result

| Baseline | Win | Tie | Loss |
| --- | ---: | ---: | ---: |
| Doubao | 77.6% | 17.2% | 5.2% |
| Gemini | 87.9% | 10.3% | 1.7% |

### Scenario-level win rate

| Scenario | vs Doubao | vs Gemini |
| --- | ---: | ---: |
| Monitoring and alerting | 100.0% | 100.0% |
| Real-time counting | 70.0% | 100.0% |
| Real-time translation | 80.0% | 100.0% |
| Time awareness | 80.0% | 50.0% |
| Live commentary and guidance | 55.6% | 100.0% |
| Long-horizon memory | 77.8% | 77.8% |

결과는 논문의 target regime에서 interaction policy가 강한 이점을 가질 수 있음을 보여준다. 특히 monitoring, counting, translation처럼 event와 output timing이 직접 연결되는 scenario에서 차이가 크다.

반면 time awareness의 Gemini 비교는 50.0% win, 40.0% tie, 10.0% loss로 가장 접전이다. 일부 case는 event가 지나간 뒤 사용자가 질문하는 형태여서 proactive timing보다 base model의 answer quality가 더 중요했다고 논문은 해석한다.

Doubao와의 live commentary 비교에서는 JoyAI-VL-Interaction의 win rate가 55.6%로 다른 scenario보다 낮다. 저자들은 larger product model의 knowledge, style, phrasing quality가 일부 case에서 유리했고, JoyAI-VL-Interaction은 commentary hallucination이 나타날 수 있다고 인정한다.

## 5-3. What really matters in the experiments

이 실험에서 가장 의미 있는 것은 전체 win rate 숫자 하나가 아니다.

### 1) Timing을 별도 평가 축으로 둠

Turn-based product와 interaction model의 차이를 보려면 final text quality만으로는 부족하다. 논문은 "맞는 말을 했는가"와 "맞는 순간에 했는가"를 분리한다. 이는 event-driven system 평가에서 재사용 가치가 높은 protocol이다.

### 2) Advantage zone을 직접 측정함

Monitoring, real-time translation, live commentary는 interaction model이 구조적으로 유리한 task다. 이 선택은 논문의 claim을 직접 검증한다는 장점이 있지만, broad multimodal intelligence를 평가하지는 않는다.

따라서 결과는 다음처럼 읽는 편이 안전하다.

- JoyAI-VL-Interaction이 모든 video assistant task에서 더 강하다는 증거는 아니다.
- Event-driven, timing-sensitive interaction에서는 turn-based product보다 유리할 수 있다는 증거다.

### 3) Product-level system을 비교함

실제 app을 사용한 비교는 deployment relevance가 높다. 동시에 baseline의 hidden model, polling policy, session limit, network condition을 통제하기 어렵다. 이는 model-only comparison이 아니라 product system comparison이다.

### 4) Memory result에 session cutoff가 개입함

논문은 Doubao가 voice input 없이 약 5분, Gemini가 약 2분 15초 뒤 session을 종료했고, memory case의 일부가 이 cutoff 이후에 놓였다고 설명한다. 이는 JoyAI-VL-Interaction의 always-on system advantage를 보여주지만, memory model 자체의 순수 성능과 product session policy가 섞인 결과다.

### 5) Emergent capability는 demonstration 수준임

Shopping app guidance와 slide-based lecture generation은 training data에 직접 포함되지 않았다고 보고된다. 흥미로운 사례지만 systematic out-of-distribution benchmark나 ablation으로 검증된 것은 아니다. "Emergence"는 가능성을 보여주는 qualitative evidence로 읽는 편이 적절하다.

# 6. Limitations

1. **평가 범위가 좁다**
   - 58개 case, 6개 scenario, 2개 product에 한정된다.
   - General VQA, broad knowledge, open-ended dialogue, multilingual robustness를 포괄하지 않는다.

2. **선별된 advantage-zone benchmark다**
   - Scenario가 proactive timing을 요구하도록 설계되어 interaction model에 유리하다.
   - 이는 정당한 target evaluation이지만 general superiority claim으로 확대하면 안 된다.

3. **Model과 system contribution이 분리되지 않는다**
   - Time-aligned data, weighted SFT, GRPO, AdaCodec, memory, vLLM cache reuse, baseline session policy가 모두 결과에 영향을 준다.
   - 각 component의 독립 contribution을 보여주는 ablation이 더 필요하다.

4. **Product comparison의 통제 한계가 있다**
   - Baseline API가 없어 app UI로 평가한다.
   - Network, hidden prompt, frame sampling, session policy, product update를 완전히 맞추기 어렵다.

5. **Commentary hallucination이 남아 있다**
   - 저자들은 compact model size와 상대적으로 적은 commentary data를 원인으로 제시한다.
   - Proactive assistant에서는 hallucination이 사용자가 묻지 않은 말을 먼저 한다는 점에서 일반 chatbot보다 더 위험할 수 있다.

6. **1 Hz default sampling의 경계가 있다**
   - 1초보다 짧은 event, 빠른 gesture, 순간적인 safety signal은 놓칠 수 있다.
   - Frame rate를 높이면 token cost와 latency가 증가한다.

7. **Latency claim의 운영 조건이 충분히 세분화되지 않는다**
   - 논문은 2시간 이상 stream과 sub-second end-to-end latency를 보고한다.
   - Hardware, concurrent session, percentile latency, memory consolidation load, TTS 포함 여부를 함께 봐야 한다.

8. **Privacy와 autonomy risk가 크다**
   - Always-on camera는 민감한 visual data를 지속적으로 처리한다.
   - Visual prompt injection, unintended surveillance, false alert, over-intervention을 고려해야 한다.

9. **Delegation은 새로운 security boundary를 만든다**
   - Background agent가 code execution이나 external API를 사용하면 permission, sandbox, audit log, timeout, result validation이 필요하다.
   - Foreground model의 delegate token 하나가 high-impact action으로 곧바로 이어지면 안 된다.

10. **Speech를 분리한 trade-off가 있다**
    - Pluggable ASR/TTS는 deployment 유연성을 높인다.
    - 반면 prosody, interruption, overlapping speech, end-to-end conversational timing은 omni model보다 제한될 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 중요한 기여는 "실시간 video를 처리했다"가 아니라, **when-to-act policy를 model behavior로 명시하고 이를 data, objective, runtime까지 일관되게 연결했다는 점**이다.

많은 AI product에서 response content는 강하지만 orchestration이 brittle하다. Trigger는 heuristic, memory는 별도 RAG, tool call은 synchronous, video input은 periodic screenshot으로 붙는다. 이 방식은 prototype에는 빠르지만 다음 문제가 생긴다.

- Trigger rule과 model understanding이 분리된다.
- Background task가 시작되면 foreground interaction이 멈춘다.
- Memory summary가 serving cache와 맞지 않는다.
- Silence와 false alarm을 학습 objective에서 다루지 않는다.
- Model benchmark와 product latency가 서로 다른 layer에서 최적화된다.

JoyAI-VL-Interaction은 이 조각들을 하나의 contract로 묶는다.

> Model은 행동 시점을 결정하고, system은 그 결정을 안전하고 교체 가능한 방식으로 실행한다.

이 문장은 visual assistant뿐 아니라 audio assistant, monitoring agent, browser copilot, robot supervisor에도 적용할 수 있다.

## 7-2. Reuse potential

### Monitoring and alerting

- Industrial camera
- Elder care
- Store operation
- Safety inspection
- Security feed

다만 high-stakes alert는 model response를 곧바로 action으로 연결하지 않는 편이 안전하다. Detection policy, confidence gate, deterministic rule, human escalation을 함께 둬야 한다.

### Accessibility

- Live scene description
- On-screen text translation
- Navigation assistance
- Event reminder

이 영역에서는 말하는 횟수보다 information priority가 중요하다. Silence policy를 user preference와 environment risk에 맞게 personalize할 필요가 있다.

### AI glasses and wearable assistant

Wearable은 user가 매번 질문하기 어려우므로 interaction model과 잘 맞는다. 동시에 battery, privacy, local inference, network loss를 고려해야 한다. Compact foreground model과 cloud background agent의 분리는 현실적인 deployment pattern이다.

### Live guidance

- Cooking
- Assembly
- Maintenance
- Education
- Sports coaching

Guidance system은 step completion detection과 intervention timing이 핵심이다. Final answer QA보다 state transition recognition dataset이 더 중요해진다.

### Agent orchestration

Delegation protocol은 multimodal task 외에도 재사용할 수 있다.

- Fast local model은 routing과 user interaction을 담당한다.
- Slow remote model은 research, coding, generation을 담당한다.
- Background result는 bounded summary로만 foreground context에 들어간다.
- Foreground는 background task가 끝날 때까지 block되지 않는다.

이는 interactive agent의 latency architecture로 꽤 실용적이다.

## 7-3. Production design considerations

실서비스에서는 model의 세 action을 그대로 최종 action으로 쓰기보다 policy layer를 더 분리하는 편이 좋다.

| Layer | Responsibility |
| --- | --- |
| Perception | 현재 scene과 event candidate 추출 |
| Interaction policy | silence, response, delegate proposal |
| Safety gate | privacy, confidence, rate limit, high-risk action 차단 |
| Response generator | 사용자에게 보여줄 content 생성 |
| Background executor | External tool, model, agent 실행 |
| Audit layer | Trigger, evidence, action, result 기록 |

특히 proactive response에는 evidence pointer가 필요하다. Model이 왜 지금 말했는지 해당 timestamp와 frame range를 함께 저장하면 false alarm debugging과 user trust에 도움이 된다.

또한 silence policy를 하나의 global threshold로 두기보다 scenario별 cost matrix로 다루는 편이 낫다.

- Fire alert에서는 false negative cost가 매우 크다.
- Casual commentary에서는 false positive cost가 더 크다.
- Accessibility assistant에서는 user preference가 중요한 prior다.
- Workplace monitoring에서는 privacy constraint가 먼저다.

즉 interaction policy의 최적 threshold는 task마다 다르다.

## 7-4. Follow-up papers

- Audio Interaction Model: audio stream에서 perceive, decide, respond loop를 다루는 유사한 문제 설정
- VideoLLM-Online: online video understanding과 streaming interaction의 초기 기준점
- EasyVideoR1: video understanding model에 RLVR을 적용하는 training framework
- Qwen3.5-Omni Technical Report: native omni-modal streaming과 speech interaction 비교
- AdaCodec: predictable video frame의 token cost를 줄이는 native streaming codec
- Mem0: production-oriented long-term memory lifecycle과 retrieval design

# 8. Summary

- JoyAI-VL-Interaction은 매초 silence, response, delegate를 선택하는 8B vision-first interaction model이다.
- 핵심 novelty는 낮은 latency보다 response timing을 model policy로 학습했다는 데 있다.
- 4M개 이상의 time-aligned clip, weighted SFT, GRPO가 interaction behavior를 만든다.
- AdaCodec, hierarchical memory, asynchronous background loop, vLLM prefix reuse가 long-running deployment를 지원한다.
- 58개 event-driven case에서 강한 product-level 결과를 보이지만, broad capability와 component-level ablation은 추가 검증이 필요하다.
