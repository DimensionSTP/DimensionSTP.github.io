---
layout: single
title: "EasyVideoR1: Easier RL for Video Understanding Review"
categories: Study-concept
tag: [MultimodalAI, VideoUnderstanding, ReinforcementLearning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.16893)

EasyVideoR1은 "video understanding 성능을 올리는 또 하나의 RL recipe"로만 읽으면 핵심을 놓치기 쉬운 논문이다. 이 논문은 새로운 VLM backbone을 제안하지 않는다. 대신 **video understanding에 RLVR(Reinforcement Learning from Verifiable Rewards)을 실제로 돌리려면 무엇이 병목이 되는가**를 시스템 관점에서 꽤 정직하게 분해한다.

LLM에서 RLVR은 DeepSeek-R1 이후 reasoning 성능을 끌어올리는 대표적인 post-training 방식이 되었다. 하지만 이 방식을 video-language model로 가져오면 상황이 훨씬 복잡해진다. 텍스트나 이미지와 달리 비디오는 decoding, frame sampling, resizing, token budget, FPS, max frame 수, prompt template, reward function, benchmark protocol이 모두 성능과 throughput에 영향을 준다. 즉 video RL은 "GRPO를 붙이면 된다"가 아니라, **데이터 전처리, rollout, reward routing, actor training, evaluation까지 모두 비디오 친화적으로 다시 설계해야 하는 문제**다.

EasyVideoR1은 이 지점을 겨냥한다. 저자들은 EasyR1/veRL 기반으로 video RL pipeline을 확장하면서, offline preprocessing과 tensor caching, task-aware reward system, hybrid online-offline training, joint image-video training, asynchronous multi-benchmark evaluation을 하나의 framework로 묶는다. 결과적으로 Qwen3-VL-8B-Instruct를 대상으로 약 100K video samples와 200 GRPO steps를 사용해 평균 benchmark accuracy를 62.1에서 64.4로 올리고, cache-based loading으로 training throughput을 1.47x 개선했다고 보고한다.

> 한 줄 요약: EasyVideoR1은 video understanding을 위한 RLVR을 실제로 학습/평가 가능한 pipeline으로 만들기 위해, video tensor caching, metadata-consistent processing, task-aware reward routing, hybrid online-offline training, joint image-video batching, asynchronous evaluation을 통합한 open-source video RL framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- multimodal reasoning의 다음 병목은 모델 크기만이 아니라 **video-specific RL infrastructure**가 될 가능성이 크다.
- 비디오는 preprocessing과 evaluation protocol이 성능을 크게 흔들기 때문에, RL algorithm보다 system hygiene가 더 중요해질 수 있다.
- EasyVideoR1은 "video RL에서 무엇을 캐시하고, 어떤 metadata를 유지하며, reward를 어떻게 라우팅해야 하는가"를 꽤 구체적으로 보여준다.
- Qwen3-VL 계열처럼 이미 강한 open VLM 위에 RL post-training을 얹을 때 필요한 실무적 체크리스트로 읽을 수 있다.
- 논문보다 codebase/repository 성격이 강해서, 실제 실험을 설계할 때 바로 가져갈 수 있는 interface가 많다.

EasyVideoR1의 핵심 메시지는 단순하다. **Video RLVR의 병목은 reward algorithm 하나가 아니라, video sample이 training loop를 통과하는 전체 경로다.** 비디오를 몇 번 decode하는지, 어떤 frame metadata를 보존하는지, image/video branch가 FSDP에서 모두 gradient graph에 연결되는지, evaluation이 official score를 재현하는지 같은 "작아 보이는" 문제들이 실제로는 RL 실험의 신뢰도를 좌우한다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 다루는 문제는 **large vision-language model(LVLM)에 RLVR을 적용해 video understanding 능력을 끌어올리는 것**이다.

텍스트 RLVR에서는 prompt를 넣고, model response를 생성하고, verifier/reward로 채점한 뒤 policy update를 수행하면 된다. 물론 텍스트 RL도 어렵지만, input pipeline 자체는 상대적으로 단순하다. 반면 video RL에서는 한 sample이 들어올 때마다 다음과 같은 처리가 필요하다.

- raw video를 읽고 decode한다.
- frame sampling을 수행한다.
- frame을 resize / normalize한다.
- visual token budget에 맞게 spatial-temporal grid를 구성한다.
- text prompt 안의 visual placeholder와 실제 visual feature 수를 맞춘다.
- rollout stage와 actor training stage에서 동일한 video representation을 재사용해야 한다.
- task type에 따라 reward function이 달라진다.
- evaluation benchmark마다 frame sampling, FPS, prompt template, scoring rule이 다르다.

즉 video RL은 optimizer만의 문제가 아니다. **video data object가 dataset loading -> rollout generation -> reward computation -> actor update -> evaluation을 지나면서 동일한 의미를 유지하도록 만드는 pipeline problem**이다.

논문은 특히 세 가지 병목을 강조한다.

1. **비디오 전처리 비용**  
   video decoding, sampling, resizing은 CPU-bound I/O bottleneck이 되기 쉽다. 기존 framework에서는 같은 raw video를 pipeline stage마다 반복 decode하는 문제가 생긴다.

2. **다양한 task type과 reward design**  
   video understanding에는 multiple choice, temporal grounding, spatial grounding, OCR, math, open-ended QA, code-like SVG/HTML task 등 서로 다른 scoring logic이 필요하다.

3. **reproducible evaluation의 어려움**  
   video benchmark는 FPS, frame 수, resolution, prompt template, max visual tokens에 민감하다. 조금만 설정이 어긋나도 upstream model의 official score와 다른 결과가 나올 수 있다.

## 1-2. Why previous approaches are insufficient

기존 RL training framework는 크게 두 부류로 볼 수 있다.

첫째, veRL, TRL, ROLL 같은 general RLHF/RLVR framework다. 이들은 distributed rollout, training/inference co-location, PPO/GRPO류 optimization을 잘 다루지만, 기본 설계는 텍스트 중심이거나 범용 multimodal support에 가깝다. video-specific preprocessing, frame cache, task-wise reward routing, benchmark-specific evaluation protocol까지 강하게 묶어 제공하지는 않는다.

둘째, EasyR1, R1-V, OneThinker처럼 multimodal RL에 더 가까운 framework다. EasyR1은 veRL 기반으로 text/image multimodal RL을 지원하고, R1-V는 visual counting/geometric reasoning 같은 image-level task에서 R1-style RL 가능성을 보여준다. OneThinker는 image/video를 포함한 heterogeneous vision task를 다루지만, 논문 기준으로는 video modality 전체를 위한 pipeline, 즉 accelerated offline preprocessing, comprehensive reward library, mixed offline-online training, joint image-video batching, asynchronous evaluation까지 완결된 형태로 제공하지 못한다.

EasyVideoR1이 지적하는 기존 접근의 한계는 다음과 같다.

- image-text RL framework는 video sequence length가 이미지보다 10~100x 길어질 수 있다는 점을 충분히 반영하지 못한다.
- raw video가 tensor로 직접 전달되지 않고 file path로 참조되면, dataset / rollout / actor training 단계에서 반복 decode가 발생한다.
- image-only micro-batch나 video-only micro-batch가 FSDP에서 특정 vision branch를 inactive하게 만들 수 있다.
- reward system이 task-specific하게 잘 분리되어 있지 않으면 새로운 video task를 붙일 때 trainer 자체를 수정해야 한다.
- evaluation protocol이 upstream official setting과 맞지 않으면 실제 model 성능을 과소평가할 수 있다.

따라서 이 논문의 문제 설정은 "video RL algorithm 하나를 더 제안한다"가 아니다. 더 정확히는 **video RL 실험이 재현 가능하고 빠르게 돌 수 있도록 framework boundary를 다시 잡는 것**이다.

# 2. Core Idea

## 2-1. Main contribution

EasyVideoR1의 핵심 기여는 크게 다섯 가지로 정리할 수 있다.

1. **Complete video RL pipeline**  
   dataset loading, rollout generation, actor training 전반을 video modality에 맞게 수정한다. 핵심은 offline preprocessing과 tensor caching, metadata-consistent frame handling, mixed-modality forward pass, independent image/video resolution budget이다.

2. **Task-aware reward system**  
   sample의 `problem_type`을 보고 reward module을 라우팅하는 dispatcher를 둔다. multiple choice, temporal grounding, OCR, math, open-ended QA 등 서로 다른 task type을 하나의 trainer interface로 처리한다.

3. **Hybrid online-offline training**  
   순수 on-policy rollout만 쓰는 대신, pre-collected offline trajectory와 current policy rollout을 같은 response group 안에 섞을 수 있게 한다. 이는 cold-start에서 sparse reward signal이 약한 task를 보완하기 위한 interface다.

4. **Joint image-video training**  
   video data가 이미지 QA data보다 부족하다는 현실을 반영해, image와 video sample을 같은 RL training loop 안에서 다룬다. 대신 image와 video의 pixel budget, frame budget은 독립적으로 설정한다.

5. **Asynchronous multi-benchmark evaluation**  
   precomputed frame cache와 vLLM `AsyncLLMEngine`, chunked prefill을 이용해 evaluation pipeline을 asynchronous streaming 형태로 만든다. 논문 기준 22개 video understanding benchmark를 하나의 interface로 지원한다.

이 다섯 가지를 묶으면 EasyVideoR1은 단순한 trainer wrapper가 아니라, **video RL의 data path와 evaluation path를 모두 다루는 system framework**에 가깝다.

## 2-2. Design intuition

EasyVideoR1의 설계 직관은 꽤 실무적이다.

첫째, video RL에서는 GPU가 항상 계산을 하고 있는 것처럼 보여도 실제 병목은 CPU-side video preprocessing일 수 있다. raw MP4를 매 stage에서 decode하면, rollout model과 actor model이 같은 video를 서로 다른 시점에 다시 읽는 일이 생긴다. 이 경우 GPU hour를 사서 CPU I/O에 낭비하는 셈이다. 그래서 저자들은 expensive한 video decoding을 training loop 밖으로 빼고, sampled/resized tensor를 cache로 저장한다.

둘째, video sample은 tensor만 있으면 충분하지 않다. frame rate, sampling indices, spatial dimension 같은 metadata가 함께 전달되어야 한다. 같은 tensor라도 processor가 다시 resize하거나 다시 sample하면 `video_grid_thw`가 달라지고, rollout과 actor training에서 다른 visual token sequence를 보게 된다. 그래서 EasyVideoR1은 cached frame과 `VideoMetadata`를 함께 전파한다.

셋째, reward design은 하나의 함수로 통일하기 어렵다. multiple choice는 exact match면 충분하지만, temporal grounding은 1D IoU, spatial grounding은 bounding-box IoU, open-ended QA는 ROUGE류 score, preference task는 LLM-as-Judge가 필요할 수 있다. 따라서 reward를 trainer 안에 hard-code하기보다, `problem_type` 기반 dispatcher와 독립 module로 분리하는 편이 맞다.

넷째, video RL은 evaluation 자체가 연구 병목이다. training이 끝났는데 benchmark마다 preprocessing과 prompt가 다르면, 성능 해석이 흔들린다. EasyVideoR1은 evaluation도 framework의 일부로 보고, precomputed frame cache와 asynchronous inference를 넣는다.

내 해석은, EasyVideoR1이 말하는 "easier"는 low-level code를 숨겨준다는 뜻이 아니다. 더 정확히는 **video RL에서 반복적으로 틀리기 쉬운 data/eval/reward plumbing을 framework level에서 고정해 실험의 자유도를 의미 있는 방향으로 제한한다**는 뜻이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | video understanding task에 RLVR을 효율적이고 재현 가능하게 적용하는 framework |
| Base framework | EasyR1 / veRL 기반 확장 |
| Main backbone support | Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5 series |
| Training engine | FSDP actor training + vLLM rollout |
| Core bottleneck | repeated video decoding/preprocessing, mixed-modality batching, task-specific rewards, slow evaluation |
| Key modules | offline tensor cache, VideoMetadata propagation, task-aware reward router, mix-policy interface, async evaluator |
| Main empirical setting | Qwen3-VL-8B-Instruct, 약 100K video samples, 200 GRPO steps, 32 GPUs |
| Main reported gain | average accuracy 62.1 -> 64.4, cache-based training 1.47x throughput improvement |

## 3-2. Module breakdown

### 1) Offline preprocessing and cache-based training

EasyVideoR1의 첫 번째 핵심은 raw video preprocessing을 training loop 밖으로 빼는 것이다.

기존 pipeline에서는 dataset stage, rollout generation stage, actor training stage가 각각 raw video file path를 받아 independently decode / sample / resize할 수 있다. 논문은 EasyR1과 OneThinker류 setup에서 video data가 pipeline stage마다 최대 세 번까지 반복 처리될 수 있다고 지적한다. 이 경우 같은 sample 하나가 들어와도 CPU decode 비용이 중복된다.

EasyVideoR1은 이를 다음 방식으로 바꾼다.

- offline batch tool이 raw video를 decode한다.
- target FPS, max frames, max pixels에 따라 frame sampling과 resizing을 수행한다.
- 결과 frame tensor를 `.pt` cache file로 저장한다.
- cache key는 `(video_path, fps, max_frames, max_pixels)`로 구성한다.
- parameter가 바뀌면 stale cache가 자동으로 무효화된다.
- preprocessing은 multi-worker process와 hash-based deduplication으로 병렬화한다.

중요한 점은 cache file을 직접 worker 간에 큰 tensor로 주고받지 않는다는 것이다. dataset stage는 cache file path만 lightweight string으로 보관하고, 실제 `.pt` loading은 각 worker가 local point-of-use에서 수행한다. 이렇게 하면 inter-node data transfer가 sample당 수 MB tensor에서 짧은 path string으로 줄어든다.

다만 storage trade-off가 있다. 논문은 10분짜리 video를 2 FPS, 최대 256 frames로 sampling하면 cache file이 약 360MB가 될 수 있다고 설명한다. 압축된 원본 MP4가 수십 MB 수준일 수 있다는 점을 감안하면 storage overhead는 크다. 저자들은 이를 "cheap storage를 expensive GPU-hour throughput과 교환하는" trade-off로 본다.

내가 보기엔 이 부분이 가장 실용적이다. training loop 안에서 video decoding이 반복되는 순간, RL algorithm을 아무리 바꿔도 throughput 해석이 흐려진다. EasyVideoR1은 먼저 data path를 deterministic하고 cache-friendly하게 만든다.

### 2) Metadata-consistent video processing

비디오 cache에서 더 중요한 것은 tensor 자체보다 metadata consistency다.

cached frame은 이미 sampling과 resizing을 거쳤다. 그런데 rollout stage나 actor training stage의 processor가 다시 sampling/resizing을 수행하면 같은 sample이 stage마다 다른 visual grid를 갖게 된다. 특히 Qwen 계열 VLM에서는 `video_grid_thw` 같은 spatial-temporal grid 정보가 positional encoding과 token alignment에 직접 연결된다.

EasyVideoR1은 cached frame과 함께 `VideoMetadata`를 pipeline 전체에 전파한다.

- dataset loading 단계에서 frame rate, sampling indices, spatial dimensions를 metadata로 붙인다.
- rollout generation에서는 vLLM에 `(tensor, VideoMetadata)` tuple을 넘긴다.
- actor training에서는 HuggingFace processor에 metadata를 전달하고, `do_resize=False`, `do_sample_frames=False`를 설정해 재처리를 막는다.
- 결과적으로 각 stage가 동일한 `video_grid_thw`를 생성하도록 한다.

이 설계는 작아 보이지만, 실제 video RL에서는 매우 중요하다. rollout policy가 본 visual token sequence와 actor update가 log-prob를 계산하는 visual token sequence가 달라지면, gradient는 같은 trajectory에 대한 update가 아니게 된다. EasyVideoR1은 이를 system-level invariant로 고정한다.

### 3) Mixed-modality pipeline adaptation

EasyVideoR1은 image와 video를 같은 training loop에서 다루기 위해 pipeline adaptation을 추가한다.

문제는 LVLM이 image encoder branch와 video encoder branch를 별도로 가질 수 있다는 점이다. mixed image-video training에서 어떤 micro-batch가 image-only라면 video branch는 forward graph에 참여하지 않을 수 있다. 반대로 video-only micro-batch에서는 image branch가 inactive할 수 있다. FSDP full sharding에서는 이런 unused parameter가 gradient synchronization failure를 만들 수 있다.

EasyVideoR1은 missing modality에 대해 zero-valued dummy tensor를 만들고, 해당 encoder output을 zero-weighted addition으로 computation graph에 연결한다. 이렇게 하면 모든 parameter가 forward graph에 참여하지만, 실제 gradient에는 spurious contribution을 주지 않는다.

또 하나의 설계는 independent resolution budget이다.

- 이미지는 high spatial resolution이 중요하다.
- 비디오는 per-frame resolution과 frame count 사이의 trade-off가 있다.
- 따라서 `image_max_pixels`, `video_max_pixels`, `video_max_frames`를 분리한다.

이 분리는 중요하다. image와 video를 같은 max pixel budget으로 묶으면 image는 해상도를 덜 쓰거나, video는 frame 수를 충분히 못 쓰는 식으로 손해가 생긴다. EasyVideoR1은 modality별 compute budget을 별도로 조정하게 만든다.

### 4) Task-aware reward system

EasyVideoR1의 reward system은 `problem_type` 기반 dispatcher 구조다. sample마다 task type을 명시하고, 중앙 dispatcher가 해당 reward module로 라우팅한다. 각 task type은 독립 module로 구현되므로 새 task를 붙일 때 trainer core를 수정하지 않아도 된다.

논문 Table 1 기준 supported task와 scoring 방식은 다음과 같다.

| Category | Task Type | Accuracy / Reward Scoring |
| --- | --- | --- |
| Multiple Choice | multiple choice | Exact match |
| Numerical | numerical, regression | Numeric comparison |
| Temporal Grounding | temporal grounding | 1D IoU |
| ST Grounding | spatial-temporal grounding | 0.5xtIoU + 0.5xmIoU |
| Spatial Grounding | spatial grounding | Bounding-box IoU |
| Open-ended | open-ended, video QA | ROUGE score |
| Math | math | Symbolic verification |
| OCR | OCR | WER / exact match |
| Boolean | boolean | Exact match |
| Code | SVG, HTML | Execution / match |
| Preference | LLaVA, critic | LLM-as-Judge |

Prompt formatting은 Jinja2 template로 task type마다 동적으로 렌더링한다.

이 부분은 논문 제목의 "Easier"와 직접 연결된다. video understanding은 task distribution이 넓어서, reward function 하나로 통일하기 어렵다. reward module을 분리해두면 multiple-choice QA에서 temporal grounding으로 넘어갈 때 trainer를 다시 뜯지 않아도 된다.

다만 이 구조는 reward quality 문제를 없애지는 않는다. exact match나 IoU는 비교적 명확하지만, open-ended QA의 ROUGE score나 preference task의 LLM-as-Judge는 task semantics를 완벽히 반영하지 못할 수 있다. EasyVideoR1은 reward plumbing을 정리하지만, reward design 자체의 어려움은 여전히 남는다.

### 5) Hybrid online-offline training

표준 GRPO류 on-policy training에서는 current policy가 rollout한 trajectories만 update에 사용한다. 하지만 video task에서는 초기 rollout이 매우 약할 수 있고, reward가 sparse하게 들어오는 task도 많다. 이 경우 on-policy only training은 cold-start 문제가 생긴다.

EasyVideoR1은 mix-policy interface를 둔다.

- 각 training sample은 optional pre-collected offline trajectory를 가질 수 있다.
- rollout 시 group size가 `n`이면, framework는 `n-1`개의 on-policy response를 current policy로 생성한다.
- 마지막 slot에는 offline trajectory를 substitute한다.
- 이렇게 구성된 `n`개 response group이 reward computation과 GRPO update를 일반적인 방식으로 통과한다.
- 이 기능은 `enable_mix_policy` flag와 optional quality threshold로 제어한다.
- 비활성화하면 standard on-policy training으로 돌아간다.

흥미로운 점은 이 interface가 GRPO algorithm 자체를 수정하지 않고 rollout layer에서 동작한다는 것이다. 즉 algorithm research를 위해 off-policy/on-policy mixture를 실험하되, trainer core의 복잡도를 크게 늘리지 않는다.

내가 보기엔 이 설계는 실제 연구에서 꽤 유용하다. video RL dataset에는 이미 더 강한 모델이나 이전 checkpoint가 만든 answer/cot/trajectory가 있을 수 있다. 이를 완전히 버리면 sample efficiency가 낮고, SFT처럼만 쓰면 exploration을 잃는다. EasyVideoR1은 그 중간 지점을 쉽게 실험하게 해준다.

### 6) Joint image-video training

EasyVideoR1은 image data와 video data를 함께 학습하는 joint training을 지원한다.

동기는 단순하다. high-quality annotated video data는 image QA data보다 부족하다. 따라서 image data로 기본적인 visual reasoning을 강화하면서, video data로 temporal understanding을 학습하는 방식이 더 현실적이다.

구현상 sample은 `data_type` field를 갖고, 이에 따라 image/video preprocessor와 decoupled resolution budget으로 라우팅된다. 또한 image와 video sample의 multimodal field schema를 통일해 data loading, rollout, reward computation, policy update 전반에서 modality-specific branching을 줄인다.

중요한 engineering choice는 strict-failure policy다. textual placeholder token 수와 vision encoder가 만든 visual feature 수가 맞지 않으면, trainer가 silent truncation/padding을 하지 않고 exception을 발생시킨다. 이는 실험을 멈추게 만들 수 있지만, long training run에서 조용히 잘못된 gradient가 쌓이는 것보다는 낫다.

내가 보기엔 이 strictness가 좋은 framework의 특징이다. video RL에서 가장 위험한 버그는 성능이 조금 낮아지는 버그가 아니라, sample alignment가 깨졌는데도 training이 계속 돌아가는 버그다.

### 7) Asynchronous multi-benchmark evaluation

EasyVideoR1은 evaluation을 별도 appendix가 아니라 core system design으로 다룬다.

비디오 benchmark evaluation은 두 가지 병목이 있다.

1. CPU-side video preprocessing이 느리다.
2. GPU inference가 batch boundary에서 idle할 수 있다.

EasyVideoR1은 이를 위해 evaluation에도 precomputed frame cache를 사용한다. video decoding, temporal sampling, resizing은 model과 무관하므로 미리 cache file로 저장한다. evaluation 시에는 cache file을 읽기만 하면 되기 때문에 per-video preprocessing latency를 크게 줄일 수 있다.

그 다음 vLLM `AsyncLLMEngine` 기반 3-stage asynchronous pipeline을 구성한다.

- **IO stage**: background thread pool이 cached video frame을 계속 읽고 input을 준비한다.
- **Prefill stage**: 새로운 input이 들어오면 바로 KV cache를 만든다.
- **Decode stage**: prefill이 끝난 request는 autoregressive decoding에 들어가고, decode token과 이후 request의 prefill이 같은 scheduling step에서 interleave된다.

또한 long video sequence가 prefill을 독점하지 않도록 chunked prefill을 사용한다. 논문은 LVBench를 예로 들어 vanilla inference framework 대비 약 6~7x speedup을 보고한다.

evaluation framework는 논문 기준 22개 video understanding benchmark를 지원한다. 범주는 general video understanding, long video understanding, video reasoning, STEM knowledge, spatial understanding, spatio-temporal grounding, streaming video를 포함한다. 각 benchmark는 data loading, prompt formatting, answer extraction, scoring function을 갖는 lightweight adapter로 등록된다.

이 부분의 핵심은 단순 speed가 아니다. video benchmark는 protocol mismatch에 민감하기 때문에, benchmark adapter를 framework에 넣어야 model 비교가 흔들리지 않는다.

# 4. Training / Data / Recipe

## 4-1. Data

논문 실험의 대표 base model은 **Qwen3-VL-8B-Instruct**다. 저자들은 Qwen3-VL-8B-Instruct를 선택한 이유로 DeepStack architecture와 interleaved M-RoPE positional encoding을 언급하고, 해당 scale에서 강한 open-source video-language model이라고 설명한다.

학습 데이터는 약 **100K video samples**다. 데이터는 공개 video RL dataset에서 구성되며, 논문은 다음을 예로 든다.

- OneThinker
- Video-R1
- VideoChat-R1

데이터 filtering도 중요하다. 저자들은 model의 learning frontier에 있는 sample만 남기기 위해 pass-rate-based filtering을 적용한다.

- 각 candidate sample에 대해 base model로 `k=8` rollouts를 수행한다.
- pass rate가 `0 < pass rate < 1`인 sample만 유지한다.
- 이미 항상 맞히는 trivial sample과 항상 실패하는 sample을 제거한다.

이 filtering은 꽤 합리적이다. RLVR에서 reward가 항상 1이거나 항상 0이면 policy gradient signal이 약하다. partial success region에 있는 sample은 model이 현재는 불안정하게 맞히지만, reward signal을 통해 개선될 여지가 있는 데이터다.

## 4-2. Training strategy

논문에서 공개한 주요 training configuration은 다음과 같다.

| Item | Value |
| --- | --- |
| Base model | Qwen3-VL-8B-Instruct |
| RL algorithm | GRPO with DAPO clipping variant |
| Clip ratios | `ε_low = 0.2`, `ε_high = 0.28` |
| KL penalty | disabled |
| Rollout group size | `n = 8` |
| Global batch size | 256 |
| Learning rate | `1e-6`, constant |
| Optimizer | AdamW, `β1=0.9`, `β2=0.999`, weight decay `0.01` |
| Video sampling | 2 FPS |
| Max video frames | 128 |
| Video pixel budget | per-frame 262,144 pixels |
| Image pixel budget | 1,048,576 pixels |
| Max response length | 4,096 tokens |
| Distributed setup | 32 GPUs, FSDP full sharding |
| Memory / throughput tricks | gradient checkpointing, padding-free attention, dynamic batching |
| Rollout engine | vLLM, tensor parallelism size 2 |

Abstract에서는 32 H200 GPUs와 약 20시간의 RL training을 사용했다고 설명한다. 본문 실험에서는 Qwen3-VL-8B-Instruct가 200 GRPO steps 이후 평가된다.

여기서 중요한 점은 이 논문이 algorithmic novelty를 GRPO objective에 두지 않는다는 것이다. GRPO는 이미 널리 쓰이는 RLVR objective이고, EasyVideoR1의 초점은 **video sample이 GRPO loop 안에서 빠르고 일관되게 처리되도록 만드는 system recipe**에 있다.

## 4-3. Engineering notes

이 논문에서 실무적으로 가져갈 만한 engineering note는 다음과 같다.

1. **cache key를 preprocessing hyperparameter까지 포함해서 잡는다**  
   `video_path`만 key로 쓰면 FPS, max frame, max pixel 변경 시 stale cache를 잘못 사용할 수 있다. EasyVideoR1은 `(video_path, fps, max_frames, max_pixels)`를 key로 둔다.

2. **cache tensor와 metadata를 반드시 같이 전파한다**  
   cached frame이 있어도 sampling indices와 spatial dimension이 빠지면 processor 단계에서 silent mismatch가 생긴다.

3. **missing modality branch를 dummy tensor로 graph에 연결한다**  
   FSDP에서 unused parameter 문제가 생기지 않게 하면서 실제 gradient는 오염시키지 않는 방식이다.

4. **reward는 central dispatcher + independent module로 분리한다**  
   video task 종류가 많기 때문에, reward function을 trainer에 직접 hard-code하면 빠르게 유지보수가 어려워진다.

5. **evaluation도 cache와 async inference를 써야 한다**  
   training만 빨라도 evaluation이 느리면 iteration loop가 느려진다. 특히 video benchmark suite 전체를 돌릴 때 evaluation pipeline 자체가 연구 생산성을 좌우한다.

6. **strict-failure policy가 필요하다**  
   visual placeholder와 visual feature 수가 mismatch되면 자동 padding/truncation으로 넘어가지 않고 exception을 내는 편이 낫다. 조용한 data bug는 RL training에서 매우 비싸다.

# 5. Evaluation

## 5-1. Main results

논문은 두 가지 질문을 중심으로 실험을 설계한다.

1. EasyVideoR1로 RL training한 instruct model이 해당 thinking variant를 넘어설 수 있는가?
2. offline preprocessing과 caching이 training throughput을 얼마나 개선하는가?

### Benchmark performance

Figure 2는 세 가지 model variant를 비교한다.

- Qwen3-VL-8B-Instruct
- Qwen3-VL-8B-Think / Thinking variant
- Qwen3-VL-8B-Instruct + EasyVideoR1, 200 GRPO steps

평가는 10개 representative benchmark에서 수행된다.

| Category | Benchmarks |
| --- | --- |
| General video understanding | Video-MME, MVBench, TempCompass |
| Long video understanding | LVBench, LongVideoBench, MLVU |
| Video reasoning | Video-Holmes |
| STEM knowledge | MMVU, Video-MMMU, VideoMathQA |

주요 결과는 다음과 같다.

- 평균 accuracy가 **62.1 -> 64.4**, 즉 **+2.3 points** 개선된다.
- 가장 큰 개선은 **Video-Holmes +6.6**, **VideoMathQA +6.7**에서 나온다.
- general video understanding에서는 Video-MME +2.1, MVBench +3.5가 개선된다.
- long video 쪽에서는 LVBench +0.7, LongVideoBench +4.1이 개선되지만, MLVU는 -0.6으로 소폭 하락한다.
- TempCompass는 -0.3, Video-MMMU는 -1.7로 하락한다.
- 논문은 RL-trained model이 standard non-thinking inference mode로도 Qwen3-VL-8B-Think와 대부분 benchmark에서 comparable or superior하다고 해석한다.

여기서 중요하게 볼 점은 평균 +2.3이 전부가 아니라는 점이다. EasyVideoR1의 RL은 특히 reasoning/math task에서 더 크게 이득을 주지만, 모든 benchmark를 균일하게 개선하지는 않는다. 따라서 이 결과는 "video RL을 하면 모든 video understanding이 좋아진다"라기보다, **verifiable reward가 잘 맞는 reasoning-heavy task에서 더 큰 효과가 난다**고 보는 편이 안전하다.

### Training efficiency

Figure 3은 cache-based loading과 on-the-fly decoding을 비교한다. 동일 조건은 다음과 같다.

- Qwen3-VL-8B
- 32 GPUs, 4 nodes x 8 GPUs
- global batch size 32
- video sequence up to 256 frames
- cache-based loading: `prefer_preprocessed`
- on-the-fly decoding: `realtime_only`
- 첫 warmup step 제외
- cache mode는 95 training steps 평균, on-the-fly mode는 68 training steps 평균

결과는 다음과 같다.

| Metric | On-the-fly decoding | Cache-based loading | Change |
| --- | ---: | ---: | ---: |
| Average step time | 194.5s | 131.9s | 1.47x faster |
| Token throughput | 797 tok/s | 1,175 tok/s | 1.47x higher |
| Rollout generation | 82.1s | 53.9s | 1.52x faster |
| Reference model forward | 53.6s | 18.8s | 2.85x faster |
| Actor parameter update | ~54s | ~54s | unchanged |
| Tokens per step | ~4.93M | ~4.93M | nearly identical |

이 결과에서 중요한 것은 actor update time이 변하지 않는다는 점이다. actor update는 token-level gradient 계산이므로 video I/O와 무관하다. 반대로 rollout generation과 reference model forward는 video decoding에 묶여 있기 때문에 caching 이득이 크다. 특히 reference model forward에서 2.85x 개선이 큰 이유는 on-the-fly mode에서는 rollout에서 이미 처리한 같은 video를 reference model stage가 다시 decode해야 하기 때문이다.

### Evaluation efficiency and coverage

EasyVideoR1의 evaluation framework는 논문 기준 22개 video understanding benchmark를 지원한다. benchmark category는 general, long video, reasoning, STEM, spatial, spatio-temporal grounding, streaming을 포함한다.

또한 precomputed frame cache와 AsyncLLMEngine, chunked prefill을 사용해 evaluation pipeline을 asynchronous하게 만든다. 논문은 LVBench에서 vanilla inference framework 대비 약 **6~7x speedup**을 보고한다.

이 결과는 학습 성능 수치보다도 framework 가치와 연결된다. video RL 연구는 training run 하나보다 benchmark suite 전체를 얼마나 자주, 일관되게 돌릴 수 있는가가 중요하다. evaluation throughput이 낮으면 hyperparameter search와 ablation 자체가 느려진다.

## 5-2. What really matters in the experiments

이 논문의 실험에서 내가 중요하게 보는 포인트는 세 가지다.

첫째, **성능 개선보다 pipeline correctness가 먼저다.**  
평균 +2.3 points는 의미 있지만, EasyVideoR1의 더 중요한 기여는 end-to-end RL loop가 video task에서 실제로 작동한다는 것을 보여준 점이다. data loading, rollout, reward computation, policy update가 일관되게 맞물린다는 것 자체가 framework 논문에서는 핵심 결과다.

둘째, **caching은 단순 speed trick이 아니라 training semantics를 보존하는 장치다.**  
Figure 3에서 tokens per step이 거의 동일하다는 점은 cache mode가 training workload 자체를 바꾸지 않고 I/O 중복만 줄였다는 의미다. speedup이 sample reduction이나 shorter video 때문이 아니라 data path 최적화에서 나온다는 점이 중요하다.

셋째, **RL gain은 task-dependent하다.**  
Video-Holmes와 VideoMathQA처럼 reasoning/math 성격이 강한 benchmark에서는 gains가 크지만, TempCompass나 Video-MMMU에서는 하락도 있다. 이는 video RLVR이 아직 reward/task alignment에 민감하다는 신호다. benchmark average만 보면 놓치기 쉽다.

넷째, **evaluation reproducibility가 framework의 일부로 들어간 것이 좋다.**  
video benchmark는 frame sampling, FPS, prompt template, resolution에 매우 민감하다. EasyVideoR1은 benchmark adapter와 async evaluation을 제공함으로써 "학습했다"와 "공정하게 평가했다"를 같은 repository 안에서 묶으려 한다.

다만 아쉬운 점도 있다. hybrid online-offline training, joint image-video training, reward module design이 각각 어느 정도 성능에 기여했는지를 분리한 ablation은 제한적이다. 이 논문은 method ablation paper라기보다 system report에 가깝다. 따라서 framework의 각 component가 어느 task에서 얼마나 중요한지는 후속 실험이 더 필요하다.

# 6. Limitations

1. **실험은 Qwen3-VL-8B 중심이다.**  
   framework는 Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5 series를 지원한다고 하지만, 논문에서 성능 개선을 자세히 보여주는 대표 실험은 Qwen3-VL-8B-Instruct다. 다른 backbone, scale, vision encoder 구조에서도 같은 정도의 gain과 throughput trade-off가 나오는지는 추가 확인이 필요하다.

2. **system report 성격이 강하고, component ablation은 제한적이다.**  
   EasyVideoR1은 많은 기능을 제공한다. offline cache, reward router, hybrid training, joint image-video training, async evaluation이 모두 중요해 보이지만, 각 component가 final accuracy에 얼마나 기여했는지는 충분히 분리되어 있지 않다. throughput ablation은 명확하지만, accuracy ablation은 더 필요하다.

3. **offline tensor cache는 storage cost가 크다.**  
   논문 예시처럼 10분 video를 2 FPS, 256 frames로 cache하면 약 360MB가 될 수 있다. 대규모 video corpus에서는 storage와 cache invalidation이 별도 운영 문제가 된다. 특히 FPS, max frame, resolution을 자주 바꾸는 연구 단계에서는 cache explosion이 생길 수 있다.

4. **reward coverage와 reward quality는 다른 문제다.**  
   EasyVideoR1은 11개 task category와 modular reward system을 제공하지만, open-ended video QA의 ROUGE score나 preference task의 LLM-as-Judge는 완전한 verifiable reward라고 보기 어렵다. task-aware routing은 plumbing을 해결하지만, reward specification 문제는 남는다.

5. **benchmark average만 보면 task별 regression을 놓칠 수 있다.**  
   Figure 2에서 평균은 +2.3이지만 TempCompass, MLVU, Video-MMMU는 하락한다. 따라서 EasyVideoR1로 RL training을 할 때는 평균 성능뿐 아니라 temporal reasoning, long video, STEM 등 세부 category별 trade-off를 따로 봐야 한다.

6. **hardware assumption이 가볍지 않다.**  
   abstract 기준 32 H200 GPUs와 약 20시간의 RL training이 언급된다. framework가 barrier를 낮추는 것은 맞지만, video RLVR 자체는 여전히 상당한 GPU/스토리지 리소스를 요구한다.

7. **evaluation protocol은 계속 변할 수 있다.**  
   논문 기준 22 benchmark를 지원하지만, repository가 업데이트되면 benchmark 수, prompt template, scoring implementation이 바뀔 수 있다. 최종 게시 전에는 논문 버전과 codebase 버전을 구분해서 확인하는 것이 좋다.

# 7. My Take

## 7-1. Why this matters for my work

내 관점에서 EasyVideoR1의 가치는 "Qwen3-VL-8B를 몇 점 올렸다"보다 **video RL 실험의 단위를 다시 정의했다**는 데 있다.

텍스트 RL에서는 sample이 대체로 prompt/answer/reward로 끝난다. 하지만 video RL에서는 sample이 훨씬 복잡하다.

```text
video sample = raw video path
             + sampled/resized frame tensor
             + frame metadata
             + prompt template
             + visual placeholder alignment
             + problem_type
             + reward module
             + benchmark-specific evaluation config
```

이 중 하나라도 stage마다 달라지면, RL signal이 의미를 잃는다. EasyVideoR1은 이 복잡한 sample object를 framework 내부에서 일관되게 다루려 한다.

특히 내가 좋게 본 부분은 "정확한 RL objective"보다 "일관된 visual data path"를 먼저 잡는 태도다. 멀티모달 RL에서는 reward algorithm보다 data mismatch가 더 큰 성능 문제를 만들 수 있다. rollout에서는 128 frames를 봤는데 actor update에서는 processor가 다시 sample해서 다른 frame grid를 본다면, 아무리 좋은 GRPO variant를 써도 실험이 깨진다.

또 하나 중요한 점은 evaluation이다. video benchmark는 prompt template과 frame sampling에 민감해서, official score 재현 자체가 어렵다. EasyVideoR1처럼 evaluation adapter를 framework에 넣는 방식은 실험 신뢰도를 높이는 데 꽤 중요하다.

## 7-2. Reuse potential

실무/연구에서 바로 가져갈 수 있는 아이디어는 다음과 같다.

1. **preprocessing cache key 설계**  
   video cache key에 `video_path`, FPS, max frames, max pixels를 모두 포함하는 방식은 바로 재사용 가능하다. image/video multimodal service에서도 cache invalidation 기준으로 쓸 수 있다.

2. **metadata propagation invariant**  
   tensor와 metadata를 함께 넘기고, downstream processor가 다시 sampling/resizing하지 못하게 막는 규칙은 video training뿐 아니라 video retrieval, video RAG, video agent memory에도 중요하다.

3. **problem_type 기반 reward router**  
   여러 task를 하나의 trainer에서 다룰 때 central dispatcher + independent reward module 구조는 유지보수성이 좋다. 특히 OCR, grounding, QA, math가 섞인 document/video AI task에도 비슷하게 쓸 수 있다.

4. **FSDP mixed-modality dummy branch 처리**  
   image/video/audio처럼 modality branch가 나뉜 모델에서 micro-batch가 특정 modality만 포함할 때 unused parameter 문제가 생길 수 있다. zero dummy tensor + zero-weighted graph connection은 깔끔한 engineering pattern이다.

5. **strict-failure policy**  
   placeholder와 visual feature 수 mismatch를 silent하게 넘기지 않는 정책은 꼭 가져갈 만하다. 장기 training에서 조용한 data bug는 GPU 비용을 크게 태운다.

6. **asynchronous evaluation design**  
   precomputed frame cache, async IO, chunked prefill, benchmark adapter를 묶는 구조는 video model 평가 자동화에 그대로 참고할 수 있다.

내가 추가로 실험해보고 싶은 방향은 reward module별 ablation이다. 예를 들어 multiple choice exact match 중심으로 RL을 돌릴 때와 temporal grounding IoU reward를 섞을 때, model이 어떤 capability를 얻고 어떤 benchmark에서 퇴화하는지 보고 싶다. EasyVideoR1은 이런 실험을 할 수 있는 기반을 제공하지만, 논문 자체는 아직 그 분석까지 깊게 들어가지는 않는다.

## 7-3. Follow-up papers

후속으로 같이 읽으면 좋은 논문/프로젝트는 다음과 같다.

- **EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework**  
  EasyVideoR1의 직접적인 기반이다. text/image multimodal RL framework 구조를 먼저 보면 EasyVideoR1의 확장 포인트가 더 잘 보인다.

- **veRL / HybridFlow**  
  RLHF/RLVR distributed training framework의 기본 구조를 이해하는 데 좋다. rollout과 actor training을 어떻게 스케줄링하는지가 핵심이다.

- **Video-R1: Reinforcing Video Reasoning in MLLMs**  
  R1-style RL을 video reasoning으로 확장하는 선행 흐름이다. EasyVideoR1의 training data source 중 하나로도 언급된다.

- **OneThinker: All-in-one Reasoning Model for Image and Video**  
  heterogeneous vision task와 reward를 함께 다루는 방향에서 연결된다.

- **Qwen3-VL Technical Report**  
  EasyVideoR1 실험의 대표 backbone이다. interleaved M-RoPE, DeepStack 구조, video understanding protocol을 함께 봐야 결과 해석이 안정적이다.

- **Scaling RL to Long Videos**  
  long video RL에서 sequence length, temporal reasoning, reward sparsity가 어떻게 문제가 되는지 보기에 좋다.

# 8. Summary

- EasyVideoR1은 새로운 VLM backbone이 아니라, video understanding RLVR을 위한 system framework다.
- 핵심은 offline tensor caching, VideoMetadata propagation, mixed-modality FSDP adaptation, task-aware reward routing, hybrid online-offline training, async evaluation이다.
- Qwen3-VL-8B-Instruct 실험에서 200 GRPO steps 후 평균 accuracy가 62.1에서 64.4로 개선되고, reasoning/math task에서 특히 큰 gain을 보인다.
- cache-based training은 on-the-fly video decoding 대비 step time과 token throughput에서 1.47x 개선을 보이며, actor update는 그대로 두고 video I/O 병목을 줄인다.
- 다만 backbone/scale 일반화, component별 accuracy ablation, reward quality, storage overhead, benchmark별 regression은 추가 확인이 필요하다.
