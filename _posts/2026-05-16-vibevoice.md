---
layout: single
title: "VibeVoice Technical Report Review"
categories: Study-concept
tag: [TTS, Speech-Generation, Diffusion]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2508.19205)

VibeVoice는 long-form multi-speaker TTS를 "short utterance TTS의 반복 호출"이 아니라, LLM 기반 sequence modeling 문제로 다시 정의하는 기술 보고서다. 기존 TTS 시스템은 짧은 단일 화자 문장을 자연스럽게 합성하는 데는 강해졌지만, 팟캐스트나 오디오북처럼 여러 화자가 긴 시간 동안 번갈아 말하는 오디오를 안정적으로 만들기는 여전히 어렵다. 문장별로 따로 합성한 뒤 이어 붙이면 음색 일관성, turn-taking, prosody, 감정 흐름, 긴 context coherence가 쉽게 깨진다.

이 논문이 흥미로운 이유는 음성 합성의 병목을 단순히 vocoder 품질이나 speaker cloning 품질에서 찾지 않는다는 점이다. VibeVoice는 긴 오디오 생성을 가능하게 만드는 핵심을 speech tokenizer의 압축률, LLM의 context modeling, continuous latent를 다루는 next-token diffusion의 결합으로 본다. 즉, speech를 discrete code sequence로만 밀어붙이는 대신, acoustic VAE latent를 diffusion head가 생성하고, LLM은 dialogue context와 speaker assignment를 길게 추적하는 역할을 맡는다.

> 한 줄 요약: VibeVoice는 7.5 Hz ultra-low frame rate speech tokenizer와 LLM-conditioned token-level diffusion head를 결합해, 최대 90분 길이와 최대 4명의 화자를 지원하는 long-form conversational TTS framework를 제안한 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- TTS가 short-form single-speaker generation에서 long-form multi-speaker content generation으로 확장될 때 어떤 bottleneck이 생기는지 잘 보여준다.
- 음성을 discrete token language modeling으로만 처리하지 않고, continuous latent를 next-token diffusion으로 생성하는 설계가 깔끔하다.
- 7.5 Hz tokenizer, 64K context, 90 minute generation이라는 수치가 long-context audio generation에서 compression budget이 얼마나 중요한지 보여준다.
- 공개 model card, project page, repository notice가 있어 논문 아이디어뿐 아니라 release availability와 responsible use 이슈까지 함께 확인할 수 있다.

VibeVoice의 핵심 메시지는 간단하다. 긴 음성 생성에서 중요한 것은 더 강한 vocoder 하나가 아니라, speech를 LLM이 감당할 수 있는 sequence length로 압축하고, 그 압축된 sequence 위에서 speaker, text, semantic context, acoustic detail을 어떤 interface로 연결할 것인가다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 문제는 long-form multi-speaker conversational speech generation이다.
- 예시는 podcast, multi-participant audiobook, multi-speaker dialogue audio처럼 여러 화자가 긴 시간 동안 문맥을 유지하며 말하는 audio content다.
- 이 설정에서는 단순한 text-to-speech 품질만으로는 부족하다.
- 같은 speaker의 timbre가 긴 시간 동안 유지되어야 한다.
- speaker turn이 자연스럽게 이어져야 한다.
- text script의 role assignment와 voice prompt가 일관되게 반영되어야 한다.
- prosody, emotion, speaking rate가 local sentence 단위가 아니라 전체 dialogue 흐름에 맞아야 한다.
- 긴 sequence를 처리해야 하므로 token count와 decoding cost가 핵심 병목이 된다.

짧은 단일 화자 TTS에서는 한 문장을 자연스럽게 읽는 것이 중심 문제다. 반면 VibeVoice가 다루는 설정에서는 "긴 대화 전체를 하나의 생성 대상"으로 보는 것이 중요하다. 이때 음성 모델은 speaker consistency와 content coherence를 동시에 만족해야 한다.

## 1-2. Why previous approaches are insufficient

기존 방식의 한계는 크게 세 가지로 정리할 수 있다.

첫째, utterance-level concatenation은 long-form audio의 구조를 충분히 모델링하지 못한다. 각 문장을 따로 생성한 뒤 이어 붙이면 문장 내부 품질은 괜찮을 수 있지만, turn boundary, 대화 리듬, 화자 간 반응, 장기적인 tone consistency가 깨질 수 있다.

둘째, discrete acoustic token 기반 모델은 sequence length가 빠르게 커진다. Encodec이나 기존 speech tokenizer가 초당 수십에서 수백 token을 만들면, 1시간 audio는 LLM context 안에 넣기 어려운 길이가 된다. Long-form TTS에서는 codec 품질만큼이나 token rate가 중요하다.

셋째, multi-speaker control은 speaker prompt, text script, role identifier, generated speech history가 하나의 modeling interface 안에서 정리되어야 한다. speaker embedding이나 prompt를 따로 붙이는 것만으로는 긴 dialogue의 speaker transition을 충분히 제어하기 어렵다.

따라서 VibeVoice의 문제 설정은 "좋은 TTS model"을 만드는 것보다 좁고 동시에 더 어렵다. 이 논문은 긴 audio를 하나의 autoregressive sequence로 처리할 수 있게 만드는 representation, context, generation head를 함께 설계한다.

# 2. Core Idea

## 2-1. Main contribution

VibeVoice의 핵심 기여는 네 가지로 볼 수 있다.

1. Ultra-low frame rate continuous speech tokenizer
   - Acoustic tokenizer와 semantic tokenizer를 분리해서 사용한다.
   - Acoustic tokenizer는 24 kHz input을 3200x downsampling해서 7.5 frames per second 수준으로 낮춘다.
   - 논문 abstract 기준으로 popular Encodec 대비 data compression을 80x 개선하면서 comparable performance를 유지한다고 주장한다.
   - 실험에서는 speech-to-text token ratio가 약 2:1이라고 설명한다. 즉 speech token 두 개가 text BPE token 하나 정도의 budget이라는 해석이다.

2. LLM-centered long-context sequence model
   - VibeVoice는 Qwen2.5 계열 LLM을 core sequence model로 사용한다.
   - voice prompt feature와 text script embedding을 speaker role identifier와 함께 하나의 sequence로 구성한다.
   - LLM은 이 hybrid context를 처리해 각 acoustic token position의 hidden state를 만든다.

3. Token-level diffusion head
   - LLM이 직접 discrete speech code를 예측하는 대신, 각 token position의 hidden state가 lightweight diffusion head를 condition한다.
   - diffusion head는 acoustic VAE feature를 denoising 방식으로 생성한다.
   - 즉 LLM은 high-level context와 sequencing을 담당하고, diffusion head는 continuous acoustic detail을 담당한다.

4. Long-form multi-speaker target
   - 논문은 최대 90분 audio와 최대 4명의 speaker를 지원한다고 보고한다.
   - 이 수치는 speech tokenizer 압축률과 64K context window가 맞물릴 때 가능해지는 결과다.

## 2-2. Design intuition

VibeVoice의 설계 직관은 꽤 명확하다. Speech를 LLM에 넣으려면 token rate를 극단적으로 낮춰야 한다. 하지만 token rate를 낮추면 reconstruction fidelity가 떨어질 수 있다. 따라서 tokenizer는 aggressively compress하면서도 acoustic fidelity를 유지해야 한다.

또 하나의 직관은 continuous audio를 discrete token classification으로만 밀어붙이지 말자는 것이다. 음성은 pitch, timbre, prosody, emotion처럼 연속적인 acoustic detail이 중요하다. VibeVoice는 이 부분을 acoustic VAE latent와 diffusion head로 처리한다. LLM은 text와 speaker structure를 이해하는 데 강하고, diffusion은 continuous latent generation에 강하다는 역할 분담이다.

마지막으로, long-form TTS에서는 generation quality가 architecture 하나로만 결정되지 않는다. Speaker prompt, role identifier, generated speech history, text script가 모두 context 안에 들어가야 한다. VibeVoice는 이를 복잡한 multi-module controller로 나누기보다, 하나의 long sequence로 정리해서 LLM이 처리하게 만든다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Long-form, multi-speaker conversational speech synthesis |
| Core backbone | Qwen2.5 based LLM |
| Key representation | Acoustic latent + semantic latent + text script + speaker role identifier |
| Main compression device | 7.5 Hz continuous speech tokenizer |
| Generation head | LLM hidden state conditioned token-level diffusion head |
| Output target | Acoustic VAE feature, decoded back to waveform by acoustic decoder |
| Main difference | Speech를 high frame rate discrete code sequence로 다루지 않고, ultra-low frame rate continuous latent sequence로 다룸 |

큰 흐름은 다음과 같다.

1. User가 voice prompt와 text script를 제공한다.
2. Voice prompt는 acoustic latent feature로 encode된다.
3. Text script는 speaker role identifier와 함께 embedding된다.
4. VibeVoice LLM이 이 hybrid sequence를 처리한다.
5. 각 generation step에서 LLM hidden state가 diffusion head를 condition한다.
6. Diffusion head가 acoustic VAE latent를 생성한다.
7. Acoustic decoder가 최종 waveform으로 복원한다.

## 3-2. Module breakdown

### 1) Acoustic tokenizer

Acoustic tokenizer는 long-form generation의 핵심 병목인 token rate를 낮추는 장치다.

- Sigma-VAE variant에서 영감을 받은 VAE 구조를 사용한다.
- Encoder와 decoder는 mirror-symmetric 구조다.
- Encoder는 7 stages of modified Transformer blocks를 사용한다.
- Self-attention 대신 1D depth-wise causal convolution을 사용해 streaming-friendly processing을 노린다.
- Six downsampling layers를 통해 24 kHz input에서 cumulative 3200x downsampling을 달성한다.
- 결과적으로 token/frame rate는 7.5 Hz다.
- Encoder와 decoder component는 각각 약 340M parameters라고 보고된다.
- Training objective는 DAC의 discriminator와 loss design을 따른다고 설명된다.

여기서 중요한 점은 tokenizer가 단순한 compression module이 아니라는 것이다. Long-form generation에서는 tokenizer의 token rate가 곧 model context budget이다. 7.5 Hz라는 수치는 VibeVoice가 64K context 안에서 90분 음성을 논할 수 있게 만드는 핵심 전제다.

### 2) Semantic tokenizer

Semantic tokenizer는 acoustic tokenizer와 별도로 존재한다.

- 구조는 acoustic tokenizer의 encoder를 mirror하지만 VAE component는 사용하지 않는다.
- 목표는 deterministic content-centric feature extraction이다.
- 학습에는 ASR proxy task를 사용한다.
- Training 중에는 Transformer decoder layers로 transcript를 예측하게 하여 semantic representation을 text semantics에 맞춘다.
- Pre-training 이후 이 decoder는 버린다.

이 분리 설계는 꽤 중요하다. Long-form speech는 acoustic quality만으로 충분하지 않다. 어떤 말을 하는지, 어떤 content structure를 유지해야 하는지가 필요하다. Acoustic tokenizer가 음색과 신호 복원을 담당한다면, semantic tokenizer는 content alignment 쪽의 역할을 보완한다.

### 3) Input representation

VibeVoice의 input은 voice font feature와 text script embedding을 speaker role identifier와 함께 구성한다.

논문의 설명을 단순화하면 다음과 같다.

- Speaker1의 voice feature
- Speaker2의 voice feature
- 각 speaker가 말할 text script
- Speaker role identifier
- 이미 생성된 speech segment의 acoustic/semantic representation

이 정보들이 하나의 hybrid context로 들어간다. 즉 model은 "speaker prompt 따로, text 따로, history 따로"가 아니라, long sequence 안에서 이들을 함께 처리한다.

이 방식의 장점은 구조가 단순하다는 것이다. 별도 speaker controller, turn controller, prosody planner를 명시적으로 많이 붙이지 않고, LLM context modeling에 최대한 맡긴다. 반대로 단점도 있다. 이런 단순성이 실제로 안정적으로 작동하려면 tokenizer, training data, context formatting이 모두 충분히 잘 맞아야 한다.

### 4) Token-level diffusion head

VibeVoice의 가장 흥미로운 부분은 LLM output을 acoustic token classification으로 끝내지 않는다는 점이다.

- LLM hidden state $h_i$가 각 token position의 condition으로 사용된다.
- Diffusion head는 clean acoustic VAE feature에 추가된 noise를 예측하도록 학습된다.
- Inference에서는 Gaussian noise vector에서 시작해 target acoustic VAE feature를 iterative denoising으로 생성한다.
- Classifier-Free Guidance를 사용해 conditional prediction과 unconditional prediction을 조절한다.
- Sampling acceleration에는 DPM-Solver++ 계열 sampler를 사용한다.
- 논문 기준 diffusion head는 4 layers다.
- VibeVoice inference에서는 guidance scale 1.3, denoising step 10을 사용한다고 보고된다.

이 설계는 next-token diffusion이라는 이름을 잘 보여준다. Autoregressive model이 다음 token position을 정하지만, 그 token의 값 자체는 continuous latent diffusion으로 생성한다. 즉 sequence modeling은 language model 방식이고, token value generation은 diffusion 방식이다.

# 4. Training / Data / Recipe

## 4-1. Data

논문 본문은 training data의 전체 구성, 규모, filtering rule을 상세하게 공개하지 않는다. 따라서 데이터 관점에서 확정적으로 말할 수 있는 부분은 제한적이다.

확인되는 내용은 다음과 같다.

- Short utterance evaluation에는 SEED test sets를 사용한다.
- Test-en은 CommonVoice에서 뽑은 약 1,000 English samples다.
- Test-zh는 CommonVoice에서 뽑은 약 2,000 Chinese samples다.
- Tokenizer reconstruction evaluation에는 LibriTTS test-clean과 test-other를 사용한다.
- Long podcast evaluation에는 총 약 1시간 분량의 8개 long conversational transcripts를 사용한다.

Training data 자체는 논문에서 충분히 자세히 풀어주지 않는다. 이 점은 재현성 관점에서 아쉬운 부분이다. 특히 long-form multi-speaker audio generation은 data distribution의 영향이 매우 크다. Script style, speaker turn distribution, noise/background 여부, multilingual ratio, role assignment quality가 모두 결과에 영향을 줄 수 있다.

## 4-2. Training strategy

VibeVoice의 training strategy는 tokenizer pre-training과 VibeVoice training을 분리한다.

### Tokenizer pre-training

- Acoustic tokenizer는 VAE 기반 reconstruction objective와 DAC-style loss를 사용한다.
- Semantic tokenizer는 ASR proxy task를 사용해 text semantics에 맞는 representation을 학습한다.
- Semantic tokenizer의 transcript decoder는 pre-training 이후 제거된다.

### VibeVoice training

- Pre-trained acoustic tokenizer와 semantic tokenizer는 frozen 상태로 둔다.
- Learnable parameter는 LLM과 diffusion head다.
- Core LLM은 Qwen2.5 1.5B와 Qwen2.5 7B version으로 instantiate된다.
- Diffusion head는 4 layers다.
- Input sequence length는 curriculum learning으로 4,096에서 65,536 tokens까지 늘린다.
- Guidance scale은 1.3이다.
- Iterative denoising step은 10이다.

이 recipe의 핵심은 frozen tokenizer다. Speech tokenizer가 안정적으로 speech를 압축하고 복원할 수 있어야, LLM과 diffusion head training이 long-form sequence modeling에 집중할 수 있다.

## 4-3. Engineering notes

### 1) Compression budget이 곧 context budget이다

VibeVoice에서 7.5 Hz tokenizer는 단순한 효율화가 아니다. Long-form audio는 초 단위로 token이 누적된다. Token rate가 높으면 64K context에서도 몇 분을 넘기기 어렵다. 반대로 7.5 Hz 수준까지 낮추면 90분 audio가 context modeling 범위 안으로 들어온다.

물론 이 계산은 실제 input formatting, speaker prompt, text script token, special token, generated history 구성에 따라 달라질 수 있다. 따라서 논문의 90분 주장은 architecture와 tokenizer compression이 결합된 system-level claim으로 읽는 편이 좋다.

### 2) LLM은 acoustic generator가 아니라 context planner에 가깝다

VibeVoice에서 LLM은 waveform을 직접 만들지 않는다. LLM은 speaker role, text script, semantic/acoustic context를 통합해 hidden state를 만든다. 실제 acoustic detail은 diffusion head와 acoustic decoder가 담당한다.

이 역할 분담은 실용적이다. LLM은 text and dialogue context modeling에 강하고, diffusion은 continuous signal detail에 강하다. VibeVoice는 이 둘을 token position 단위로 연결한다.

### 3) Simplicity가 장점이지만, prompt/interface 의존도도 커진다

논문은 voice latent feature와 text script를 하나의 sequence로 concat하는 단순한 구조를 강조한다. 이는 복잡한 controller를 줄이는 장점이 있다. 하지만 실제 사용에서는 speaker label, punctuation, turn formatting, voice prompt quality가 결과를 크게 좌우할 가능성이 있다. Model card에서도 text normalization을 별도로 하지 않는다고 설명하므로, input formatting은 사용자가 더 신경 써야 한다.

# 5. Evaluation

## 5-1. Main results

VibeVoice의 evaluation은 크게 세 축으로 나뉜다.

1. Long podcast generation
2. Short utterance generation
3. Tokenizer reconstruction

### 1) Long podcast generation

Long-form conversational speech generation 평가에서는 최근 open-source와 proprietary system을 비교한다. 비교군에는 Nari Labs Dia, Mooncast, SesameAILabs-CSM, Higgs Audio V2, Elevenlabs v3 alpha, Gemini 2.5 pro preview tts가 포함된다.

평가 설정은 다음과 같다.

- 8개 long conversational transcripts
- 총 duration 약 1시간
- Speech prompt를 사용해 timbre consistency를 맞춤
- Gemini 2.5 pro preview tts는 speech-prompt control을 지원하지 않아 default male/female voice 사용
- Objective metric: WER by Whisper-large-v3, WER by Nemo ASR, speaker similarity by WavLM-large
- Subjective metric: 24 human annotators, MOS over Realism, Richness, Preference

핵심 결과만 줄이면 다음과 같다.

| Model | Average MOS | WER Whisper | WER Nemo | SIM |
| --- | ---: | ---: | ---: | ---: |
| SesameAILabs-CSM | 2.89 | 2.66 | 3.05 | 0.685 |
| Higgs Audio V2 | 2.99 | 5.94 | 5.97 | 0.543 |
| Elevenlabs v3 alpha | 3.40 | 2.39 | 2.47 | 0.623 |
| Gemini 2.5 pro preview tts | 3.66 | 1.73 | 2.43 | - |
| VibeVoice-1.5B | 3.54 | 1.11 | 1.82 | 0.548 |
| VibeVoice-7B | 3.76 | 1.29 | 1.95 | 0.692 |

이 표에서 흥미로운 점은 VibeVoice-1.5B와 VibeVoice-7B의 trade-off다. VibeVoice-1.5B는 WER에서 더 낮은 값을 보고하고, VibeVoice-7B는 subjective MOS와 SIM에서 더 높은 값을 보고한다. 즉 scaling은 단순히 transcription accuracy만 올리는 것이 아니라, richness, realism, speaker similarity 쪽에 더 잘 반영되는 것으로 보인다.

다만 이 평가는 compact test set이라는 점을 같이 봐야 한다. 8개 transcript, 총 1시간, 24 annotators는 long-form TTS에서 의미 있는 출발점이지만, 다양한 domain, speaker style, language mixture, noisy prompt 조건을 모두 대표한다고 보기는 어렵다.

### 2) Short utterance generation

SEED test sets에서는 약 1,000 English samples와 약 2,000 Chinese samples를 사용한다. Table 2 기준 VibeVoice-1.5B는 7.5 frame rate에서 다음 결과를 보고한다.

| Model | Frame rate | test-zh CER | test-zh SIM | test-en WER | test-en SIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| MaskGCT | 50 | 2.27 | 0.774 | 2.62 | 0.714 |
| Seed-TTS | - | 1.12 | 0.796 | 2.25 | 0.762 |
| FireRedTTS | 25 | 1.51 | 0.635 | 3.82 | 0.460 |
| CosyVoice 2 | 25 | 1.45 | 0.748 | 2.57 | 0.652 |
| Spark TTS | 50 | 1.20 | 0.672 | 1.98 | 0.584 |
| VibeVoice-1.5B | 7.5 | 1.16 | 0.744 | 3.04 | 0.689 |

여기서는 VibeVoice가 모든 metric에서 최고라고 읽으면 안 된다. Chinese CER은 Seed-TTS와 Spark TTS에 가까운 강한 수치를 보이지만, English WER은 Spark TTS, Seed-TTS, CosyVoice 2보다 높다. 대신 VibeVoice는 frame rate가 7.5로 훨씬 낮다. 이 실험의 핵심은 short utterance absolute score보다, long-form 중심 모델이 ultra-low frame rate에서도 short-form benchmark에서 어느 정도 generalization을 보인다는 점이다.

### 3) Tokenizer reconstruction

Tokenizer reconstruction은 LibriTTS test-clean과 test-other에서 PESQ, STOI, UTMOS를 측정한다.

| Tokenizer | Nq | Token rate | test-clean PESQ | test-clean STOI | test-clean UTMOS | test-other PESQ | test-other STOI | test-other UTMOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Encodec | 8 | 600 | 2.720 | 0.939 | 3.040 | 2.682 | 0.924 | 2.657 |
| DAC | 4 | 400 | 2.738 | 0.928 | 3.433 | 2.595 | 0.908 | 2.945 |
| WavTokenizer | 1 | 75 | 2.373 | 0.914 | 4.049 | 2.261 | 0.891 | 3.431 |
| WavTokenizer | 1 | 40 | 1.703 | 0.862 | 3.602 | 1.662 | 0.834 | 3.055 |
| VibeVoice acoustic | 1 | 7.5 | 3.068 | 0.828 | 4.181 | 2.848 | 0.823 | 3.724 |

이 표는 VibeVoice의 tokenizer 주장을 이해하는 데 가장 중요하다. VibeVoice acoustic tokenizer는 token rate 7.5에서도 PESQ와 UTMOS에서 강한 결과를 보고한다. 다만 STOI는 Encodec이나 DAC보다 낮다. 따라서 이 결과는 "모든 reconstruction metric에서 압도적"이라기보다, long-form generation에 필요한 compression rate를 확보하면서 perceptual quality 지표를 높게 유지했다는 식으로 읽는 것이 안전하다.

## 5-2. What really matters in the experiments

### 1) 90분 생성 claim의 핵심은 tokenizer다

90분 speech generation은 모델이 단순히 긴 audio를 출력했다는 결과가 아니다. 64K context window에서 speech를 다룰 수 있으려면, speech token rate가 극단적으로 낮아야 한다. VibeVoice에서 가장 중요한 실험은 subjective MOS보다도 Table 3의 tokenizer reconstruction일 수 있다. 7.5 Hz가 무너지면 long-form claim도 같이 약해진다.

### 2) Subjective score는 중요하지만 compact setting이다

Long-form TTS는 WER만으로 평가하기 어렵다. Realism, richness, preference처럼 사람이 듣고 판단해야 하는 요소가 핵심이다. VibeVoice-7B가 subjective average 3.76을 기록한 것은 의미 있다. 하지만 8개 transcript와 24 annotators라는 설정은 넓은 domain coverage를 보장하지 않는다. 특히 podcast style, audiobook style, spontaneous dialogue style, multilingual dialogue는 분포가 다를 수 있다.

### 3) WER와 MOS가 같은 방향으로만 움직이지 않는다

VibeVoice-1.5B는 WER에서 더 좋은 값을 보이고, VibeVoice-7B는 MOS와 SIM에서 더 좋다. 이는 speech generation 평가에서 transcription accuracy 하나만 보면 안 된다는 점을 보여준다. Listener가 선호하는 speech는 단순히 ASR이 잘 읽는 speech가 아니라, natural turn, timbre richness, emotion, pacing이 함께 좋은 speech다.

### 4) Proprietary baseline 비교는 조건 차이를 함께 봐야 한다

Gemini 2.5 pro preview tts는 speech-prompt control을 지원하지 않아 default voices를 사용했다고 논문이 설명한다. 이 때문에 SIM 비교가 비어 있다. Proprietary system과 open model을 비교할 때는 input control interface가 다르면 metric interpretation이 달라진다.

# 6. Limitations

1. Language coverage가 제한적이다.
   - 논문은 English와 Chinese 외 언어 transcript가 unexpected audio output을 만들 수 있다고 명시한다.
   - Cross-lingual capability가 보인다고 해도, 안정적 multilingual TTS로 해석하면 안 된다.

2. Non-speech audio를 명시적으로 다루지 않는다.
   - Background noise, music, sound effect generation은 목표가 아니다.
   - Model card 예시에서 spontaneous singing이나 BGM 현상이 언급되더라도, 이는 controllable non-speech generation capability로 보면 안 된다.

3. Overlapping speech를 명시적으로 modeling하지 않는다.
   - 실제 대화나 회의에서는 말 겹침이 자주 발생한다.
   - VibeVoice는 multi-speaker turn-taking에는 강점을 보이지만, simultaneous speech까지 안정적으로 생성한다고 보기는 어렵다.

4. Safety risk가 크다.
   - High-quality synthetic voice는 impersonation, fraud, disinformation에 악용될 수 있다.
   - 논문과 model card 모두 responsible use를 강조한다.
   - 상용 또는 실제 서비스 적용에는 consent, watermarking, disclosure, abuse monitoring, jurisdiction-specific compliance가 필요하다.

5. Training data와 ablation이 제한적으로 공개된다.
   - Tokenizer와 architecture 설명은 비교적 명확하지만, long-form training data distribution과 filtering rule은 충분히 자세하지 않다.
   - 각 구성 요소가 얼마나 기여했는지 보여주는 ablation도 제한적이다.
   - 특히 semantic tokenizer, diffusion head depth, context curriculum, token rate 선택에 대한 세부 ablation이 더 있으면 좋았을 것이다.

6. Evaluation coverage가 아직 좁다.
   - Long podcast evaluation은 compact test set이다.
   - Real-world deployment에서는 speaker prompt quality, text formatting, multilingual script, noisy prompt, very long generation stability, latency, memory usage가 모두 중요하다.
   - 논문 결과만으로 production reliability를 단정하기는 어렵다.

# 7. My Take

## 7-1. Why this matters for my work

VibeVoice는 speech generation을 LLM system design 관점에서 보게 만든다. 이 논문의 가장 큰 가치는 TTS 품질 수치 자체보다, long-form audio를 LLM context 안으로 가져오기 위해 어떤 representation budget이 필요한지 보여준다는 점이다.

LLM 기반 multimodal generation에서 자주 나오는 문제는 modality마다 token rate가 다르다는 것이다. Text는 매우 압축된 symbolic sequence지만, audio와 video는 훨씬 길고 dense하다. VibeVoice의 7.5 Hz tokenizer는 이 문제에 대한 하나의 강한 답이다. Audio를 text와 비슷한 token budget으로 낮춰야 LLM의 long-context ability를 제대로 활용할 수 있다.

또한 next-token diffusion이라는 설계도 중요하다. Discrete token prediction은 LLM과 잘 맞지만, continuous acoustic detail을 모두 discrete code로 양자화하면 품질 손실이나 codebook bottleneck이 생길 수 있다. VibeVoice는 sequence는 autoregressive하게, token value는 diffusion으로 생성한다. 이 구조는 audio뿐 아니라 image, video, robotics trajectory 같은 continuous modality에도 다시 생각해볼 만한 패턴이다.

## 7-2. Reuse potential

실무나 연구에서 재사용할 수 있는 포인트는 다음과 같다.

- Long-form generation system에서는 tokenizer compression ratio를 first-class design variable로 둬야 한다.
- Multi-speaker generation에서는 speaker prompt와 role assignment를 별도 metadata로 흩뜨리지 말고, sequence input format 자체에 통합하는 방식을 고려할 수 있다.
- LLM은 continuous modality를 직접 출력하기보다, latent generator를 condition하는 planner로 쓰는 편이 안정적일 수 있다.
- Evaluation에서는 WER, SIM, MOS를 함께 봐야 한다. 특히 long-form speech에서는 subjective listening test를 빼기 어렵다.
- Safety layer는 모델 성능 이후에 붙이는 부가 기능이 아니라, voice synthesis product의 기본 설계 조건으로 봐야 한다.

내가 직접 후속 실험을 한다면 다음을 보고 싶다.

1. Token rate를 7.5 Hz, 15 Hz, 30 Hz로 바꿨을 때 long-form coherence와 reconstruction metric이 어떻게 trade-off 되는가.
2. Semantic tokenizer를 제거했을 때 content coherence와 WER가 얼마나 떨어지는가.
3. Diffusion head step 수를 줄였을 때 latency와 MOS가 어떻게 변하는가.
4. Speaker count가 2명에서 4명으로 늘어날 때 speaker confusion rate가 어떻게 변하는가.
5. Input script punctuation과 role formatting이 generation stability에 얼마나 영향을 주는가.

## 7-3. Follow-up papers

- Multimodal Latent Language Modeling with Next-Token Diffusion
- Seed-TTS: A Family of High-Quality Versatile Speech Generation Models
- CosyVoice 2
- WavTokenizer
- SpeechTokenizer
- VibeVoice-ASR Technical Report

# 8. Summary

- VibeVoice는 long-form multi-speaker TTS를 LLM 기반 next-token diffusion framework로 다룬다.
- 핵심은 7.5 Hz ultra-low frame rate speech tokenizer이며, 이것이 64K context와 90분 generation claim의 기반이다.
- LLM은 speaker role과 text script를 긴 context에서 처리하고, token-level diffusion head가 acoustic VAE latent를 생성한다.
- 실험에서는 long podcast subjective evaluation, SEED short utterance, LibriTTS tokenizer reconstruction을 통해 quality와 compression trade-off를 보여준다.
- 다만 English/Chinese 중심, non-speech와 overlapping speech 제한, safety risk, compact evaluation, training data 공개 부족은 반드시 같이 봐야 한다.
