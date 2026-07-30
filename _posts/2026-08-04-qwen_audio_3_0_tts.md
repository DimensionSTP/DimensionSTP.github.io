---
layout: single
title: "Qwen-Audio-3.0-TTS Review"
categories: Study-concept
tag: [TTS, SpeechGeneration, VoiceCloning, ReinforcementLearning, AudioAI]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2607.23938)

> 한 줄 요약: Qwen-Audio-3.0-TTS는 12.5 Hz low-frame-rate speech tokenizer, continuous LM hidden-state conditioning, LM과 flow-matching model을 분리해 최적화하는 five-stage training을 결합해 multilingual voice cloning, free-style control, long-form generation, degraded-prompt robustness를 하나의 production-oriented TTS system으로 묶은 기술 보고서다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- TTS를 단순한 text-to-waveform mapping이 아니라 content, speaker, prosody, quality, control, latency, robustness의 multi-objective system으로 다룬다.
- Autoregressive sequence를 12.5 Hz로 줄이면서 커진 FSQ codebook과 supervised audio task로 representation loss를 보완한다.
- Discrete token만 FM에 전달하지 않고 LM의 continuous hidden state를 연결해 quantization bottleneck과 component mismatch를 줄인다.
- LM RL과 FM RL을 분리해 semantic planning과 acoustic realization을 서로 다른 reward space에서 최적화한다.
- Noisy, reverberant, narrow-band prompt를 inference-time denoiser 없이 voice-cloning path 자체에서 처리하도록 학습한다.
- 16개 언어, 20개 중국어 방언 지역, 3분 수준의 one-pass long-form, natural-language style control을 함께 평가한다.

최근 TTS system은 여러 capability를 빠르게 흡수하고 있다. Zero-shot voice cloning, multilingual synthesis, emotion control, non-verbal event, long-form narration, noisy reference adaptation이 하나의 model 안으로 들어간다. 하지만 capability가 많아질수록 하나의 metric만 최적화해서는 전체 품질을 설명하기 어려워진다.

예를 들어 WER를 지나치게 낮추면 발음은 정확하지만 억양이 기계적으로 변할 수 있다. Speaker similarity를 높이면 prompt의 잡음이나 microphone characteristic까지 복제할 수 있다. Token rate를 낮추면 autoregressive latency는 줄지만 semantic and acoustic detail이 빠질 수 있다. Prompt denoising을 강하게 적용하면 음질은 좋아져도 speaker cue가 손실될 수 있다.

Qwen-Audio-3.0-TTS가 흥미로운 지점은 이 trade-off를 하나의 end-to-end loss로 밀어 넣지 않는다는 점이다. 저자들은 LM, FM, tokenizer, vocoder가 담당하는 capability를 나누고, 각 stage에서 특정 module을 freeze하거나 연결 방식을 바꾸면서 순차적으로 문제를 푼다.

즉 이 보고서의 핵심 질문은 다음과 같다.

> Production TTS의 서로 충돌하는 목표를 어떤 representation과 training stage로 분해해야 하는가?

# 1. Problem Setting

## 1-1. Problem definition

Qwen-Audio-3.0-TTS가 겨냥하는 system objective는 하나가 아니다.

| Axis | Production requirement |
| --- | --- |
| Content consistency | Text의 character, word, number, symbol을 빠뜨리지 않아야 한다 |
| Speaker similarity | Reference voice의 timbre와 speaker cue를 유지해야 한다 |
| Prosody | Emotion, rhythm, pause, speed, intonation이 자연스러워야 한다 |
| Audio quality | Noise, artifact, high-frequency defect가 적어야 한다 |
| Controllability | Natural-language instruction과 inline tag를 따라야 한다 |
| Multilinguality | 여러 언어와 cross-lingual cloning을 지원해야 한다 |
| Long-form stability | 수분 길이에서도 content and speaker drift가 작아야 한다 |
| Robustness | Noisy or reverberant reference에서도 안정적으로 clone해야 한다 |
| Efficiency | Autoregressive token sequence와 acoustic decoding cost가 낮아야 한다 |

일반적인 LM-based TTS pipeline은 크게 다음 순서를 가진다.

1. Text와 reference speech를 condition으로 받는다.
2. LM이 discrete speech token을 autoregressive하게 생성한다.
3. Acoustic model이 token을 mel or latent feature로 복원한다.
4. Vocoder가 feature를 waveform으로 바꾼다.

이 구조는 component를 나누기 쉽지만, 세 가지 bottleneck이 생긴다.

첫째, LM decode cost는 speech token rate에 직접 비례한다. 같은 duration의 speech를 25 Hz 대신 12.5 Hz token으로 표현하면 autoregressive step 수를 절반 수준으로 줄일 수 있다. 반대로 token 하나가 더 긴 시간 구간을 대표하므로 information density가 커지고 quantization error가 커질 수 있다.

둘째, discrete token은 LM과 acoustic model 사이의 narrow interface다. Token ID는 content planning에는 유용하지만, speaker, local prosody, instruction nuance와 같은 정보가 codebook에 완전히 보존된다고 보장하기 어렵다.

셋째, LM과 acoustic model이 독립적으로 학습되면 objective mismatch가 생긴다. LM은 token accuracy를 높이지만 FM이 듣기에 좋은 representation을 만들지 않을 수 있고, FM은 ground-truth token distribution에는 잘 맞아도 LM rollout의 error pattern에는 약할 수 있다.

## 1-2. Why previous approaches are insufficient

### 1) Token rate만 낮추면 representation quality가 무너질 수 있다

Low-frame-rate token은 decode latency에 유리하다. 하지만 동일한 codebook size에서 25 Hz를 12.5 Hz로 낮추면 token 하나가 더 많은 content and acoustic information을 담아야 한다.

논문의 tokenizer ablation에서 6,561-code vocabulary를 그대로 둔 12.5 Hz variant는 25 Hz reference보다 ASR error와 speaker similarity가 모두 나빠진다. 즉 low rate 자체가 공짜 효율화는 아니다.

### 2) Discrete token interface만으로는 rich acoustic context가 부족할 수 있다

LM output token을 embedding으로 바꾸어 FM에 주는 cascade는 modular하다. 하지만 quantization 과정에서 사라진 information은 FM이 다시 복원하기 어렵다.

특히 free-style instruction, fine-grained emotion, pause timing, speaker detail을 다루려면 token ID 외의 continuous context가 필요할 수 있다.

### 3) WER-only optimization은 naturalness와 diversity를 훼손할 수 있다

TTS RL에서 ASR-based reward만 쓰면 content consistency는 개선될 수 있다. 그러나 model이 지나치게 보수적인 pronunciation과 timing으로 수렴하거나, duration collapse, repeated token, missing stop token 같은 behavior를 만들 수 있다.

따라서 reward는 content만 아니라 duration, diversity, prosody를 함께 봐야 한다.

### 4) Prompt enhancement를 별도 denoiser로 두면 speaker cue가 손실될 수 있다

Noisy reference를 먼저 clean speech로 바꾼 뒤 cloning하는 pipeline은 직관적이다. 하지만 denoiser가 noise와 함께 speaker-specific spectral cue도 제거할 수 있다. 실제 비교에서도 denoising mode가 DNSMOS를 높이는 대신 speaker similarity를 낮추는 경우가 관찰된다.

### 5) LM과 FM을 한 번에 RL하는 것은 credit assignment가 어렵다

Waveform-level reward는 최종 결과를 직접 측정하지만 rollout cost가 크고, 어떤 error가 LM token planning에서 왔는지 FM acoustic reconstruction에서 왔는지 분리하기 어렵다.

Qwen-Audio-3.0-TTS는 LM RL과 FM RL을 다른 stage로 분리한다. Semantic token quality는 token-domain reward로 먼저 최적화하고, waveform quality는 FM rollout에서 별도로 다룬다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 네 가지로 정리할 수 있다.

### 1) 12.5 Hz supervised speech tokenizer

- Causal SenseVoice encoder와 FSQ를 사용한다.
- ASR, LID, SER, AED, speaker analysis, general audio analysis를 함께 학습한다.
- 10-dimensional FSQ에 dimension당 3 levels를 사용해 59,049-code vocabulary를 만든다.
- Continuous representation을 먼저 학습한 뒤 quantization을 활성화하는 curriculum을 사용한다.

### 2) Continuous hidden-state conditioned LM-FM architecture

- LM은 semantic token을 autoregressive하게 예측한다.
- FM은 discrete token embedding뿐 아니라 LM continuous hidden state를 condition으로 받는다.
- FM reconstruction loss가 hidden-state path를 통해 upstream LM representation에도 영향을 준다.
- Causal BigVGAN vocoder가 waveform을 생성한다.

### 3) Five-stage progressive training

1. LM and FM independent pretraining
2. Joint LM-FM training with high-quality annealing
3. LM reinforcement learning
4. FM acoustic robustness training
5. FM reinforcement learning

각 stage는 이전 checkpoint에서 시작하고, 어떤 module을 freeze할지와 어떤 reward를 쓸지를 달리한다.

### 4) Deployment-oriented evaluation and adaptation

- Standard zero-shot benchmark뿐 아니라 text normalization, long-form, acoustic robustness, instruction following을 별도 diagnostic set으로 평가한다.
- 16-language multilingual cloning과 12-direction cross-lingual cloning을 본다.
- Speaker adaptation을 joint LM-FM SFT와 FM-only SFT의 두 단계로 구성한다.

## 2-2. Design intuition

이 system의 설계 직관은 capability를 module ownership에 맞춰 분해하는 것이다.

| Capability | Main owner | Main optimization stage |
| --- | --- | --- |
| Text and semantic planning | LM | Independent pretraining, joint training, LM RL |
| Token efficiency | Speech tokenizer | Continuous-to-quantized tokenizer training |
| Acoustic reconstruction | FM | Independent pretraining, joint training |
| Noisy prompt recovery | FM | Robustness training |
| Speaker similarity and perceptual quality | FM | FlowTTS-GRPO |
| Waveform detail | Vocoder | Causal vocoder and adaptation vocoder training |
| Style and dialect control | LM plus FM | Data curriculum and attribute-aware alignment |

첫째, low-frame-rate token은 semantic planning의 sequence length를 줄인다. 이때 codebook capacity와 supervised task를 늘려 compression loss를 보완한다.

둘째, discrete token은 stable semantic supervision으로 유지하되, FM에는 continuous hidden state도 전달한다. 즉 interface를 token-only에서 token plus hidden state로 확장한다.

셋째, general coverage와 high-quality speech를 같은 mixture weight로 끝까지 학습하지 않는다. Broad data로 coverage를 만든 뒤 clean and expressive subset으로 anneal한다.

넷째, LM과 FM의 RL objective를 분리한다. LM은 waveform을 전부 생성하지 않고 token domain에서 빠르게 rollout할 수 있고, FM은 fixed LM condition 아래 acoustic objective에 집중할 수 있다.

이 논문의 가장 중요한 부분은 새로운 단일 module보다 이 ownership 분해다. Production TTS의 실패를 하나의 model score로 보지 않고, semantic planning, acoustic realization, prompt robustness, waveform quality를 서로 다른 optimization problem으로 취급한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Multilingual, controllable, long-form, robust zero-shot TTS |
| Tokenizer | 12.5 Hz supervised tokenizer with 59,049-code FSQ |
| Semantic generator | Autoregressive bi-streaming LM |
| Acoustic generator | Chunk-based flow-matching model |
| LM-FM bridge | Discrete semantic path plus continuous LM hidden-state conditioning |
| Waveform decoder | Causal BigVGAN vocoder |
| Control | Free-style natural language and fine-grained inline tags |
| Language coverage | 16 languages |
| Dialect coverage | 20 Chinese dialect regions |
| Long-form target | One-pass synthesis up to about 3 minutes |
| Training | Five progressive stages plus two-stage speaker adaptation |

## 3-2. Module breakdown

### 1) Low-frame-rate speech tokenizer

Tokenizer는 Mel feature를 12.5 Hz discrete sequence로 바꾼다. Frame rate가 낮을수록 LM이 생성해야 하는 token 수가 줄어든다.

Codebook size는 다음과 같다.

$$
|\mathcal{V}_{\mathrm{speech}}| = 3^{10} = 59{,}049
$$

위 수식의 의미는 10-dimensional FSQ가 각 dimension에서 3개 level을 선택한다는 것이다. Product space를 통해 59,049개의 discrete combination을 만든다.

Tokenizer training은 두 단계로 볼 수 있다.

1. Continuous training
   - FSQ를 bypass한다.
   - Tokenizer component를 직접 학습한다.
   - Qwen2.5-7B-Instruct initialized LM은 LoRA로 adaptation한다.

2. Quantization training
   - FSQ를 활성화한다.
   - LM weight를 freeze한다.
   - Supervised audio task의 cross-entropy objective로 discrete representation을 정착시킨다.

이 curriculum은 처음부터 hard quantization을 강제해 representation learning을 불안정하게 만드는 것을 피한다.

### 2) Language model

LM은 text와 prompt context를 받아 semantic speech token을 예측한다. 이 단계가 pronunciation, duration planning, pause progression, style realization의 high-level path를 담당한다.

논문은 LM을 bi-streaming이라고 설명한다. 핵심은 streaming-oriented token generation을 유지하면서, low-rate token으로 autoregressive burden을 줄이는 것이다.

LM output은 두 경로로 쓰인다.

- Discrete path: semantic token prediction objective
- Continuous path: hidden state를 FM condition으로 전달

Discrete path는 stable target을 제공하고, continuous path는 codebook에 압축되지 않은 rich context를 전달한다.

### 3) Flow-matching model

FM은 semantic condition과 prompt information을 사용해 acoustic feature를 복원한다. Condition에는 다음 정보가 포함된다.

- LM continuous hidden state
- Prompt mel feature
- Speaker embedding
- Generated semantic structure

Joint training에서 FM loss는 LM hidden representation까지 gradient를 전달한다. 따라서 LM은 token accuracy만 맞추는 representation이 아니라 downstream acoustic reconstruction에 유용한 representation을 학습할 수 있다.

### 4) Causal BigVGAN vocoder

Vocoder는 FM output을 waveform으로 바꾼다. Causal design은 streaming and low-latency synthesis에 유리하다.

Speaker adaptation에서는 별도의 48 kHz super-resolution vocoder를 사용한다. Multi-scale STFT discriminator를 사용해 여러 time-frequency resolution에서 adversarial supervision을 주고, high-frequency stripe artifact를 줄인다.

### 5) Control interface

Model은 두 종류의 control을 함께 지원한다.

1. Free-style natural-language instruction
   - Speaker role
   - Emotion
   - Speaking style
   - Rate
   - Timbre
   - Accent

2. Fine-grained inline tag
   - Local emotion
   - Style transition
   - Speed change
   - Laughter
   - Coughing
   - Breathing
   - Sighing

Natural-language instruction은 global behavior를 설명하기 좋고, inline tag는 utterance 내부의 local event를 지정하기 좋다. 두 interface를 함께 두는 것은 control granularity를 분리한다는 의미가 있다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 training data를 다음 다섯 capability axis로 설명한다.

### 1) Multilingual and dialect coverage

- 16 languages
- 20 Chinese dialect regions
- Malay, Tagalog, Arabic, Portuguese, Indonesian, Thai, Vietnamese가 predecessor 대비 추가된다.

### 2) Free-style instruction following

- Role
- Emotion
- Style
- Speaking rate
- Timbre
- Accent

### 3) Fine-grained control

- Emotion, style, speed의 localized tag
- Laughter, cough, breath, sigh와 같은 non-verbal event

### 4) Long-form speech

- Narration-like data
- Single-pass generation up to about 3 minutes

### 5) Hard-case robustness

- Polyphonic character
- Rare and archaic character
- Number and symbol normalization
- LaTeX mathematical expression
- Degraded reference speech

중요한 점은 논문이 전체 audio hour, source composition, filtering ratio, model parameter count, stage별 compute를 공개하지 않는다는 것이다. 따라서 recipe의 구조는 상세하지만 scale reproducibility는 제한된다.

## 4-2. Five-stage progressive training

### Stage 1) Independent LM and FM pretraining

LM과 FM을 large-scale diverse speech data에서 따로 학습한다.

- LM: text and prompt context에서 semantic token을 예측한다.
- FM: tokenizer-derived token에서 continuous acoustic feature를 복원한다.

장점은 안정적인 initialization과 modular scaling이다. LM은 linguistic planning에 집중하고, FM은 acoustic fidelity와 speaker reconstruction에 집중할 수 있다.

### Stage 2) Joint LM-FM training with high-quality annealing

독립 pretrained checkpoint를 연결해 end-to-end로 학습한다.

Joint objective는 개념적으로 다음처럼 볼 수 있다.

$$
\mathcal{L}_{\mathrm{joint}}
=
\mathcal{L}_{\mathrm{token}}
+
\lambda_{\mathrm{FM}}\mathcal{L}_{\mathrm{flow}}
$$

논문은 exact coefficient를 공개하지 않는다. 중요한 구조는 FM이 discrete token embedding 대신 LM hidden state를 직접 condition으로 받는다는 점이다.

Training은 두 distribution phase를 가진다.

1. Broad-coverage mixture
   - Language, speaker, style coverage를 만든다.

2. High-quality annealing
   - Clean and expressive subset 비중을 높인다.
   - Naturalness, fidelity, expression, instruction realization을 강화한다.

### Stage 3) Language model reinforcement learning

FM과 vocoder를 freeze하고 LM만 최적화한다. Online GRPO는 frozen reference policy에 대한 KL regularization과 함께 사용된다.

Base reward는 다음과 같다.

$$
R_{\mathrm{base},i}
=
\lambda_{\mathrm{content}}R_{\mathrm{content},i}
+
\lambda_{\mathrm{dur}}R_{\mathrm{dur},i}
+
\lambda_{\mathrm{div}}R_{\mathrm{div},i}
+
\lambda_{\mathrm{prosody}}R_{\mathrm{prosody},i}
$$

각 reward의 역할은 다음과 같다.

- Content: token-domain ASR로 text consistency를 본다.
- Duration: 지나치게 짧거나 긴 rollout을 억제한다.
- Diversity: mechanical collapse를 줄인다.
- Prosody: alignment progression과 pause timing을 평가한다.

이 reward는 FM and vocoder inference 전에 계산된다. Waveform rollout을 매번 수행하지 않아도 되므로 RL cost를 줄일 수 있다.

추가로 Gumbel-Softmax 기반 DiffRO branch를 사용한다.

$$
\mathcal{L}_{\mathrm{RL}}
=
\mathcal{L}_{\mathrm{GRPO}}
+
\lambda_{\mathrm{diff}}\mathcal{L}_{\mathrm{DiffRO}}^{+}
$$

GRPO는 sequence-level relative preference를 제공하고, DiffRO는 선택된 token에 differentiable corrective gradient를 제공한다. DiffRO는 group-relative advantage가 non-negative인 candidate에만 적용한다.

Extreme repetition이나 missing stop token rollout은 GRPO update에서 제외한다. 이는 low-quality outlier가 group statistic과 policy gradient를 오염시키는 것을 막기 위한 filtering이다.

LM RL은 curriculum으로 진행된다.

1. General generation optimization
   - Instruction, fine-grained control, dialect sample을 우선 제외한다.
   - Base reward가 측정하지 못하는 attribute를 잘못 최적화하는 것을 피한다.

2. Multi-task alignment
   - Dialect classification correctness를 attribute reward로 추가한다.
   - General robustness를 유지하면서 dialect authenticity를 강화한다.

### Stage 4) Acoustic robustness training with frozen LM

LM을 freeze하고 FM만 degraded prompt에 적응시킨다.

Augmentation에는 다음이 포함된다.

- Additive noise
- Reverberation
- Phone, Bluetooth, laptop microphone response
- Far-field recording
- Mask or hand blockage
- Codec, DAC, amplifier artifact
- Packet loss
- Strong echo
- Compound corruption

핵심은 clean-up module을 inference path 밖에 따로 두지 않는다는 점이다. FM이 degraded prompt에서 clean high-quality target을 복원하도록 직접 학습한다.

### Stage 5) Flow-matching reinforcement learning

LM을 freeze하고 FM을 FlowTTS-GRPO로 최적화한다. Deterministic ODE sampling을 early step에서 marginal-preserving SDE로 바꾸어 on-policy exploration을 만든다.

간단히 쓰면 stochastic update는 다음 형태다.

$$
x_{t+\Delta t}
=
x_{t,\mathrm{mean}}
+
\sigma_t\sqrt{\Delta t}\epsilon
$$

$$
\sigma_t
=
a\sqrt{\frac{1-t}{t}},
\qquad
\epsilon \sim \mathcal{N}(0,\mathbf{I})
$$

Prompt마다 $G$개의 waveform을 생성하고 group-normalized advantage를 계산한다.

$$
\hat{A}^{i}
=
\frac{R(\hat{x}_{1}^{i},c)-\mathrm{mean}_{j}R(\hat{x}_{1}^{j},c)}
{\mathrm{std}_{j}R(\hat{x}_{1}^{j},c)}
$$

Reward는 세 축을 결합한다.

$$
R
=
\lambda_1\frac{R_{\mathrm{SS}}}{\mathrm{std}(R_{\mathrm{SS}})}
+
\lambda_2\frac{R_{\mathrm{ASR}}}{\mathrm{std}(R_{\mathrm{ASR}})}
+
\lambda_3\frac{R_{\mathrm{MOS}}}{\mathrm{std}(R_{\mathrm{MOS}})}
$$

- $R_{\mathrm{SS}}$: speaker verification similarity
- $R_{\mathrm{ASR}}$: intelligibility and content
- $R_{\mathrm{MOS}}$: DNSMOS-based perceptual quality

각 reward를 batch standard deviation으로 normalize해 raw scale 차이가 weight interpretation을 지배하지 않게 한다.

SDE exploration은 early-step window에만 적용하고 later step은 ODE로 돌아간다. Training rollout에서는 classifier-free guidance를 사용하지 않아 exploration을 넓힌다.

## 4-3. Speaker adaptation

Speaker adaptation은 두 단계 SFT다.

### Stage 1) Joint adaptation

- LM과 FM을 함께 fine-tune한다.
- Target speaker data와 duration-matched replay subset을 섞는다.
- Replay subset을 round마다 refresh한다.
- Language and expression coverage collapse를 줄인다.

### Stage 2) FM-only refinement

- LM을 freeze한다.
- Target speaker speech만 사용한다.
- Timbre와 local prosody를 집중적으로 맞춘다.

이 구조는 speaker-specific adaptation이 semantic capability를 과도하게 덮어쓰지 않게 한다.

# 5. Evaluation

## 5-1. Evaluation setup

Standard benchmark와 자체 diagnostic benchmark를 함께 사용한다.

| Evaluation | Main question |
| --- | --- |
| SEED-TTS-Eval | Zero-shot content and speaker similarity |
| CV3-Eval | Multilingual and cross-lingual voice cloning |
| Text normalization | Number, code, symbol, formula를 올바르게 읽는가 |
| Long-form | One-pass 1.5-3 minute synthesis가 안정적인가 |
| Acoustic robustness | Degraded prompt에서도 content, speaker, quality를 지키는가 |
| Instruction following | Natural-language and structured control을 따르는가 |
| Speaker adaptation | Target speaker SFT가 content를 유지하는가 |
| Human evaluation | Dialect authenticity, pronunciation, prosody를 사람이 어떻게 평가하는가 |

Content는 Chinese CER와 English WER로 평가한다. Speaker similarity에는 ERes2Net과 WavLM이 사용되고, audio quality에는 DNSMOS가 사용된다.

여기서 metric disagreement를 주의해야 한다. 같은 system도 WavLM과 ERes2Net ranking이 다르다. Speaker similarity를 하나의 encoder score로 단정하면 안 된다.

## 5-2. Tokenizer ablation

같은 12.5 Hz에서 codebook을 키우면 low-rate compression loss가 회복된다.

| Tokenizer | Codebook | zh CER | zh SIM | en WER | en SIM | hard CER | hard SIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12.5 Hz small | 6,561 | 2.59 | 72.44 | 3.21 | 61.64 | 7.94 | 69.78 |
| 12.5 Hz medium | 19,683 | 1.48 | 83.25 | 2.56 | 77.58 | 6.70 | 80.85 |
| 12.5 Hz final | 59,049 | 1.23 | 83.09 | 2.37 | 77.49 | 6.68 | 80.61 |

59,049 code configuration은 content consistency가 가장 좋다. 19,683 configuration은 일부 speaker similarity에서 아주 조금 높다. 따라서 final choice는 content, similarity, frame rate의 trade-off다.

이 ablation은 논문의 가장 명확한 causal evidence 중 하나다. 단순히 frame rate를 낮춘 것이 아니라 code capacity를 함께 바꾸어야 한다는 주장을 직접 지지한다.

## 5-3. Zero-shot TTS on SEED-TTS-Eval

Qwen-Audio-3.0-TTS의 주요 결과는 다음과 같다.

| Split | Content error | WavLM SIM | ERes2Net SIM |
| --- | ---: | ---: | ---: |
| test-zh | CER 0.84 | 0.792 | 0.847 |
| test-en | WER 1.54 | 0.762 | 0.815 |
| test-hard | CER 7.00 | 0.768 | 0.824 |

이 결과를 모든 column의 단일 SOTA로 읽으면 안 된다. 일부 system은 특정 CER or WER가 더 낮다. Qwen-Audio-3.0-TTS는 content와 speaker similarity의 balance를 주장한다.

특히 ERes2Net 기준으로 세 split 모두 가장 높은 similarity를 보고하지만, WavLM ranking은 다르다. 논문도 두 metric이 complementary aspect를 본다고 해석한다.

## 5-4. Multilingual and cross-lingual voice cloning

CV3-Eval multilingual subset은 16개 언어를 포함한다. Qwen-Audio-3.0-TTS는 Japanese, Korean, Russian, Arabic, Malay, Thai를 포함한 여러 언어에서 가장 낮은 error를 보이고, 나머지 언어에서도 competitive한 결과를 보고한다.

Cross-lingual subset에서는 12개 transfer direction 중 8개에서 best, 4개에서 second-best다. Reference language와 generated language가 다를 때도 speaker cue를 유지하면서 target-language content를 생성하는 능력을 본다.

다만 language별 test size, training resource balance, accent diversity가 같은지는 별개 문제다. 16-language support를 동일한 품질 보장으로 해석해서는 안 된다.

## 5-5. Text normalization

Qwen-Audio-TTS-Eval의 text normalization set은 1,375개의 Chinese and English case로 구성된다.

Category는 다음과 같다.

- Number, date, time
- Currency and financial expression
- Acronym and mixed reading
- Serial number, code, address
- Formula, unit, symbol

Overall accuracy는 Chinese 68.7%, English 65.7%로 비교 model 중 가장 높다.

하지만 scoring은 Gemini-2.5-Pro가 original text, ASR transcript, acceptable verbalization list를 보고 binary judgment를 내리는 방식이다. 따라서 이 결과는 evaluator prompt, acceptable answer coverage, ASR transcript quality의 영향을 함께 받는다.

## 5-6. One-pass long-form generation

Chinese and English 각각 100개 paragraph-level input을 사용한다. External segmentation이나 audio stitching 없이 one-pass로 생성한다.

| Language | Content error | Prompt SIM | Segment SIM |
| --- | ---: | ---: | ---: |
| Chinese | CER 2.22 | 78.85 | 93.16 |
| English | WER 5.00 | 82.35 | 93.45 |

Prompt SIM은 reference prompt와 generated segment의 similarity다. Segment SIM은 같은 generated utterance 내부 segment끼리의 consistency다.

Qwen-Audio-3.0-TTS는 CosyVoice3보다 content error를 크게 줄이면서 높은 segment-level speaker consistency를 유지한다. 다만 English에서는 length bucket에 따라 WER가 단조롭게 좋아지는 것은 아니므로, `3분 생성 가능`과 `3분 전체가 균일하게 정확`은 구분해야 한다.

## 5-7. Acoustic robustness

Noisy, reverberant, unclear prompt에서 inference-time denoising 없이 평가한다.

| Condition | WER | SIM | DNSMOS |
| --- | ---: | ---: | ---: |
| Noisy | 1.18 | 76.14 | 3.962 |
| Reverb | 0.69 | 74.12 | 3.925 |
| Unclear | 1.61 | 76.53 | 3.305 |

중요한 해석은 한 metric의 최고점보다 trade-off다. 일부 system은 denoising mode에서 WER or DNSMOS를 더 높이지만 speaker similarity가 크게 떨어진다. Qwen-Audio-3.0-TTS는 prompt enhancement를 FM training에 통합해 quality와 speaker preservation을 함께 유지하려 한다.

## 5-8. Instruction following

440개의 bilingual zero-shot voice-cloning case를 사용한다.

- SA: single attribute
- NL: natural-language multi-attribute
- ST: structured multi-attribute

Overall instruction-following score는 Chinese 78.94, English 80.45다. 비교된 IndexTTS2와 CosyVoice3보다 aggregate가 높다.

다만 speaker similarity는 CosyVoice3보다 낮은 column이 있다. Control을 더 강하게 반영하는 과정에서 prompt voice preservation과 trade-off가 남아 있음을 보여준다.

Evaluator는 Gemini-2.5-Pro이며 human calibration에서 single-attribute agreement는 70.0%, complex criterion-level agreement는 56.7%다. 따라서 small score gap은 과도하게 해석하면 안 된다.

## 5-9. Dialect subjective evaluation

Human rating 결과는 다음과 같다.

| Dimension | Perfect | Mean score |
| --- | ---: | ---: |
| Dialect authenticity | 66.7% | 3.639 |
| Pronunciation accuracy | 93.5% | 3.935 |
| Prosodic naturalness | 68.1% | 3.680 |

Pronunciation은 안정적이지만 dialect authenticity와 prosodic naturalness에는 mild error가 더 남는다. 즉 dialect character를 정확히 읽는 능력과 해당 지역의 자연스러운 말투를 재현하는 능력은 분리해서 봐야 한다.

## 5-10. What really matters in the experiments

### 1) Low-rate tokenizer claim에는 직접 ablation이 있다

Frame rate와 codebook size를 바꾼 비교가 있어 efficiency-quality trade-off를 확인할 수 있다.

### 2) Five-stage recipe 전체의 causal attribution은 제한적이다

Tokenizer ablation은 명확하지만, Stage 2부터 Stage 5까지 각 stage를 하나씩 제거한 full system ablation은 충분히 제공되지 않는다. 최종 성능이 어느 stage에서 얼마나 왔는지 정밀하게 분리하기 어렵다.

### 3) Aggregate performance는 single-metric domination과 다르다

Qwen-Audio-3.0-TTS는 여러 benchmark에서 강하지만 모든 metric의 best는 아니다. 저자도 content, naturalness, speaker similarity의 balance를 강조한다.

### 4) Deployment diagnostic은 유용하지만 in-house 비중이 크다

Text normalization, long-form, acoustic robustness, instruction following은 실무적이다. 반면 benchmark construction과 evaluator protocol의 external reproduction이 필요하다.

# 6. Limitations

1. Training scale이 공개되지 않는다.
   - 전체 audio hour, language별 data ratio, stage별 sample count, compute, model parameter가 명시되지 않는다.
   - Recipe를 구조적으로 이해할 수는 있지만 independent reproduction은 어렵다.

2. Five-stage contribution을 분리한 ablation이 제한적이다.
   - Joint training, high-quality annealing, LM RL, robustness training, FM RL의 incremental gain을 같은 table에서 비교하지 않는다.
   - 어떤 stage가 어떤 benchmark gain을 만들었는지 정량적 attribution이 부족하다.

3. 자체 benchmark와 LLM judge 의존이 크다.
   - Text normalization과 instruction following은 deployment-oriented이지만 dataset and evaluator가 완전히 independent하지 않다.
   - Complex instruction의 criterion-level human agreement가 56.7%이므로 작은 차이는 조심해서 봐야 한다.

4. Speaker similarity metric이 일관된 ranking을 주지 않는다.
   - WavLM과 ERes2Net이 다른 system order를 만든다.
   - Voice cloning quality를 하나의 cosine score로 환원하기 어렵다.

5. Long-form generation의 boundary가 아직 제한적이다.
   - 보고된 target은 약 1.5-3 minute range다.
   - Podcast or audiobook 수준의 수십 분 generation과는 다른 setting이다.

6. Language coverage가 quality parity를 보장하지 않는다.
   - 16개 언어를 지원하지만 low-resource language, accent, code-switching의 coverage가 동일하다는 근거는 없다.

7. Voice cloning safety 논의가 충분하지 않다.
   - Strong zero-shot cloning과 degraded-prompt robustness는 유용하지만 impersonation, consent, provenance, watermarking risk를 함께 다뤄야 한다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 multi-stage training을 단순한 recipe list가 아니라 module-specific optimization으로 보는 데 가치가 있다.

LLM-based generation system에서 자주 생기는 문제는 final output reward를 모든 component에 동시에 걸려는 것이다. 하지만 semantic planner와 continuous generator가 담당하는 error가 다르면 reward domain도 달라야 한다.

Qwen-Audio-3.0-TTS는 다음처럼 나눈다.

- LM: token-domain content, duration, diversity, prosody planning
- FM: waveform-domain speaker, intelligibility, perceptual quality
- Robustness: degraded prompt recovery
- Vocoder: waveform realization and high-frequency detail

이 분해는 TTS 밖에도 적용할 수 있다. 예를 들어 text planner와 image diffusion decoder를 가진 system, discrete action planner와 continuous controller를 가진 robot policy에서도 stage별 owner를 분리할 수 있다.

## 7-2. Reuse potential

### 1) Low-rate token plus larger codebook

Sequence length를 줄일 때 vocabulary capacity와 supervision을 같이 키운다. Audio token뿐 아니라 video token, action token에서도 같은 trade-off를 검토할 수 있다.

### 2) Discrete target plus continuous bridge

Discrete token은 stable objective로 유지하면서 downstream generator에는 hidden state를 추가 condition으로 준다. Compression interface의 안정성과 continuous information을 함께 쓰는 방식이다.

### 3) Token-domain RL before expensive decoder rollout

Final output을 매번 생성하지 않고 semantic token에서 cheap reward를 계산한다. Expensive renderer를 가진 multimodal generation system의 RL cost를 줄이는 아이디어로 재사용할 수 있다.

### 4) Component freeze schedule

각 stage에서 target module만 update한다. Multi-objective gradient interference와 credit assignment를 줄이는 practical pattern이다.

### 5) Integrated robustness training

Noisy input을 별도 preprocessing model로 clean-up하지 않고 main conditional generator가 직접 처리하게 한다. Input enhancement와 identity preservation이 충돌하는 task에 유용하다.

### 6) Diagnostic benchmark design

Standard benchmark 외에 실제 failure mode를 분리한 set을 만든다.

- Text normalization
- Long-form drift
- Degraded reference
- Multi-attribute instruction

Product-oriented model evaluation에서 바로 재사용 가능한 구조다.

## 7-3. Follow-up papers

- CosyVoice 3: Towards In-the-Wild Speech Generation via Scaling-Up and Post-Training
- Qwen3-TTS Technical Report
- JoyVoice: Long-Context Conditioning for Anthropomorphic Multi-Speaker Conversational Synthesis
- FlowTTS-GRPO: Online Reinforcement Learning with Multi-Objective Reward Optimization for Flow-Matching Based Text-to-Speech
- F5R-TTS: Improving Flow Matching Based Text-to-Speech with Group Relative Policy Optimization
- Seed-TTS: A Family of High-Quality Versatile Speech Generation Models
- IndexTTS2: Emotionally Expressive and Duration-Controlled Zero-Shot Text-to-Speech

# 8. Summary

- Qwen-Audio-3.0-TTS는 12.5 Hz tokenizer, LM, flow-matching model, causal vocoder를 결합한 production-oriented TTS system이다.
- 59,049-code FSQ와 supervised audio task로 low-frame-rate compression의 quality loss를 보완한다.
- Continuous LM hidden state를 FM에 전달해 discrete token bottleneck과 component mismatch를 줄인다.
- Five-stage training은 independent pretraining, joint training, LM RL, robustness training, FM RL로 capability owner를 분리한다.
- 실험은 multilingual, cross-lingual, long-form, degraded prompt, instruction following까지 넓지만 training scale과 stage별 ablation은 제한적이다.
