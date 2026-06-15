---
layout: single
title: "Language Modeling Is Compression Review"
categories: Study-concept
tag: [LLM, Compression, InformationTheory]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2309.10668)

[OpenReview](https://openreview.net/forum?id=jznbgiynus)

[Code link](https://github.com/google-deepmind/language_modeling_is_compression)

Language Modeling Is Compression은 language model을 text generator로만 보지 말고, lossless compressor로 보자는 논문이다. 더 정확히는 predictive model과 compressor 사이의 고전적인 등가성을 foundation model 시대의 scaling, tokenization, in-context learning 관점에서 다시 실험한 논문이다.

이 논문은 2023년 arXiv에 올라왔고, ICLR 2024 poster로 공개되었다. 시간이 조금 지난 논문이지만 지금 다시 볼 가치가 있다. 최근 LLM 연구에서 evaluation, tokenization, long-context, multimodal representation, model scaling을 모두 따로 보는 경우가 많은데, 이 논문은 이들을 compression이라는 하나의 관점으로 묶는다.

> 한 줄 요약: Language Modeling Is Compression은 next-token prediction을 잘하는 모델이 arithmetic coding을 통해 lossless compressor가 될 수 있고, 반대로 compressor도 conditional generative model처럼 사용할 수 있음을 text, image, audio 실험으로 보여주는 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM의 cross entropy loss를 compression length로 해석하는 가장 직접적인 실험 사례다.
- Chinchilla 70B 같은 text-trained foundation model을 text뿐 아니라 image patch와 audio sample에도 compressor로 적용한다.
- tokenization, scaling law, in-context learning을 compression lens로 다시 해석한다.
- model size까지 포함한 adjusted compression이라는 관점이 practical model selection과 연결된다.
- gzip 같은 고전 compressor도 conditional generative model로 바꿀 수 있다는 대칭성을 보여준다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 다음 질문이다.

Language model이 좋은 predictor라면, 그 자체를 좋은 compressor로 볼 수 있는가?

일반적인 language modeling은 sequence $x = (x_1, ..., x_T)$에 대해 다음 token 확률을 예측한다.

$$
L(x) = - \sum_{t = 1}^{T} \log_2 p_{theta}(x_t \mid x_{1:t-1})
$$

이 값은 bits 단위의 code length처럼 볼 수 있다. 모델이 다음 token을 높은 확률로 맞히면 필요한 bits가 줄고, 예측이 틀리면 bits가 늘어난다. 따라서 prediction loss를 줄이는 것은 compression length를 줄이는 것과 연결된다.

논문의 핵심은 이 연결을 단순한 비유가 아니라 실제 compressor 구현으로 가져가는 것이다. 모델이 각 symbol에 확률을 주면 arithmetic coding을 통해 lossless bitstream으로 바꿀 수 있다. Decoder도 같은 모델 확률을 재현할 수 있으면 원본 sequence를 정확히 복원할 수 있다.

## 1-2. Why previous approaches are insufficient

기존에는 language model과 compressor를 서로 다른 시스템으로 보는 경향이 강했다.

- Compressor는 gzip, LZMA, PNG, FLAC처럼 domain-specific algorithm으로 다룬다.
- Language model은 text generation이나 downstream prediction 모델로 다룬다.
- Image와 audio compression은 각 modality에 맞는 codec 중심으로 평가한다.
- LM scaling law는 loss와 compute 중심으로 보고, compressed size와 직접 연결하지 않는 경우가 많다.

이 분리는 실용적으로는 자연스럽지만, 연구적으로는 몇 가지 관점을 놓치게 만든다.

첫째, cross entropy가 실제 storage cost와 어떻게 연결되는지 직관이 약해진다. LM loss 0.1 개선이 실제 bits per byte 관점에서 어떤 의미인지 따로 환산해야 한다.

둘째, tokenization을 단순 preprocessing으로 보기 쉽다. 하지만 tokenization은 사실상 sequence를 더 짧은 symbol stream으로 바꾸는 pre-compression stage다. Tokenization이 모델 학습에는 유리해 보여도 최종 lossless compression 관점에서는 항상 좋은 선택이 아닐 수 있다.

셋째, scaling law를 model-only metric으로 보기 쉽다. Compression에서는 data를 압축하는 bit cost뿐 아니라 model 자체를 전달하는 bit cost도 중요하다. 특히 작은 dataset에서는 큰 model을 쓰는 것이 raw compression은 좋지만 adjusted compression에서는 손해일 수 있다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 크게 네 가지다.

1. Predictive model과 lossless compressor의 등가성을 foundation model 실험으로 보여준다.
2. Chinchilla 계열 pretrained model을 text, image, audio compressor로 평가한다.
3. Compression 관점에서 scaling law와 optimal model size를 다시 해석한다.
4. Compressor를 conditional generative model로 사용할 수 있음을 보여준다.

가장 중요한 수식적 연결은 간단하다.

$$
\text{code length} \approx - \log_2 p_{theta}(x)
$$

Sequence 전체에 대해서는 다음처럼 쓸 수 있다.

$$
- \log_2 p_{theta}(x) = - \sum_t \log_2 p_{theta}(x_t \mid x_{1:t-1})
$$

즉 model이 다음 symbol distribution을 잘 맞히면, arithmetic coder가 더 짧은 code를 만든다. 이때 compression은 lossy reconstruction이 아니라 원본 sequence를 정확히 복원하는 lossless compression이다.

## 2-2. Design intuition

이 논문의 설계 직관은 다음 세 가지로 정리할 수 있다.

첫째, prediction은 compression이다. 다음 token을 맞히는 능력은 distribution의 구조를 더 잘 안다는 뜻이고, 구조를 잘 알수록 짧은 code를 만들 수 있다.

둘째, foundation model은 general-purpose predictor다. Chinchilla 70B는 text로 주로 학습되었지만, byte 또는 token stream으로 표현된 image patch와 audio sample에서도 nontrivial statistical structure를 포착할 수 있다.

셋째, compression은 evaluation lens다. Perplexity, bits per byte, model size, context length, tokenization이 모두 하나의 cost로 연결된다. 특히 adjusted compression은 model capacity를 공짜로 보지 않게 만든다.

이 논문의 가장 좋은 점은 LLM을 더 잘 압축한다는 결과 자체보다, LLM evaluation을 information cost 관점으로 옮겨놓는다는 데 있다. 모델이 잘한다는 말을 compressed length로 바꾸면, 서로 다른 modality와 preprocessing choice를 더 엄격하게 비교할 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | LM을 arithmetic coding과 결합해 lossless compressor로 평가한다. |
| Core equivalence | Prediction probability를 code length로 바꾼다. |
| Main models | Classical compressors, small Transformer, Chinchilla family models |
| Data modalities | Text, image patches, audio samples, random data |
| Key analysis | Raw compression, adjusted compression, scaling, tokenization, generation |
| Practical caveat | Chinchilla weights are not released in the public repo. |

## 3-2. Module breakdown

### 1) Language model as compressor

Language model은 sequence prefix를 보고 next symbol distribution을 낸다. Compression pipeline은 이 distribution을 arithmetic coder에 넣어 bitstream을 만든다.

개념적으로 encoder와 decoder는 같은 predictor를 공유한다.

1. Encoder는 원본 sequence를 앞에서부터 읽는다.
2. 각 step에서 model이 $p_{theta}(x_t | x_{1:t-1})$를 계산한다.
3. Arithmetic encoder는 실제 symbol $x_t$를 해당 probability interval로 encode한다.
4. Decoder는 같은 prefix와 같은 model로 probability interval을 재구성한다.
5. Bitstream에서 symbol을 복원하고 prefix를 갱신한다.

이 구조에서는 model prediction이 좋아질수록 code length가 줄어든다. 단, lossless compression이므로 generation quality처럼 주관적 metric이 아니라 복원 가능성과 bit length가 핵심이다.

### 2) Compressor as generative model

논문은 반대 방향도 보여준다. Predictor를 compressor로 바꿀 수 있다면, compressor도 predictor처럼 사용할 수 있다.

고전 compressor는 명시적인 probability distribution을 바로 내지 않을 수 있다. 하지만 특정 continuation을 붙였을 때 compressed length가 얼마나 바뀌는지를 보면, 그 continuation이 context에 얼마나 잘 맞는지 score처럼 사용할 수 있다.

직관적으로는 다음과 같다.

$$
\text{score}(y | x) = - C(x, y)
$$

여기서 $C(x, y)$는 context $x$ 뒤에 continuation $y$를 붙였을 때의 compressed length다. 더 잘 맞는 continuation은 더 짧게 압축될 가능성이 높다. 이 방식으로 gzip 같은 compressor도 conditional generation 또는 selection model처럼 사용할 수 있다.

### 3) Adjusted compression

Raw compression rate만 보면 큰 pretrained model이 유리하다. 하지만 실제 compressor라면 decoder도 같은 model을 가져야 한다. 따라서 model parameters를 전달하는 cost를 무시하면 fairness 문제가 생긴다.

논문은 adjusted compression 관점에서 data cost와 model cost를 함께 본다.

$$
L_{total}(D, theta) = L_{data}(D | theta) + L_{model}(theta)
$$

이 식의 의미는 단순하다. Dataset이 작으면 큰 model의 parameter cost가 부담이 된다. Dataset이 크면 model cost가 많은 examples에 amortize되어 더 큰 model을 쓰는 것이 유리해진다. 이 관점은 scaling law를 compression problem으로 다시 쓰는 부분과 연결된다.

### 4) Tokenization as pre-compression

Tokenization도 compression 관점에서 다시 볼 수 있다. BPE 같은 tokenizer는 byte sequence를 더 긴 symbol vocabulary 위의 짧은 sequence로 바꾼다. 표면적으로는 sequence length가 줄어들기 때문에 유리해 보인다.

하지만 최종 lossless compression에서는 vocabulary size, token distribution, model prediction difficulty가 함께 중요하다. 더 큰 vocabulary가 항상 더 좋은 것은 아니다. Token stream이 짧아져도 model이 해당 token distribution을 충분히 잘 예측하지 못하면 최종 compressed length가 좋아지지 않을 수 있다.

이 논문이 tokenization을 흥미롭게 만드는 이유는, tokenizer를 단순한 engineering choice가 아니라 pre-compression transform으로 본다는 점이다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 여러 modality에서 compression을 평가한다.

| Data | Role |
| --- | --- |
| enwik9 | Text compression benchmark |
| ImageNet patches | Image byte stream에 대한 compression 평가 |
| LibriSpeech samples | Audio sample compression 평가 |
| Random data | Structure가 없는 data에서 compressor가 어떻게 동작하는지 확인 |

Image와 audio는 domain-specific codec과 비교하기 좋다. PNG는 image lossless compressor이고, FLAC은 audio lossless compressor다. Chinchilla 70B가 text 중심 pretrained model임에도 image와 audio에서 strong compression result를 보인다는 점이 논문의 대표적인 메시지다.

## 4-2. Training strategy

논문의 핵심 실험은 pretrained foundation model을 compressor로 쓰는 것이다. 하지만 repo에는 small Transformer를 enwik8에서 학습하고 compression rate를 평가하는 코드도 포함되어 있다.

실험 구성은 크게 세 그룹으로 볼 수 있다.

1. Classical compressor baseline
   - gzip
   - LZMA2
   - PNG
   - FLAC

2. Small Transformer trained for text compression
   - enwik8 기반 training
   - arithmetic coder와 결합해 compression rate 평가

3. Pretrained foundation model
   - Chinchilla family model
   - text, image, audio byte sequence에 대해 in-context compression 평가

Public GitHub repo는 Chinchilla weights를 제공하지 않는다. 대신 compressor protocol, arithmetic coder, compression script, small Transformer training script를 포함한다. 따라서 논문의 full Chinchilla result를 그대로 재현하려면 원문과 내부 model access가 필요하다.

## 4-3. Engineering notes

이 논문을 실무 관점에서 볼 때 중요한 engineering point는 네 가지다.

1. Arithmetic coding이 필요하다
   - LM probability를 실제 bitstream으로 바꾸려면 arithmetic encoder와 decoder가 필요하다.
   - 단순히 perplexity를 계산하는 것과 실제 lossless compression 구현은 다르다.

2. Context length가 compression quality를 제한한다
   - 긴 dependency를 보려면 긴 context가 필요하다.
   - Context가 짧으면 large model도 long-range structure를 충분히 활용하지 못한다.

3. Model size를 포함한 cost를 따로 봐야 한다
   - Raw compression은 model을 무료로 가정한다.
   - Adjusted compression은 model parameters cost를 함께 고려한다.

4. Tokenization은 end-to-end로 평가해야 한다
   - Token count만 줄었다고 좋은 tokenizer라고 말하기 어렵다.
   - 최종 compressed bits 기준으로 봐야 한다.

# 5. Evaluation

## 5-1. Main results

arXiv abstract와 OpenReview 기준으로 가장 많이 인용되는 결과는 다음과 같다.

| Setting | Reported result |
| --- | --- |
| ImageNet patches | Chinchilla 70B compresses to 43.4% of raw size. |
| PNG on ImageNet patches | PNG is reported as 58.5% of raw size. |
| LibriSpeech samples | Chinchilla 70B compresses to 16.4% of raw size. |
| FLAC on LibriSpeech samples | FLAC is reported as 30.3% of raw size. |

이 수치의 해석은 조심해야 한다. 논문이 말하는 핵심은 Chinchilla 70B가 practical image or audio codec을 대체한다는 것이 아니다. 더 중요한 메시지는 text 중심으로 학습된 large predictor가 다른 modality의 byte-level structure도 상당히 잘 포착한다는 점이다.

또 하나 중요한 결과는 compressor를 conditional generative model처럼 쓸 수 있다는 부분이다. 논문은 gzip 기반 text generation, audio continuation, image continuation 예시를 통해 compression score가 generation에도 연결될 수 있음을 보여준다.

## 5-2. What really matters in the experiments

이 논문에서 진짜 봐야 할 것은 SOTA 수치 하나가 아니라 비교 관점이다.

1. Raw compression vs adjusted compression
   - Raw compression은 data만 얼마나 줄였는지 본다.
   - Adjusted compression은 model size cost를 포함한다.
   - 큰 pretrained model은 raw compression에는 강하지만, 작은 dataset에서는 adjusted compression이 불리할 수 있다.

2. General-purpose predictor vs domain-specific codec
   - PNG와 FLAC은 특정 modality에 맞춘 lossless compressor다.
   - Chinchilla는 text-trained foundation model이다.
   - 그럼에도 image와 audio에서 의미 있는 compression을 한다는 점이 surprising result다.

3. Tokenization as compression
   - Tokenizer가 sequence length를 줄이는 것은 사실이다.
   - 하지만 final bits 기준에서는 vocabulary size와 model prediction quality가 같이 들어간다.
   - 따라서 tokenizer choice는 training convenience뿐 아니라 compression objective 관점에서도 평가해야 한다.

4. Context length and in-context learning
   - Compression은 context를 잘 활용할수록 좋아진다.
   - LM의 in-context learning 능력은 새로운 sequence의 local pattern을 빠르게 포착하는 compression ability로 해석할 수 있다.

# 6. Limitations

1. Practical codec으로 바로 쓰기 어렵다
   - Chinchilla 70B 같은 큰 model은 compression quality가 좋아도 inference cost와 model transfer cost가 크다.
   - 실제 storage system에서 PNG나 FLAC을 대체하려면 latency, memory, decoder availability 문제가 생긴다.

2. Chinchilla weights가 공개 repo에 포함되지 않는다
   - repo는 implementation과 small Transformer 재현 경로를 제공하지만, 대표적인 foundation model result를 그대로 재현하려면 Chinchilla access가 필요하다.

3. Compression result는 data representation에 민감하다
   - Image patch 크기, audio chunking, byte representation, context length에 따라 결과가 달라질 수 있다.
   - Modality별 preprocessing이 benchmark interpretation에 영향을 준다.

4. Domain-specific codec과의 비교는 목적이 다르다
   - PNG와 FLAC은 production codec이다.
   - Chinchilla compression은 general predictor의 capability probe에 가깝다.
   - 따라서 결과를 practical codec superiority로 과장하면 안 된다.

5. Adjusted compression 관점에서는 large model의 이점이 data scale에 의존한다
   - Dataset이 충분히 크지 않으면 model parameter cost가 raw compression gain을 상쇄할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 LLM을 평가할 때 loss, perplexity, benchmark accuracy만 보지 말고 information cost로 다시 보게 만든다. 특히 한국어 LLM, document AI, OCR, multimodal model을 다룰 때도 같은 질문을 던질 수 있다.

모델이 정말 domain structure를 이해했다면, 그 domain을 더 짧게 encode할 수 있어야 한다. OCR 모델이 문서 layout을 잘 이해한다면 document token stream을 더 잘 압축할 수 있을까. Video model이 temporal structure를 잘 이해한다면 frame sequence의 residual을 더 짧게 encode할 수 있을까. 이런 식으로 compression lens는 representation quality를 보는 보조 지표가 될 수 있다.

## 7-2. Reuse potential

실무적으로 바로 재사용 가능한 포인트는 다음과 같다.

1. Evaluation metric으로 bits per byte를 추가하기
   - Text domain adaptation이나 tokenizer 비교에서 perplexity뿐 아니라 compressed bits를 같이 볼 수 있다.

2. Tokenizer를 end-to-end compression 관점으로 평가하기
   - Vocabulary size, sequence length, final loss를 분리해서 보지 말고 compressed size로 묶어 볼 수 있다.

3. Dataset scale에 따른 model size 선택
   - 작은 domain dataset에서는 큰 model이 raw metric은 좋아도 adjusted cost에서 불리할 수 있다.
   - 반대로 큰 corpus에서는 model cost가 amortize되며 bigger predictor가 유리해질 수 있다.

4. Compressor 기반 generative baseline 만들기
   - gzip, LZMA 같은 compressor도 conditional scoring baseline으로 사용할 수 있다.
   - 강력한 neural model이 없어도 compression-based retrieval or continuation baseline을 구성할 수 있다.

## 7-3. Follow-up papers

- Chinchilla: Training Compute-Optimal Large Language Models
- Neural Compression and Information Theory 기반 lossless coding 논문들
- Tokenization and scaling law 관련 recent LLM tokenizer 분석 논문
- Multimodal representation을 compression objective로 평가하는 논문들

# 8. Summary

- Language Modeling Is Compression은 prediction과 compression의 등가성을 foundation model 실험으로 다시 보여준다.
- LM probability를 arithmetic coding과 결합하면 lossless compressor를 만들 수 있다.
- Chinchilla 70B는 text-trained model이지만 ImageNet patch와 LibriSpeech sample에서도 강한 compression result를 보인다.
- Compression lens는 scaling law, tokenization, in-context learning을 같은 information cost 축에서 해석하게 한다.
- Practical codec 대체 논문이라기보다, LLM capability를 information-theoretic view로 재해석하는 논문으로 읽는 것이 맞다.
