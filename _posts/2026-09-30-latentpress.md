---
layout: single
title: "LatentPress: Context Compression Beyond Text and Vision Review"
categories: Study-concept
tag: [ContextCompression, SoftTokens, LongContext]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/pdf/2609.01507)

[Code link](https://github.com/HJSang/LatentPress)

LLM context compression은 보통 두 가지 representation을 사용한다.

첫 번째는 text다. 긴 history를 summary로 바꾸거나 중요 token만 남긴다. Human-readable and portable하다는 장점이 있지만, exact fact와 formatting detail이 사라질 수 있다.

두 번째는 image다. Document나 conversation을 image로 render하고 vision encoder로 압축한 뒤 OCR or multimodal reader로 다시 읽는다. Token count를 줄일 수 있지만 rendering, visual encoding, OCR reconstruction cost가 들어간다.

LatentPress는 세 번째 representation을 제안한다.

> Context를 text로 다시 만들지도 않고 image로 우회하지도 않은 채, frozen language model이 input embedding으로 직접 읽는 continuous memory token으로 저장할 수 있는가?

작은 writer가 context를 decoder embedding dimension의 soft token으로 바꾸고, frozen reader는 질문 token 앞에 이 vector sequence를 붙여 바로 answer를 생성한다. Inference-time text reconstruction은 없다.

> 한 줄 요약: LatentPress는 frozen reader의 bottom layers를 재사용하는 reader-matched writer와 작은 linear adapter로 conversation or document를 continuous memory token으로 압축하고, reader가 embedding interface에서 이를 직접 읽게 하는 context representation framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Context storage를 text and image 사이의 선택이 아니라 direct latent interface 문제로 다시 정의한다.
- Downstream decoder를 고치지 않고 trainable parameter를 약 0.1% 수준으로 제한한다.
- Conversation에서는 user and assistant turn에 다른 compression rate를 주는 role-aware policy를 사용한다.
- Reconstruction loss와 forward KL을 결합해 compressed reader가 full-context reader의 output distribution을 유지하도록 학습한다.
- Accuracy뿐 아니라 write latency, read latency, trainable footprint를 함께 측정한다.
- Oracle memory QA뿐 아니라 role structure가 없는 long-document QA까지 transfer를 확인한다.

# 1. Problem Setting

## 1-1. Text summary는 readable하지만 lossy하다

Long-term chat memory나 long-document QA에서 text summary는 가장 자연스러운 compression baseline이다. 그러나 summary objective와 downstream QA objective는 다르다.

Summary model은 다음을 잘할 수 있다.

- Main topic 유지
- Repeated detail 제거
- Narrative flow 정리
- Human-readable memory 생성

반면 QA reader는 다음을 필요로 할 수 있다.

- 한번만 등장한 exact fact
- User and assistant 발화 주체
- Number, name, date
- Negation and exception
- Formatting and local relation

Summary가 읽기 좋은 text를 만들었다고 해서 downstream answer에 필요한 information을 보존했다는 보장은 없다.

## 1-2. Visual compression은 decode path가 길다

Image-based compression은 text token을 pixel layout으로 바꾸고 visual encoder의 patch representation을 활용한다. Document AI에서는 강력하지만 conversational memory에 적용하면 다음 cost가 생긴다.

1. Text rendering
2. Image encoding
3. Visual token generation
4. OCR or multimodal decoding
5. Reader input reconstruction

또한 image resolution and layout이 새로운 bottleneck이 된다. Text가 짧아도 blank area가 생길 수 있고, OCR error가 downstream answer error로 이어질 수 있다.

## 1-3. Reader adaptation cost

Continuous context compression 자체는 새로운 분야가 아니다. Gist token, ICAE, AutoCompressor, xRAG류 method는 latent vector를 사용한다.

LatentPress가 겨냥하는 practical constraint는 다음과 같다.

- Reader LLM은 그대로 유지한다.
- LLM-scale encoder를 새로 학습하지 않는다.
- Reader별로 작은 writer만 붙인다.
- Output text를 복원하지 않는다.
- Compression rate를 deployment policy로 조절한다.

즉 핵심은 새로운 latent space보다 frozen decoder와 호환되는 write-read interface다.

# 2. Core Idea

## 2-1. Direct-read soft context

Original context를 $x$, question을 $q$, compressed memory token을 $m$, frozen reader를 $f_\theta$라고 하자.

LatentPress는 다음 mapping을 학습한다.

$$
m
=
\operatorname{Write}_{\phi}(x;\pi)
$$

Frozen reader는 memory vector와 question embedding을 바로 연결해 answer를 생성한다.

$$
y
=
f_{\theta}
\left(
[m;\operatorname{emb}(q)]
\right)
$$

여기서 $\pi$는 segment별 compression policy이고, $\theta$는 고정된다. 학습되는 것은 writer parameter $\phi$뿐이다.

## 2-2. Reader-matched writer

Writer는 reader와 완전히 별도인 generic encoder가 아니다. Frozen reader의 bottom two transformer layers를 빌려 context-aware representation을 만든다.

Token $i$의 literal input embedding을 $E_i$, borrowed layer가 만든 contextual abstraction을 $c_i$라고 하면 writer는 두 정보를 결합한다.

$$
h_i
=
H(E_i,c_i)
$$

그 뒤 compression block 안의 hidden state를 pooling하고, reader embedding dimension과 같은 output을 만드는 linear adapter를 적용한다.

이 설계의 장점은 다음과 같다.

- Reader가 이미 이해하는 representation geometry를 재사용한다.
- 별도 large encoder를 학습하지 않는다.
- Compressed token dimension이 reader input embedding과 바로 맞는다.
- Reader-specific compatibility를 작은 adapter가 담당한다.

## 2-3. Uniform compression and role-aware compression

Long document에는 uniform mean pooling을 사용한다.

Compression factor가 $k$라면 $k$개의 original token을 하나의 soft token으로 줄인다.

$$
n
\rightarrow
\left\lceil
\frac{n}{k}
\right\rceil
$$

Conversation에서는 모든 role을 같은 비율로 줄이지 않는다.

- User turn: $k_{\mathrm{user}}=1$
- Assistant turn: $k_{\mathrm{assistant}}\in\{8,16,32\}$

User fact and request는 lossless하게 남기고, 일반적으로 더 길고 중복이 많은 assistant response를 강하게 압축한다.

Role-aware compression ratio는 fixed number가 아니다. Conversation별 user and assistant token mix에 따라 달라진다. LongMemEval에서 평균 ratio는 약 4.62x to 7.70x다.

## 2-4. Full-context reader를 teacher로 사용한다

Compressed reader가 answer token을 맞히는 reconstruction loss만 사용하면, sparse target sequence에 맞춰 과도하게 task-specific한 shortcut을 배울 수 있다.

LatentPress는 full-context reader distribution을 teacher로 사용한다.

$$
\mathcal L(\phi)
=
\mathcal L_{\mathrm{rec}}
+
\lambda
\mathcal L_{\mathrm{fkl}}
$$

Teacher-forced reconstruction loss는 다음과 같다.

$$
\mathcal L_{\mathrm{rec}}
=
-\frac{1}{N}
\sum_{t=1}^{N}
\log p_{\mathrm{comp},t}(y_t)
$$

Forward KL은 full-context distribution과 compressed-context distribution을 맞춘다.

$$
\mathcal L_{\mathrm{fkl}}
=
\frac{1}{N}
\sum_{t=1}^{N}
\operatorname{KL}
\left(
p_{\mathrm{full},t}
\parallel
p_{\mathrm{comp},t}
\right)
$$

이 objective는 exact context reconstruction이 아니라 reader behavior preservation을 학습한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Reader | Frozen Qwen decoder |
| Context encoder | Reader bottom two layers, frozen |
| Trainable module | Identity-initialized linear adapter |
| Output interface | Reader input-embedding dimension |
| Compression | Uniform or role-aware mean pooling |
| Supervision | Reconstruction plus forward KL |
| Inference reconstruction | None |
| Deployment unit | Reader-specific writer |

## 3-2. Write path

Write path는 다음과 같다.

1. Original context를 token embedding으로 변환한다.
2. Frozen bottom two reader layers로 contextual representation을 만든다.
3. Literal embedding and contextual abstraction을 결합한다.
4. Segment policy $\pi$에 따라 pooling한다.
5. Linear adapter로 memory token을 만든다.
6. Memory token을 cacheable continuous vector sequence로 저장한다.

Conversation memory에서는 role boundary를 보존해 user and assistant compression rate를 다르게 적용한다.

## 3-3. Read path

Read path는 더 단순하다.

1. Stored memory token을 load한다.
2. Question text를 normal embedding으로 변환한다.
3. Memory vector and question embedding을 concatenate한다.
4. Frozen reader가 answer를 생성한다.

Text summary generation, image rendering, OCR decoding이 없다.

## 3-4. Frozen encoder가 fine-tuned encoder보다 좋다

Bottom layers를 writer와 함께 fine-tune하는 것이 직관적으로는 더 flexible해 보인다. 그러나 LongMemEval ablation에서는 frozen encoder가 모든 compression rate에서 더 높다.

Qwen2.5-7B result:

| Encoder setting | $k_a=8$ | $k_a=16$ | $k_a=32$ |
| --- | ---: | ---: | ---: |
| Fine-tuned | 0.454 | 0.460 | 0.438 |
| Frozen | 0.476 | 0.478 | 0.504 |

Fine-tuning은 UltraChat training distribution에 overfit하고, 높은 compression에서 general representation을 손상시킨 것으로 해석된다.

이 결과는 reader-matched writer의 핵심이 reader representation을 바꾸는 것이 아니라, 이미 학습된 geometry를 작은 adapter로 압축하는 데 있음을 보여준다.

# 4. Training / Data / Recipe

## 4-1. Main hyperparameters

| Setting | Value |
| --- | --- |
| Frozen readers | Qwen2.5-7B, Qwen2.5-14B, Qwen3-8B, Qwen3-1.7B |
| Borrowed encoder layers | 2 |
| Trainable head | Linear $d\times d$ adapter |
| Trainable parameters | 4.196M to 26.220M |
| Optimizer | AdamW |
| Learning rate | $1\times 10^{-4}$ |
| Training steps | 1,000 |
| Maximum chunk length | 2,048 |
| Batch size | 1 |
| KL weight | $\lambda=1.0$ |
| Precision | Reader bf16, compressor head fp32 |

Trainable parameter count는 reader size에 따라 달라지며 decoder parameter의 약 0.1% 수준이다.

## 4-2. Zero-shot conversational memory training

Zero-shot LongMemEval experiment에서 writer는 QA label을 보지 않는다.

- Training corpus: UltraChat 2,000 conversations
- Input type: text-only conversations
- QA supervision: none
- Evaluation: LongMemEval 500 oracle-evidence questions
- Judge: Llama-3.1-70B-Instruct
- Seeds: 5

Writer는 generic conversation reconstruction and full-reader KL만 학습한 뒤, unseen memory QA에 transfer된다.

## 4-3. Long-document QA training

LongBench-QA English는 role structure가 없는 긴 document를 사용한다.

Subsets:

- narrativeqa
- qasper
- multifieldqa_en
- hotpotqa
- 2wikimqa
- musique

두 setting을 비교한다.

### Cross-domain

LongMemEval-derived QA supervision으로 writer를 학습하고 LongBench로 transfer한다.

### In-domain

Target LongBench domain QA example로 writer를 adaptation한다.

두 setting 모두 uniform compression factor $4$, $8$, $16$을 사용한다. Qwen3-8B의 raw, OCR, LatentPress run은 모두 non-thinking decoding mode로 맞춘다.

## 4-4. Deployment recipe

Practical pipeline에서는 writer output을 reusable memory artifact로 볼 수 있다.

- Conversation이 끝날 때 once-write한다.
- Reader request마다 compressed vector를 prefix로 재사용한다.
- Reader가 바뀌면 writer도 다시 학습해야 한다.
- Compression rate는 context value and latency budget에 따라 선택한다.
- Human audit가 필요한 item은 raw text와 latent token을 함께 보존할 수 있다.

# 5. Evaluation

## 5-1. LongMemEval

Qwen2.5-7B headline result:

| Method | Compression | Accuracy |
| --- | ---: | ---: |
| Raw oracle evidence | 1.00x | 0.490 |
| LatentPress | 4.62x | 0.476 |
| LatentPress | 6.27x | 0.478 |
| LatentPress | 7.70x | 0.504 |
| Text summary | 12.1x | 0.184 |
| DeepSeek-OCR | 2.33x | 0.426 |
| DeepSeek-OCR | 5.97x | 0.390 |
| DeepSeek-OCR | 9.34x | 0.312 |

가장 aggressive role-aware point가 raw context보다 높은 0.504를 기록한다. 다만 0.490 and 0.504의 차이를 곧바로 compression이 information을 추가했다고 해석하면 안 된다. Five-seed variance, reader inductive bias, compression regularization이 함께 들어간 결과다.

Reader에 따라 ordering도 달라진다. Qwen3-8B에서는 DeepSeek-OCR의 low-compression point가 0.542로 LatentPress 0.506보다 높다. Latent representation의 우위가 모든 reader and rate에서 절대적이지 않다.

Text summary는 모든 reader에서 가장 약하다. Human-readable summary가 exact memory QA에 필요한 evidence를 충분히 보존하지 못했음을 보여준다.

## 5-2. Judge-free Token-F1

LLM judge score만 사용하면 judge bias 우려가 있다. 논문은 token-F1도 함께 보고한다.

Qwen2.5-7B에서 role-aware LatentPress $k_a=32$는 0.251을 기록한다. Uniform soft-token compression은 0.052 to 0.072이고, DeepSeek-OCR은 compression이 커질수록 0.233에서 0.160으로 내려간다.

Role-aware policy가 단순 uniform latent compression보다 훨씬 강한 이유는 user fact를 lossless하게 남긴다는 데 있다.

## 5-3. LongBench-QA cross-domain

Cross-domain transfer에서는 mild compression이 가장 안정적이다.

| Reader | Raw | 4x | 8x | 16x |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | 43.80 | 45.13 | 40.69 | 32.94 |
| Qwen3-8B | 30.80 | 32.79 | 24.41 | 20.05 |
| Qwen2.5-14B | 47.93 | 49.88 | 37.08 | 30.34 |

4x에서는 raw baseline을 match or exceed하지만 8x and 16x에서는 degradation이 커진다. Conversation memory의 7.70x headline을 long document에 그대로 적용할 수 없다는 뜻이다.

## 5-4. LongBench-QA in-domain

In-domain adaptation에서는 compressed reader가 raw baseline을 크게 넘는 경우가 있다.

- Qwen3-8B raw 30.80, 4x 39.62, 8x 36.93
- Qwen2.5-14B raw 47.93, 4x 57.99, 8x 52.18

그러나 이 결과는 순수 compression comparison이 아니다. Writer가 target-domain QA supervision을 학습했기 때문에 task adapter 역할도 한다.

따라서 올바른 해석은 다음과 같다.

> Soft-token writer는 compression module이면서 frozen reader 앞의 task-specific input adapter가 될 수 있다.

## 5-5. Efficiency

Average write latency는 conversation당 약 43 ms다.

Warm-loaded LongBench-QA inference latency:

| Reader | Raw context | LatentPress 8x | Cached OCR |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B | 2.44 s | 0.49 s | 2.71 s |
| Qwen2.5-14B | 4.14 s | 0.49 s | 4.34 s |
| Qwen3-8B | 3.97 s | 0.43 s | 4.03 s |

Compressed prefix read는 raw context보다 약 5x to 9x 빠르다.

Whole-job comparison에서도 in-domain LatentPress는 nearest-compression cold-cache DeepSeek-OCR pipeline보다 6.0x to 13.7x 짧다. 이 비교에는 OCR cache generation cost가 포함되므로 warm-cache production scenario와는 분리해서 봐야 한다.

# 6. Limitations

1. **Oracle evidence setting**

   LongMemEval은 relevant session이 이미 주어진 oracle setting이다. Retrieval, memory update, conflict resolution, stale memory removal은 평가하지 않는다.

2. **Reader-specific writer**

   Qwen2.5-7B용 memory token을 다른 reader가 그대로 읽을 수 있다고 보장되지 않는다. Model upgrade마다 writer and stored representation migration이 필요할 수 있다.

3. **Human interpretability 부족**

   Continuous token은 text summary처럼 사람이 읽거나 직접 수정할 수 없다. Audit, compliance, deletion request, provenance tracking을 위해 raw context를 별도로 보존해야 할 수 있다.

4. **Aggressive compression degradation**

   Long document에서 16x compression은 raw context보다 크게 낮다. Conversation result를 모든 modality and task에 일반화하면 안 된다.

5. **In-domain result의 confound**

   Target QA supervision을 받은 writer는 compression뿐 아니라 task adaptation을 수행한다. Raw reader와의 차이가 information preservation만을 의미하지 않는다.

6. **Formatting failure**

   높은 compression에서 `unanswerable` collapse, excessive repetition, format drift가 발생한다. 일부 score loss는 semantic information loss가 아니라 decoder behavior pathology다.

7. **Storage and serving interface 미정**

   Soft token serialization format, versioning, quantization, encryption, cross-device portability는 본문의 중심 범위가 아니다.

8. **Security and privacy**

   Human-readable하지 않다고 privacy-safe한 것은 아니다. Latent vector inversion, membership leakage, unauthorized reader access를 별도로 검증해야 한다.

# 7. My Take

## 7-1. Context compression을 third representation으로 만든다

LatentPress의 좋은 점은 text summary와 visual OCR 중 하나를 고르는 문제가 아니라는 점이다. Context consumer가 language model이라면 human-readable intermediate를 반드시 거칠 필요가 없다는 주장은 자연스럽다.

이 관점은 embedding index와도 다르다.

- Retrieval embedding은 relevant item을 고르는 representation이다.
- LatentPress memory token은 frozen decoder가 answer generation에 직접 사용하는 representation이다.

즉 retrieval key가 아니라 generative prefix다.

## 7-2. Compression and task adaptation의 경계

In-domain LongBench result는 writer가 단순 compressor를 넘어 input-side adapter가 될 수 있음을 보여준다.

Reader weight를 바꾸지 않고도 soft prefix가 다음을 할 수 있다.

- Evidence selection
- Task formatting
- Domain-specific abstraction
- Answer behavior steering

이 특성은 PEFT의 다른 형태로 볼 수 있다. LoRA가 weight delta를 저장한다면 LatentPress는 context-conditioned input delta를 저장한다.

다만 성능이 좋아질수록 memory token이 original context를 faithfully encode하는지, task-specific shortcut을 encode하는지 구분해야 한다.

## 7-3. 실제 long-term memory system에 넣으려면

Full system은 다음 layer가 필요하다.

1. Retrieval layer
   - 어떤 memory segment를 꺼낼지 결정한다.

2. Writer layer
   - Segment를 reader-compatible token으로 압축한다.

3. Version layer
   - Reader checkpoint and writer version을 함께 기록한다.

4. Audit layer
   - Raw source, provenance, deletion, conflict를 관리한다.

5. Read layer
   - Latent prefix and question을 reader에 전달한다.

6. Refresh layer
   - Reader upgrade or memory conflict 때 latent token을 재생성한다.

LatentPress는 이 중 write and read interface를 잘 보여주지만 full memory lifecycle을 해결하지는 않는다.

## 7-4. Follow-up papers

- xRAG
- ICAE
- AutoCompressor
- Gisting
- LLMLingua-2
- DeepSeek-OCR
- LongMemEval

함께 읽으면 discrete text pruning, continuous soft token, visual compression이 각각 어떤 information and deployment cost를 가지는지 비교하기 좋다.

# 8. Summary

- LatentPress는 context를 text or image가 아닌 continuous memory token으로 저장한다.
- Reader bottom two layers는 frozen encoder로 재사용하고 작은 linear adapter만 학습한다.
- Conversation에서는 user turn을 보존하고 assistant turn을 강하게 압축하는 role-aware policy를 사용한다.
- Reconstruction plus forward KL로 full-context reader behavior를 compressed prefix에 옮긴다.
- Mild compression은 강하지만 aggressive compression, reader portability, auditability는 남은 과제다.
