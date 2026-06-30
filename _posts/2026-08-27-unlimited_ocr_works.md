---
layout: single
title: "Unlimited OCR Works Review"
categories: Study-concept
tag: [OCR, DocumentAI, VLM, EfficientAttention, LongContext]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.23050)

[Code link](https://github.com/baidu/Unlimited-OCR)

Unlimited OCR Works는 "OCR 성능을 몇 점 더 올린 기술 리포트"로 읽으면 핵심을 놓치기 쉽다. 이 논문이 실제로 묻는 질문은 OCR model이 한 페이지를 잘 읽는가가 아니라, **LLM decoder를 가진 end-to-end OCR model이 긴 문서를 한 번에 계속 읽을 수 있는가**다.

최근 OCR 흐름은 pipeline OCR에서 VLM/LLM 기반 end-to-end parsing으로 이동하고 있다. Detection, crop, recognition, layout heuristics를 여러 단계로 나누는 대신, image encoder와 LLM decoder가 하나의 sequence를 바로 생성한다. 이 방식은 language prior를 활용할 수 있고, Markdown-like linearization에도 유리하다. 하지만 output이 길어질수록 decoder KV cache가 계속 커진다는 문제가 있다.

이 논문은 이 병목을 Reference Sliding Window Attention, 이하 R-SWA로 푼다. 핵심은 간단하다. Visual reference token과 prompt는 항상 볼 수 있게 유지하고, 이미 생성된 output token은 최근 $n$개만 보게 한다. OCR에서 필요한 것은 전체 이전 출력이 아니라, 원본 문서 전체와 방금 쓴 주변 context라는 직관이다.

> 한 줄 요약: Unlimited OCR Works는 DeepSeek-OCR의 high-compression encoder를 유지하면서 decoder attention을 R-SWA로 바꿔, reference token에는 global access를 유지하고 generated token에는 bounded sliding window를 적용해 long-horizon OCR에서 KV cache를 constant하게 만드는 technical report다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- OCR을 single-page document parsing이 아니라 **long-horizon reference-conditioned decoding** 문제로 다시 정의한다.
- Standard full attention을 단순히 sliding window로 바꾸는 것이 아니라, reference token과 output token의 역할을 분리한다.
- DeepSeek-OCR 계열의 optical compression idea와 decoder KV cache optimization을 하나로 연결한다.
- Long context를 늘리는 대신, decoding state를 soft-forgetting 방식으로 설계하는 attention pattern을 제안한다.
- R-SWA를 OCR뿐 아니라 ASR, translation 같은 reference-based parsing task로 확장할 수 있는 general mechanism으로 제시한다.

이 글에서는 Unlimited OCR을 OCR leaderboard paper보다, **reference는 항상 보되 생성 history는 제한적으로 보자는 decoder attention design paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

End-to-end OCR model은 보통 다음 형태로 볼 수 있다.

$$
y_{1:T}
=
\mathrm{Decoder}
\left(
\mathrm{Encoder}(I),
p
\right)
$$

여기서 $I$는 document image, $p$는 prompt, $y_{1:T}$는 OCR output sequence다. 문제는 $T$가 매우 길어질 때 발생한다. 일반 causal self-attention decoder는 token $t$를 생성할 때 모든 이전 output token을 attend한다.

$$
\mathcal{N}_{\mathrm{MHA}}(t)
=
\{1,\ldots,L_m+t-1\}
$$

여기서 $L_m$은 visual token and prompt로 구성된 prefix length다. 이 경우 KV cache는 output length와 함께 선형으로 커진다.

$$
C_{\mathrm{MHA}}(T)
=
L_m+T
$$

문서가 한 페이지라면 괜찮다. 하지만 20-40페이지 이상의 OCR을 한 번에 하려면 generated text가 매우 길어진다. DeepSeek-OCR처럼 visual token을 강하게 압축하더라도, decoder side KV cache는 계속 증가한다.

현재 많은 OCR system은 이 문제를 page-by-page loop로 피한다.

1. Page 1을 OCR한다.
2. Memory를 reset한다.
3. Page 2를 OCR한다.
4. 다시 reset한다.

이 방식은 engineering workaround로는 유용하지만, 문서 전체를 하나의 long-horizon parsing process로 다루지는 못한다. Page boundary마다 state가 끊기기 때문에 cross-page continuity, repeated structure, progress tracking, book-level parsing 같은 관점에서는 제한이 생긴다.

## 1-2. Why previous approaches are insufficient

### 1) Pipeline OCR

Traditional document OCR pipeline은 detection, crop, rectification, recognition, layout reconstruction을 나눈다. 이 방식은 module별로 튜닝하기 쉽지만, dense PDF parsing에서는 heuristic이 많아지고 end-to-end language prior를 충분히 활용하기 어렵다.

특히 table, formula, reading order, multi-column layout을 Markdown-like text로 선형화하려면 단순 character recognition보다 높은 수준의 layout-to-language conversion이 필요하다.

### 2) End-to-end VLM OCR

End-to-end OCR은 image encoder와 LLM decoder를 붙여 page content를 한 번에 생성한다. Language prior를 사용하기 때문에 noisy OCR, math, table caption, list, Markdown formatting에서 이점이 있다.

하지만 decoder가 standard full attention이면 output이 길수록 KV cache와 attention cost가 늘어난다. Multi-page parsing에서는 visual token보다 output token이 병목이 된다.

### 3) Vanilla sliding window attention

Output token history를 sliding window로 제한하면 cache 문제는 줄어든다. 그러나 OCR에서는 visual reference token이 핵심이다. Visual token까지 recurrent state처럼 압축하거나 window 밖으로 밀어내면 원본 image fidelity가 점차 흐려질 수 있다.

R-SWA는 이 지점을 분리한다.

- Reference token은 항상 static and globally visible하다.
- Generated token은 최근 window만 유지한다.

즉 OCR에서 memory를 줄여야 하는 대상은 reference image가 아니라 generated output history다.

# 2. Core Idea

## 2-1. Main contribution

Unlimited OCR의 핵심 contribution은 세 가지다.

1. **Reference Sliding Window Attention**
   - 모든 generated token은 visual/prompt prefix 전체를 attend한다.
   - Generated output token은 최근 $n$개만 attend한다.
   - Decode-side KV cache는 $L_m+n$으로 bounded된다.

2. **Unlimited OCR architecture**
   - DeepSeek-OCR의 DeepEncoder를 유지한다.
   - LLM decoder의 모든 standard attention layer를 R-SWA로 교체한다.
   - DeepEncoder의 high compression과 constant decoder cache를 결합한다.

3. **Long-horizon OCR validation**
   - OmniDocBench v1.5와 v1.6에서 single-page document parsing 성능을 확인한다.
   - In-house multi-page document set에서 2, 5, 10, 15, 20, 40+ pages parsing을 평가한다.
   - Throughput and memory behavior가 output length에 따라 안정적인지 본다.

## 2-2. Design intuition

이 논문의 설계 직관은 사람의 필사 행동에서 온다. 사람이 책을 베껴 쓸 때 이미 쓴 모든 문자를 매번 다시 읽지는 않는다. 원본 문서를 보고, 방금 쓴 몇 글자만 확인하면서 다음 글자를 이어간다.

이를 OCR decoder attention으로 바꾸면 다음과 같다.

- 원본 문서 image token은 항상 볼 수 있어야 한다.
- 바로 앞 output context는 progress tracking을 위해 필요하다.
- 오래된 output history는 대부분 필요하지 않다.
- 오래된 output을 계속 보관하면 cost만 커진다.

R-SWA는 이 intuition을 다음 attention set으로 표현한다.

$$
\mathcal{N}(t)
=
\mathcal{P}
\cup
\mathcal{D}_{n}(t)
$$

$$
\mathcal{P}
=
\{1,\ldots,L_m\}
$$

$$
\mathcal{D}_{n}(t)
=
\left\{
j
\mid
\max(L_m+1,L_m+t-n)
\leq
j
\leq
L_m+t-1
\right\}
$$

여기서 $\mathcal{P}$는 visual token and prompt prefix, $\mathcal{D}_{n}(t)$는 generated output의 recent window다.

Attention weight는 이 restricted set 위에서 계산된다.

$$
\alpha_{tj}
=
\frac{
\exp
\left(
\mathbf{q}_t^\top \mathbf{k}_j / \sqrt{d_k}
\right)
}{
\sum_{i \in \mathcal{N}(t)}
\exp
\left(
\mathbf{q}_t^\top \mathbf{k}_i / \sqrt{d_k}
\right)
}
$$

$$
\mathbf{o}_t
=
\sum_{j \in \mathcal{N}(t)}
\alpha_{tj}
\mathbf{v}_j
$$

핵심은 reference와 decode history의 asymmetry다. OCR task에서 static reference는 항상 informative하지만, 오래된 generated history는 local progress만 넘으면 빠르게 덜 중요해진다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Multi-page OCR를 page-by-page loop 없이 long-horizon parsing으로 처리 |
| Baseline | DeepSeek-OCR |
| Encoder | DeepEncoder with high visual token compression |
| Decoder | MoE LLM decoder, 3B total and 0.5B activated parameters |
| Main change | Decoder attention을 R-SWA로 전면 교체 |
| Prefix access | Visual token과 prompt를 항상 볼 수 있음 |
| Decode access | 가장 최근 $n$개 generated token만 사용, default $n=128$ |
| KV cache | $L_m+\min(n,T)$ |
| Training target | End-to-end detection and parsing text sequence |
| Main evaluation | OmniDocBench v1.5/v1.6 plus in-house multi-page parsing |

## 3-2. Module breakdown

### 1) DeepEncoder

Unlimited OCR은 DeepSeek-OCR의 DeepEncoder를 유지한다. DeepEncoder는 SAM-ViT and CLIP-ViT를 cascade하고, bridge에서 16x token compression을 적용한다.

Paper는 DeepEncoder가 1024 x 1024 PDF image를 256 tokens로 압축할 수 있다고 설명한다. 이 compression은 multi-page OCR에서 중요하다. Prefix token이 너무 길면 R-SWA를 써도 prefix attention cost가 커진다.

R-SWA가 줄이는 것은 generated output side cache다. Visual prefix side가 너무 크면 여전히 long-horizon parsing의 상한이 낮아진다. 따라서 Unlimited OCR의 실용성은 encoder compression과 decoder cache bound가 함께 있어야 나온다.

### 2) Decoder MoE LLM

Unlimited OCR은 DeepSeek-OCR baseline의 MoE decoder를 사용한다.

- Total parameter: 3B
- Activated parameter: 0.5B
- Attention: standard MHA에서 R-SWA로 교체
- DeepEncoder: freeze
- Decoder LLM parameters: continue training

MoE activation이 0.5B라는 점은 serving cost에 중요하다. OCR task에서는 LLM decoder가 길게 token을 생성하므로 activated parameter와 KV cache가 모두 latency and memory에 직접적인 영향을 준다.

### 3) Reference Sliding Window Attention

R-SWA의 KV cache는 다음처럼 bounded된다.

$$
C_{\mathrm{R-SWA}}(T)
=
L_m+\min(n,T)
\leq
L_m+n
$$

Standard MHA와 비교한 cache ratio는 다음과 같다.

$$
\rho(T)
=
\frac{
C_{\mathrm{R-SWA}}(T)
}{
C_{\mathrm{MHA}}(T)
}
=
\frac{
L_m+\min(n,T)
}{
L_m+T
}
$$

$T \gg n$이면 다음처럼 근사된다.

$$
\rho(T)
\approx
\frac{
L_m+n
}{
L_m+T
}
$$

즉 output length가 길어질수록 R-SWA의 상대적인 cache advantage가 커진다.

다만 절대 cost가 0이 되는 것은 아니다. Prefix length $L_m$은 여전히 남아 있고, 모든 decode token은 reference prefix를 attend한다. 따라서 unlimited라는 표현은 실제 무한 길이가 아니라 engineering sense의 "훨씬 긴 one-shot parsing"으로 읽는 편이 안전하다.

### 4) KV cache queue

R-SWA cache는 prefix KV와 output KV buffer로 나뉜다.

- Prefix KV: visual tokens and prompt, fixed
- Output KV: recent $n$ generated tokens, queue

New token이 생성되면 output KV queue에 들어가고, window 밖 token의 KV는 제거된다. 이 때문에 output-side memory가 constant하게 유지된다.

이 구조는 page-by-page reset과 다르다. Document reference는 계속 유지되고, recent output state만 sliding한다. 따라서 model은 문서 전체 image를 보면서 current writing progress만 local memory로 추적한다.

### 5) Difference from ordinary SWA

Ordinary SWA는 모든 token history를 local window로 제한한다. R-SWA는 prefix token을 special reference로 둔다. 이 차이가 OCR에서는 핵심이다.

| Attention type | Reference token | Output history | Risk |
| --- | --- | --- | --- |
| Full attention | Always visible | All previous output | KV grows with output |
| Vanilla SWA | May be out of window or compressed | Recent output only | Reference fidelity can decay |
| R-SWA | Always visible and static | Recent output only | Prefix cost remains |

R-SWA는 OCR의 reference-conditioned nature에 맞춘 attention이다. 일반 LLM generation에 그대로 적용하면 오래된 textual context를 잃을 수 있다. 따라서 task family를 잘 구분해야 한다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 약 2M document OCR data sample을 구성한다.

| Data type | Description |
| --- | --- |
| Single-page data | Paddle OCR annotation으로 block coordinate and content 구성 |
| Multi-page data | Single-page data를 concatenate하여 synthetic multi-page sample 생성 |
| Ratio | Single-page:multi-page = 9:1 |
| Multi-page sample count | Around 200K |
| Page count range | 2 to 50 pages |
| Page separator | `<page>` token |
| Sequence length | 32K packed sequence |

Single-page data에는 element coordinate가 0-1000 range로 normalized된다. 이는 OCR을 text recognition뿐 아니라 detection and parsing target으로 학습한다는 뜻이다.

Multi-page data가 synthetic concatenation이라는 점은 중요하다. Long-horizon behavior를 만들기에는 효율적이지만, real book-level layout, cross-page footnote, repeated header, bibliography, index, chapter structure 같은 natural multi-page phenomena를 충분히 반영하는지는 별도 확인이 필요하다.

## 4-2. Training strategy

Unlimited OCR은 DeepSeek-OCR checkpoint에서 continue training한다.

| Item | Value |
| --- | --- |
| Training steps | 4,000 |
| Global batch size | 256 |
| Max sequence length | 32K |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Initial learning rate | 1e-4 |
| Encoder | DeepEncoder frozen |
| Trainable part | LLM decoder parameters |
| Parallelism | DeepEP, expert parallelism 4 |
| Framework | Megatron-LM |
| Inference support | Transformers and SGLang |

논문은 8 x 16 A800 GPUs라고 적고 있다. 이 표현이 정확히 128 A800을 의미하는지, node/GPU notation인지 publish 전 원문과 code release를 다시 확인하는 편이 좋다.

## 4-3. Engineering notes

실무적으로 중요한 포인트는 다음과 같다.

1. **Reference token과 output token을 다르게 다룬다**
   - OCR에서는 visual reference가 source of truth다.
   - Generated output은 progress tracking memory에 가깝다.

2. **KV cache optimization을 decoder-only problem으로 보지 않는다**
   - Encoder compression이 충분히 좋아야 prefix cost가 낮아진다.
   - Decoder cache가 bounded되어도 prefix attention은 남아 있다.

3. **Training sequence length와 inference target이 맞아야 한다**
   - 32K packed sequence로 train한다.
   - Multi-page sample은 `<page>` separator로 page boundary를 표현한다.

4. **SWA width는 task hyperparameter다**
   - 논문 default는 $n=128$이다.
   - OCR language locality가 강하면 작게 둘 수 있지만, table continuation or long formula에서는 더 큰 window가 필요할 수 있다.

5. **R-SWA는 source가 고정된 task에서 가장 자연스럽다**
   - Document OCR, ASR with fixed audio features, translation with fixed source tokens처럼 reference가 고정된 task와 잘 맞는다.
   - Free-form dialogue나 long-context reasoning에는 old output history가 더 중요할 수 있다.

6. **SGLang support가 중요하다**
   - Attention mechanism이 paper idea로만 끝나지 않으려면 actual serving engine의 KV eviction and kernel path가 맞아야 한다.

# 5. Evaluation

## 5-1. Main benchmark

논문은 OmniDocBench v1.5 and v1.6을 main benchmark로 사용한다. OmniDocBench는 text recognition, formula recognition, table extraction, reading order를 함께 평가한다.

| Metric | Meaning |
| --- | --- |
| Text Edit Distance | Character-level text accuracy |
| Formula CDM | Mathematical formula recognition quality |
| Table TEDS | Table structure extraction with content |
| Table TEDS-S | Table structure extraction without content |
| Reading Order Edit Distance | Predicted reading sequence correctness |
| Overall | Weighted aggregate over text, formula, table tasks |

v1.5는 DeepSeek-OCR baseline과 classic end-to-end model comparison에 사용되고, v1.6은 newer benchmark 비교에 사용된다.

## 5-2. Main results

논문 table 기준 주요 수치는 다음과 같다.

| Setting | Model | Overall |
| --- | --- | ---: |
| OmniDocBench v1.5 | DeepSeek-OCR | 87.01 |
| OmniDocBench v1.5 | DeepSeek-OCR 2 | 89.17 |
| OmniDocBench v1.5 | Unlimited-OCR | 93.23 |
| OmniDocBench v1.6 | Qianfan-OCR | 93.90 |
| OmniDocBench v1.6 | Unlimited-OCR | 93.92 |

v1.5에서는 DeepSeek-OCR 87.01에서 Unlimited-OCR 93.23으로 올라간다. Text edit distance는 0.073에서 0.038로 낮아지고, table TEDS는 84.97에서 90.93으로 높아진다.

v1.6에서는 Unlimited-OCR이 93.92 overall을 기록한다. Qianfan-OCR 93.90과 차이는 매우 작다. 따라서 v1.6 result는 압도적 차이라기보다 end-to-end VLM OCR group 안에서 top-tier에 있는 정도로 읽는 것이 적절하다.

## 5-3. Subcategory study

OmniDocBench v1.5의 9개 document type에서 DeepSeek-OCR series와 비교한다.

- PPT
- Academic paper
- Book
- Colorful textbook
- Exam paper
- Magazine
- Newspaper
- Note
- Research report

논문은 Unlimited OCR이 text edit distance와 reading order에서 전반적인 개선을 보인다고 설명한다. 다만 일부 subtype에서는 DeepSeek-OCR 2가 더 나은 cell도 존재한다. 따라서 "모든 metric에서 완전 우세"라기보다, 대부분의 document type에서 stable gain이 있다는 방향으로 보는 편이 안전하다.

## 5-4. Long-horizon parsing

Long-horizon OCR 평가에서는 2, 5, 10, 15, 20, 40+ pages로 page count를 나누고, 각 category에 최소 10권 이상의 book/document를 사용한다.

논문은 Distinct-n과 Edit Distance를 보고한다. Distinct-n은 generated text에서 unique n-gram ratio를 본다. Repetition collapse가 생기면 Distinct-n이 낮아질 수 있다.

Table 3 일부 수치는 다음과 같다.

| Pages | Distinct-20 | Distinct-35 |
| --- | ---: | ---: |
| 2 | 99.76% | 99.87% |
| 5 | 99.78% | 99.98% |
| 10 | 97.49% | 99.83% |
| 15 | 99.92% | 99.99% |
| 20 | 98.73% | 99.89% |
| 40+ | 96.08% | 96.90% |

40+ pages에서도 높은 Distinct-n을 유지한다는 점은 long generation이 repetitive collapse로 무너지는 것을 어느 정도 피한다는 evidence다. 다만 Distinct-n은 correctness를 직접 측정하지 않는다. 반복이 적다고 OCR이 정확하다는 뜻은 아니다. 반드시 edit distance와 manual case study를 함께 봐야 한다.

## 5-5. Efficiency

논문은 OmniDocBench에서 Unlimited OCR의 throughput을 DeepSeek-OCR과 비교한다.

| Model | TPS |
| --- | ---: |
| DeepSeek-OCR | 4951 |
| Unlimited OCR | 5580 |

이는 12.7% 속도 향상으로 보고된다. 중요한 점은 short benchmark보다 long output에서 advantage가 더 커진다는 주장이다. R-SWA는 output length가 길어질수록 standard MHA 대비 cache advantage가 커지는 구조이기 때문이다.

## 5-6. What really matters in the experiments

### 1) Single-page 성능이 떨어지지 않는가

R-SWA는 cache를 줄이는 mechanism이다. 만약 OCR accuracy가 크게 떨어지면 의미가 없다. 논문은 OmniDocBench에서 R-SWA가 accuracy를 유지하거나 개선한다고 보고한다. 이는 R-SWA가 OCR decoder에는 충분한 context를 제공한다는 근거다.

### 2) Long-horizon에서 repetitive collapse를 피하는가

Long generation은 반복, drift, page order error가 생길 수 있다. Distinct-n은 이 중 반복 collapse를 보는 간단한 지표다. 하지만 reading order와 content accuracy를 더 직접적으로 측정하는 long-document benchmark가 필요하다.

### 3) Prefix cost는 여전히 남는다

R-SWA의 cache는 $L_m+n$으로 bounded되지만, $L_m$은 page count와 resolution에 따라 증가한다. DeepEncoder compression이 강하지 않으면 one-shot parsing limit가 낮아진다.

### 4) Multi-page data가 synthetic인 점

Training multi-page sample은 single-page concatenation 기반이다. 실제 long document의 cross-page structure까지 학습했다고 단정하기 어렵다.

# 6. Limitations

1. **Unlimited는 문자 그대로 무한 길이를 뜻하지 않는다**
   - Training과 inference는 32K max length setting을 사용한다.
   - Prefix length와 output length limit는 여전히 있다.

2. **Prefix attention cost가 남아 있다**
   - Visual reference token은 항상 visible하므로 page count가 늘면 prefix cost가 증가한다.
   - Encoder compression이 충분히 강해야 한다.

3. **Multi-page training data는 synthetic이다**
   - Single-page sample을 concatenate해 만든 multi-page data가 real book structure를 충분히 대표하는지는 불확실하다.

4. **Long-horizon correctness metric은 제한적이다**
   - Distinct-n은 repetition을 보지만 OCR correctness를 직접 보장하지 않는다.
   - Long-document edit distance와 human inspection이 더 중요하다.

5. **R-SWA는 task-specific하다**
   - Static reference와 local output state만으로 충분한 task에는 강하다.
   - General long-context reasoning이나 dialogue에 그대로 쓰면 long-range output dependency를 잃을 수 있다.

6. **Baseline 의존성**
   - Architecture는 DeepSeek-OCR의 DeepEncoder와 MoE decoder에 강하게 의존한다.
   - 다른 OCR backbone에서도 같은 gain이 나오는지 확인이 필요하다.

7. **OCR benchmark saturation**
   - v1.6 overall에서 Qianfan-OCR과 차이가 0.02 point 수준이다.
   - Statistical variance와 evaluation noise를 고려해야 한다.

8. **Language and domain coverage**
   - Paper에서 benchmark coverage가 다국어, handwriting, historical document, noisy scan까지 충분히 대표하는지 추가 확인이 필요하다.

9. **Serving 구현 복잡도**
   - R-SWA는 KV cache eviction과 attention mask가 serving engine과 맞아야 한다.
   - Custom kernel path가 production support에 영향을 준다.

10. **ASR/translation 확장은 아직 실험으로 보이지 않았다**
    - Paper는 R-SWA의 generality를 주장하지만 main empirical result는 OCR 중심이다.

# 7. My Take

## 7-1. Why this matters for my work

Unlimited OCR의 핵심은 OCR 자체보다 **reference-conditioned generation의 memory design**이다.

많은 multimodal parsing task는 다음 형태를 갖는다.

$$
\text{static reference}
+
\text{long output stream}
$$

이때 generated history 전체를 계속 들고 있는 것은 낭비일 수 있다. Source reference가 충분히 보존된다면, output side는 local progress tracking만으로 충분한 task가 있다.

대표적으로 다음 task가 그렇다.

- OCR
- ASR transcription
- Source-to-target translation
- Data-to-report linearization
- Visual document to Markdown
- Long form copy editing with fixed source

R-SWA는 이런 task를 위한 attention primitive로 읽을 수 있다.

## 7-2. Reuse potential

### Document AI

Private PDF를 page-by-page OCR한 뒤 concat하는 방식은 여전히 많이 쓰인다. R-SWA style model이 안정화되면 다음 workflow가 가능해진다.

- 여러 page image를 one prompt에 넣는다.
- Output은 full Markdown document로 생성한다.
- Header/footer repetition과 page boundary를 model이 직접 처리한다.
- Post-processing scheduler가 줄어든다.

### Long-context compression

DeepSeek-OCR이 optical compression을 강조했다면, Unlimited OCR은 decoder memory를 함께 줄인다. Document AI에서 long context 문제는 encoder와 decoder 양쪽을 동시에 봐야 한다.

### Reference-based ASR

Audio encoder output이 static reference로 주어지고 transcript를 길게 생성하는 ASR도 유사한 structure다. 다만 ASR은 timestamp alignment, speaker turn, streaming chunk boundary가 있어 OCR보다 더 복잡할 수 있다.

### Translation

Source sentence나 document를 prefix로 고정하고 target translation을 생성한다면 R-SWA가 가능할 수 있다. 하지만 translation은 target long-range consistency가 OCR보다 더 중요할 수 있어 window size와 document-level term consistency를 주의해야 한다.

### Serving

Production에서는 다음을 직접 측정해야 한다.

- Page count별 prefix token length
- Output length별 KV memory
- TPS percentile
- Batch concurrency
- SGLang kernel support
- 큰 $n$과 작은 $n$에서의 accuracy degradation
- Page boundary 주변 error

## 7-3. Follow-up papers

- DeepSeek-OCR: Contexts Optical Compression
- olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models
- PaddleOCR-VL
- MinerU2.5
- DOLPHIN
- LightOnOCR / OCRFlux
- Sliding Window Attention and Longformer-style sparse attention
- StreamingLLM and attention sink studies
- ASR long-form transcription models with source-conditioned decoding

# 8. Summary

- Unlimited OCR은 decoder full attention을 R-SWA로 대체한다.
- R-SWA는 visual/prompt prefix를 항상 볼 수 있게 유지하고, generated output history는 최근 $n$ token으로 제한한다.
- KV cache는 $L_m+T$에서 최대 $L_m+n$으로 바뀐다.
- DeepSeek-OCR encoder compression과 R-SWA를 결합하면 더 긴 one-shot document parsing이 가능해진다.
- 가장 중요한 open question은 synthetic multi-page training과 OCR metric이 실제 long-document correctness를 충분히 포착하는지다.
