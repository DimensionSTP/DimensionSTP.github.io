---
layout: single
title: "Comparing Transformers and Hybrid Models at the Token Level Review"
categories: Study-concept
tag: [Transformer, HybridModel, TokenLevelAnalysis, OLMo, SequenceModeling]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.20936)

Comparing Transformers and Hybrid Models at the Token Level는 "hybrid model이 transformer보다 validation loss가 낮다"에서 멈추지 않는다. 이 논문이 실제로 묻는 질문은 더 세밀하다. **hybrid가 이기는 token은 어떤 token이고, transformer가 여전히 강한 token은 어떤 token인가**다.

최근 attention layer와 recurrent layer를 섞는 hybrid language model은 꽤 설득력 있는 방향으로 보인다. Attention은 prefix 안의 token을 직접 다시 볼 수 있으므로 copy, recall, bracket matching에 강하다. 반면 recurrent layer는 hidden state를 계속 업데이트하므로 ordered state tracking에 유리하다는 이론적 동기가 있다. 실제로 OLMo Hybrid 같은 모델은 같은 pretraining budget에서 pure transformer보다 aggregate loss나 benchmark가 좋아질 수 있다.

하지만 aggregate loss 하나만 보면 어떤 능력이 개선됐는지 알 수 없다. Open-class content word에서 좋아진 것인지, bracket closing에서 좋아진 것인지, repeated n-gram copy에서 좋아진 것인지, entity state tracking에서 좋아진 것인지가 모두 한 숫자에 섞인다.

이 논문은 OLMo 3 7B와 OLMo Hybrid 7B의 released weights를 사용해 같은 prefix, 같은 target token에서 paired token loss를 비교한다. 각 token 위치마다 transformer loss와 hybrid loss의 차이를 계산하고, 이를 POS tag, code/markup token category, copy feature, delimiter role, synthetic probe로 나눠 분석한다.

> 한 줄 요약: 이 논문은 OLMo 3 7B와 OLMo Hybrid 7B를 같은 token 위치에서 직접 비교해, hybrid gain이 open-class semantic state prediction에 집중되고, repeated n-gram copy나 closing delimiter 같은 visible-prefix retrieval/structural closure에서는 transformer가 여전히 경쟁적이거나 더 유리하다는 것을 보여주는 token-level diagnostic study다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같다.

- Hybrid architecture의 이점을 aggregate validation loss가 아니라 token family별로 분해한다.
- Attention과 recurrence의 이론적 역할이 실제 natural token prediction에서 어디에 나타나는지 확인한다.
- POS tag, parser tag, repeated n-gram, delimiter open/close, controlled probe를 한꺼번에 사용한다.
- Pretraining run 중 architecture 비교에 쓸 수 있는 filtered token loss diagnostic을 제안한다.
- Hybrid model이 항상 좋은 것이 아니라, 어떤 token regime에서 이기고 지는지 보여준다.

이 글에서는 이 논문을 "새 hybrid architecture 제안"이 아니라, **sequence mixer를 평가할 때 평균 loss를 어떤 axis로 쪼개야 하는지 보여주는 분석 논문**으로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Language model validation loss는 모든 target token의 negative log-likelihood를 평균한다.

$$
L
=
\frac{1}{N}
\sum_{i=1}^{N}
-\log p(x_i \mid x_{<i})
$$

이 값은 모델 비교에 유용하지만, 어떤 token prediction이 좋아졌는지는 말해주지 않는다. Hybrid model이 transformer보다 average loss가 낮아도 그 이유는 여러 가지일 수 있다.

- 더 나은 semantic state representation
- 더 나은 long-range entity tracking
- 더 나은 copy ability
- 더 나은 delimiter matching
- 더 나은 frequent token smoothing
- 특정 domain의 tokenizer artifact
- 단순 overall calibration 차이

논문은 이를 token-level paired comparison으로 바꾼다. 같은 prefix $x_{<i}$와 같은 observed target $x_i$에 대해 transformer와 hybrid의 loss를 각각 계산한다.

$$
\ell_i^{\mathrm{Tr}}
=
-\log p_{\mathrm{Tr}}(x_i \mid x_{<i})
$$

$$
\ell_i^{\mathrm{Hyb}}
=
-\log p_{\mathrm{Hyb}}(x_i \mid x_{<i})
$$

그리고 paired gap을 정의한다.

$$
\Delta_i
=
\ell_i^{\mathrm{Tr}}
-
\ell_i^{\mathrm{Hyb}}
=
\log p_{\mathrm{Hyb}}(x_i \mid x_{<i})
-
\log p_{\mathrm{Tr}}(x_i \mid x_{<i})
$$

$\Delta_i > 0$이면 해당 token 위치에서 hybrid가 observed token에 더 높은 probability를 준 것이다.

이 정의의 장점은 모델 차이를 token occurrence 단위로 볼 수 있다는 점이다. 평균 loss 차이를 다음 질문으로 바꾼다.

> Hybrid의 gain은 어떤 token 위치에서 실제로 발생하는가?

## 1-2. Why previous approaches are insufficient

### 1) Aggregate loss는 capability mixture다

Validation loss는 prose, code, markup, content word, function word, repeated token, delimiter, numeric literal, identifier가 모두 섞인 값이다. Architecture가 바뀌어도 어떤 computation이 좋아졌는지 보기 어렵다.

예를 들어 hybrid가 average loss에서 이겼다고 해도, 그 이득이 content word prediction에서 온 것인지, repeated boilerplate에서 온 것인지에 따라 architecture 해석이 완전히 달라진다.

### 2) Downstream benchmark는 attribution이 어렵다

Downstream benchmark score는 task-level 결과를 주지만, 어떤 token prediction primitive가 좋아졌는지는 알기 어렵다. Hybrid layer가 state tracking을 개선했는지, 단순 calibration이나 domain mismatch가 원인인지 분리하기 어렵다.

### 3) Theoretical expressivity와 natural token behavior 사이 gap이 있다

이론적으로 attention은 visible prefix retrieval과 structural matching에 강하고, recurrence는 ordered state tracking에 강하다. 그러나 실제 language model token distribution에서 이 regime이 어떻게 나타나는지는 별도 empirical analysis가 필요하다.

이 논문은 바로 이 gap을 메운다. 이론적 primitive를 natural token tag와 synthetic probe로 연결한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 contribution은 네 가지다.

1. **Token-level paired loss decomposition**
   - 같은 prefix와 target token에서 transformer와 hybrid의 loss gap을 계산한다.
   - Average loss가 아닌 token occurrence별 contribution을 본다.

2. **Natural-token stratification**
   - Prose에서는 Brown POS tag를 사용한다.
   - Python, HTML, LaTeX에서는 lightweight tokenizer/parser tag를 사용한다.
   - Whole word, subword position, token frequency, copy feature를 함께 기록한다.

3. **Controlled synthetic probes**
   - Pronoun memory
   - Entity tracking
   - Structural closure
   - Distance $d \in \{32,64,128,256,512,1024\}$를 바꾸며 비교한다.

4. **Filtered pretraining diagnostics**
   - 1B-scale Transformer, Hybrid, Pure RNN development checkpoints에서 filtered token loss를 비교한다.
   - Aggregate loss가 숨기는 architecture regime을 보여준다.

## 2-2. Design intuition

이 논문의 intuition은 next-token prediction을 하나의 homogeneous task로 보지 않는다는 것이다.

어떤 target token은 prefix에서 이미 보이는 material을 copy하면 된다.

```text
The National Hamburger Association of America ...
```

어떤 target token은 열린 bracket이나 tag를 닫으면 된다.

```text
<header> ... </header>
```

또 다른 token은 discourse state를 유지해야 한다. 예를 들어 role과 person을 연결해 두고 나중에 pronoun을 고르거나, attribute와 entity를 연결해 둔 뒤 나중에 entity를 회상해야 한다.

이 관점에서 architecture primitive는 다음처럼 나뉜다.

| Primitive | 강한 영역 | 약한 영역 |
| --- | --- | --- |
| Attention | Visible prefix의 copy, recall, opener matching | 순서가 있는 state update |
| Recurrence | Latent state 구성, role/entity tracking | 임의 prefix retrieval |
| Hybrid | State 구성과 recall의 결합 | Component ratio와 task mix에 민감 |

따라서 hybrid가 전체적으로 이겨도 모든 token에서 이긴다고 기대하면 안 된다. 이 논문은 그 non-uniformity를 정량화한다.

# 3. Architecture / Method

## 3-1. Overview

| 항목 | 내용 |
| --- | --- |
| Goal | Hybrid gain이 어떤 token prediction에서 발생하는지 분석 |
| Models | OLMo 3 7B vs OLMo Hybrid 7B |
| Unit | 같은 prefix와 같은 target token의 paired NLL gap |
| Domains | Prose, Python, HTML, LaTeX |
| Prose tags | Brown POS tag family |
| Structured tags | Identifier, string, comment, text node, attribute, command, delimiter, tag |
| Synthetic probes | Pronoun memory, entity tracking, structural closure |
| Diagnostic output | Raw tag gap, regression-adjusted effect, filtered token loss |

## 3-2. Natural token analysis

### 1) Paired loss gap

각 token 위치에서 $\Delta_i$를 계산한다.

$$
\Delta_i
=
\ell_i^{\mathrm{Tr}}
-
\ell_i^{\mathrm{Hyb}}
$$

Positive $\Delta_i$는 hybrid가 해당 token을 더 잘 예측한다는 뜻이다.

Tag $\tau$에 대한 raw gap은 다음이다.

$$
\widehat{\Delta}(\tau)
=
\frac{1}{|I_{\tau}|}
\sum_{i \in I_{\tau}}
\Delta_i
$$

이 값은 실제 corpus에서 해당 tag에 속한 token 위치의 average hybrid advantage다.

### 2) Tagging and alignment

Prose에서는 word span을 Brown POS tag로 labeling한다. Structured domain에서는 Python, HTML, LaTeX에 맞는 lightweight tagger/parser를 사용한다.

LM target token은 source text span과 align된다. 하나의 LM token이 여러 source tag와 overlap하면 multi-tag attribution을 사용한다.

이 방식은 tokenization artifact를 완전히 제거하지는 못하지만, BPE token을 source-level category와 연결하는 practical compromise다.

### 3) Regression robustness check

Raw tag mean은 difficulty, frequency, position, subword status, local copy 여부에 영향을 받을 수 있다. 그래서 논문은 linear regression을 사용한다.

$$
\Delta_i
\sim
\mathrm{DOMAIN}_i
+
\mathrm{TAG}_i
+
\mathrm{WPOS}_i
+
\mathrm{RELPOS}_i
+
\bar{\ell}_i
+
\bar{\ell}_i^2
+
\sum_{k \in \{1,2,3,4\}}
\mathrm{COPY}_{k,i}
+
\log \mathrm{PREVDIST}_i
+
\log \mathrm{FREQ}(x_i)
$$

이 regression은 tag effect가 단순히 token difficulty나 frequency 때문에 생긴 것이 아닌지 확인한다.

주요 control 변수는 다음과 같다.

| Control | 의미 |
| --- | --- |
| DOMAIN | Prose source fixed effect |
| TAG | POS 또는 word class |
| WPOS | Whole/prefix/middle/suffix subword status |
| RELPOS | Packed sequence 내 relative position |
| mean NLL | Difficulty proxy |
| COPY-k | Repeated k-gram completion 여부 |
| PREVDIST | 같은 token type의 이전 occurrence까지 거리 |
| FREQ | Target token type frequency |

## 3-3. Structured token analysis

Structured domain에서는 다음 category를 본다.

- Python identifier
- String
- Comment
- HTML text node
- Attribute value
- LaTeX command
- Opening delimiter
- Closing delimiter
- Formatting token

논문의 핵심 관찰은 hybrid가 유리한 category가 semantic token이나 content-bearing token에 많이 나타난다는 점이다. 반면 closing delimiter와 rigid formatting token에서는 hybrid advantage가 작거나 transformer가 경쟁적이다.

이 결과는 "recurrent layer가 document/program state를 업데이트하는 데 도움을 준다"는 intuition과 맞지만, "attention은 structural closure와 prefix reuse에 여전히 강하다"는 점도 같이 보여준다.

## 3-4. Synthetic probes

Natural token analysis는 observational이다. Surface tag는 모델이 실제로 어떤 computation을 해야 하는지 직접 말하지 않는다. 그래서 논문은 세 synthetic probe를 추가한다.

### 1) Pronoun memory

Prompt가 두 사람과 역할을 소개한다. Filler token 뒤에 특정 role을 다시 언급하고 pronoun을 골라야 한다.

예시 구조는 다음과 같다.

```text
Liam is the violinist.
Naomi is the pilot.
... filler ...
the violinist reviewed the report, and <target>
```

Target은 `he` vs `she` 같은 contrastive choice다.

### 2) Entity tracking

Prompt가 entity와 attribute를 bind한다. Filler 뒤에 특정 attribute를 가진 entity를 물어본다.

```text
Julia carried the orange notebook.
Sofia carried the green folder.
... filler ...
Who carried the green folder?
```

이 task는 entity-attribute binding을 latent state로 유지해야 한다.

### 3) Structural closure

Prompt가 bracket이나 tag region을 열고 filler 뒤에 closing token을 예측하게 한다.

```text
<header>
...
</header>
```

이 task도 delayed dependency를 갖지만, answer는 visible opener와 structural stack에 의해 결정된다. 따라서 recurrence보다 attention-based matching이 더 유리할 수 있다.

### 4) Distance sweep

세 probe 모두 distance $d$를 바꾼다.

$$
d \in \{32,64,128,256,512,1024\}
$$

Pronoun probe와 entity probe는 accuracy와 log-prob margin을 보고, closure probe는 closing token NLL을 본다.

# 4. Training / Data / Recipe

## 4-1. Data and evaluation corpus

논문은 OLMo 3 7B와 OLMo Hybrid 7B의 released checkpoints를 사용한다. 두 model은 tokenizer, data mixture, training recipe가 closely matched되어 있다고 설명된다. 따라서 token-level gap이 주로 sequence mixer 차이를 반영한다고 가정한다.

Evaluation domains는 다음과 같다.

| Domain type | Sources |
| --- | --- |
| Prose | PG-19, News, Wikipedia, essay, textbooks, scientific papers |
| Code | Python |
| Markup | HTML, LaTeX |

Text는 length $L=8192$ packed sequence로 구성된다. 모든 target position에서 next-token NLL을 계산한다.

## 4-2. Model comparison recipe

이 논문은 training recipe를 제안하지 않는다. 대신 released model을 diagnostic 방식으로 비교한다.

1. 같은 packed sequence를 두 model에 넣는다.
2. 같은 target token에서 $\ell_i^{\mathrm{Tr}}$와 $\ell_i^{\mathrm{Hyb}}$를 계산한다.
3. Paired gap $\Delta_i$를 구한다.
4. Tag, copy feature, delimiter role, synthetic probe family별로 aggregate한다.
5. Regression control을 사용해 robustness를 확인한다.
6. 1B-scale development checkpoint에서는 filtered validation loss curve를 본다.

## 4-3. Filtered evaluation

Pretraining diagnostic으로 세 filter를 제안한다.

| Filter | 의미 |
| --- | --- |
| All tokens | 표준 validation loss |
| Top-10 and No-Copy | Hybrid가 유리한 non-copy open-class target |
| Copy-5 only | 반복된 5-gram completion target |

All-token loss는 여러 capability regime을 섞는다. Top-10 and No-Copy는 state-conditioned readout에 더 민감하고, Copy-5 only는 visible-prefix reuse에 더 민감하다.

이 filter는 new benchmark가 아니다. 같은 validation NLL에서 subset을 잘라 보는 방식이므로 overhead가 낮다.

## 4-4. Engineering notes

실무적으로 가져갈 수 있는 포인트는 다음과 같다.

1. **Architecture comparison에는 filtered loss가 필요하다**
   - Aggregate validation loss만으로는 sequence mixer의 성격을 보기 어렵다.

2. **Copy feature를 반드시 control해야 한다**
   - Repeated n-gram은 attention-friendly regime이다.
   - 이를 섞어 평균하면 recurrence gain이 희석될 수 있다.

3. **Opening delimiter와 closing delimiter를 나눠야 한다**
   - 같은 bracket character라도 target role이 다르다.
   - Opener는 새 scope나 state update이고, closer는 structural obligation을 만족시키는 token이다.

4. **Synthetic probe는 delayed dependency만으로 충분하지 않다**
   - Long dependency라도 state-conditioned choice와 visible-prefix closure는 다르다.

5. **Hybrid design은 layer ratio를 함께 봐야 한다**
   - Hybrid가 어디서 좋은지 알아야 attention/recurrent ratio를 조정할 수 있다.

# 5. Evaluation

## 5-1. Natural token results

논문은 hybrid가 대부분의 tag family에서 lower loss를 보인다고 보고한다. 하지만 gain은 균일하지 않다.

### Content vs function words

Prose에서 content words의 raw gap은 function words보다 크다.

| Token family | Raw gap |
| --- | ---: |
| Content words | 0.0384 nats |
| Function words | 0.0238 nats |
| Difference | 0.0146 nats |

논문은 이 차이를 61% larger로 설명한다. Regression control 이후에도 content-function contrast는 유지된다.

Interpretation은 다음과 같다.

- Content words는 open-class vocabulary에 속하고 document/discourse state에 더 의존한다.
- Function words는 closed-class grammar pattern과 local syntax에 더 많이 의존한다.
- Hybrid recurrent layer는 content-bearing state prediction에 더 도움이 될 수 있다.

### Structured domains

Code와 markup에서도 비슷한 pattern이 나타난다.

Hybrid-favored categories는 다음에 많다.

- Identifier
- String
- Comment
- Text node
- Attribute value
- Command

반대로 advantage가 작은 영역은 다음이다.

- Closing delimiter
- Rigid formatting token
- Repeated n-gram continuation

## 5-2. Opening vs closing delimiter

Opening bracket/tag는 hybrid가 더 유리한 반면, closing bracket/tag에서는 advantage가 줄어든다. 이 차이는 prose, Python, HTML, LaTeX 전반에서 관찰된다.

논문 해석은 다음과 같다.

- Opening delimiter는 새 region이나 scope를 시작한다.
- Closing delimiter는 이미 prefix에 있는 opener와 structural stack을 만족한다.
- Closing target은 attention-based visible-prefix matching이 강한 regime이다.

이 결과는 surface symbol 자체가 아니라 **predictive role**이 중요하다는 점을 보여준다.

## 5-3. Repeated n-grams

Repeated n-gram completion에서는 hybrid advantage가 거의 사라진다. Repetition-adjusted regression effect는 negative다.

이는 자연스러운 결과다. Repeated n-gram은 이전 prefix에 이미 나온 continuation을 reuse하면 되므로 attention이 직접 강한 primitive를 가진다. Recurrent state가 더 좋다고 해서 꼭 이길 필요가 없다.

## 5-4. Synthetic probes

Synthetic probes는 natural-token observation을 더 직접적으로 테스트한다.

| Probe | 필요한 computation | 유리한 model |
| --- | --- | --- |
| Pronoun memory | Role/person state readout | Hybrid |
| Entity tracking | Entity-attribute binding | Hybrid |
| Structural closure | Visible opener matching | Transformer |

중요한 점은 closure probe도 long-distance dependency를 갖는다는 것이다. 따라서 difference는 "long vs short"가 아니다.

더 정확한 구분은 다음이다.

- State-conditioned choice
- Visible-prefix copy/reuse
- Structural closure obligation

첫 번째 regime에서는 hybrid가 더 강하고, 나머지 두 regime에서는 transformer가 여전히 강하다.

## 5-5. Filtered pretraining diagnostics

1B-scale development run에서 all-token loss만 보면 Transformer와 Hybrid의 차이는 작다. 최대 gap이 대략 0.06 nats 수준이라고 보고된다.

하지만 Top-10 and No-Copy filter에서는 gap이 대략 0.12 nats로 커지고, ordering은 다음처럼 바뀐다.

```text
Hybrid < Pure RNN < Transformer
```

여기서 loss가 낮을수록 좋다.

Copy-5 only filter에서는 Pure RNN이 attention-based model보다 약 0.10-0.20 nats 나쁘다. 이 filter는 visible-prefix retrieval regime을 분리한다.

결론은 명확하다. Aggregate loss에서는 Pure RNN과 Transformer가 비슷해 보일 수 있지만, filtered loss를 보면 서로 다른 강점과 약점이 드러난다.

## 5-6. What really matters in the experiments

### 1) Hybrid gain은 균일하지 않다

Hybrid가 전체적으로 좋더라도 모든 token family에서 같은 gain을 주지 않는다. Open-class semantic token과 state tracking probe에서는 강하지만, copy와 closure에서는 attention이 강하다.

### 2) Token-level decomposition은 실용적인 diagnostic이다

Architecture iteration에서 "loss가 좋아졌다"보다 "어떤 filtered regime이 좋아졌는가"가 중요하다. Hybrid ratio, recurrent mixer type, attention layer placement를 조정할 때 diagnostic으로 쓸 수 있다.

### 3) Synthetic probe는 predictive role을 분명히 한다

Long dependency라고 모두 recurrence-friendly한 것이 아니다. Long-distance closure도 attention-friendly일 수 있다. 모델 비교는 dependency length보다 computation type을 기준으로 해야 한다.

### 4) Same-prefix paired loss는 좋은 비교 설계다

두 model이 같은 target token과 prefix를 보므로, output sampling variance가 없다. Architecture comparison에 깔끔한 unit이다.

# 6. Limitations

1. **두 model 비교에 기반한다**
   - OLMo 3 7B와 OLMo Hybrid 7B pair에 기반한다.
   - 다른 hybrid ratio, 다른 recurrent mixer, 다른 training recipe에 일반화하려면 추가 실험이 필요하다.

2. **Observational analysis가 많다**
   - Natural token tag 결과는 causal claim이 아니다.
   - Tag family가 difficulty, frequency, tokenization과 완전히 분리되지 않는다.

3. **Regression control도 완전하지 않다**
   - 포함된 covariate로 설명되지 않는 confound가 남을 수 있다.

4. **Tagger와 alignment noise가 있다**
   - POS tag와 code/markup parser가 source text 기준이고, LM token과 span overlap으로 align한다.
   - BPE token이 여러 tag를 걸칠 수 있다.

5. **Synthetic probe는 작고 stylized되어 있다**
   - Pronoun memory, entity tracking, closure가 real discourse/program state tracking 전체를 대표하지는 않는다.

6. **Loss gap은 downstream task score가 아니다**
   - Token-level diagnostic은 architecture mechanism을 보는데 유용하지만 user-facing task success와 직접 같지 않다.

7. **Hybrid architecture 내부 contribution은 분리하지 않는다**
   - 어떤 recurrent layer, 어떤 attention layer가 gain을 만드는지 layer-level causality는 제한적이다.

8. **Copy filter definition에 민감할 수 있다**
   - Repeated n-gram feature가 visible-prefix reuse의 proxy지만 모든 copying behavior를 포괄하지는 않는다.

9. **Tokenizer와 model family 의존성**
   - OLMo tokenizer와 corpus property가 result shape에 영향을 줄 수 있다.

10. **Filtered eval은 diagnostic이지 benchmark를 대체하지 않는다**
    - 특정 filter에 과최적화하면 aggregate capability를 해칠 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 중요한 메시지는 hybrid model의 우열이 아니라, **architecture comparison을 token-level capability accounting으로 바꿔야 한다는 점**이다.

요즘 sequence model 연구는 Transformer, Mamba, DeltaNet, hybrid, linear attention, recurrent depth가 복잡하게 섞인다. 그런데 validation loss 하나로 비교하면 다음 질문에 답하기 어렵다.

- Attention layer가 실제로 copy/retrieval을 맡고 있는가.
- Recurrent layer가 state tracking을 맡고 있는가.
- Open-class semantic prediction에서 gain이 나는가.
- Code delimiter와 syntax closure에서 손해가 나는가.
- Pure RNN이 평균 loss에서는 괜찮지만 copy regime에서 실패하는가.

이 논문은 그런 질문을 실제 pretraining diagnostic으로 바꾸는 방법을 보여준다.

## 7-2. Reuse potential

### Architecture ablation

새 sequence mixer를 제안할 때 aggregate loss 외에 다음 filter를 같이 보고 싶다.

- Content word loss
- Function word loss
- Identifier와 string loss
- Opening delimiter와 closing delimiter loss
- Repeated n-gram completion loss
- Copy-only loss
- No-copy content loss
- Synthetic state-tracking probe

### Data curation

만약 model이 content word에서 좋아지고 copy에서 나빠진다면, data mixture나 architecture가 semantic state에 치우친 것일 수 있다. 반대로 copy-only에서는 강하지만 no-copy content에서 약하면 retrieval은 강하지만 state representation이 약한 것이다.

### Hybrid layer design

Hybrid model에서 attention/recurrent layer ratio를 정할 때, "overall perplexity"만 보지 않고 retrieval-oriented filter와 state-oriented filter를 함께 봐야 한다.

### Training monitoring

Pretraining 중 filtered loss curve를 로그로 남기면 architecture가 어떤 capability를 먼저 배우고 어느 지점에서 saturated되는지 볼 수 있다.

## 7-3. Follow-up papers

- OLMo 3 Technical Report
- OLMo Hybrid
- Empirical Study of Hybrid Language Models
- Transformer와 RNN의 theoretical expressivity
- DeltaNet과 Gated DeltaNet
- Mamba-2
- RULER and NIAH benchmarks
- Token-level loss decomposition for data mixture analysis

# 8. Summary

- Hybrid model gain은 모든 token에서 균일하지 않다.
- OLMo Hybrid는 open-class content word, identifier, string, comment, text node에서 더 큰 gain을 보인다.
- Transformer는 repeated n-gram continuation과 structural closure에서 여전히 강하다.
- Synthetic probe에서도 hybrid는 pronoun/entity state tracking에, transformer는 closure에 유리하다.
- Filtered token loss는 sequence mixer pretraining diagnostic으로 유용하다.
