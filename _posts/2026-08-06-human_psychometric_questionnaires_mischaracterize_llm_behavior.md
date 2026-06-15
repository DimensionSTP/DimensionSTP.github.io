---
layout: single
title: "Human Psychometric Questionnaires Mischaracterize LLM Behavior Review"
categories: Study-concept
tag: [LLM, Evaluation, Psychometrics]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2509.10078)

Human Psychometric Questionnaires Mischaracterize LLM Behavior는 "LLM에게 사람용 심리검사를 풀리면 그 모델의 성향을 알 수 있는가?"라는 질문을 정면으로 다루는 논문이다. 이 논문을 단순히 "LLM은 personality test를 잘 못 푼다" 정도로 읽으면 아쉽다. 더 중요한 메시지는 LLM behavior evaluation에서 self-report style instrument와 실제 generation behavior를 분리해서 봐야 한다는 점이다.

최근 LLM safety, persona simulation, value alignment, user-facing assistant evaluation에서는 모델의 value나 personality를 측정하려는 시도가 많다. 그중 가장 손쉬운 방법은 사람에게 쓰던 questionnaire를 그대로 모델에게 주고 Likert score를 받는 것이다. 하지만 이 방법이 정말 모델이 실제 user query에 어떻게 답할지를 예측하는지는 별개의 문제다.

이 논문은 그 간극을 비교적 깔끔한 실험 구조로 보여준다. Established questionnaire에서 얻은 self-report profile과, realistic user query에 대한 candidate response의 generation probability에서 얻은 profile을 같은 construct space에서 비교한다. 결론은 분명하다. questionnaire profile은 꽤 일관되어 보이지만, 그 일관성이 실제 generation behavior로 잘 이어지지 않는다.

> 한 줄 요약: 이 논문은 PVQ와 BFI 같은 human psychometric questionnaire가 LLM의 실제 user interaction behavior를 안정적으로 예측하지 못하며, questionnaire item의 textual transparency와 persona prompt artifact가 profile을 그럴듯하게 보이게 만든다는 점을 generation probability profiling으로 보인다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM personality, value, persona evaluation에서 자주 쓰이는 questionnaire 기반 측정의 validity 문제를 직접 다룬다.
- PVQ-40/21, BFI-44/10 self-report profile과 Value Portrait 기반 generation probability profile을 같은 construct space에서 비교한다.
- "모델이 성향을 갖는다"보다 "측정 도구가 무엇을 실제로 측정하는가"라는 evaluation 관점으로 문제를 바꾼다.
- persona prompting이 questionnaire에서는 사람 demographic pattern처럼 보이는 shift를 만들지만, generation probability에서는 그 shift가 유지되지 않는다는 점을 보여준다.
- 서비스형 assistant, social simulation, safety evaluation에서 questionnaire score를 그대로 product metric으로 쓰면 왜 위험한지 설명하기 좋다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 LLM behavior를 human psychometric questionnaire로 측정할 수 있는가다.

사람의 가치관이나 성격을 측정할 때는 Portrait Values Questionnaire, Big Five Inventory 같은 도구가 오래 쓰여 왔다. 인간 대상에서는 이런 self-report가 특정 상황의 behavior와 어느 정도 연결될 수 있다. 그래서 LLM 연구에서도 자연스럽게 비슷한 시도를 한다. 모델에게 "당신은 이런 사람입니까?" 또는 "이 설명이 당신과 얼마나 비슷합니까?" 같은 질문을 주고 Likert response를 받는다. 그런 다음 그 점수를 기반으로 모델의 value profile이나 personality profile을 해석한다.

문제는 LLM에게 questionnaire를 푸는 행위가 실제 user interaction behavior와 같은 종류의 signal인지 불분명하다는 점이다. LLM은 questionnaire item을 읽고 "이 문항이 어떤 construct를 묻는지"를 알아낼 수 있다. 그러면 모델의 답변은 stable disposition의 발현이라기보다, 문항이 요구하는 socially desirable answer를 맞히는 behavior일 수 있다.

따라서 이 논문의 problem setting은 다음 질문으로 정리된다.

- LLM의 Likert self-report profile은 실제 generation behavior profile과 일치하는가?
- questionnaire item 내부에서 보이는 construct consistency가 실제 generation probability에서도 유지되는가?
- questionnaire item의 wording 자체가 target construct를 너무 명시적으로 드러내는가?
- demographic persona prompt로 생긴 questionnaire profile shift가 실제 generation behavior에도 전이되는가?

## 1-2. Why previous approaches are insufficient

기존 psychometric evaluation의 약점은 측정 방식이 너무 쉽고, 그 쉬움이 오히려 artifact를 만든다는 점이다.

첫째, questionnaire는 reflective response를 요구한다. 모델에게 "나는 외향적이다", "나는 규칙을 잘 지킨다", "나는 다른 사람을 돕는 것을 중요하게 여긴다" 같은 문항을 주면, 모델은 그 문항이 무엇을 측정하는지 쉽게 추론할 수 있다. 그 결과는 실제 behavior tendency보다 instruction following과 alignment prior를 더 많이 반영할 수 있다.

둘째, established questionnaire item은 lexical cue가 강하다. 예를 들어 BFI item에는 forgiving, organized, talkative 같은 trait cue가 직접 들어간다. PVQ item도 achievement, security, benevolence 같은 value dimension과 연결되는 단어와 상황을 비교적 노골적으로 포함한다. 모델이 이런 단서를 잡아 construct-consistent answer를 생성하는 것은 어렵지 않다.

셋째, persona prompting은 questionnaire response를 더 그럴듯하게 만들 수 있다. "elderly", "right-wing", "university educated" 같은 persona를 system prompt로 주면, 모델은 그 persona에 대해 학습된 stereotype을 questionnaire item에 적용할 수 있다. 하지만 그 shift가 realistic user query에 대한 generation behavior에도 유지되는지는 별도로 확인해야 한다.

넷째, 기존 연구 중 일부는 questionnaire와 behavior gap을 보였지만, behavior probe가 실제 user interaction과 멀거나, target construct와 독립적으로 검증된 item을 쓰지 못한 경우가 많았다. 이 논문은 Value Portrait dataset을 사용해 construct space를 유지하면서도, realistic query-response setting으로 behavior proxy를 만든다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 LLM psychometric profile을 두 방식으로 만들고 직접 비교하는 것이다.

| Profile type | Measurement | What it captures |
| --- | --- | --- |
| Established questionnaire profile | PVQ-40, PVQ-21, BFI-44, BFI-10에 대한 Likert response | 모델이 명시적 심리검사 문항에 어떻게 self-report하는가 |
| Generation probability profile | Value Portrait scenario-response pair에서 response log-probability | 모델이 realistic user query에서 어떤 value-laden response에 probability를 주는가 |

이 비교를 통해 논문은 네 가지 research question을 다룬다.

1. RQ1: established questionnaire profile과 generation probability profile의 construct ranking이 일치하는가?
2. RQ2: questionnaire에서 보이는 within-construct consistency가 generation probability에서도 나타나는가?
3. RQ3: LLM은 item text만 보고 target construct를 인식할 수 있는가?
4. RQ4: demographic persona prompt로 생긴 profile shift가 human demographic pattern과 behavior level에서도 맞는가?

이 구조가 좋은 이유는 "LLM에게 심리검사를 하면 이상한 답을 한다"가 아니라, 같은 construct를 다른 measurement channel로 측정했을 때 결과가 얼마나 일치하는지를 본다는 점이다.

## 2-2. Design intuition

이 논문의 설계 직관은 behavior measurement에서 "무엇을 묻는가"보다 "어떤 행동을 관측하는가"가 중요하다는 것이다.

Established questionnaire는 모델에게 자신을 평가하라고 요구한다. 이것은 사람이 자기 성향을 보고하는 setting과 겉모양은 비슷하지만, LLM에게는 다른 task가 된다. 모델은 문항을 semantic classification task처럼 풀 수 있다. "이 문항은 Agreeableness를 묻는구나", "이 문항은 Security를 묻는구나"를 알아내고, alignment-consistent answer를 고르는 식이다.

반대로 generation probability profiling은 모델에게 construct label을 직접 묻지 않는다. 현실적인 user query와 plausible response candidates가 있고, 모델이 각 response에 얼마나 높은 probability를 부여하는지를 본다. 이 방법도 완전한 free generation은 아니지만, questionnaire self-report보다는 실제 assistant behavior에 더 가까운 signal을 준다.

이 논문의 가장 중요한 포인트는 "questionnaire score가 틀렸다"가 아니라 "questionnaire score가 무엇을 측정하는지 다시 정의해야 한다"는 데 있다. Likert response는 model psychology의 직접 관측값이 아니라, model이 transparent item에 대해 수행한 constrained QA 결과일 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Human questionnaire 기반 LLM profile이 실제 generation behavior를 예측하는지 검증 |
| Constructs | Schwartz 10 values, Big Five 5 traits |
| Questionnaires | PVQ-40, PVQ-21, BFI-44, BFI-10 |
| Behavior proxy | Value Portrait의 realistic query-response pair에 대한 log-probability |
| Models | Gemma3 4B/27B, GPT-OSS 20B/120B, Qwen2.5 7B/72B, Qwen3 30B/235B |
| Main analyses | Ranking agreement, item-level construct structure, textual transparency, persona shift |
| Key claim | questionnaire profile coherence는 stable disposition보다 item transparency를 반영할 가능성이 크다 |

## 3-2. Module breakdown

### 1) Established questionnaire profiling

Established questionnaire profiling은 사람용 psychometric instrument를 LLM에게 그대로 적용한다.

논문은 PVQ-40과 PVQ-21로 Schwartz value profile을, BFI-44와 BFI-10으로 Big Five trait profile을 만든다. 각 item은 원래 문항 wording을 유지하되, pronoun은 gender-neutral form으로 바꾼다. PVQ는 1-6 Likert scale, BFI는 1-5 Likert scale을 사용한다.

LLM은 option order에 민감할 수 있으므로, 논문은 high-to-low와 low-to-high 두 가지 prompt variant를 사용하고 결과를 평균낸다. 이 설계는 단순하지만 중요하다. option order artifact를 줄이지 않으면, questionnaire profile 차이가 construct difference인지 formatting sensitivity인지 구분하기 어렵다.

### 2) Generation probability profiling

Generation probability profiling은 Value Portrait dataset을 사용한다.

Value Portrait는 104개 real-world query에서 파생된 520개 query-response pair를 포함한다. Source는 ShareGPT, LMSYS, Reddit, Dear Abby로 구성되며, everyday request, opinion question, brainstorming, open-ended discussion, interpersonal dilemma를 포함한다. 각 query에는 5개의 plausible candidate response가 붙고, response는 Schwartz values와 Big Five traits에 대해 human validation을 거친다.

논문은 모델이 각 candidate response를 실제로 생성하도록 sampling하지 않는다. 대신 prompt와 candidate response를 붙인 sequence에 대해 response token의 log-probability를 계산한다. 그러면 특정 construct에 tagged된 response들에 모델이 얼마나 높은 probability를 주는지로 generation-based profile을 만들 수 있다.

개념적으로는 아래처럼 볼 수 있다.

$$
score(c) = mean_{s in S_c} mean_{r in R_{s,c}} log p_model(r | prompt(s))
$$

여기서 $c$는 value 또는 trait construct, $s$는 scenario, $r$은 해당 construct에 tag된 candidate response다. 이 formulation은 free-form generation을 직접 평가하지는 않지만, construct-validated response candidate 위에서 model distribution을 읽는다는 장점이 있다.

### 3) Ranking agreement and item structure

RQ1은 construct ranking agreement를 본다.

논문은 established questionnaire profile을 reference ranking으로 두고, generation probability profile과의 agreement를 Spearman rank correlation과 NDCG로 측정한다. Spearman은 전체 ranking 일관성을 보고, NDCG는 top-ranked construct가 얼마나 맞는지를 더 크게 본다.

RQ2는 item-level construct structure를 본다.

questionnaire에서 같은 construct를 측정하는 item들이 비슷한 score를 받는다면, 이것은 stable disposition의 증거처럼 보일 수 있다. 하지만 그 구조가 generation probability profile에서는 사라진다면, questionnaire consistency가 실제 behavior tendency가 아니라 item transparency에서 왔을 가능성이 커진다.

논문은 between-construct differentiation과 within-model variance 계열 지표를 사용하고, construct label permutation baseline과 비교한다. 이 부분은 "평균 score가 다르다"보다 더 중요한 분석이다. 같은 construct끼리 정말로 묶이는지 보기 때문이다.

### 4) Textual transparency test

RQ3는 이 논문의 mechanism analysis에 가깝다.

논문은 LLM에게 item text와 construct definition을 주고, 이 item이 해당 construct를 측정하는지 yes/no로 판단하게 한다. Established questionnaire item에서는 모델이 target construct를 꽤 잘 알아본다. 반대로 VP scenario-response item에서는 recognition이 near chance에 가깝다.

여기에 sentence embedding analysis도 붙는다. all-mpnet-base-v2로 item text와 construct definition을 embedding하고, item이 자기 construct definition에 더 가까운지, 같은 construct item끼리 textually cluster되는지 본다. Established item은 diagonal structure가 뚜렷하지만, VP item에서는 그런 구조가 거의 없다.

이 분석이 중요한 이유는 questionnaire coherence의 원인을 behavior consistency가 아니라 measurement item의 surface cue로 설명하기 때문이다.

### 5) Persona shift test

RQ4는 demographic persona prompting이 실제 behavior shift를 만들었는지를 본다.

논문은 Gender, Age, Political orientation, Education 네 category에서 두 조건씩, 총 8개 demographic condition을 만든다. 예를 들어 male/female, 20-39/80+, right-wing/left-wing, below university/university+ 같은 조건이다. 각 persona prompt를 model에 넣고, vanilla profile 대비 value profile shift를 계산한다.

Human reference는 European Social Survey의 Schwartz value profile을 사용한다. ESS는 Big Five를 포함하지 않기 때문에, RQ4는 Schwartz values에만 적용된다. 이후 LLM persona-induced shift와 human demographic delta의 cosine similarity, direction match, normalized magnitude를 비교한다.

# 4. Training / Data / Recipe

이 논문은 모델을 새로 training하는 논문이 아니라 measurement and evaluation paper다. 따라서 여기서 중요한 것은 training objective가 아니라 evaluation recipe다.

## 4-1. Data

| Data / instrument | Role |
| --- | --- |
| PVQ-40, PVQ-21 | Schwartz basic values에 대한 Likert self-report profile |
| BFI-44, BFI-10 | Big Five personality traits에 대한 Likert self-report profile |
| Value Portrait | realistic user query와 construct-tagged candidate response 기반 behavior proxy |
| ESS Round 11 | persona-induced value shift와 비교할 human demographic reference |

Value Portrait의 구성이 특히 중요하다.

- 104개 source query를 사용한다.
- 각 query에 5개 candidate response를 붙여 총 520개 query-response pair를 만든다.
- 681명 human rater가 response similarity와 PVQ-21, BFI-10을 함께 평가해 construct association을 검증한다.
- 결과적으로 286개 value-associated tag와 228개 trait-associated tag가 생성된다.
- VP item은 realistic scenario를 유지하면서도, construct annotation을 human validation으로 확보한다.

이 덕분에 논문은 "실제 behavior처럼 보이는 아무 text"가 아니라, psychometric construct와 연결된 response candidate 위에서 generation probability를 측정할 수 있다.

## 4-2. Evaluation strategy

모델은 4 family에서 작은 variant와 큰 variant를 하나씩 포함한다.

| Family | Models |
| --- | --- |
| Gemma3 | 4B, 27B |
| GPT-OSS | 20B, 120B |
| Qwen2.5 | 7B, 72B |
| Qwen3 | 30B-A3B, 235B-A22B |

Inference setup은 다음과 같다.

- vLLM v0.16.0으로 local serving을 수행한다.
- Single node, 4 x NVIDIA A100 80 GB PCIe를 사용한다.
- 대부분 bfloat16으로 load하며, Qwen3-235B-A22B는 officially released FP8-quantized checkpoint를 사용한다.
- Likert scoring은 `/v1/chat/completions`, temperature 0.0, max_tokens 1024로 deterministic response를 받는다.
- VP log-probability scoring은 `/v1/completions`, `echo=True`, `logprobs=1`, `max_tokens=1`, `temperature=1.0` 설정을 사용한다.
- 실제 scoring에서는 prompt boundary 이후 response token log-probability만 추출하고, 마지막에 생성된 trailing token은 버린다.

이 설계는 꽤 engineering-heavy하다. 특히 generation probability를 측정하려면 closed API에서 단순 completion만 받는 것으로는 부족하다. token-level log-probability에 접근할 수 있어야 한다.

## 4-3. Engineering notes

실무적으로 재사용하려면 세 가지를 봐야 한다.

1. Questionnaire prompt artifact control
   - option order를 reverse해서 평균내는 것은 최소한의 control이다.
   - 실제 internal evaluation에서도 Likert option order, wording, persona prompt, system prompt를 모두 sensitivity test로 돌려야 한다.

2. Candidate response probability scoring
   - free-form answer를 생성한 뒤 사람이 annotation하는 방식은 noisy하다.
   - construct-validated candidate response set을 만들고, response log-probability를 읽는 방식은 더 controlled하다.
   - 다만 token log-probability API 접근이 필요하므로 model serving infra가 필요하다.

3. Behavior metric과 self-report metric의 분리
   - questionnaire score는 "모델이 이렇게 말한다"를 측정한다.
   - generation probability score는 "모델이 이런 response를 distribution상 얼마나 선호한다"를 측정한다.
   - product evaluation에서는 이 둘을 같은 dashboard에 넣되, 같은 metric으로 해석하면 안 된다.

# 5. Evaluation

## 5-1. Main results

### RQ1: questionnaire profile과 generation probability profile은 잘 맞지 않는다

Table 2의 핵심은 within-method agreement는 높지만 cross-method agreement는 낮다는 점이다.

| Comparison | Average Spearman |
| --- | ---: |
| PVQ-40 vs PVQ-21 | 0.74 |
| BFI-44 vs BFI-10 | 0.77 |
| Gen vs PVQ-40 | 0.31 |
| Gen vs PVQ-21 | 0.28 |
| Gen vs BFI-44 | 0.26 |
| Gen vs BFI-10 | 0.11 |

NDCG도 비슷한 방향이다. Within-method reference는 values에서 0.91, traits에서 0.89인데, cross-method NDCG는 대략 0.69-0.81 범위로 내려간다. 즉 questionnaire끼리는 서로 비슷한 profile을 만들지만, generation probability로 읽은 behavior-facing profile과는 일관성이 낮다.

여기서 중요한 점은 divergence가 모든 model family에서 나타난다는 것이다. 다만 어느 construct가 올라가고 내려가는지는 model별로 달라진다. 이는 단순한 universal correction factor로 보정하기 어렵다는 뜻이다.

### RQ2: item-level construct structure는 questionnaire에서만 나타난다

Table 3은 더 날카롭다. Established questionnaire에서는 같은 construct item들이 비슷한 score를 받는 structure가 나타난다. 하지만 generation probability에서는 그 structure가 permutation baseline과 크게 다르지 않다.

논문은 PVQ-40과 BFI-44에서 between-construct differentiation 평균이 각각 0.526, 0.492로 높게 나온다고 보고한다. WMV도 각각 0.603, 0.592로 questionnaire item이 construct별로 묶이는 듯한 모습을 보인다. 반대로 generation probability에서는 같은 construct로 tag된 response들이 random grouping과 비슷하게 흩어진다.

이 결과는 매우 중요하다. questionnaire에서 "같은 construct에 일관된 답을 했다"는 사실만으로 LLM의 stable disposition을 주장하기 어렵다. 그 일관성이 실제 generation behavior distribution에서는 유지되지 않기 때문이다.

### RQ3: established questionnaire item은 textually transparent하다

Table 4에서 LLM의 item-construct recognition은 established questionnaire에서 높고, VP에서는 낮다.

| Instrument | Mean recognition |
| --- | ---: |
| PVQ-40 | 0.69 |
| PVQ-21 | 0.72 |
| BFI-44 | 0.83 |
| BFI-10 | 0.80 |
| VP | 0.09 |

즉 모델은 established item text만 보고 "이 문항이 어떤 construct를 묻는지"를 꽤 잘 맞힌다. 반대로 VP scenario-response item은 construct cue가 약해서 near chance에 가깝다.

Sentence embedding 결과도 같은 방향이다. Established questionnaire item은 item-definition top-1 accuracy가 77-81% 수준이고, discrimination과 clustering gap도 뚜렷하다. VP item은 top-1 accuracy가 11-26% 수준이고, clustering gap은 거의 0에 가깝다. Figure 1의 heatmap도 이 차이를 시각적으로 보여준다.

이 분석은 논문의 mechanism을 뒷받침한다. questionnaire response가 coherent해 보이는 이유는 모델이 실제 성향을 안정적으로 드러내서라기보다, 문항이 target construct를 너무 잘 알려주기 때문일 수 있다.

### RQ4: persona shift는 questionnaire에서는 그럴듯하지만 generation probability에서는 유지되지 않는다

RQ4는 persona prompt의 위험을 보여준다. Demographic persona를 주면 PVQ response는 ESS human demographic pattern과 어느 정도 같은 방향으로 움직인다. 하지만 VP generation probability profile에서는 그런 pattern이 유지되지 않는다.

| Source | Mean cosine with human demographic delta | Direction match |
| --- | ---: | ---: |
| PVQ-40 | 0.60 | 62/80 |
| PVQ-21 | 0.47 | 55/80 |
| VP generation probability | 0.03 | 40/80 |

PVQ-40과 PVQ-21에서는 persona-induced shift가 사람 demographic pattern과 양의 cosine을 보인다. 하지만 VP에서는 mean cosine이 거의 0이고, direction match도 40/80으로 chance level에 가깝다. 논문은 이것을 persona prompt가 questionnaire item의 explicit cue와 결합해 stereotype-consistent self-report를 만든 결과로 해석한다.

또한 established questionnaire의 shift magnitude는 human reference보다 더 크게 보인다. 논문은 between-value normalized magnitude 기준으로 PVQ-40 0.665, PVQ-21 0.711, VP 0.373, human ESS 0.202를 보고한다. 단, 이 값은 classical Cohen's d가 아니라 within-profile spread로 normalize한 상대 magnitude이므로 과해석하면 안 된다.

## 5-2. What really matters in the experiments

### 1) 이 논문은 LLM psychology를 부정하는 논문이 아니다

이 논문은 "LLM에는 value나 trait이 없다"를 증명하지 않는다. 더 정확히는 human questionnaire score만으로 LLM behavior profile을 해석하면 안 된다는 측정론적 주장이다. 즉 object-level claim보다 measurement validity claim에 가깝다.

### 2) consistency는 validity가 아니다

Questionnaire에서 같은 construct item에 일관된 답을 하는 것은 얼핏 좋아 보인다. 하지만 그 consistency가 item wording의 lexical cue에서 온 것이라면, 실제 behavior predictability를 보장하지 않는다. 이 논문은 RQ2와 RQ3를 통해 그 차이를 잘 보여준다.

### 3) persona prompt는 simulation이 아니라 stereotype activation일 수 있다

Demographic persona를 넣었을 때 questionnaire score가 human demographic pattern처럼 움직이면, 모델이 해당 demographic을 잘 simulate한다고 해석하기 쉽다. 하지만 VP generation probability에서 같은 shift가 나타나지 않는다면, 이는 realistic behavior simulation보다는 transparent questionnaire item에 대한 stereotype matching에 가깝다.

### 4) generation probability profiling도 완전한 정답은 아니다

이 논문이 제안하는 generation probability profile은 questionnaire보다 behavior-facing이지만, free generation 전체를 포착하지는 않는다. Fixed candidate response 위에서 token log-probability를 읽는 controlled proxy다. 따라서 실제 deployment behavior를 보려면 open-ended output analysis, long-context interaction, safety-critical scenario testing과 함께 봐야 한다.

# 6. Limitations

1. Token-level log-probability access가 필요하다.
   - 이 방법은 open-weight model에는 적용하기 쉽지만, closed model에는 API가 log-probability를 제공하지 않으면 바로 적용하기 어렵다.
   - 논문도 closed-source model 확장은 API-level log-probability support나 alternative probability estimation이 필요하다고 본다.

2. Value Portrait coverage가 제한적이다.
   - VP는 Schwartz values와 Big Five traits를 unified benchmark로 다룬다.
   - 하지만 더 다양한 psychological construct, 더 넓은 situational context, domain-specific interaction까지 포함하는 것은 아니다.

3. VP scoring은 free-form generation이 아니다.
   - fixed candidate response의 probability를 읽기 때문에 sampling temperature나 paraphrase variation에는 덜 민감하다.
   - 대신 실제 open-ended output에서 trait이나 value가 어떻게 드러나는지는 직접 포착하지 못한다.

4. RQ4의 human reference는 Schwartz values 중심이다.
   - ESS는 PVQ 기반 value data를 제공하지만 Big Five를 포함하지 않는다.
   - 따라서 demographic persona shift 분석은 values에 한정된다.

5. RQ1-RQ3에 matched human baseline이 없다.
   - human self-report와 behavior 사이에도 gap은 존재한다.
   - 이 논문의 gap이 LLM 특유의 문제인지, 인간 psychometrics에서도 비슷하게 나타나는 일반적 self-report/behavior gap인지는 추가 비교가 필요하다.

6. 평가 모델 범위가 open-source model 8개로 제한된다.
   - Gemma3, GPT-OSS, Qwen2.5, Qwen3 family를 다루지만, closed frontier model이나 service-tuned model까지 일반화하려면 추가 실험이 필요하다.

7. Code and data release 상태를 확인해야 한다.
   - arXiv v4 abstract footnote는 code and data가 publication 이후 release될 예정이라고 적고 있다.
   - 최종 발행 전에 GitHub, Hugging Face, project page release 여부를 다시 확인하는 것이 좋다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 중요한 이유는 evaluation artifact를 꽤 현실적인 product risk로 연결해주기 때문이다.

LLM 서비스에서 persona, value, safety profile을 만들 때 가장 쉬운 방법은 questionnaire를 던지고 score를 얻는 것이다. 그런데 이 논문이 맞다면, 그 score는 실제 user interaction behavior보다 "모델이 transparent survey item에 얼마나 alignment-consistent하게 답하는가"를 더 많이 반영할 수 있다.

특히 user simulation, mental health assistant, education assistant, child-facing chatbot, value-sensitive recommendation 같은 영역에서는 이 차이가 중요하다. questionnaire score가 좋아 보여도, 실제 query에서 어떤 answer distribution을 갖는지 따로 봐야 한다.

이 논문의 메시지는 service evaluation에도 그대로 들어온다. 모델에게 "당신은 안전한가?"라고 묻는 것보다, safety-critical scenario에서 어떤 response에 probability를 주는지 보는 편이 더 직접적인 behavior metric이 된다.

## 7-2. Reuse potential

재사용해볼 만한 포인트는 다음과 같다.

1. Self-report metric과 behavior metric을 분리한 dashboard
   - persona, value, safety, helpfulness 평가에서 questionnaire score와 scenario-conditioned response probability를 별도 panel로 둔다.
   - 두 metric의 divergence 자체를 warning signal로 볼 수 있다.

2. Item transparency audit
   - evaluation item이 target construct를 너무 노골적으로 드러내는지 LLM classifier와 sentence embedding으로 먼저 검사한다.
   - item-definition top-1 accuracy가 너무 높으면, 그 benchmark는 behavior보다 construct recognition을 측정할 가능성이 있다.

3. Persona prompt transfer test
   - persona prompt로 self-report가 변했을 때, 실제 generation behavior도 같은 방향으로 변하는지 따로 본다.
   - 특히 demographic simulation에서는 questionnaire shift와 behavior shift를 반드시 분리해야 한다.

4. Candidate response probability profiling
   - open-ended generation annotation이 noisy한 경우, construct-validated candidate response set을 만들고 log-probability로 profile을 읽을 수 있다.
   - 다만 closed model에서는 log-probability API 접근 문제가 있으므로, pairwise preference query나 calibrated judge를 대체 proxy로 검토해야 한다.

5. Internal benchmark construction
   - 사내 서비스에서 "모델 성향"을 측정하고 싶다면, 사람용 questionnaire를 그대로 쓰기보다 실제 user log에서 scenario를 뽑고 candidate response를 검증하는 방식이 더 낫다.

## 7-3. Follow-up papers

- Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items
- Quantifying Data Contamination in Psychometric Evaluations of LLMs
- Self-assessment tests are unreliable measures of LLM personality
- Limited Ability of LLMs to Simulate Human Psychological Behaviours: a Psychometric Analysis
- AIPsychoBench: Understanding the Psychometric Differences between LLMs and Humans

# 8. Summary

- 이 논문은 human psychometric questionnaire가 LLM behavior profile을 안정적으로 예측하는지 검증한다.
- PVQ/BFI self-report profile은 서로 잘 맞지만, Value Portrait 기반 generation probability profile과는 낮은 agreement를 보인다.
- Questionnaire item은 target construct를 textually transparent하게 드러내며, 모델은 이를 인식해 socially desirable answer를 낼 수 있다.
- Demographic persona prompt는 questionnaire에서는 human pattern처럼 보이는 shift를 만들지만, realistic generation probability에서는 그 shift가 유지되지 않는다.
- 따라서 LLM personality/value evaluation에서는 self-report score보다 behavior-facing probability profile과 item transparency audit이 더 중요하다.
