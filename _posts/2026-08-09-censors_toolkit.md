---
layout: single
title: "Position: The Alignment Community is Unintentionally Building a Censor's Toolkit Review"
categories: Study-concept
tag: [AI-Alignment, AI-Governance, AI-Safety]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://openreview.net/forum?id=dy2HwmOvFX)

[Project page](https://s-ball-10.github.io/censors-toolkit)

[ICML 2026 award announcement](https://blog.icml.cc/2026/07/05/announcing-the-icml-2026-awards/)

> 한 줄 요약: 이 position paper는 pre-training filtering, post-training preference alignment, inference-time guardrail이 harmful output을 줄이는 safety mechanism인 동시에, control objective를 정하는 actor에 따라 censorship과 manipulation에 재사용될 수 있는 dual-use control stack이라고 주장한다. 저자들은 alignment 중단이 아니라 transparency, independent audit, standardized benchmark, model pluralism, user and researcher awareness를 요구한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Alignment technique의 unintended bias가 아니라 human operator에 의한 deliberate misuse를 별도의 threat model로 정의한다.
- Pre-training, post-training, inference-time이라는 세 층에서 누가 어떤 access와 resource로 model behavior를 바꿀 수 있는지 구조적으로 비교한다.
- Safety refusal과 censorship을 같은 surface behavior로 보지 않고, 목적, 권력 관계, transparency, recourse를 포함한 governance problem으로 확장한다.
- Model policy disclosure만으로 부족하며 provider cooperation 없이도 적용 가능한 black-box benchmark가 필요하다고 주장한다.
- 하나의 perfectly neutral model보다 multiple model과 provider의 pluralism이 더 현실적인 safeguard일 수 있다는 관점을 제시한다.
- ICML 2026 Outstanding Position Paper로 선정되어, AI safety와 freedom of information 사이의 tension을 alignment community 내부 의제로 끌어올렸다는 점에서 읽을 가치가 크다.

AI alignment는 일반적으로 model이 harmful instruction을 거절하고, human preference를 따르며, deployment policy를 준수하도록 만드는 기술로 설명된다. 같은 기술을 다른 관점에서 보면 model의 available information, preferred answer, refusal boundary를 조직적으로 제어하는 방법이기도 하다.

이 논문은 바로 이 dual-use property를 문제 삼는다. Alignment method 자체에는 benevolent objective가 내장되어 있지 않다. 누가 value와 policy를 정의하는지에 따라 같은 filtering, fine-tuning, system prompt, output classifier가 protection에도 쓰이고 information control에도 쓰일 수 있다는 주장이다.

중요한 점은 이 논문이 "모든 refusal은 censorship"이라고 주장하지 않는다는 것이다. Harmful instruction 차단과 privacy protection은 alignment의 중요한 목적이다. 저자들이 문제 삼는 것은 국제적으로 인정된 기본권을 침해하거나, 소수의 powerful actor가 수십억 user에게 보이지 않는 방식으로 자신의 private interest를 강제하면서 이의를 제기할 recourse도 제공하지 않는 경우다.

또한 이 논문은 새로운 model, dataset, benchmark score를 제시하지 않는다. 기존 alignment literature, policy document, survey, news report, documented incident를 하나의 control-stack argument로 묶은 position paper다. 따라서 읽을 때는 technical mapping의 설득력과 normative boundary의 타당성을 분리해 봐야 한다.

# 1. Problem Setting

## 1-1. Problem definition

### 1) Alignment method는 purpose-agnostic control tool이다

논문이 출발하는 premise는 단순하다.

> AI alignment는 AI system이 human intention과 value에 맞게 행동하도록 만드는 과정이다.

문제는 여기서 "human"이 하나의 합의된 actor가 아니라는 점이다. Model의 value를 실제로 정하는 주체는 dataset curator, annotator pool, policy author, model provider, regulator, deployment operator일 수 있다.

따라서 alignment의 technical success와 social legitimacy는 같은 개념이 아니다.

- Objective를 정확히 학습했는가.
- 그 objective가 누구의 value를 반영하는가.
- 다른 viewpoint를 부당하게 suppress하는가.
- User가 policy와 intervention을 알 수 있는가.
- 잘못된 suppression에 appeal할 수 있는가.

Model이 주어진 policy에 완벽히 aligned되어도, policy 자체가 oppressive하면 outcome은 safety가 아닐 수 있다.

### 2) 두 가지 threat: censorship과 manipulation

논문은 information control을 두 축으로 나눈다.

#### Censorship

특정 information을 user가 접근하지 못하도록 suppress하는 경우다.

- 특정 fact를 답하지 않는다.
- 특정 viewpoint를 항상 refusal한다.
- 특정 historical or political topic을 training data에서 제거한다.
- Generated answer를 runtime filter가 삭제하거나 refusal로 교체한다.

Surface behavior는 safety refusal과 비슷할 수 있다. 차이는 어떤 content가 왜 차단되며 누가 그 boundary를 정했는지에 있다.

#### Manipulation

Information을 완전히 숨기지 않고 user perception을 covertly distort하는 경우다.

- 특정 claim을 일관되게 더 신뢰하도록 framing한다.
- Counter-evidence를 약화하거나 생략한다.
- One-sided preference data로 answer tone과 conclusion을 steer한다.
- User가 intervention을 인지하지 못한 채 특정 political or commercial interest를 따르게 한다.

Manipulation은 refusal보다 detection이 어렵다. Answer가 fluent하고 fact-like하게 보일 수 있기 때문이다.

### 3) 무엇을 misuse라고 부를 것인가

모든 alignment는 output distribution을 바꾸므로, 단순히 behavior modification이 있다는 이유만으로 misuse라고 할 수 없다. 저자들은 strict boundary로 두 조건을 제시한다.

1. 국제적으로 인정된 freedom of expression, thought, access to information 같은 human right를 침해하는 경우
2. User에게 visibility와 recourse가 없는 상태에서, 소수의 powerful actor의 private interest를 수십억 user의 interest보다 우선하는 경우

이 정의는 perfect하지 않다. Hate speech, dangerous instruction, political propaganda의 boundary는 society마다 다르게 판단될 수 있다. 논문도 sociocultural background가 assessment에 영향을 준다는 점을 인정한다.

그래도 misuse를 "자신이 동의하지 않는 policy"로 넓히지 않고, rights, power asymmetry, transparency, recourse라는 기준에 묶으려는 시도는 중요하다.

### 4) 가능한 actor

논문은 large-scale behavior를 결정할 수 있는 주요 actor로 두 집단을 둔다.

#### State actors

- Law와 regulation으로 provider에게 alignment objective를 강제할 수 있다.
- Approved model만 허용하거나 alternative access를 제한할 수 있다.
- 직접 model을 만들지 않아도 provider를 intermediary로 사용해 control할 수 있다.

#### Foundation model providers

- Pre-training corpus, post-training data, reward model, system policy, safety classifier를 직접 통제한다.
- Technical team과 compute를 보유해 깊은 intervention이 가능하다.
- Single policy change가 매우 많은 user에게 동시에 적용될 수 있다.

Downstream fine-tuner도 model을 steer할 수 있지만, 논문은 reach와 resource가 큰 original provider에 주로 초점을 맞춘다.

## 1-2. Why previous approaches are insufficient

### 1) Alignment misuse는 보통 accidental harm 뒤에 가려진다

Alignment research는 bias, representation failure, over-refusal, cultural mismatch 같은 unintended harm을 많이 다뤄 왔다. 이는 중요하지만, malicious or self-interested actor가 system을 의도적으로 steer하는 threat와는 다르다.

- Accidental bias: dataset와 policy의 blind spot에서 발생한다.
- Deliberate misuse: actor가 원하는 suppression or manipulation을 목표로 control stack을 설계한다.

두 경우는 audit requirement도 다르다. Accidental bias는 representative data와 error analysis가 중요하다. Deliberate misuse는 policy provenance, independent verification, power concentration, user recourse가 더 중요하다.

### 2) Output-only evaluation은 intervention layer를 알려주지 않는다

같은 refusal도 여러 원인에서 나올 수 있다.

- Pre-training corpus에 관련 information이 없었다.
- Post-training에서 특정 answer를 negative preference로 학습했다.
- System prompt가 topic을 금지했다.
- Output classifier가 generation 뒤 answer를 교체했다.

Behavioral test만으로 suppression의 존재를 볼 수는 있지만, 어느 layer가 원인인지 정확히 알기 어렵다. 반대로 weight access가 있어도 proprietary training data와 policy history가 없으면 full provenance를 복원하기 어렵다.

따라서 transparency와 black-box audit이 함께 필요하다.

### 3) Left-right political bias benchmark만으로는 부족하다

Existing political bias evaluation은 특정 국가의 ideological axis나 limited contentious topic에 묶이는 경우가 많다. 논문은 다음 coverage가 필요하다고 주장한다.

- Multiple country and language
- Authoritarian tendency와 state power
- Historical information suppression
- Minority and opposition viewpoint
- Commercial and provider self-interest
- Time-varying policy and current event
- Refusal뿐 아니라 subtle framing and factual distortion

단일 left-right score는 censorship과 manipulation을 충분히 설명하지 못한다.

### 4) Transparency alone도 충분하지 않다

Provider가 policy document를 공개하더라도 user가 alternative model을 선택할 수 없다면 practical control은 남는다. 또한 state가 malicious actor라면 domestic disclosure rule이 독립적 oversight를 보장하지 못할 수 있다.

논문은 transparency, evaluation, pluralism을 complementary safeguard로 본다.

# 2. Core Idea

## 2-1. Main contribution

이 position paper의 contribution은 네 가지다.

### 1) Deliberate misuse를 alignment threat model에 포함

Alignment method가 robust해질수록 jailbreak와 unauthorized modification은 어려워진다. 이는 safety에는 유리하지만, malicious policy owner가 설정한 restriction도 더 robust해질 수 있다.

논문은 이 asymmetry를 강조한다.

- User는 policy를 우회하기 어려워진다.
- Provider나 state는 policy definition을 계속 통제한다.
- Alignment robustness가 power holder의 control robustness로 바뀔 수 있다.

### 2) Three-stage control stack

Modern LLM behavior를 세 층으로 나눈다.

1. Pre-training data filtering
2. Post-training preference alignment
3. Inference-time control

각 layer는 access requirement, compute, expertise, modification ease, depth가 다르다. 이 mapping은 censorship risk를 하나의 "biased answer" 문제가 아니라 system lifecycle 전체의 control surface로 본다는 점에서 유용하다.

### 3) 왜 지금 논의해야 하는가

논문은 세 변화가 risk를 키운다고 주장한다.

1. LLM이 information source로 빠르게 채택된다.
2. Foundation model market과 infrastructure가 소수 provider에 집중된다.
3. Digital space에서 authoritarian control pressure가 커진다.

논문은 Reuters의 6개국 survey를 인용해 weekly generative AI use가 2024년 18%에서 2025년 34%로 늘었고 information seeking이 dominant use case라고 설명한다. 이 수치는 paper가 인용한 secondary source이며, 본 논문이 직접 수행한 survey가 아니다.

### 4) 중단이 아니라 verifiable alignment

저자들의 결론은 alignment research를 멈추자는 것이 아니다. Alignment는 harmful use와 autonomous system risk를 줄이는 데 여전히 필요하다고 본다.

대신 다음 방향을 제안한다.

- Oversight, transparency, independent audit
- Standardized censorship and manipulation benchmark
- Model and provider pluralism
- User literacy and researcher responsibility
- Substantive impact statement

## 2-2. Design intuition

논문의 design intuition은 alignment를 model capability가 아니라 governance infrastructure로 보는 데 있다.

Alignment pipeline은 다음 질문에 답한다.

- 어떤 knowledge가 model에 들어가는가.
- 어떤 answer가 desirable한가.
- 어떤 request를 refusal하는가.
- 어떤 output을 runtime에서 차단하는가.
- User에게 어떤 policy explanation을 제공하는가.

이 질문은 technical optimization인 동시에 institutional choice다. Loss가 잘 수렴하고 jailbreak robustness가 높아도, policy legitimacy와 auditability는 자동으로 생기지 않는다.

이 논문의 가장 중요한 point는 safety와 freedom을 하나의 scalar로 trade-off하지 않는다는 점이다. 어떤 content를 차단할지 논쟁하는 것만으로는 부족하며, 누가 rule을 정하고, 변경을 공개하고, independent party가 검증하며, user가 alternative와 appeal path를 갖는지를 함께 설계해야 한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Paper type | Position paper |
| Main claim | Alignment method는 protection과 information control에 모두 쓰일 수 있는 dual-use technology |
| Threats | Censorship, manipulation |
| Main actors | State actors, foundation model providers |
| Control stack | Pre-training filtering, post-training alignment, inference-time control |
| Analysis axes | Access, compute, expertise, modification ease, modification depth |
| Proposed safeguards | Transparency, independent audit, benchmark, pluralism, awareness |
| Explicit non-goal | Alignment research 중단 주장 |
| Evidence | Prior research, policy text, survey, news report, documented case |

## 3-2. Module breakdown

### 1) Pre-training data filtering

Pre-training filtering은 base model이 어떤 information에 노출되는지 결정한다.

Common legitimate purpose는 다음과 같다.

- De-duplication
- Personally identifiable information removal
- Low-quality or boilerplate filtering
- Unsafe or adult content filtering
- Domain block list
- Model-based quality classifier

Dual-use risk는 unwanted fact, history, political viewpoint, minority language source를 corpus에서 선택적으로 제거할 수 있다는 점이다.

Pre-training intervention의 특징은 다음과 같다.

- Access: pre-training pipeline 필요
- Compute: very high
- Expertise: high
- Ease: moderate to difficult
- Depth: fundamental

한번 knowledge가 base model에 들어가지 않으면 post-training prompt만으로 복원하기 어렵다. 또한 filtered corpus가 public dataset으로 재배포되면 다른 model도 같은 omission을 inherit할 수 있다.

논문은 Chinese-language corpus와 state-approved data construction에 관한 media report와 prior research를 evidence로 인용한다. 이 사례들은 paper가 직접 audit한 결과가 아니므로, blog에서는 control mechanism을 설명하는 cited example로만 해석하는 것이 안전하다.

### 2) Post-training preference alignment

Post-training은 base model을 instruction-following assistant로 만들고, preferred answer와 refusal policy를 학습한다.

논문이 다루는 method는 다음을 포함한다.

- RLHF
- Preference modeling
- Reward model training
- Constitutional AI
- Guideline-based alignment
- Deliberative Alignment
- Supervised safety fine-tuning

Dual-use leverage는 preference data와 policy definition에 있다.

- 어떤 annotator를 선택하는가.
- 어떤 answer pair를 preferred로 label하는가.
- 어떤 worldview를 constitution에 넣는가.
- 어떤 topic을 unsafe로 정의하는가.
- 어떤 disagreement를 misinformation으로 처리하는가.

Post-training intervention의 특징은 다음과 같다.

- Access: model weights 필요
- Compute: moderate to high
- Expertise: moderate to high
- Ease: moderate
- Depth: persistent

Pre-training보다 싸고 빠르며, inference prompt보다 behavior에 깊게 남는다. Provider가 preference collection과 reward model을 통제하면 policy를 broad behavior pattern으로 internalize할 수 있다.

### 3) Inference-time control

Inference-time layer는 deployment에서 가장 쉽게 바꿀 수 있다.

- System prompt
- Hidden instruction
- Prompt wrapper
- Keyword filter
- Safety classifier
- Output moderation
- Response replacement
- Routing and model selection

특징은 다음과 같다.

- Access: runtime access
- Compute: negligible to moderate
- Expertise: low to moderate
- Ease: easy
- Depth: superficial

Weight를 바꾸지 않아도 즉시 policy를 적용하거나 rollback할 수 있다. 반면 user가 model card나 weight를 보더라도 actual deployment behavior를 알기 어렵게 만든다.

논문은 generation 뒤 critical answer가 refusal로 교체된 것으로 보이는 reported case를 inference-time filtering의 example로 인용한다. 실제 product audit에서는 streaming token, raw model output, moderation log가 없으면 intervention point를 확정하기 어렵다.

### 4) Control-stack comparison

논문의 Table 1을 정리하면 다음과 같다.

| Characteristic | Pre-training filtering | Post-training alignment | Inference-time control |
| --- | --- | --- | --- |
| Required access | Pre-training pipeline | Model weights | Runtime access |
| Compute | Very high | Moderate to high | Negligible to moderate |
| Expertise | High | Moderate to high | Low to moderate |
| Ease of modification | Moderate to difficult | Moderate | Easy |
| Depth | Fundamental | Persistent | Superficial |

이 table은 "가장 위험한 layer"를 하나 고르기 위한 것이 아니다. Actor resource와 intended persistence에 따라 선택 가능한 attack surface가 다르다는 뜻이다.

- Resource가 큰 provider는 세 layer를 모두 사용할 수 있다.
- State는 regulation을 통해 provider의 stack을 indirect하게 통제할 수 있다.
- Downstream deployer는 inference control부터 쉽게 바꿀 수 있다.
- User는 final behavior를 보지만 intervention provenance는 거의 보지 못할 수 있다.

### 5) Oversight and transparency

논문은 proprietary model의 alignment policy, data, method, internals가 충분히 공개되지 않아 scrutiny가 어렵다고 지적한다.

제안은 full public release 하나로 단순화되지 않는다.

- Policy objective disclosure
- Training and post-training method disclosure
- Independent auditor access
- Versioned behavior change log
- Evaluation report
- User-facing notice
- Appeal or correction channel

Sensitive safety detail을 모두 공개하면 jailbreak나 attack을 돕는 문제가 생길 수 있다. 따라서 independent auditor와 public summary를 구분하는 tiered transparency가 현실적이다.

### 6) Standardized black-box benchmark

Provider가 cooperation하지 않거나 state가 malicious actor인 경우 internal audit만으로 부족하다. 논문은 black-box model을 대상으로 information suppression과 political bias를 측정하는 standardized benchmark를 제안한다.

Required properties는 다음과 같다.

1. Global coverage
   - 특정 국가나 language에 한정하지 않는다.

2. Multiple ideological axes
   - Left-right뿐 아니라 authoritarian tendency, state criticism, minority right, historical revision을 포함한다.

3. Dynamic maintenance
   - Current event와 policy change를 반영한다.

4. Multiple behavior types
   - Refusal, omission, factual distortion, asymmetric caveat, source selection, framing을 구분한다.

5. Independent verification
   - Provider self-report만으로 score를 만들지 않는다.

6. Context-aware legitimacy
   - Legitimate safety refusal와 political suppression을 구분하는 annotation rule이 필요하다.

### 7) Pluralism and competition

Transparency가 있어도 alternative가 없으면 user choice는 제한된다. 논문은 single neutral model을 찾기보다 diverse model ecosystem을 유지해야 한다고 주장한다.

이를 "Neutrality Through Diversity"라는 표현으로 설명한다.

- 서로 다른 provider
- Open and closed model
- Multiple country and legal jurisdiction
- Different policy profile
- User-selectable behavior

Pluralism은 하나의 model이 완전히 neutral하다는 가정을 피한다. 다만 model 수가 많다고 자동으로 meaningful diversity가 생기는 것은 아니다. 같은 base model, same filtered corpus, same cloud distribution channel을 공유하면 apparent diversity만 늘 수 있다.

### 8) Awareness and research practice

User는 LLM answer가 neutral information retrieval이 아니라 policy-conditioned output일 수 있음을 알아야 한다. Model 선택 시 privacy, safety, political bias, refusal profile을 비교할 literacy가 필요하다.

연구자에게는 perfunctory ethics statement를 넘어 자신의 method가 누구에게 어떤 control capability를 제공하는지 구체적으로 쓰라고 요구한다.

- 어떤 access가 필요한가.
- Modification이 얼마나 persistent한가.
- Detection과 rollback이 가능한가.
- Malicious operator가 어떤 objective를 넣을 수 있는가.
- Affected user에게 visibility와 recourse가 있는가.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 새로운 training dataset을 구축하지 않는다. Evidence base는 heterogeneous하다.

- Alignment and AI safety research
- Bias and representation literature
- Censorship benchmark research
- AI governance and regulation
- Human rights framework
- Industry and market analysis
- User adoption survey
- News report and documented incident

이 구성은 broad argument를 만드는 데 유리하지만 evidence strength가 균일하지 않다.

- Peer-reviewed technical result
- Policy document
- Survey
- Journalistic observation
- Public statement

을 같은 수준의 causal evidence로 취급하면 안 된다.

## 4-2. Training strategy

Model training recipe는 없다. 대신 paper의 analytical recipe를 정리하면 다음과 같다.

1. Alignment lifecycle을 pre-training, post-training, inference로 나눈다.
2. 각 layer의 access, compute, expertise, ease, depth를 비교한다.
3. Censorship과 manipulation으로 이어질 수 있는 control leverage를 식별한다.
4. Reported case와 prior literature를 layer별로 mapping한다.
5. Adoption, concentration, political context가 impact를 확대하는 경로를 설명한다.
6. Transparency, evaluation, pluralism, awareness를 mitigation으로 제안한다.
7. Stop-alignment view와 risk-exaggeration view를 alternative perspective로 검토한다.

## 4-3. Engineering notes

이 position을 operational evaluation으로 바꾸려면 benchmark design이 필요하다.

### 1) Paired prompt construction

같은 information need를 safety context와 political context로 paired하게 설계한다.

- Legitimate safety refusal
- Benign factual inquiry
- Criticism of powerful actor
- Minority or opposition viewpoint
- Historical fact verification
- Commercial conflict of interest

단순 refusal rate가 아니라 condition 간 asymmetry를 본다.

### 2) Behavior taxonomy

Output을 최소 다음 category로 나눌 수 있다.

| Category | Description |
| --- | --- |
| Direct refusal | Answer 자체를 거절 |
| Selective omission | 핵심 fact 또는 viewpoint를 생략 |
| Factual distortion | Verifiable claim을 다르게 서술 |
| Asymmetric uncertainty | 한쪽 claim에만 과도한 caveat 적용 |
| Source steering | 특정 source만 반복적으로 우선 |
| Tone manipulation | 같은 fact를 actor별로 다른 framing으로 표현 |
| Runtime replacement | Generation 뒤 answer가 filter output으로 교체된 징후 |

### 3) Cross-layer audit

가능하다면 같은 base model에 대해 세 layer를 분리한다.

- Raw base checkpoint
- Post-trained checkpoint
- Deployment API

차이를 비교하면 knowledge absence, learned preference, runtime guardrail을 더 잘 구분할 수 있다.

### 4) Temporal regression

Inference policy는 server-side에서 빠르게 바뀔 수 있다. 동일 prompt suite를 version별로 반복해야 한다.

- Model version
- Date and region
- System policy version
- Safety classifier version
- Refusal template
- Source citation behavior

을 함께 기록해야 한다.

### 5) Multilingual parity

같은 question을 multiple language로 평가해야 한다. Pre-training corpus와 moderation classifier가 language별로 다르기 때문에 answer availability와 framing도 달라질 수 있다.

### 6) Human review and recourse

Political and rights-related benchmark는 annotation disagreement가 크다. Single label보다 rationale, jurisdiction, harm model, appeal process를 남기는 것이 중요하다.

# 5. Evaluation

## 5-1. Main results

이 paper에는 model accuracy나 benchmark score가 없다. Evaluation 대상은 position의 argument quality다.

### 1) Technical mapping은 설득력이 있다

Pre-training, post-training, inference-time layer가 서로 다른 access와 persistence를 갖는다는 mapping은 실제 LLM system design과 잘 맞는다.

특히 다음 구분이 유용하다.

- Pre-training omission은 knowledge foundation을 바꾼다.
- Post-training은 preferred behavior를 persistent하게 만든다.
- Inference control은 가장 쉽게 변경되며 user에게 가장 불투명할 수 있다.

### 2) Misuse boundary를 제시한다

Position paper가 흔히 빠지는 문제는 모든 disagreement를 harm으로 부르는 것이다. 이 논문은 rights violation과 extreme power asymmetry라는 minimum boundary를 제시한다.

완전한 operational definition은 아니지만, safety refusal와 censorship을 구분하기 위한 출발점은 제공한다.

### 3) Mitigation이 anti-alignment로 흐르지 않는다

저자들은 alignment 중단을 명시적으로 거부한다. Alignment가 harmful instruction, criminal use, autonomous system risk를 줄이는 데 필수라는 premise를 유지한다.

따라서 제안은 safeguard removal이 아니라 safeguard governance다.

### 4) Alternative view를 포함한다

논문은 두 반론을 다룬다.

1. Dual-use risk가 크므로 alignment를 멈춰야 한다.
2. Censorship에는 다른 수단도 있고 law가 있으므로 risk가 과장되었다.

첫 번째에 대해서는 safety와 autonomous harm 때문에 alignment가 필요하다고 답한다. 두 번째에 대해서는 strong rule of law, enforcement capacity, foreign provider, private actor 문제 때문에 legal safeguard만으로 충분하지 않다고 답한다.

### 5) Evidence는 broad하지만 uneven하다

Paper는 technical paper, governance research, survey, news report를 폭넓게 연결한다. 이는 문제의 scope를 보여주지만 causal proof의 강도는 낮춘다.

특히 specific model behavior 사례는 다음 질문이 남는다.

- Raw generation과 post-filter output을 구분했는가.
- Policy change 전후 controlled test가 있는가.
- Region, language, account setting을 통제했는가.
- Provider explanation과 independent replication이 있는가.

따라서 case를 definitive proof보다 plausibility evidence로 보는 것이 적절하다.

## 5-2. What really matters in the experiments

Position paper를 평가할 때 중요한 지점은 다음과 같다.

### 1) Alignment robustness의 ownership

Robustness는 누구를 상대로 한 robustness인가.

- Malicious user의 jailbreak를 막는가.
- External attacker의 fine-tuning을 막는가.
- Policy owner의 restriction을 더 강하게 만드는가.
- Legitimate researcher의 audit도 어렵게 만드는가.

Same technical property가 actor에 따라 benefit과 risk를 동시에 만들 수 있다.

### 2) Refusal과 false answer를 분리해야 한다

Censorship benchmark가 refusal만 보면 manipulation을 놓친다. Model이 answer를 거절하지 않고 one-sided or false narrative를 생성하면 user는 intervention을 더 알아차리기 어렵다.

평가는 availability, correctness, calibration, source coverage, framing을 함께 봐야 한다.

### 3) Benchmark 자체도 value-laden하다

어떤 prompt를 controversial로 고르고, 어떤 answer를 balanced하다고 label할지에도 worldview가 들어간다. Standardized benchmark가 provider bias를 측정하면서 benchmark author의 bias를 새로운 standard로 만들 수 있다.

따라서 benchmark는 다음을 공개해야 한다.

- Prompt source
- Annotation guideline
- Annotator distribution
- Disagreement rate
- Jurisdiction and language
- Gold evidence provenance
- Update and appeal process

### 4) Pluralism은 distribution layer까지 봐야 한다

Model artifact가 여러 개 있어도 access channel이 하나면 practical choice는 제한된다.

- Cloud provider concentration
- App store and search distribution
- API pricing
- Regulatory approval
- Compute access
- Open-weight availability

을 함께 봐야 한다.

### 5) Transparency와 security 사이 trade-off가 있다

Safety classifier rule, attack threshold, exact system prompt를 모두 공개하면 evasion이 쉬워질 수 있다. 반대로 아무것도 공개하지 않으면 misuse audit이 불가능하다.

현실적인 방향은 layered disclosure다.

- Public: policy objective, broad category, evaluation result, change log
- Accredited auditor: detailed data and rule access
- Internal secure review: attack-sensitive implementation

# 6. Limitations

1. Position paper이며 새로운 causal experiment가 없다.
   - Alignment method가 실제 large-scale opinion manipulation을 일으켰다는 controlled evidence를 제시하지 않는다.
   - Main contribution은 threat framing과 control-stack mapping이다.

2. Evidence source의 강도가 균일하지 않다.
   - Peer-reviewed study와 news report, public anecdote가 함께 사용된다.
   - Specific case의 mechanism attribution은 추가 verification이 필요하다.

3. Misuse definition은 normative judgment를 완전히 제거하지 못한다.
   - Human right language도 concrete content moderation case에서 해석 conflict가 생긴다.
   - Hate speech, public safety, national security, political speech boundary는 jurisdiction마다 다르다.

4. Censorship과 safety refusal의 operational separation이 부족하다.
   - Rights and power criteria는 useful하지만 benchmark label로 바로 쓰기 어렵다.
   - Dual-use prompt가 legitimate harm prevention인지 political suppression인지 case-by-case review가 필요하다.

5. Manipulation measurement가 특히 어렵다.
   - Framing, omission, source selection은 single-answer correctness로 잡히지 않는다.
   - Longitudinal user belief change까지 보려면 expensive human study가 필요하다.

6. Transparency가 항상 안전한 것은 아니다.
   - Detailed guardrail disclosure는 jailbreak와 evasion을 돕는다.
   - Auditor access를 누가 인증하고 통제할지 governance problem이 남는다.

7. Pluralism이 neutrality를 보장하지 않는다.
   - 여러 model이 같은 data, provider, cloud, regulatory pressure를 공유할 수 있다.
   - Low-quality or extremist model의 증가를 beneficial diversity로 볼 수도 없다.

8. Competition policy와 technical recommendation의 연결이 약하다.
   - Open-weight release, interoperability, portability, antitrust, public infrastructure 중 무엇이 가장 효과적인지 비교하지 않는다.

9. Proposed benchmark도 political capture에 취약하다.
   - Topic selection과 gold answer가 특정 ideology에 편향될 수 있다.
   - Dynamic update 과정이 투명하지 않으면 benchmark가 새로운 control tool이 될 수 있다.

10. Risk magnitude가 정량화되지 않는다.
    - Adoption과 concentration이 risk를 키운다는 argument는 plausible하지만, probability와 expected harm estimate는 없다.
    - 다른 censorship channel과 비교한 marginal risk도 측정하지 않는다.

11. Model provider와 state actor 외의 ecosystem actor가 덜 다뤄진다.
    - Cloud provider, app distributor, enterprise deployer, data vendor, benchmark owner도 behavior와 access를 control할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

LLM system을 실제로 운영하면 alignment는 model weight 하나가 아니라 layered policy system이 된다.

- Training data filter
- SFT or preference data
- Reward and refusal objective
- System prompt
- Retrieval source policy
- Output validator
- Human escalation

이 중 하나만 바뀌어도 user가 받는 answer가 달라진다. 하지만 model card는 흔히 base model과 benchmark score만 설명하고, deployment policy의 version history는 충분히 보여주지 않는다.

이 논문의 실무적 의미는 alignment evaluation을 safety pass rate 하나로 끝내지 말라는 데 있다. Harmful request refusal이 높아도 benign information access, political symmetry, correction channel이 함께 건강한지 봐야 한다.

특히 enterprise RAG나 internal document QA에서는 다른 형태의 information control risk가 있다.

- Retrieval source가 특정 document를 제외한다.
- Access-control bug가 필요한 evidence를 숨긴다.
- Policy classifier가 compliance-sensitive topic을 과도하게 refusal한다.
- Citation이 selective evidence만 노출한다.
- Administrator가 user에게 보이지 않게 source priority를 바꾼다.

따라서 alignment audit은 generation model뿐 아니라 retrieval, ranking, policy, citation layer까지 포함해야 한다.

## 7-2. Reuse potential

### 1) Three-layer audit matrix

각 product release마다 control point를 inventory할 수 있다.

| Layer | Questions |
| --- | --- |
| Pre-training | 어떤 source와 language가 제거되었는가 |
| Post-training | 어떤 preference와 refusal policy를 학습했는가 |
| Inference | 어떤 system prompt, classifier, filter가 적용되는가 |
| Retrieval | 어떤 corpus와 rank policy가 answer evidence를 결정하는가 |
| Governance | 누가 policy를 변경하고 누가 승인하는가 |

### 2) Legitimate-safety vs suppression paired test

같은 topic에서 harmful operational request와 benign informational request를 paired한다.

- Harmful instruction은 refusal되어야 한다.
- Historical, scientific, journalistic context는 evidence와 함께 answer되어야 한다.
- Model은 refusal reason과 allowed scope를 설명해야 한다.

이 방식은 blanket refusal을 줄이는 데 유용하다.

### 3) Refusal and manipulation dashboard

단일 refusal rate 대신 다음을 version별로 추적한다.

- Refusal rate
- Unsupported refusal rate
- Factual omission rate
- Cross-language inconsistency
- Viewpoint asymmetry
- Citation diversity
- Post-generation replacement rate
- User appeal success rate

### 4) Policy change log

Runtime behavior는 weight release 없이 바뀔 수 있다. Model version과 policy version을 분리해 기록해야 한다.

- Changed category
- Reason for change
- Expected user impact
- Regression benchmark result
- Rollback condition
- Auditor sign-off

### 5) Independent red-team split

Provider 내부 safety team만 평가하면 objective blindness가 생길 수 있다. External civil society, domain expert, human-rights researcher, regional and multilingual reviewer가 참여하는 split audit가 필요하다.

### 6) Benchmark governance

Censorship benchmark 자체가 capture되지 않도록 다음 mechanism이 필요하다.

- Multiple stakeholder board
- Public prompt provenance
- Versioned annotation rule
- Disagreement preservation
- Challenge and appeal process
- Regional sub-benchmark
- Periodic topic rotation

## 7-3. Follow-up papers

- Constitutional AI: Harmlessness from AI Feedback
- Collective Constitutional AI: Aligning a Language Model with Public Input
- Whose Opinions Do Language Models Reflect?
- The Impact of Online Censorship on LLMs
- Generative Monoculture in Large Language Models
- DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models
- Position: Pluralistic Alignment Should Be a Core Research Goal
- Auditing Large Language Models: A Three-Layered Approach

# 8. Summary

- 이 논문은 alignment method를 safety mechanism인 동시에 censorship과 manipulation에 재사용될 수 있는 dual-use control technology로 본다.
- Threat를 information suppression과 covert perception distortion으로 나누고, misuse를 rights violation과 extreme power asymmetry에 연결한다.
- Pre-training filtering, post-training alignment, inference-time control은 access, compute, modification depth가 서로 다른 control surface다.
- State actor와 foundation model provider가 large-scale behavior를 결정할 수 있는 주요 actor로 분석된다.
- 저자들은 alignment 중단이 아니라 transparency, independent audit, standardized benchmark, pluralism, awareness를 제안한다.
- 설득력 있는 threat framing이지만 position paper이므로 causal evidence, operational benchmark, risk magnitude는 future work로 남는다.
