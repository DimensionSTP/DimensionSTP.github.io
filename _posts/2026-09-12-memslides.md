---
layout: single
title: "MemSlides: A Hierarchical Memory Driven Agent Framework for Personalized Slide Generation with Multi-turn Local Revision Review"
categories: Study-concept
tag: [MemSlides, AgentMemory, SlideGeneration, Personalization, MultiTurnRevision]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.17162)

[Code link](https://github.com/huohua325/Memslides)

[Project page](https://memslides.github.io/)

MemSlides는 presentation generation 논문이지만, 핵심은 slide rendering 자체보다 **personalized authoring agent에서 memory를 어떻게 나눠야 하는가**에 있다. 좋은 slide deck은 current prompt만으로 만들어지지 않는다. 사용자는 발표 목적, 선호하는 layout, content density, evidence style, visual convention, revision habit을 반복적으로 가진다. 그리고 이 preference는 첫 prompt에서 전부 드러나는 것이 아니라, revision 과정에서 점진적으로 드러난다.

기존 slide agent는 complete deck을 만들고, feedback이 들어오면 큰 context에 다시 넣어 수정하거나 많은 부분을 regenerate하는 경향이 있다. 이 방식은 multi-turn authoring에서 문제가 크다. "이 slide title만 파란색으로 바꿔줘" 같은 local request가 deck-level regeneration을 유발하면, 이미 맞춰진 content나 style이 drift할 수 있다. 또 이전 revision에서 드러난 temporary preference를 나중 revision에 유지하기 어렵다.

MemSlides는 이를 hierarchical memory와 scoped local revision으로 푼다. Memory는 long-term memory와 working memory로 나뉜다. Long-term memory는 다시 user profile memory와 tool memory로 나뉜다. User profile memory는 intent-conditioned recurring preference를 저장한다. Working memory는 current session constraint와 temporary preference를 revision round 동안 유지한다. Tool memory는 localized editing을 더 reliably 수행하기 위한 reusable execution experience를 저장한다.

> 한 줄 요약: MemSlides는 personalized slide generation을 one-shot prompt-to-deck 문제가 아니라 multi-turn authoring process로 보고, user profile memory, working memory, tool memory를 분리한 hierarchical memory와 scoped slide-local revision을 결합해 persistent personalization과 reliable local editing을 지원하는 agent framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Agent memory를 slide generation product의 actual bottleneck에 맞춰 분해한다.
- Long-term personalization과 current-session preference carryover를 같은 buffer에 넣지 않는다.
- Local revision을 full-deck regeneration이 아니라 smallest affected region update로 다룬다.
- Plan-Act-Guard protocol로 revision scope, selector, patch, verification을 명시화한다.
- Persona-alignment, DeepPresenter-style general quality, diagnostic matched-pair modify evaluation을 함께 본다.
- Presentation generation agent에서 "memory quality"를 실제 edit reliability와 user preference alignment로 연결한다.

이 글에서는 MemSlides를 "PPT agent paper"보다, **personalized document authoring에서 persistent profile, session state, execution memory를 어떻게 분리할지 보여주는 agent memory paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Personalized presentation generation은 source material $x$, user profile memory $P_u$, optional template $\tau$에서 initial deck $S_0$를 만든다.

$$
S_0 = G_{\mathrm{init}}(x,P_u,\tau)
$$

그 뒤 user feedback $f_t$가 들어오면 session state $z_t$를 update하고 current deck을 수정한다.

$$
z_t = U(z_{t-1},f_t;S_{t-1})
$$

$$
S_t = G_{\mathrm{edit}}(S_{t-1},x,P_u,\tau,z_t)
$$

이 formulation이 중요한 이유는 slide generation이 one-shot task가 아니라는 점이다. 실제 authoring은 다음을 포함한다.

- round-0 personalized draft generation
- multi-turn revision
- temporary preference carryover
- local element update
- deck-level rule update
- job 종료 후 user profile consolidation

따라서 memory system은 preference lifetime을 구분해야 한다.

## 1-2. Why previous approaches are insufficient

### 1) Prompt-only personalization

Prompt에 "나는 financial manager다" 같은 persona를 넣을 수는 있다. 하지만 recurring preferences over evidence type, slide density, layout rhythm, closing convention은 prompt만으로 안정적으로 유지되기 어렵다.

### 2) Reference slide 또는 template conditioning

SlideTailor류 approach는 task-time reference slide나 template으로 personalization을 지원한다. 하지만 이는 accumulated user profile을 학습하는 것이 아니라 provided example/template에 의존한다.

### 3) Full-deck regeneration for revision

Small edit request마다 whole deck을 다시 읽거나 다시 쓰면, context pressure와 unintended drift가 생긴다. Local change는 local하게 끝나야 한다.

### 4) Single homogeneous memory buffer

Long-term user preference, temporary session rule, tool execution experience를 같은 memory buffer에 넣으면 conflict resolution과 routing이 어렵다. MemSlides는 memory lifetime과 use를 나눈다.

# 2. Core Idea

## 2-1. Main contribution

MemSlides의 기여는 세 가지다.

1. **Hierarchical memory**
   - Long-term memory와 working memory로 나눈다.
   - Long-term memory는 user profile memory와 tool memory를 포함한다.
   - Working memory는 active session constraint를 유지한다.

2. **Scoped slide-local revision**
   - Revision request를 영향을 받는 가장 작은 slide region으로 project한다.
   - Local layout snapshot, selector, exposed style rule을 읽는다.
   - Entire deck을 regenerate하지 않고 scoped patch를 쓴다.

3. **Plan-Act-Guard modify execution**
   - Plan은 explicit execution contract를 만든다.
   - Act는 minimal effective edit을 적용한다.
   - Guard는 coverage를 검증하고 premature finalize를 막는다.

## 2-2. Design intuition

MemSlides의 memory hierarchy는 lifetime과 function에 따른 분리다.

| Memory | Lifetime | Function |
| --- | --- | --- |
| User profile memory | Across jobs | 안정적인 user preference와 intent-conditioned style |
| Working memory | Current session | Temporary constraint와 carryover feedback |
| Tool memory | Across operations/jobs | Local edit을 위한 reusable execution experience |

이 분리는 중요하다. 사용자는 business deck 전반에서 dense evidence slide를 선호할 수 있지만, 현재 deck에서는 명시적으로 "minimal text"를 요구할 수 있다. 이 job에서는 current session request가 profile보다 우선해야 한다. Working memory는 long-term profile을 즉시 오염시키지 않고 active override를 처리한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Multi-turn local revision을 포함한 personalized slide generation |
| Framework | MemSlides |
| Memory hierarchy | Long-term memory와 working memory |
| Long-term modules | User profile memory와 tool memory |
| Revision strategy | Scoped slide-local revision |
| Executor | Plan-Act-Guard modify execution |
| Evaluation | Persona alignment, deck quality, localized revision diagnostics |
| Main baselines | DeepPresenter와 SlideTailor |
| Main claim | Profile, session, tool memory를 분리하면 personalization과 local editing이 개선됨 |

## 3-2. User profile memory

User profile memory는 user와 intent별 recurring preference를 저장한다. 논문은 preference를 theme, content, visual, layout, template, general 같은 dimension으로 정리한다.

Job 시작 시 MemSlides는 intent에 맞는 profile bucket을 선택하고 current request에서 constraint를 추출한다.

$$
\tilde{P}_u = \mathcal{S}(P_u,i_0)
$$

$$
C_0 = \mathcal{E}(q_0)
$$

$$
A_0 = \mathcal{R}(\tilde{P}_u,C_0)
$$

Current request와 명시적으로 충돌하는 항목은 현재 deck에서 stored profile item보다 우선한다. Compatible preference는 working memory 안에서 함께 유지된다.

## 3-3. Working memory

Working memory는 active temporary preference item을 저장한다.

$$
\mathcal{M}^{pref}_t=(P_u,A_t)
$$

Revision 중 active temporary memory는 feedback에 따라 evolve한다.

$$
A_t = \mathcal{U}(A_{t-1},r_t)
$$

새로 드러난 preference를 append하고, 실제 conflict는 supersede하며, conflict가 없는 item은 이후 round를 위해 active하게 유지한다.

이 module이 multi-turn feedback이 잊히는 것을 막는다.

## 3-4. Tool memory

Tool memory는 reusable execution experience를 저장한다. 이는 user style에 관한 것이 아니라 slide를 안정적으로 edit하는 방법에 관한 것이다.

예시는 다음과 같다.

- 어떤 selector pattern이 작동했는가
- body를 건드리지 않고 title style을 patch하는 방법
- stale snapshot을 처리하는 방법
- full rewrite를 피한 tool path

Tool memory는 persona alignment가 아니라 localized revision reliability를 지원한다.

## 3-5. Scoped local revision

MemSlides는 필요한 경우가 아니면 full-deck rewrite를 피한다. Revision request는 explicit execution contract로 변환된다.

Contract에는 다음이 포함될 수 있다.

- inferred scope
- target slide paths
- active rule identifiers
- selector hints
- coverage requirement

System은 이후 local repair surface만 읽고, explicit selector 또는 style rule에 scope가 제한된 patch를 쓴다.

## 3-6. Plan-Act-Guard

| Stage | Role |
| --- | --- |
| Plan | Execution contract를 만들고 target scope를 결정 |
| Act | Slide structure/tool을 사용해 minimal effective edit 적용 |
| Guard | Coverage를 검증하고 stale snapshot을 감지하며 premature finalize를 차단 |

이는 중요한 engineering idea다. Revision은 "LLM이 끝났다고 판단하는 것"이 아니다. Completion은 검증되는 state다.

# 4. Training / Data / Recipe

## 4-1. Data와 resources

논문은 code, website, project page, video link를 제공한다. Evaluation을 위해 multi-persona, multi-intent user profile bank를 구성한다.

정확한 profile bank construction detail은 appendix에 있으며, reproduction 전에 확인해야 한다.

## 4-2. Evaluation protocol

Experiment는 다음을 평가한다.

| Evaluation | Purpose |
| --- | --- |
| Persona-alignment judgments | User profile memory가 personalization을 개선하는지 평가 |
| DeepPresenter-style general quality | Personalization이 overall deck quality를 해치지 않는지 평가 |
| Diagnostic matched-pair modify evaluation | Tool memory가 localized edit을 개선하는지 평가 |
| Qualitative carryover cases | Working memory가 session constraint를 보존하는지 확인 |

## 4-3. Engineering notes

1. **Preference lifetime을 분리한다**
   - Long-term preference와 temporary current-deck constraint를 섞지 않는다.

2. **Local edit은 local하게 유지한다**
   - Scoped patch, selector, coverage check를 사용한다.

3. **Completion을 guard한다**
   - Target coverage가 검증되기 전에 agent가 finalize하지 못하게 한다.

4. **Tool experience는 user preference와 분리해 저장한다**
   - Editing skill은 user style과 같지 않다.

5. **Long-term memory consolidation은 신중하게 한다**
   - 모든 temporary preference를 persistent profile에 쓰면 안 된다.

# 5. Evaluation

## 5-1. Personalization

논문은 user profile memory가 여러 dimension에서 round-0 persona alignment를 개선한다고 보고한다. GLM-5와 Gemini 3.1 Pro judge에서는 MemSlides가 두 baseline 대비 모든 column에서 이긴다. GPT-5 기준으로도 SlideTailor보다 Content, Structure, Specificity에서 앞서고, DeepPresenter보다 Content, Visual, Specificity에서 앞선다.

Model family 전체 평균에서 MemSlides는 DeepPresenter보다 다음만큼 개선된다.

| Dimension | Improvement |
| --- | ---: |
| Content | +1.37 |
| Structure | +0.53 |
| Visual | +1.66 |
| Specificity | +1.19 |

SlideTailor와 비교하면 다음과 같다.

| Dimension | Improvement |
| --- | ---: |
| Content | +2.73 |
| Structure | +2.95 |
| Visual | +2.79 |
| Specificity | +3.08 |

이는 단순히 더 예쁜 slide가 아니라 persona-alignment gain이다.

## 5-2. General deck quality

논문은 같은 generated deck에 대해 DeepPresenter-style quality check도 수행한다. 결과는 MemSlides가 competitive한 PPT generation quality를 유지하면서 personalization을 개선함을 보여준다. Personalization이 broken deck quality를 대가로 얻어져서는 안 되기 때문에 이 구분이 중요하다.

## 5-3. Localized revision

Tool-memory injection은 diagnostic matched-pair setting에서 modify behavior를 개선한다.

보고된 값은 다음과 같다.

| Metric | Without/low tool memory | With tool memory |
| --- | ---: | ---: |
| Closed-Loop Completion | 0.815 | 0.963 |
| Strict Verify | 0.310 | 0.534 |
| Time to First Correct Edit | 609.5s | 242.5s |
| Core Tool Time Ratio | 1.0x baseline | 0.327x |

Pair-level count는 모든 pair에서 이기지는 않음을 보여준다. 이는 좋은 reporting이다. 논문의 claim은 universal dominance가 아니라 overall reliability와 search efficiency 개선이다.

## 5-4. Working memory evidence

논문은 working memory carryover에 대한 qualitative evidence를 제공한다. 이전 feedback에서 나온 active temporary preference가 같은 deck의 이후 local edit에 영향을 줄 수 있다. 또한 반복된 local feedback이 reusable profile preference가 되는 cross-job profile consolidation case도 보여준다.

이는 별도의 quantitative metric이 아니라 qualitative evidence다.

## 5-5. What really matters in the experiments

### 1) Persona gain은 planning-level이다

논문은 Structure가 template retrieval accuracy를 제외하고 Specificity가 distractor persona를 사용하기 때문에 gain이 단순한 template matching만은 아니라고 주장한다.

### 2) Locality 자체가 metric이다

System은 requested edit을 만족하면서도 non-target region을 건드릴 수 있다. MemSlides는 local revision을 final visual success뿐 아니라 scope control로도 평가해야 한다고 주장한다.

### 3) Tool memory는 process memory다

Tool memory는 user가 무엇을 좋아하는지가 아니라 agent가 어떻게 edit하는지를 개선한다. 이는 유용한 분리다.

### 4) Working memory는 아직 정량 평가가 부족하다

Qualitative carryover는 case evidence로는 설득력이 있지만, 더 체계적인 multi-turn metric이 있으면 claim이 강화될 것이다.

# 6. Limitations

1. **Evaluation은 일부 judge-based다**
   - Persona-alignment와 deck quality는 model judge에 의존한다.

2. **Working memory evidence는 qualitative하다**
   - 더 quantitative한 delayed-carryover metric이 있으면 도움이 된다.

3. **Slide generation stack dependency가 있다**
   - 결과는 underlying slide rendering 및 editing tool에 의존할 수 있다.

4. **Profile consolidation risk가 있다**
   - Temporary preference를 long-term memory로 잘못 승격하면 future job에 해가 될 수 있다.

5. **Privacy 문제가 있다**
   - User profile memory는 recurring preference와 민감할 수 있는 work style을 저장한다.

6. **Template과 design diversity가 제한적일 수 있다**
   - User profile bank가 모든 real-world presentation style을 포괄하지 못할 수 있다.

7. **Latency와 tool cost가 있다**
   - Plan-Act-Guard와 verification step은 process cost를 추가한다.

8. **Conflict policy가 어렵다**
   - Current request, template, profile, working memory가 충돌할 수 있다.

9. **Full product study가 없다**
   - 실제 team을 대상으로 한 longitudinal human use가 필요하다.

10. **Local edit scope가 global intent를 놓칠 수 있다**
    - Local하게 들리는 일부 user request가 global consistency를 함의할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

MemSlides의 핵심은 "memory를 붙인 slide agent"가 아니다. 더 중요한 점은 **authoring agent memory를 lifetime과 function으로 분리했다는 것**이다.

이는 slide를 넘어 재사용 가능하다. 모든 creative/document agent에는 다음이 필요하다.

- long-term profile
- current-session working state
- tool execution memory
- scoped revision protocol

하나의 dialogue buffer만으로는 충분하지 않다.

## 7-2. Reuse potential

### Document editing agents

Local revision과 Plan-Act-Guard는 report, blog post, paper, design document에도 적용할 수 있다.

### Personalized assistants

User profile memory는 모든 prompt에 맹목적으로 주입할 것이 아니라 intent-conditioned되고 routed되어야 한다.

### Tool-use agents

Tool memory는 execution experience를 user preference와 분리해 저장할 수 있다. 이는 agent가 안정적으로 edit하는 법을 배우는 데 도움이 된다.

### Multi-turn creative workflows

Working memory는 temporary preference가 revision turn 사이에서 사라지는 것을 막는다.

## 7-3. Production considerations

- Profile memory에 대한 user control을 추가해야 한다.
- 어떤 preference가 적용되었는지 보여줘야 한다.
- Session-only memory와 persistent memory를 명시적으로 분리해야 한다.
- Preference를 consolidate하기 전에 confirmation을 요구해야 한다.
- 모든 local patch와 touched element를 log로 남겨야 한다.
- Non-target drift를 first-class metric으로 추적해야 한다.
- User profile에는 privacy-preserving storage를 사용해야 한다.

## 7-4. Follow-up papers

- PPTAgent
- DeepPresenter
- SlideTailor
- AutoPresent
- MemGPT / Letta
- MemoryBank
- Mem0
- A-MEM
- Agent-Native Memory System papers

# 8. Summary

- MemSlides는 personalized slide generation을 multi-turn authoring으로 다룬다.
- User profile memory, working memory, tool memory를 분리한다.
- Scoped local revision은 작은 edit이 full-deck regeneration으로 번지는 것을 막는다.
- Plan-Act-Guard는 edit scope와 completion을 명시적으로 만든다.
- 핵심 insight는 personalization에 memory lifetime separation과 locality-aware revision이 필요하다는 점이다.
