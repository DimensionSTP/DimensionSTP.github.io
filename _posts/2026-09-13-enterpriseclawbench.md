---
layout: single
title: "EnterpriseClawBench: Benchmarking Agents from Real Workplace Sessions Review"
categories: Study-concept
tag: [EnterpriseClawBench, AgentBenchmark, WorkplaceAgent, Evaluation, CodingAgent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.23654)

[Code link](https://github.com/FrontisAI/EnterpriseClawBench)

[Leaderboard](https://frontisai.github.io/EnterpriseClawBench/)

EnterpriseClawBench는 "enterprise agent benchmark"라는 이름보다 훨씬 중요한 문제를 다룬다. 이 논문의 핵심은 **real workplace session을 benchmark로 바꾸려면 무엇을 보존하고, 무엇을 비공개로 남기고, 어떤 evaluation protocol을 공개해야 하는가**다.

Enterprise agent는 단순 coding benchmark와 다르게 persistent workspace 안에서 동작한다. Agent는 heterogeneous files를 읽고, spreadsheet를 수정하고, 문서를 만들고, web page를 생성하고, artifact를 제출한다. 사용자는 "이 파일들을 바탕으로 보고서 만들어줘"처럼 business artifact를 요구한다. 이런 task에서는 final text response보다 generated artifacts, visual quality, hard rule compliance, runtime cost, skill transfer가 중요하다.

하지만 real workplace session에는 proprietary data가 들어 있다. 따라서 benchmark를 그대로 공개할 수 없다. EnterpriseClawBench는 이 tension을 construction과 evaluation protocol 공개로 푼다. Full private benchmark data는 release하지 않지만, sanitized raw-session example, construction pipeline, local evaluation pipeline, sandbox run-directory protocol, reference run, aggregate leaderboard data를 공개한다. 이 점이 이 논문의 핵심이다. Reusable contribution은 data 자체가 아니라 **real sessions를 reproducible enterprise tasks로 바꾸는 protocol**이다.

> 한 줄 요약: EnterpriseClawBench는 proprietary real workplace agent sessions에서 852 reproducible tasks를 구성하고, fixtures, rewritten prompts, role classes, skill subclasses, hard rules, semantic rubrics, visual artifact judging, cost/runtime reporting을 포함한 enterprise agent evaluation protocol을 제안하는 benchmark methodology paper다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같다.

- Enterprise agent benchmark를 synthetic task가 아니라 real workplace sessions에서 출발해 만든다.
- 데이터 privacy 때문에 full benchmark를 공개하지 않지만, construction과 evaluation pipeline을 reusable artifact로 공개한다.
- Base model 단독이 아니라 harness-model combination을 평가 단위로 둔다.
- Generated artifacts, visual evidence, semantic rubrics, hard rules, cost, runtime을 함께 본다.
- Best configuration도 0.663에 그친다고 보고해 real workplace automation의 headroom을 보여준다.
- Agent benchmark에서 "score 하나"보다 protocol, artifact delivery, skill-transfer behavior가 중요하다는 점을 강조한다.

이 글에서는 EnterpriseClawBench를 "비공개 benchmark라 재현이 애매한 논문"보다, **real enterprise session을 benchmark로 바꾸는 construction/evaluation protocol paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Enterprise agent task는 prompt-response QA가 아니다. Agent는 workspace $W$ 안에서 files, tools, intermediate outputs, generated artifacts를 다룬다.

$$
y, A
=
\mathrm{Agent}(p,W,\mathcal{T})
$$

- $p$: user request
- $W$: uploaded file과 workspace state
- $\mathcal{T}$: available tools
- $y$: final text response
- $A$: generated artifacts

Evaluation은 response text만으로 끝나지 않는다.

$$
Score
=
F(y,A,r,h,c)
$$

- $r$: semantic rubric
- $h$: hard rules
- $c$: cost와 runtime metadata
- $A$: rendered artifact 또는 textual artifact

EnterpriseClawBench의 문제 설정은 다음이다.

> Proprietary real workplace sessions를 privacy-preserving 방식으로 reproducible task로 만들고, complete agent system의 artifact delivery quality를 평가할 수 있는가?

## 1-2. Why previous approaches are insufficient

### 1) Synthetic enterprise tasks

Synthetic task는 깔끔하고 공유하기 쉽지만, ambiguous instruction, mixed file format, implicit business constraint, visual deliverable, artifact dependency 같은 messy real workplace detail을 놓칠 수 있다.

### 2) Coding-only benchmarks

Coding benchmark는 repository edit이나 code execution을 측정한다. Enterprise task는 slide, spreadsheet, document, HTML, PDF, image, business report를 포함할 수 있다. 따라서 visual evaluation과 file-artifact evaluation이 필요하다.

### 3) Model-only leaderboards

Enterprise agent performance는 harness와 model에 함께 의존한다. File mounting, sandbox, tool availability, artifact export, history management가 중요하다. 따라서 leaderboard는 base model만이 아니라 harness-model system을 비교해야 한다.

### 4) Public data release assumption

많은 benchmark는 data를 공개할 수 있다고 가정한다. Enterprise data는 그렇지 않은 경우가 많다. 따라서 reusable contribution은 raw task가 아니라 protocol, pipeline, aggregate analysis가 되어야 한다.

# 2. Core Idea

## 2-1. Main contribution

EnterpriseClawBench의 기여는 다음과 같다.

1. **Real-session construction pipeline**
   - Proprietary workplace agent session에서 시작한다.
   - Recovered fixture와 rewritten prompt를 사용해 reproducible task를 만든다.
   - Role class, skill subclass, hard rule, semantic rubric을 부여한다.

2. **Evaluation protocol**
   - Task-specific run directory 안에 response와 artifact를 요구한다.
   - Text-only case와 visual artifact case를 적절한 judge로 route한다.
   - Visual artifact에는 screenshot 또는 converted evidence를 사용한다.

3. **Harness-model leaderboard**
   - Complete agent configuration을 비교한다.
   - Score-cost와 runtime trade-off를 보고한다.

4. **Public reproducibility shell**
   - Full private benchmark data는 공개하지 않는다.
   - Repository는 sanitized example, construction pipeline, local evaluation pipeline, sandbox run-directory protocol, small reference run, aggregate leaderboard data와 figure를 공개한다.

## 2-2. Design intuition

EnterpriseClawBench의 설계 직관은 다음과 같다.

```text
Enterprise agent evaluation은 answer text만이 아니라 artifact workflow를 보존해야 한다.
```

Workplace task는 artifact가 실제로 usable할 때에만 성공하는 경우가 많다.

예시는 다음과 같다.

- Spreadsheet formula와 formatting이 정확하다.
- Generated HTML이 answer를 시각적으로 전달한다.
- Document가 required section을 포함하고 forbidden claim을 담지 않는다.
- Chart가 readable하고 source data와 consistent하다.
- Final report가 role-specific constraint를 지킨다.

따라서 evaluation은 semantic rubric과 artifact evidence를 모두 포함해야 한다.

# 3. Architecture / Method

## 3-1. Overview

| 항목 | 내용 |
| --- | --- |
| Goal | Realistic workplace workflow에서 enterprise agent 평가 |
| Source | Proprietary real-world agent session |
| Public data | Full private benchmark data는 공개하지 않음 |
| Tasks | 852 reproducible tasks |
| Evaluation unit | Harness-model system |
| Task artifacts | File, document, spreadsheet, webpage, multimodal deliverable |
| Task annotation | Fixtures, rewritten prompts, role classes, skill subclasses, hard rules, semantic rubrics |
| Best reported score | 0.663, Codex with GPT-5.5 |
| Public contribution | Construction/evaluation protocol과 sanitized example |

## 3-2. Construction pipeline

Repository README는 construction pipeline을 다음처럼 설명한다.

1. Raw session을 filtering한다.
2. Fixture를 recover한다.
3. Prompt를 self-contained task로 rewrite한다.
4. Taxonomy를 부여한다.
5. Hard rule을 생성한다.
6. Semantic rubric을 생성한다.

이는 privacy constraint를 보존하면서 real session을 benchmark task로 바꾼다.

각 task row는 다음과 같은 field를 포함한다.

- `task_id`
- `agent_prompt`
- `evaluation.rule_based`
- `semantic_rubric`
- role class, subclass, input fixture path 같은 optional metadata

## 3-3. Agent run protocol

External user는 각 task가 run directory를 쓰기만 하면 어떤 agent harness든 사용할 수 있다.

필수 항목은 다음과 같다.

```text
runs/<run_name>/<task_id>/
  response.txt
  artifacts/
```

권장 또는 optional 항목은 다음과 같다.

```text
task.json
prompt.txt
history.jsonl
metadata.json
```

이 protocol은 단순하지만 중요하다. Agent execution과 judging을 분리한다. Benchmark는 하나의 sandbox implementation을 강제하지 않는다.

## 3-4. Judge routing

Evaluation routing은 엄격하다.

| Case | Judge |
| --- | --- |
| Text-only output | Text judge |
| Any visual artifact suffix | Visual judge |
| Visual evidence failure | main_score를 비워 두고 error 기록 |
| Visual case | Text judge로 조용히 fallback하지 않음 |

Visual suffix에는 HTML, PDF, PPTX, DOCX, XLSX, PNG, JPG, SVG 및 관련 format이 포함된다. Visual evidence는 rendering 또는 conversion으로 준비한다. HTML/SVG는 Playwright screenshot을 사용하고, office document는 PDF와 screenshot으로 변환하며, spreadsheet는 sheet HTML로 render하고, image는 직접 upload한다.

이는 핵심 design decision이다. Enterprise artifact는 visual하거나 layout-dependent한 경우가 많으므로 text judge fallback은 misleading score를 만들 수 있다.

## 3-5. Metrics와 reporting

논문은 evaluation이 하나의 score 이상을 보고해야 한다고 주장한다.

| Axis | 중요한 이유 |
| --- | --- |
| Main quality score | Rubric 아래에서의 overall task success |
| Artifact delivery | Usable file이 생성되었는가 |
| Visual quality | Rendered artifact가 requirement를 만족하는가 |
| Hard rules | 위반하면 안 되는 constraint |
| Cost | Deployment feasibility |
| Runtime | Workflow latency |
| Harness-model pairing | System-level capability |
| Skill transfer | Skill이 role/class 전반으로 generalize되는가 |

# 4. Training / Data / Recipe

## 4-1. Data

Private benchmark는 852개의 reproducible task를 포함한다. Task는 real workplace session에서 출발하지만 self-contained evaluation case로 처리된다.

Session에 internal enterprise content가 포함되므로 full benchmark data는 공개되지 않는다. Public repository에는 다음이 포함된다.

| Public component | 목적 |
| --- | --- |
| Sanitized raw-session example | Construction input을 보여줌 |
| Construction pipeline | Pipeline logic을 재현 |
| Local evaluation pipeline | External run을 judge |
| Sandbox run-directory protocol | Artifact를 표준화 |
| Small reference run | Evaluation smoke test |
| Aggregate leaderboard data | 보고된 pattern을 제시 |
| Public visualizations | Task/artifact distribution 분석 |

## 4-2. Evaluation recipe

Typical evaluation flow는 다음과 같다.

1. `eval_tasks.jsonl`을 만들거나 load한다.
2. 각 `agent_prompt`와 fixture로 agent를 실행한다.
3. `response.txt`와 `artifacts/`를 저장한다.
4. 필요하면 `task.json`, `prompt.txt`, `history.jsonl`, metadata를 저장한다.
5. Run directory를 validate한다.
6. Artifact에 따라 text judge 또는 visual judge를 실행한다.
7. Score, cost, runtime, error를 aggregate한다.

## 4-3. Engineering notes

1. **Harness-model pair를 평가한다**
   - Model만으로는 충분하지 않다.

2. **Actual prompt와 mounted path를 기록한다**
   - Runtime에 prompt를 rewrite하면 task condition이 바뀔 수 있다.

3. **Visual artifact를 조용히 text judge로 평가하지 않는다**
   - Visual evaluation 누락은 fallback이 아니라 error여야 한다.

4. **Hard rule과 semantic rubric을 분리한다**
   - 어떤 requirement는 binary constraint이고, 다른 requirement는 graded quality다.

5. **Audit trace를 유지한다**
   - `history.jsonl`과 metadata는 failure diagnosis에 도움이 된다.

6. **Cost와 runtime을 보고한다**
   - Enterprise deployment는 둘 모두에 의존한다.

# 5. Evaluation

## 5-1. Main result

arXiv abstract는 best configuration, 구체적으로 Codex with GPT-5.5가 0.663에 도달한다고 보고한다. 중요한 점은 특정 configuration이 이긴다는 사실이 아니다. 현재 best available configuration조차 realistic workplace session에서 상당한 headroom을 남긴다는 점이다.

## 5-2. Harness-model evaluation

Public README는 leaderboard가 base model만이 아니라 harness-model system을 비교한다고 설명한다. 이는 enterprise agent에 정확히 맞는 설정이다. Artifact, file handling, visual output, tool orchestration이 harness에 크게 의존하기 때문이다.

Model이 강하더라도 harness가 file을 mount하지 못하거나, output을 render하지 못하거나, history를 보존하지 못하거나, 올바른 artifact를 export하지 못하면 실패할 수 있다.

## 5-3. Artifact and visual evaluation

EnterpriseClawBench는 visual artifact를 visual judge로 route한다. 이는 PPTX, XLSX, PDF, HTML, image, document에서 중요하다. Evaluation module은 screenshot 또는 conversion을 통해 evidence를 준비하고, failure를 명시적으로 기록한다.

이는 쉬우면서도 위험한 shortcut을 피한다. 즉 spreadsheet나 slide deck을 final natural-language response만으로 평가하는 일을 막는다.

## 5-4. What really matters in the experiments

### 1) Real session origin은 benchmark distribution을 바꾼다

Task는 benchmark 작성자가 상상한 것만이 아니라 enterprise user가 실제로 agent에게 요청한 일을 반영한다.

### 2) Data privacy는 benchmark design을 바꾼다

Raw task를 공개할 수 없기 때문에 reusable artifact는 protocol과 pipeline이다.

### 3) Score는 multi-dimensional하다

단일 scalar는 artifact failure, cost explosion, runtime delay, skill-transfer weakness를 숨길 수 있다.

### 4) Visual evidence를 핵심 평가 요소로 다룬다

Business artifact는 file existence나 text summary만이 아니라 rendered output으로 평가되어야 한다.

# 6. Limitations

1. **Full benchmark data가 공개되지 않는다**
   - 이는 private leaderboard score의 independent replication을 제한한다.

2. **Proprietary session source에 의존한다**
   - Distribution은 provider의 workplace session에 의존한다.

3. **Judge dependence가 있다**
   - Semantic scoring과 visual scoring은 model judge에 의존한다.

4. **Harness-model coupling이 있다**
   - Realism에는 유용하지만 comparison 해석을 어렵게 만든다.

5. **Privacy-preserving rewriting이 task를 바꿀 수 있다**
   - Prompt rewriting과 fixture recovery가 original intent를 바꿀 수 있다.

6. **Artifact rendering이 실패할 수 있다**
   - LibreOffice, Playwright, PDF conversion, screenshot preparation이 evaluator error를 만들 수 있다.

7. **Public smoke test가 제한적이다**
   - Public repository는 full task reproduction이 아니라 protocol reproduction을 지원한다.

8. **Cost와 runtime variability가 있다**
   - Provider pricing과 infrastructure latency는 바뀔 수 있다.

9. **Skill taxonomy가 domain-specific일 수 있다**
   - Role class와 subclass가 enterprise 간에 transfer되지 않을 수 있다.

10. **Human usefulness는 더 넓은 문제다**
    - Judge된 artifact가 높은 score를 받더라도 실제 business stakeholder를 만족시키지 못할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

EnterpriseClawBench의 핵심은 "enterprise benchmark가 있다"가 아니다. 더 중요한 점은 **real workplace agent evaluation은 artifact protocol과 harness protocol 없이는 성립하지 않는다는 것**이다.

Agent output은 text만이 아니다. File, document, screenshot, log, cost, runtime, final response로 구성된 evidence directory다.

이 논문은 그 점을 드러낸다.

## 7-2. Reuse potential

### Internal enterprise agent evaluation

기업은 raw data를 공유하지 않고도 자체 private session 위에서 construction protocol과 run-directory protocol을 재현할 수 있다.

### Artifact-centric benchmark design

Deck, spreadsheet, report, dashboard를 생성하는 agent라면 evaluator가 rendered artifact를 검사해야 한다.

### Harness evaluation

Agent framework는 complete system으로 비교해야 한다. Model-only benchmark는 sandbox와 artifact delivery failure를 놓친다.

### Audit trail design

Run directory protocol은 agent observability를 위한 실용적인 출발점이다.

## 7-3. Production considerations

- Task fixture, prompt, output, artifact를 versioning해야 한다.
- Agent에게 보낸 exact prompt를 저장해야 한다.
- Judging 전에 artifact를 render해야 한다.
- Text-only judge path와 visual judge path를 분리해야 한다.
- Score와 함께 cost 및 runtime을 보고해야 한다.
- High-stakes artifact에는 human spot-check를 추가해야 한다.
- Private benchmark를 internal governance asset으로 다뤄야 한다.

## 7-4. Follow-up papers

- RealClawBench
- OpenClawBench
- EnterpriseClawBench
- WorkArena
- MiniWoB and web-agent benchmarks
- Agent-as-a-Router
- Verification Horizon
- LLM-as-a-Judge reliability studies
- Agent-Native Memory System

# 8. Summary

- EnterpriseClawBench는 real workplace session에서 enterprise agent evaluation을 구축한다.
- Fixture, rewritten prompt, role class, hard rule, rubric을 갖춘 852개의 reproducible private task를 만든다.
- Proprietary content 때문에 full data는 공개되지 않으므로 reusable artifact는 protocol과 tooling이다.
- Evaluation은 harness-model combination, artifact, visual evidence, cost, runtime을 핵심 요소로 다룬다.
- 핵심 교훈은 enterprise agent benchmark가 final text answer만이 아니라 delivered business artifact를 평가해야 한다는 점이다.
