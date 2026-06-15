---
layout: single
title: "COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation Review"
categories: Study-concept
tag: [AI-Agents, AgentSkills, KnowledgeDistillation, Personalization]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2605.31264)

[PDF](https://arxiv.org/pdf/2605.31264)

[Code](https://github.com/titanwings/colleague-skill)

COLLEAGUE.SKILL은 LLM agent가 사람의 업무 지식, 판단 기준, 상호작용 스타일을 어떻게 재사용 가능한 skill artifact로 만들 수 있는가를 다루는 논문이다. 제목만 보면 colleague persona를 만드는 프로젝트처럼 보일 수 있지만, 논문의 핵심은 사람을 복제하는 것이 아니다. 핵심은 흩어진 trace를 읽어서 inspectable, correctable, portable, governable한 agent skill package로 바꾸는 workflow를 제안한다는 데 있다.

요즘 agent system에서 memory, tool, skill, workflow가 점점 분리되고 있다. Tool은 외부 action을 담당하고, memory는 과거 정보를 보존하며, skill은 특정 절차와 지식을 재사용 가능한 단위로 포장한다. 그런데 실제 조직에서 중요한 지식은 깨끗한 instruction manual로 남아 있지 않다. code review comment, incident note, design doc, chat decision, email, screenshot 같은 파편에 섞여 있다.

이 논문은 이 파편을 hidden memory에 넣거나 긴 prompt에 붙이는 대신, 파일과 metadata가 있는 skill artifact로 렌더링하자는 쪽에 가깝다.

> 한 줄 요약: COLLEAGUE.SKILL은 사람 또는 역할에 대한 heterogeneous trace를 capability track과 bounded behavior track으로 분리해 versioned skill package로 만들고, 이를 inspect, correct, rollback, install, share할 수 있게 만든 trace-to-skill distillation system이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- agent skill이 단순 prompt snippet이 아니라 배포 가능한 software artifact로 바뀌는 흐름을 잘 보여준다.
- person-grounded agent를 memory나 persona simulation이 아니라 governance 가능한 package 문제로 재정의한다.
- capability와 behavior를 분리해서, 업무 판단력과 말투 모사를 같은 층에 섞지 않는다.
- correction, rollback, deletion, optional gallery 같은 lifecycle operation을 논문 기여의 일부로 본다.
- benchmark score보다 product surface와 artifact contract가 중요한 system paper라는 점에서 읽을 가치가 있다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 person-grounded skill generation이다. 여기서 person-grounded skill은 특정 사람이나 역할의 trace에 근거하지만, 그 사람 자체를 대체하거나 복제한다고 주장하지 않는 bounded artifact를 뜻한다.

논문은 입력과 출력을 다음처럼 정리한다.

- 입력은 lightweight profile $p$, source scope $c$, source materials $D = {d_1, ..., d_n}$이다.
- 출력은 skill package $S = (A, M, L)$이다.
- $A$는 생성된 파일, $M$은 metadata와 install 정보, $L$은 version, update time, correction count, rollback history 같은 lifecycle state다.

즉 문제는 "이 사람처럼 답하는 agent를 만들자"가 아니라, "선택된 trace에서 재사용 가능한 판단 기준, 절차, 상호작용 규칙을 추출해서 검토 가능한 package로 만들자"에 가깝다.

논문이 요구하는 operational property는 다섯 가지다.

| Property | Meaning |
| --- | --- |
| Portable | skill-compatible agent host가 일반 skill mechanism으로 load할 수 있어야 한다. |
| Inspectable | 사용자가 rule, example, limitation, metadata를 읽을 수 있어야 한다. |
| Composable | full, work-only, persona-only entrypoint를 분리해서 호출할 수 있어야 한다. |
| Correctable | 새 evidence나 user feedback으로 package를 고칠 수 있어야 한다. |
| Governable | source boundary, disclaimer, deletion, sharing decision을 metadata로 다룰 수 있어야 한다. |

이 framing이 중요한 이유는 person-grounded agent를 평가 가능한 software object로 만들기 때문이다. hidden prompt나 memory store 안에 들어간 representation은 사용자가 무엇이 들어갔는지 알기 어렵다. 반대로 skill package는 파일, metadata, correction log, version state를 통해 검토할 수 있다.

## 1-2. Why previous approaches are insufficient

기존 접근은 크게 세 갈래로 볼 수 있다.

첫째, memory 기반 personalization은 과거 정보를 retrieve하거나 context에 넣는 데 강하지만, 그 representation이 agent-usable instruction으로 정리되어 있는지는 별개의 문제다. memory는 trace를 보관할 수 있지만, 업무 기준이나 decision heuristic을 명시적인 runtime artifact로 만들지는 않는다.

둘째, persona prompt는 surface style을 빠르게 흉내 낼 수 있다. 하지만 말투, 사실 지식, 업무 절차, 의사결정 기준이 한 덩어리 prompt에 섞이면 감사와 수정이 어렵다. "그 사람처럼 말하기"와 "그 사람이 쓰던 review standard 적용하기"는 다른 문제인데, persona prompt는 이 둘을 자주 섞는다.

셋째, skill framework는 portable packaging format을 제공하지만, skill 자체를 어떻게 생성할지는 별도의 문제다. 즉 SKILL.md와 metadata format이 있어도, chat logs나 design docs에서 어떤 정보를 뽑고 어떤 boundary를 달아야 하는지는 자동으로 해결되지 않는다.

COLLEAGUE.SKILL은 이 빈틈을 trace-to-skill distillation workflow로 채우려 한다. 이 논문에서 skill은 지식 저장소가 아니라, 생성, 검토, 수정, 설치, 삭제, 공유까지 포함하는 lifecycle object다.

# 2. Core Idea

## 2-1. Main contribution

COLLEAGUE.SKILL의 핵심 기여는 크게 네 가지다.

1. Person-grounded trace-to-skill distillation formulation

   사람의 trace를 unrestricted simulation으로 보지 않고, source, usage, governance constraint가 있는 skill artifact로 만든다. 이 framing 덕분에 논문의 claim이 과장되지 않는다. 논문은 faithful person reproduction을 주장하지 않는다.

2. Dual-track representation

   생성 artifact를 capability track과 bounded behavior track으로 나눈다. Capability track은 업무 절차, mental model, decision heuristic을 담고, behavior track은 communication style, interaction rule, correction history를 담는다.

3. Versioned artifact contract

   출력은 SKILL.md 하나로 끝나지 않는다. work.md, persona.md, work_skill.md, persona_skill.md, manifest.json, meta.json 같은 파일을 포함하고, schema version과 lifecycle state를 둔다.

4. Correction and deployment lifecycle

   자연어 feedback을 받아 patch 또는 correction record로 바꾸고, archive, rollback, regenerate를 수행한다. 또한 Claude Code, OpenClaw, Codex, Hermes 같은 agent host에 설치하거나 optional gallery distribution으로 연결할 수 있게 한다.

## 2-2. Design intuition

이 논문의 설계 직관은 "사람을 흉내내는 agent"보다 "사람의 trace에서 나온 bounded artifact"가 더 안전하고 실용적이라는 데 있다.

예를 들어 퇴사한 backend engineer의 code review style을 생각해보자. 실제로 필요한 것은 그 사람의 모든 대화 습관을 복제하는 것이 아니다. 더 중요한 것은 authentication, input validation, rate limiting, response schema, sensitive-data exposure 같은 review priority를 유지하는 것이다. 말투는 secondary feature일 수 있고, 어떤 상황에서는 아예 제거하는 편이 낫다.

그래서 capability-only entrypoint와 behavior-only entrypoint를 분리하는 것이 중요하다. 업무 지식이 필요한 상황에서는 work-only skill을 쓰고, 표현 style이 필요한 상황에서는 persona-only skill을 쓸 수 있다. full skill은 두 track을 함께 쓰지만, 그 역시 source boundary와 disclaimer를 가져야 한다.

이 논문의 가장 좋은 지점은 "persona를 잘 만들자"가 아니라 "persona와 capability를 분리해서 관리하자"라고 말한다는 점이다. 실제 agent product에서 위험한 것은 사람이름이 붙은 멋진 demo가 아니라, 출처, 권한, 수정 이력, 삭제 방법이 없는 opaque behavior package이기 때문이다.

# 3. Architecture / Method

## 3-1. Overview

COLLEAGUE.SKILL pipeline은 다섯 단계로 볼 수 있다.

| Stage | Role |
| --- | --- |
| Trace intake | docs, email, screenshots, chats, reviews, public materials를 local knowledge directory로 정리한다. |
| Preset router | colleague, celebrity, relationship preset 중 source boundary와 command semantics를 고른다. |
| Dual distill | capability track과 persona track을 분리해서 work.md와 persona.md를 만든다. |
| Artifact writer | SKILL.md, manifest.json, meta.json, sub-skills, slash commands를 package로 렌더링한다. |
| Productization | agent host 설치, invocation, optional gallery sharing으로 연결한다. |

이 구조의 핵심은 governance rail이다. 논문은 local-first storage, provenance and evidence, correction log, version and rollback, optional gallery를 pipeline 바깥 장식이 아니라 필수 rail로 둔다.

## 3-2. Module breakdown

### 1) Application presets

논문은 세 가지 preset을 설명한다.

| Preset | Source | Main governance issue | Runtime use |
| --- | --- | --- | --- |
| colleague | work docs, reviews, decisions, incidents | organizational access, handover, consent | full, work-only, persona-only skill |
| celebrity | first-person public works, interviews, speeches | citation, source boundary, no private inference | source-grounded mental model skill |
| relationship | private interpersonal traces | consent, local control, deletion, non-public default | local editable interaction rule state |

Preset은 별도 시스템이 아니라 같은 artifact workflow의 configuration이다. 새 domain을 추가할 때도 core pipeline을 다시 만드는 대신 source boundary, prompt bundle, storage root, command alias를 바꾸는 방식이다.

이 점이 꽤 실무적이다. agent skill generation을 product로 운영하려면, task마다 완전히 다른 implementation을 만드는 것보다 같은 lifecycle contract 위에 domain policy를 얹는 편이 낫다.

### 2) Dual representation

Generated artifact는 capability track과 behavior track을 나눈다.

Capability track은 다음 내용을 담는다.

- responsibilities
- workflows
- technical standards
- review criteria
- decision heuristics
- lessons from past work

Behavior track은 다음 내용을 담는다.

- expression preference
- interaction posture
- communication boundary
- correction records
- usage limits

이 분리는 단순히 파일을 두 개로 나누는 것이 아니다. agent runtime에서 어떤 정보를 언제 적용할지 정하는 abstraction이다. 논문은 full, capability-only, behavior-only entrypoint를 노출해서 사용자가 risk-utility trade-off를 선택할 수 있게 한다.

### 3) Artifact schema and writer

Artifact writer는 schema version 3을 사용한다. 주요 출력은 다음과 같다.

| Artifact | Consumer | Contents |
| --- | --- | --- |
| SKILL.md | agent runtime, user | capability track, behavior track, operating rules가 결합된 invokable skill |
| work.md | user, updater | procedure, standard, heuristic, task pattern을 담은 editable capability document |
| persona.md | user, updater | style, interaction posture, boundary, correction log를 담은 editable behavior document |
| work_skill.md | agent runtime | capability-only entrypoint |
| persona_skill.md | agent runtime | persona-only entrypoint |
| manifest.json | installer, gallery | entrypoint, artifact list, compatible runtime, slash command, toolchain metadata |
| meta.json | lifecycle tools | schema, provenance, lifecycle version, correction count, compatibility field |

Agent Skills format과 맞춰 SKILL.md를 중심 entrypoint로 두고, 자세한 지시는 invocation 시점에 progressive disclosure로 load하는 구조다. 이 덕분에 skill이 단순 text blob이 아니라 host가 이해할 수 있는 package가 된다.

### 4) Correction and update workflow

논문은 생성 artifact가 처음부터 완벽하다고 보지 않는다. Correction handler는 자연어 feedback을 받아 두 방식으로 반영한다.

1. Expert work correction

   업무 기준이나 절차가 틀린 경우 Markdown patch를 만든다. 같은 level-2 heading이 있으면 해당 section을 replace하고, 없으면 append한다.

2. Behavior correction

   말투나 interaction rule이 틀린 경우 `{scene, wrong, correct}` 형태의 normalized correction record를 만든다.

그 뒤 writer는 현재 version을 archive하고, patch 또는 correction을 적용하며, lifecycle version을 올리고, derived artifact를 다시 생성한다. Version manager는 list, backup, rollback, archive cleanup을 담당한다.

이 부분은 agent product 관점에서 중요하다. 사람 기반 skill은 반드시 틀린다. 중요한 것은 처음부터 완벽하게 만드는 것이 아니라, 틀렸을 때 누가 어떻게 고쳤는지 남기고 되돌릴 수 있는가다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문에는 일반적인 supervised training dataset이나 RL benchmark가 없다. 대신 source material과 ingestion recipe가 중요하다.

지원하는 source category는 다음과 같다.

- chat logs
- work documents
- code review comments
- incident notes
- email archives
- screenshots
- PDFs
- Markdown files
- direct paste text
- public research materials
- subtitles and interviews for public-figure preset

Repository-supported collector와 import path로는 Feishu, DingTalk, Slack, WeChat SQLite export, email archive, PDF, screenshot, Markdown, direct paste가 언급된다.

중요한 점은 source quality가 skill quality를 결정한다는 것이다. 일반 잡담이 많은 trace보다, 실제 의사결정, review rationale, incident handling, architecture trade-off가 들어 있는 trace가 더 가치 있다.

## 4-2. Training strategy

여기서 training은 model weight를 업데이트하는 의미가 아니다. 더 정확히는 expert knowledge distillation workflow다.

흐름은 다음과 같다.

1. 사용자가 alias, optional profile, source scope, materials를 제공한다.
2. Preset router가 domain-specific prompt bundle을 선택한다.
3. Analyzer가 capability evidence와 behavior evidence를 분리한다.
4. Builder가 work.md와 persona.md를 만든다.
5. Writer가 SKILL.md, sub-skills, manifest, meta file로 package를 만든다.
6. 사용자가 artifact를 inspect하고 natural-language feedback으로 correction을 넣는다.
7. Version manager가 archive, patch, regenerate, rollback을 처리한다.

이 recipe는 neural distillation이라기보다 artifact distillation이다. Teacher는 특정 model이 아니라 target person or role의 trace이고, student는 agent host가 읽을 수 있는 skill package다.

## 4-3. Engineering notes

1. Source boundary를 먼저 정해야 한다.

   어떤 trace를 사용할 수 있는지, 어떤 trace는 제외해야 하는지, public evidence인지 private evidence인지가 prompt보다 먼저 정해져야 한다.

2. Capability와 behavior를 섞지 않는 것이 중요하다.

   업무 판단을 배우고 싶은데 말투까지 강제로 가져오면 risk가 늘어난다. 반대로 말투만 필요한데 업무 판단을 붙이면 권한 없는 inference가 생길 수 있다.

3. Correction log는 product feature가 아니라 safety feature다.

   누가 어떤 오류를 고쳤고, 어떤 version부터 적용됐는지 추적할 수 있어야 skill의 신뢰도를 논의할 수 있다.

4. Gallery는 adoption metric이 아니다.

   논문은 public counters를 distribution surface의 evidence로만 사용한다. 이것을 task performance나 behavioral fidelity로 해석하면 안 된다.

# 5. Evaluation

## 5-1. Main results

이 논문은 benchmark score로 성능을 주장하는 논문이 아니다. 대신 deployed system과 public distribution surface를 보여준다.

논문 작성 시점 기준 public deployment counter는 다음과 같다.

| Signal | Value | How to read it |
| --- | --- | --- |
| GitHub stars | about 18.5k | repository visibility |
| Forks | about 1.8k | community interest |
| Commits | 104 | implementation activity |
| Skills in gallery | 215 | shareable package surface |
| Meta-skills | 55 | higher-level reusable skill surface |
| Contributors | 165 | public participation |
| Cumulative gallery stars | more than 100k | order-of-magnitude public signal |

논문도 이 숫자를 performance claim으로 쓰지 않는다. 이 숫자는 generated skill이 local folder에서 끝나지 않고, gallery와 host installation까지 이어지는 public deployment surface를 가졌다는 증거에 가깝다.

Application case는 세 가지다.

1. Colleague skill

   design document, chat decision, review comment, incident note에서 업무 기준을 추출한다. 예시로 authentication, input validation, rate limiting, response schema, sensitive-data exposure 같은 review criteria를 lower-priority issue보다 먼저 보게 할 수 있다.

2. Celebrity skill

   public first-person source, interview, speech 등을 기반으로 mental model과 expression boundary를 만들지만, private inference는 피해야 한다.

3. Relationship skill

   private interpersonal trace를 local editable state로 다루지만, consent, deletion, emotional overattachment, non-consensual simulation risk를 가장 강하게 봐야 한다.

## 5-2. What really matters in the experiments

이 논문에서 진짜 중요한 것은 score가 아니라 claim boundary다.

논문이 주장하는 것은 다음이다.

- package format을 정의했다.
- generation and update workflow를 구현했다.
- correction and rollback state를 노출했다.
- multiple agent hosts와 gallery distribution setting을 연결했다.
- colleague, celebrity, relationship preset을 같은 mechanism으로 다룰 수 있음을 보였다.

논문이 주장하지 않는 것은 다음이다.

- generated skill이 실제 사람을 충실히 재현한다.
- generated skill이 downstream work performance를 향상한다.
- capability-only variant가 persona risk 없이 항상 충분하다.
- correction이 behavior를 항상 단조롭게 개선한다.
- public-figure extension이 motive hallucination을 완전히 막는다.

따라서 이 논문을 읽을 때는 "성능이 얼마나 좋은가"보다 "어떤 artifact handle이 생겼는가"를 봐야 한다. Work.md, persona.md, correction record, manifest, meta.json, rollback state가 생기면 후속 연구에서 source quality, correction quality, invocation mode, governance policy를 비교할 수 있다.

# 6. Limitations

1. Behavioral fidelity evaluation이 없다.

   논문은 artifact-level claim을 한다. Generated skill이 실제 source expert와 같은 review issue를 잡는지, 같은 trade-off를 내는지, user가 신뢰도를 정확히 calibration하는지는 별도 human and task-based study가 필요하다.

2. Source quality에 매우 민감하다.

   Trace가 부정확하거나 편향되어 있으면 skill도 그 편향을 반영한다. 특히 correction이 editor bias를 artifact에 굳힐 수 있다.

3. Consent와 access control은 기술만으로 해결되지 않는다.

   Local-first, inspectable, versioned design은 governance affordance를 주지만, lawful source use, consent, redaction, retention limit은 별도 조직 절차가 필요하다.

4. Public counter를 adoption quality로 읽으면 안 된다.

   GitHub star나 gallery count는 distribution signal이지 task impact가 아니다. 논문도 이 점을 명시적으로 구분한다.

5. Relationship preset은 특히 위험하다.

   Private trace, emotional overattachment, non-consensual simulation, deletion right가 걸린다. 이 영역은 demo value보다 safety and policy design이 먼저 와야 한다.

6. Skill artifact가 agent host를 완전히 통제하지 않는다.

   SKILL.md와 metadata가 잘 작성되어도 실제 runtime model이 instruction을 어떻게 따르는지는 host, model, context, tool permission에 따라 달라질 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 agent를 만들 때 기억해야 할 중요한 방향을 제시한다. 앞으로 agent system의 차이는 base model 하나보다, 외부화된 context와 capability를 어떤 artifact contract로 관리하느냐에서 날 가능성이 크다.

특히 연구 조직이나 engineering team에서는 암묵지가 많다. 좋은 code review 기준, 장애 대응 우선순위, 실험 로그 해석법, paper reading heuristic은 문서에 깔끔하게 남지 않는다. 이런 지식을 RAG index에 넣는 것만으로는 부족하고, agent가 바로 사용할 수 있는 operating rule로 바꿔야 한다.

COLLEAGUE.SKILL은 그 방향을 보여준다. Trace를 모으고, capability와 behavior를 나누고, 파일로 보여주고, 수정하고, rollback하고, host에 설치한다. 이 흐름은 실제 product에서 memory보다 skill package가 더 좋은 abstraction이 될 수 있음을 보여준다.

## 7-2. Reuse potential

실무적으로 재사용할 만한 포인트는 다음과 같다.

1. Team skill package

   개인 한 명을 복제하기보다, team convention이나 project-specific review standard를 skill로 만드는 쪽이 안전하고 유용할 수 있다.

2. Capability-only mode

   말투를 제거하고 work-only skill만 쓰는 모드는 enterprise agent에서 특히 중요하다. 업무 기준은 필요하지만 person simulation은 불필요한 경우가 많다.

3. Correction-first design

   Agent skill은 처음부터 맞을 수 없다. Correction record와 rollback을 기본 설계에 넣는 것은 매우 재사용 가능하다.

4. Governance metadata

   Source scope, consent status, publication permission, retention limit 같은 metadata를 skill package에 포함시키는 설계는 다른 agent workflow에도 적용할 수 있다.

5. Benchmark design hint

   후속 benchmark는 open-ended imitation이 아니라, source-bounded judgment preservation을 봐야 한다. 예를 들어 source expert의 code review issue detection과 skill의 issue detection을 비교하는 식이다.

## 7-3. Follow-up papers

- Agent Skills specification
- Agent Skills: A Data-Driven Analysis of Claude Skills for Extending Large Language Model Functionality
- SkillGen: Verified Inference-Time Agent Skill Synthesis
- AutoSkill: Lifelong Personalized Agents with Skill Memory
- Voyager: An Open-Ended Embodied Agent with Large Language Models
- LaMP: When Large Language Models Meet Personalization

# 8. Summary

- COLLEAGUE.SKILL은 사람을 복제하는 논문이 아니라, 사람 또는 역할의 trace를 bounded skill artifact로 만드는 논문이다.
- 핵심 설계는 capability track과 bounded behavior track을 분리하는 dual representation이다.
- 출력은 SKILL.md뿐 아니라 work.md, persona.md, sub-skills, manifest.json, meta.json, lifecycle state를 포함한다.
- Evaluation은 benchmark score보다 deployment surface, artifact contract, correction lifecycle을 보여주는 쪽에 가깝다.
- 가장 중요한 limitation은 behavioral fidelity, task performance, consent, source quality, runtime safety가 아직 별도 검증 문제로 남아 있다는 점이다.
