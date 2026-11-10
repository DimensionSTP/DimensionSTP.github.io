---
layout: single
title: "Reimagining research papers as interactive and reliable AI agents Review"
categories: Study-concept
tag: [Paper2Agent, MCP, AI-Agents]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://www.nature.com/articles/s41586-026-11044-y)

- 논문: Reimagining research papers as interactive and reliable AI agents
- 연구명: Paper2Agent
- 저자: Jiacheng Miao, Joe R. Davis, Yaohui Zhang, Jonathan K. Pritchard, James Zou
- 공개일: 2026-09-16, Nature 658, 1272-1280 (2026)
- [Code repository](https://github.com/jmiao24/Paper2Agent)

> 한 줄 요약: Paper2Agent는 연구 논문과 그에 연결된 코드, 데이터, tutorial을 분석하여 실행 가능한 MCP server와 논문 전용 AI agent를 구성한다. 단순 질의응답을 넘어 원래 연구의 분석 과정을 재실행하고 새로운 입력에 적용하는 것을 목표로 한다.

논문을 읽고 새로운 방법을 실제 데이터에 적용하려면 생각보다 많은 작업이 필요하다. 저장소를 찾고, 환경을 설치하고, tutorial을 실행하고, 함수의 입력과 출력을 이해해야 한다. 논문에 소개된 성능이 뛰어나도 이 과정에서 막히면 연구 결과는 다른 프로젝트에서 활용되기 어렵다.

Paper2Agent가 주목할 만한 이유는 논문을 요약하는 챗봇을 만드는 것이 아니라, **논문이 제공하는 계산 가능한 기능을 재사용 가능한 도구로 전환**한다는 점이다.

이 연구를 읽을 때 구분해야 할 세 가지 범위가 있다.

1. 논문의 내용을 질문으로 검색하는 일
2. 논문의 공개 코드를 새로운 입력에 맞게 실행하는 일
3. 실행 결과가 과학적으로 타당한지 판단하는 일

Paper2Agent는 주로 1과 2를 자동화한다. 3은 여전히 별도의 전문가 검증과 연구 설계가 필요하다.

# 1. Problem Setting

## 1-1. Research paper의 재현 가능성 문제

연구 논문에는 알고리즘 설명과 실험 결과가 담겨 있지만, 실제 사용 가능한 소프트웨어 인터페이스는 없는 경우가 많다. 관련 코드가 공개되더라도 설치 환경이 불안정하거나 tutorial이 오래된 상태일 수 있다.

기존 해결 방법은 크게 세 가지로 볼 수 있다.

- 논문 텍스트에 RAG를 적용해 질문에 답한다.
- 코드 저장소와 실행 환경을 공유한다.
- 일반 코딩 에이전트가 저장소를 직접 읽고 필요한 분석 코드를 작성하게 한다.

각 방식에는 한계가 있다. RAG는 계산을 실제로 실행하지 않으며, 저장소 공유만으로는 사용법이 보장되지 않는다. 일반 코딩 에이전트는 매 요청마다 구현을 다시 해석하고 코드를 작성해야 하므로 반복 비용과 실행 오류가 누적될 수 있다.

## 1-2. 목표를 '설명'에서 '실행'으로 바꾸기

Paper2Agent의 목표는 논문과 연관된 계산 자산을 agent가 호출할 수 있는 MCP 인터페이스로 만든 뒤, 해당 인터페이스가 원래 코드의 결과를 재현하는지 검사하는 것이다.

중요한 차이는 사용자가 매번 코드를 새로 작성하도록 LLM에 맡기지 않는다는 점이다. 미리 구성하고 검증한 도구를 자연어로 호출하도록 만든다.

# 2. Core Idea

## 2-1. Paper2MCP와 agent layer의 분리

원문의 전체 시스템은 크게 두 층이다.

- Paper2MCP: 논문과 코드에서 MCP server를 생성하고 실행 가능한 기능을 노출한다.
- Agent layer: 만들어진 MCP server에 연결된 대화형 agent가 사용자의 요청을 해석하고 필요한 도구를 호출한다.

이 분리는 중요하다. MCP server는 특정 chat interface에 종속되지 않으므로, 같은 논문의 기능을 여러 agent framework에서 재사용할 수 있다.

## 2-2. MCP의 세 가지 구성요소

Paper2Agent가 생성하는 MCP server는 tools, resources, prompts로 구성된다.

| 구성요소 | 내용 | 역할 |
| --- | --- | --- |
| Tools | 분석 함수, 시각화, 예측 등의 실행 가능한 기능 | 실제 계산 수행 |
| Resources | 논문, 코드, 표, 보조 자료, 데이터 참조 | 출처와 문맥 제공 |
| Prompts | 여러 도구를 올바른 순서로 연결하는 작업 절차 | 복잡한 workflow 안내 |

예를 들어 단일 세포 분석에서는 quality control, filtering, normalization, dimensionality reduction, clustering을 각각 독립된 실행 기능으로 노출할 수 있다. 여기에 prompts가 적절한 실행 순서를 설명하면 사용자는 전체 파이프라인을 자연어로 요청할 수 있다.

## 2-3. 코드 생성보다 검증을 먼저 생각하기

이 연구의 핵심은 자동 생성한 tool을 곧바로 공개하지 않는다는 점이다.

Paper2Agent는 원래 tutorial의 실행 결과를 기준으로 자동 검증을 수행한다. 파일 생성 여부, 수치 결과의 오차, 그림의 유사도 등을 검사하며 실패하면 수정 과정을 반복한다. 검증에 계속 실패하는 tool은 최종 서버에서 제외한다.

따라서 신뢰성은 LLM의 설명 능력만으로 확보하는 것이 아니라, 원본 코드와 결과를 활용한 실행 테스트에 근거한다. 다만 tutorial 재현에 성공했다고 해서 새로운 데이터의 해석까지 과학적으로 보증되는 것은 아니다.

# 3. Architecture / Method

## 3-1. End-to-end 흐름

원문 Figure 1과 Methods의 파이프라인을 작업 단위로 정리하면 다음과 같다.

1. 논문에서 코드 저장소를 찾아 내려받는다.
2. 저장소를 실행할 수 있는 격리된 환경을 구성한다.
3. 실행 가능한 tutorial과 예제를 찾아 분류한다.
4. Tutorial을 실행해 입력, 출력, 필요한 환경 가정을 기록한다.
5. 분석 기능을 MCP tool로 만들고 자동 테스트를 수행해 오류를 수정한다.
6. 검증한 tool을 resources, prompts와 함께 MCP server로 조립하고 agent에 연결한다.

각 단계는 독립 역할을 가진 sub-agent가 수행하며, 중앙 orchestrator가 작업과 중간 산출물을 관리한다. 저자는 Claude Code 기반 agent SDK를 이용해 구현했다.

## 3-2. Environment manager

가장 먼저 해결해야 할 문제는 종속성이다. 논문의 README에 있는 설치 절차를 그대로 실행해도 패키지 버전이나 파일 경로가 맞지 않는 경우가 있다.

Environment manager는 저장소 구조를 분석하고 독립적인 실행 환경을 만든다. 단순히 코드를 생성하기보다 먼저 원본 프로젝트를 실행 가능한 상태로 만드는 데 집중한다. 이는 이후의 tool 테스트가 저장소 환경 오류에 의해 오염되지 않도록 하기 위한 설계다.

## 3-3. Tutorial scanner와 executor

Tutorial scanner는 저장소에서 사용할 만한 notebook, 예제, 문서를 찾는다. 이어서 tutorial executor가 실제 예제를 실행해 기준 출력을 수집한다.

이 과정에서 입력 데이터 경로, 환경 변수, 중간 파일, figure 생성 등 문서에 명확히 적히지 않은 가정도 드러난다. 다음 단계는 이러한 암묵적 계약을 도구의 명시적 입출력 인터페이스로 바꾸는 것이다.

## 3-4. Test verifier-improver

검증 agent는 생성된 기능이 원본 tutorial과 같은 작업을 수행하는지 확인한다. Methods에 따르면 주요 검증 기준은 다음과 같다.

- 기대한 파일이 실제로 생성되는가?
- Float 결과가 기준값의 허용 오차 안에 있는가? 원문에서는 3% tolerance를 사용한다.
- 생성한 그림이 참조 그림과 유사한가? Perceptual hash의 Hamming distance가 20 미만인지 평가한다.
- 동일 기능이 반복적으로 실패할 경우 제외할 수 있는가?

함수당 최대 6회까지 테스트, 실패 진단, 수정 단계를 수행한다. 계속 실패하면 해당 MCP tool을 최종 서비스에서 제외하고 실패 기록을 남긴다.

여기서 테스트 통과는 **tutorial과의 기능적 일치**를 의미한다. 통계적 검정의 적절성, 연구 가설의 타당성, 데이터 편향까지 자동으로 검증했다는 뜻은 아니다.

## 3-5. 배포 후 agent의 사용 방식

검증한 기능은 MCP Python server로 패키징할 수 있다. 논문에서는 Hugging Face Spaces와 같은 원격 환경에 배포하고, Claude Code 등 agent를 연결해 실행한다.

사용자는 '이 데이터를 분석해 달라'고 요청하지만, 실제로는 agent가 고정된 MCP tool과 workflow를 호출한다. 이는 매 요청마다 유사한 코드를 새로 만들어 실행하는 방식과 다르다.

복수의 MCP server를 하나의 agent에 연결할 수도 있다. 이 구조가 서로 다른 논문의 방법과 데이터를 조합하는 후속 실험의 기반이 된다.

# 4. Training / Data / Recipe

## 4-1. 새로운 foundation model을 학습하는 연구인가

Paper2Agent는 자체 대형 모델을 새로 사전 학습하는 논문이 아니다. 기존 LLM을 도구 생성, 환경 설정, 테스트 수정, 자연어 요청 수행에 사용하는 systems 연구다.

원문은 Paper2Agent 구성에 Claude Code를 사용하고, 평가에서는 Claude Sonnet 4를 주요 모델로 사용한다. 따라서 개선된 결과를 새로운 base LLM의 학습 성능으로 돌려서는 안 된다. 미리 생성한 MCP 인터페이스와 검증 과정의 효과를 해석해야 한다.

## 4-2. 실제 입력 자료

시스템이 활용하는 입력은 다음과 같다.

- 논문 본문과 supplementary material
- 논문이 공개한 소스 코드와 관련 저장소
- 실행 가능한 tutorial 및 예제 데이터
- 원본 코드가 생성하는 수치 및 figure
- 분석 workflow를 설명하는 prompt

코드가 없는 논문에도 structured resource layer는 구성될 수 있다. 그러나 실행 가능한 MCP tool이 만들어진다는 보장은 없다. 원문은 이 두 성공 수준을 구분한다.

## 4-3. 검증 데이터를 분리하는 방식

원문의 100편 computational biology 평가에서는 원래 코드의 tutorial에서 질문을 만들고, 그 tutorial을 실행한 결과를 기준 정답으로 사용한다.

한편 26편의 data/discovery 중심 논문에서는 본문과 보조 자료를 종합해야 하는 질문으로 resource layer를 평가한다. 다른 학문 영역의 10편에 대해서는 실행 기반 과제를 별도로 구성한다.

이처럼 평가 종류가 다르므로 91.2%, 89.0%, 98.1%를 하나의 동일한 능력 점수로 합치면 안 된다.

# 5. Evaluation

## 5-1. 대규모 agentification의 성공과 실패

원문은 세 그룹을 인간의 수동 수정 없이 end-to-end로 처리한다.

| 그룹 | 입력 논문 수 | 평가 목적 | 주요 결과 |
| --- | ---: | --- | --- |
| Computational biology | 100 | 실행 가능한 MCP tool 생성 및 활용 | 74편 성공, 26편 실패 |
| Data/discovery 중심 | 26 | Structured resource 기반 종합 질문 | 100문항 평가 |
| Non-biology computational | 10 | 코드 실행을 필요로 하는 과제 | 42문항 평가 |

성공한 74편의 computational biology 논문에서 599개의 tool 후보가 생성됐고, 그중 593개가 자동 검증을 통과했다. 이 비율을 '전체 100편 논문의 99%가 완전 자동화됐다'고 말하면 잘못이다. 도구 검증 통과율은 **이미 성공적으로 처리된 논문 내부의 tool 후보**를 분모로 한다.

실패한 26편의 주요 원인은 실행 가능한 코드나 모델 자료의 부재, 환경 및 종속성 문제, 특정 환경에 지나치게 맞춰진 스크립트였다.

## 5-2. 300개 tutorial 기반 질문

동일한 300개 tutorial-derived 질문에서 저자들이 보고한 평균 accuracy와 표준오차는 다음과 같다.

| 시스템 | Accuracy (%) |
| --- | ---: |
| Paper2Agent + Sonnet 4 | 91.2 +/- 1.6 |
| Claude + Repo, Sonnet 4 | 80.3 +/- 2.3 |
| Claude + Repo, Sonnet 4.6 | 86.3 +/- 1.1 |

이 결과는 이미 만들어 둔 MCP tool의 효과를 보여준다. Baseline은 논문과 전체 저장소에 접근할 수 있지만 같은 MCP server, resource, prompt 구조는 사용하지 않는다.

원문은 Sonnet 4 기준으로 질의당 비용 약 0.20달러, 지연 약 1.6분을 보고하며, 직접 paper와 repo를 이용하는 비교군은 0.38달러, 4.3분이다.

중요한 조건은 이 수치가 **질의 시점의 비용과 지연**이라는 점이다. MCP server를 최초 생성하고 검증하는 준비 비용까지 합친 total cost of ownership과 동일하지 않다.

## 5-3. 다른 분야로의 확장

10편의 비생물학 계산 논문에서 구성한 42개 실행 기반 과제에 대해 Paper2Agent는 5회 독립 실행 평균 accuracy 98.1 +/- 0.8%를 보고한다.

26편의 data/discovery 중심 논문에 대해서는 실행 도구 생성 여부와 무관하게 구조화된 resource를 이용해 100개 synthesis 질문에 응답했다. 이 설정에서는 89.0 +/- 3.1%이며, browser-use 기반 비교군은 82.0 +/- 3.8%다. 논문은 해당 설정에서 34배 낮은 비용과 15배 빠른 수행도 보고한다.

다만 이 평가들 역시 논문별 기능과 질문에 대해 정의된 정확도다. 모든 연구 주제의 open-ended scientific reasoning을 검증한 결과는 아니다.

## 5-4. AlphaGenome와 Scanpy 사례

AlphaGenome agent는 genomics variant scoring 및 시각화 등 22개 MCP tool을 약 45분, 약 14달러의 생성 비용으로 만들었고, 모두 자동 검증을 통과했다고 보고한다.

Scanpy agent는 single-cell preprocessing과 clustering의 주요 기능을 7개 tool로 구성했다. 사용자가 dataset path를 제공하면 quality control부터 clustering까지 여러 단계를 연결한다. 연구진은 일부 데이터에 대해 사람이 수행한 분석과 주요 산출물이 일치한다고 보고한다.

이 수치는 구체적인 사례의 생성 비용과 작업 범위를 보여주는 것이며, 모든 논문을 동일한 시간과 비용으로 agent화할 수 있다는 보장은 아니다.

## 5-5. 여러 논문을 결합한 discovery 사례

원문은 AlphaGenome, MPRA-coupled scCRISPRi, CD4+ T-cell Perturb-seq에 대응하는 agent를 조합해 psoriasis 위험 변이와 관련된 후보 유전자를 우선순위화하는 사례를 제시한다. 연구진은 rs887314 locus에 대해 GPR137을 후보로 분석한다.

이 사례는 논문별 도구와 데이터가 하나의 agent interface에서 연결될 수 있음을 보여준다. 그러나 계산 기반의 후보 우선순위화와 실제 생물학적 인과 관계의 실험적 확증은 구분해야 한다. 원문도 open-ended 과학적 해석에는 human-in-the-loop가 필요하다고 강조한다.

# 6. Limitations

## 6-1. 자동 생성의 성공률은 100%가 아니다

100편 중 26편의 computational biology 논문에서는 end-to-end tool 생성을 완료하지 못했다. 누락된 실행 코드나 데이터, 환경 재현 실패는 LLM의 추론 능력만으로 해결하기 어렵다.

따라서 Paper2Agent의 성능은 논문 자체의 명료성뿐 아니라 저장소와 tutorial의 품질에도 크게 의존한다.

## 6-2. Tool test와 과학적 신뢰성은 다르다

원본 tutorial 결과를 재현하는 검증은 의미가 크다. 그러나 원래 코드가 잘못된 통계 분석을 수행하거나, 중요한 교란 변수를 놓치거나, 특정 데이터에서만 작동한다면 그 오류도 충실하게 재현될 수 있다.

논문은 open-ended 과학적 판단에서 인간의 책임을 남겨 두고 있다. 따라서 'reliable AI agents'의 reliable은 모든 과학적 결론을 보증한다는 의미로 읽으면 안 된다.

## 6-3. 배포와 운영상의 문제

- 제3자 연구 코드 실행에 따르는 보안 위험이 있다.
- API key, 라이선스, 데이터 접근 권한, 개인정보 처리가 필요할 수 있다.
- 서버의 도구 버전과 upstream repository의 버전이 달라질 수 있다.
- 연구용 notebook이 원격 서비스로 배포될 때 리소스와 비용 관리가 필요하다.
- 논문과 연결된 도구가 유지보수되지 않으면 재현 가능성도 시간이 지남에 따라 낮아진다.

# 7. My Take

## 7-1. 논문 이해보다 연구 기능의 표준화

가장 중요한 변화는 논문을 검색 가능한 지식 문서에 머무르게 하지 않고, 검증 가능한 기능 단위로 포장한다는 점이다. 기존 RAG 시스템과의 차이는 답변의 자연스러움이 아니라 **실행 결과와 원본 자산을 연결하는 방식**에 있다.

AI/ML 연구 환경이라면 새로운 벤치마크, 데이터 전처리, 모델 평가 방법을 읽는 것에서 끝내지 않고 agent-callable tool로 만드는 작업에 적용할 수 있다.

## 7-2. 내부 연구 파이프라인에 적용한다면

다음과 같은 조건에서 작은 시범 프로젝트를 시작하는 것이 현실적이다.

1. 공개 코드와 실행 예제가 잘 갖춰진 논문 하나를 선택한다.
2. 환경을 고정하고 기준 실행 결과를 저장한다.
3. 각 기능의 입력 schema와 출력 artefact를 명시한다.
4. MCP tool 생성 후 unit test와 regression test를 수행한다.
5. 정확도뿐 아니라 최초 변환 비용, 질의당 비용, 실패 복구 비용을 측정한다.
6. 논문 버전, 저장소 commit, model version, 데이터 라이선스를 기록한다.
7. 새로운 결과의 과학적 타당성은 별도의 human review를 거치게 한다.

이는 원문에 보고된 결과가 아니라 실제 연구 조직에서 도입할 때 필요한 운영 기준에 대한 제안이다.

## 7-3. 후속 읽을거리

- [Paper2Agent repository](https://github.com/jmiao24/Paper2Agent): 구현과 benchmark 질문 확인
- [Model Context Protocol](https://modelcontextprotocol.io/): tools, resources, prompts의 표준 인터페이스
- [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0): Paper2Agent가 적용한 genomics 기반 모델의 연구 배경. 2026년 Nature 게재 원문

# 8. Summary

- Paper2Agent는 논문과 코드를 MCP server 및 논문 전용 AI agent로 변환한다.
- 중요한 설계는 자동 tool 생성보다 원본 tutorial에 기반한 실행 검증과 실패한 도구의 제외다.
- 100편 computational biology 논문 중 74편에서 agentification에 성공했다.
- 300개 tutorial 기반 질문에서 Paper2Agent + Sonnet 4의 정확도는 91.2%로, 동일 모델의 직접 저장소 접근 baseline보다 높았다.
- 연구 기능의 재사용성과 과학적 결론의 타당성은 별도로 평가해야 한다.
