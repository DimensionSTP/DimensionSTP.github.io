---
layout: single
title: "K-BrowseComp: A Web Browsing Agent Benchmark Grounded in Korean Contexts Review"
categories: Study-concept
tag: [AgentBenchmark, WebAgent, KoreanLLM, Evaluation, BrowseComp]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.02404)

K-BrowseComp는 "한국어 에이전트 평가는 어디에서 어려워지는가"를 꽤 직접적으로 묻는 benchmark paper다. 요즘 agent benchmark는 단순 QA나 instruction following에서 벗어나, 검색하고, 비교하고, 중간 가설을 수정하고, 최종 답을 검증하는 compositional capability를 보려는 쪽으로 이동하고 있다. 그런데 한국어와 한국 맥락을 가진 web browsing task는 아직 많지 않다.

이 논문은 바로 그 빈 곳을 겨냥한다. K-BrowseComp는 400개 문제로 구성된 한국 맥락 기반 web-browsing agent benchmark이며, 그중 300개는 native Korean speaker가 수동으로 구성하고 검증한 K-BrowseComp-Verified subset이다. 추가로 100개 synthetic split은 hard few-shot exemplar와 failure-mode-targeted generation을 이용해 구성하고, adversarial filtering을 거쳐 별도 diagnostic stress test로 보고한다.

> 한 줄 요약: K-BrowseComp는 한국어와 한국 맥락에서 web browsing agent가 실제로 얼마나 오래, 정확하게, 근거 있게 검색할 수 있는지 평가하기 위해 만든 400문항 benchmark이며, frontier model도 verified subset에서 30.00-45.67 percent 수준에 머무른다는 점을 보여준다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- 한국어 agent benchmark를 단순 번역 문제가 아니라, **한국 맥락 기반 web browsing 문제** 로 따로 세운다.
- frontier model과 국내 모델 사이의 성능 차이를 보여주는 동시에, frontier model도 여전히 낮은 absolute accuracy를 보인다는 점을 제시한다.
- 300개 manual verified split과 100개 synthetic diagnostic split을 분리해, benchmark construction과 stress test를 구분한다.
- BrowseComp류 문제를 한국어 환경으로 가져올 때 생기는 locality, search path, verification 문제를 생각하게 만든다.

이 논문의 핵심 메시지는 단순하다. 한국어 agent 평가에서 중요한 것은 한국어로 답을 잘 쓰는 능력만이 아니다. 실제 병목은 **한국 맥락의 정보를 찾고, 서로 얽힌 단서를 연결하고, 짧은 정답으로 검증 가능한 형태로 수렴하는 능력** 에 있다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 한국어 web browsing agent의 평가 부재다.

기존 LLM benchmark는 크게 세 부류로 나눌 수 있다.

| Type | What it measures | Why it is insufficient here |
| --- | --- | --- |
| Static QA | 주어진 문맥이나 model memory 기반 QA | 실제 web navigation과 search persistence를 거의 보지 못함 |
| Reasoning benchmark | 수학, 코드, 논리 reasoning | 한국 맥락의 entity search와 evidence chaining을 직접 평가하지 못함 |
| General web agent benchmark | 영어 중심 web search 능력 | 한국어 표현, 한국 맥락, local source discovery가 충분히 반영되지 않을 수 있음 |

K-BrowseComp는 이 중 세 번째 축을 한국어와 한국 맥락으로 가져온다. 즉 문제는 단순히 "정답을 아는가"가 아니라, web을 사용해서 답을 찾을 수 있는가다.

이때 중요한 점은 benchmark가 답변 생성 능력만 보지 않는다는 것이다. web browsing agent task에서는 아래 능력이 한꺼번에 필요하다.

- query reformulation
- entity disambiguation
- source discovery
- evidence aggregation
- multi-hop verification
- final answer normalization

정답이 짧게 검증 가능하더라도, 그 정답까지 가는 경로는 길고 불확실할 수 있다. K-BrowseComp의 문제 설정은 바로 이 friction을 한국어 환경에서 측정하려는 시도에 가깝다.

## 1-2. Why previous approaches are insufficient

기존 접근이 부족한 이유는 세 가지다.

첫째, 한국어 benchmark의 많은 부분은 language understanding이나 static knowledge 평가에 머물기 쉽다. 하지만 browsing agent는 language understanding만으로는 충분하지 않다. 검색 query를 바꾸고, 실패한 search path를 버리고, 새 단서를 찾아야 한다.

둘째, 영어 web agent benchmark의 성능을 그대로 한국어 agent 성능으로 해석하기 어렵다. 한국어 web에는 한국어 고유명사, 행정 용어, 기관명, 지역 정보, 뉴스/공공자료 스타일, 검색 엔진의 indexing 편향이 함께 들어간다. 이런 요소는 model 내부 지식보다 browsing policy와 source selection 능력을 더 강하게 요구할 수 있다.

셋째, synthetic benchmark만으로는 충분하지 않다. web browsing task는 문제 생성보다 문제 검증이 더 어렵다. 틀린 단서, 중복 entity, 시간 변화, source drift가 존재하기 때문이다. 그래서 이 논문은 manually constructed and validated subset과 synthetic diagnostic split을 분리한다.

이 구성이 꽤 중요하다. benchmark paper에서 synthetic split을 하나로 섞어 overall score만 내면, 데이터 품질과 model stress test가 뒤섞인다. K-BrowseComp는 300개 verified subset을 main benchmark로 두고, 100개 synthetic split을 targeted stress test로 분리함으로써 이 혼동을 줄이려 한다.

# 2. Core Idea

## 2-1. Main contribution

K-BrowseComp의 핵심 기여는 크게 4가지로 볼 수 있다.

1. **Korean-context web browsing benchmark**
   - 총 400개 문제를 구성한다.
   - 한국어와 한국 맥락에서 agentic browsing capability를 본다.

2. **Manual verified main split**
   - 300개 K-BrowseComp-Verified subset을 native Korean speaker가 수동 구성하고 검증한다.
   - 논문은 이 subset을 main evaluation target으로 둔다.

3. **Synthetic diagnostic split**
   - 100개 synthetic split을 별도로 구성한다.
   - hard few-shot exemplar와 failure-mode-targeted generation을 이용한다.
   - adversarial filtering 뒤에 targeted stress test로 보고한다.

4. **Frontier and Korean model evaluation**
   - frontier LLM은 verified subset에서 30.00-45.67 percent에 머무른다.
   - Korea's Proprietary AI Foundation Model program을 통해 공개된 Korean LLM은 0.00-10.33 percent 범위로 보고된다.
   - synthetic diagnostic split에서는 strongest model이 26.00 percent에 그친다.

이 결과는 benchmark가 단순히 한국어라서 어려운 것이 아니라, web search와 reasoning이 결합된 agentic task라는 점에서 어렵다는 해석을 가능하게 한다.

## 2-2. Design intuition

이 논문의 설계 직관은 BrowseComp류 benchmark를 한국어로 단순 번역하는 것이 아니다. 핵심은 **local context가 search problem의 구조를 바꾼다** 는 점이다.

예를 들어 agent가 어떤 entity를 찾아야 한다고 하자. 영어권 benchmark에서는 global web source와 영어 query가 잘 맞을 수 있다. 하지만 한국어 맥락에서는 같은 entity라도 한글 표기, 영문 표기, 약칭, 기관명 변경, 지역명, 행정 구역 표현이 섞일 수 있다. 그러면 agent는 단순히 web search tool을 호출하는 것이 아니라, 어떤 검색어를 어떤 순서로 시도할지 결정해야 한다.

또 하나의 직관은 solving and creating asymmetry다. 좋은 browsing 문제를 만드는 일은 어렵지만, 강한 few-shot exemplar와 known failure mode를 이용하면 모델이 틀리기 쉬운 synthetic problem을 만들 수 있다. 저자들은 이 비대칭성을 이용해 synthetic diagnostic split을 만든다.

내 해석으로는, K-BrowseComp의 진짜 contribution은 문제 수 자체보다 split design에 있다. verified split은 benchmark의 신뢰성을 담당하고, synthetic split은 model failure mode를 압박하는 stress test 역할을 한다. 이 둘을 분리해 둔 것이 실험 해석에 중요하다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | 한국어와 한국 맥락에서 web browsing agent capability 평가 |
| Benchmark size | 400 problems |
| Main split | 300-problem K-BrowseComp-Verified |
| Diagnostic split | 100-problem synthetic split |
| Human validation | Native Korean speaker 기반 수동 구성 및 검증 |
| Main metric | 문제별 정답 여부 기반 accuracy로 해석 가능 |
| Key comparison | Frontier LLM, Korean LLM, BrowseComp 대비 난도 |
| Difference from prior work | Korean context를 중심으로 web browsing task를 재구성 |

## 3-2. Module breakdown

### 1) K-BrowseComp-Verified

K-BrowseComp-Verified는 300개 문제로 구성된 main split이다. abstract 기준으로 이 subset은 native Korean speaker가 manually constructed and validated 했다.

이 split의 역할은 benchmark의 core validity를 잡는 것이다. web browsing benchmark에서는 문제 자체가 틀리면 model score를 믿기 어렵다. 특히 한국어 web context에서는 source가 바뀌거나, entity가 중복되거나, 답이 시점에 따라 달라질 수 있다. 그래서 manual validation은 선택 사항이 아니라 benchmark quality의 핵심이다.

verified split은 다음 질문에 답한다.

- 실제 사람이 검증한 한국어 browsing task에서 model은 얼마나 잘 푸는가?
- frontier model은 영어 BrowseComp와 비슷한 수준으로 generalize하는가?
- 국내 Korean LLM은 local context에서 competitive한 agent behavior를 보이는가?

논문이 보고한 결과는 꽤 강하다. frontier model도 45.67 percent를 넘지 못하고, Korean LLM은 10.33 percent 이하에 머문다. 즉 이 benchmark는 단순한 localization benchmark가 아니라, 한국어 web agent capability gap을 보여주는 diagnostic benchmark에 가깝다.

### 2) Synthetic diagnostic split

Synthetic split은 100개 문제로 구성된다. 저자들은 hard few-shot exemplar와 failure-mode-targeted generation을 이용해 synthetic problem을 만들고, adversarial filtering을 거쳐 별도 stress test로 보고한다.

이 구성의 장점은 명확하다. verified split은 품질이 높지만 비용이 크다. synthetic split은 비용을 낮추면서도 모델이 자주 실패하는 지점을 압박할 수 있다. 다만 synthetic data는 benchmark의 main score로 섞이면 위험할 수 있다. 생성 모델이 만든 문제는 편향이나 artifact를 가질 수 있기 때문이다.

그래서 K-BrowseComp가 synthetic split을 main split과 분리해 보고하는 점은 좋은 설계다.

- verified split은 benchmark headline score에 적합하다.
- synthetic split은 failure-mode stress test에 적합하다.
- 두 결과를 합쳐 하나의 overall score로 해석하면 안 된다.

이 synthetic split에서 strongest model이 26.00 percent라는 결과는, adversarial filtering이 실제로 난도를 끌어올렸을 가능성을 보여준다. 다만 이 수치는 main benchmark 성능과 직접 비교하기보다, targeted diagnostic signal로 보는 편이 맞다.

### 3) Evaluation protocol

논문 abstract만으로는 exact prompt, browser tool, search backend, grading script의 세부는 모두 확인되지 않는다. 다만 benchmark의 성격상 평가 절차는 아래 구성으로 이해할 수 있다.

1. agent가 web browsing을 수행한다.
2. agent가 최종 short answer를 낸다.
3. reference answer와 비교해 correct 여부를 판정한다.
4. split별 accuracy를 보고한다.

개념적으로 score는 아래처럼 볼 수 있다.

$$
Accuracy = \frac{N_{correct}}{N_{total}}
$$

여기서 중요한 것은 $N_{total}$이 split마다 다르다는 점이다. verified split에서는 $N_{total}=300$이고, synthetic diagnostic split에서는 $N_{total}=100$이다. 따라서 1문항 차이가 각각 0.33 point와 1.00 point에 해당한다.

이 차이는 실험 해석에서 중요하다. synthetic split의 26.00 percent는 26개 정답이라는 뜻으로 볼 수 있고, verified split의 45.67 percent는 137개 안팎의 정답 규모로 해석할 수 있다. 단, exact rounding은 원문 table에서 다시 확인하는 것이 좋다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 model training paper라기보다 benchmark construction paper다. 그래서 여기서의 data recipe는 training data가 아니라 evaluation data 구성이다.

확인 가능한 구성은 아래와 같다.

| Split | Size | Construction | Role |
| --- | ---: | --- | --- |
| K-BrowseComp-Verified | 300 | Native Korean speaker가 수동 구성 및 검증 | Main benchmark |
| Synthetic diagnostic | 100 | Hard few-shot exemplar와 failure-mode-targeted generation | Stress test |
| Total | 400 | Manual + synthetic | Korean-context browsing evaluation |

이 데이터 설계의 핵심은 manual quality와 synthetic scalability를 분리한 것이다. verified split은 작은 대신 신뢰성을 담당하고, synthetic split은 모델 failure mode를 압박한다.

## 4-2. Training strategy

별도의 model training strategy는 핵심이 아니다. 이 논문은 여러 모델을 benchmark에 올려 비교하는 evaluation work다.

다만 benchmark를 실제 post-training이나 agent training에 재사용한다면, 다음 식으로 사용할 수 있다.

- evaluation only
  - agent의 web browsing policy를 측정한다.
  - model별 local-context search ability를 비교한다.

- diagnostic data
  - synthetic split의 failure case를 분석한다.
  - query rewrite, source selection, answer normalization error를 분류한다.

- training signal 후보
  - trajectory가 공개된다면 search-agent RL이나 supervised trajectory distillation에 쓸 수 있다.
  - 하지만 benchmark contamination을 피하려면 train/eval 분리가 필요하다.

K-BrowseComp의 가장 직접적인 활용은 training dataset이 아니라 **eval harness + error taxonomy** 다. 특히 한국어 agent를 개발하는 팀은 이 benchmark를 score competition보다 failure analysis tool로 먼저 쓰는 편이 낫다.

## 4-3. Engineering notes

K-BrowseComp류 benchmark를 실제로 돌릴 때는 paper score보다 engineering details가 더 중요할 수 있다.

1. Browser/search backend 고정
   - 같은 model이라도 search engine, region, language setting에 따라 결과가 달라질 수 있다.
   - 한국어 web search에서는 검색 결과 ranking drift가 score variance를 만들 가능성이 크다.

2. Time-sensitive answer 관리
   - web browsing task는 시간이 지나며 정답이 바뀔 수 있다.
   - benchmark snapshot, access date, accepted answer normalization이 중요하다.

3. Final answer normalization
   - 한국어 고유명사, 숫자, 날짜, 기관명 약칭은 normalization rule이 필요하다.
   - exact match만 쓰면 과소평가가 생길 수 있고, 느슨한 matching을 쓰면 과대평가가 생길 수 있다.

4. Trajectory logging
   - web agent benchmark에서는 final answer만 보면 실패 원인을 알기 어렵다.
   - query sequence, visited pages, cited evidence, intermediate hypotheses를 저장해야 디버깅이 가능하다.

5. Contamination control
   - 문제와 정답이 공개되면 model memory나 retrieval cache에 들어갈 수 있다.
   - 평가용 private split이나 time-based refresh가 필요할 수 있다.

# 5. Evaluation

## 5-1. Main results

abstract 기준으로 확인되는 주요 결과는 아래와 같다.

| Split | Models | Reported result |
| --- | --- | --- |
| K-BrowseComp-Verified | Frontier LLMs including GPT-5.5, DeepSeek-V4-Pro, GLM-5.1 | 30.00-45.67 percent |
| K-BrowseComp-Verified | Korean LLMs released through Korea's Proprietary AI Foundation Model program | 0.00-10.33 percent |
| Synthetic diagnostic | Strongest model | 26.00 percent |

이 결과에서 중요한 포인트는 두 가지다.

첫째, frontier model도 절반을 넘지 못한다. verified subset에서 최고 45.67 percent라는 것은, agentic browsing이 아직 frontier model에게도 안정적인 능력이 아니라는 뜻이다.

둘째, Korean LLM의 낮은 점수는 단순 언어 이해 문제보다 agent pipeline 문제가 클 수 있음을 시사한다. 한국어를 잘한다는 것과 한국 web에서 필요한 정보를 끈질기게 찾는 것은 다른 능력이다.

## 5-2. What really matters in the experiments

이 논문에서 진짜 봐야 할 것은 rank 자체보다 **drop의 성격** 이다.

- frontier model이 BrowseComp 대비 크게 떨어진다.
- Korean LLM은 local language context에서도 충분한 advantage를 보이지 못한다.
- synthetic diagnostic split에서 strongest model도 26.00 percent에 그친다.

이 세 가지를 합치면, K-BrowseComp는 언어 benchmark라기보다 agent benchmark에 가깝다. 즉 한국어 유창성만으로는 충분하지 않고, 아래 capability가 같이 필요하다.

- search query를 여러 단계로 재구성하는 능력
- ambiguous entity를 좁혀 가는 능력
- evidence source를 비교하고 버리는 능력
- 최종 답을 짧고 검증 가능한 형태로 정리하는 능력
- 실패한 path에서 빠져나오는 능력

이 benchmark의 좋은 점은 Korean model에게 유리한 benchmark가 아니라는 데 있다. 오히려 한국어 환경에서도 local model이 자동으로 유리하지 않다는 점을 드러낸다. 이것은 한국어 LLM 개발에서 매우 중요한 신호다. local language capability와 local agent capability를 분리해서 봐야 한다는 뜻이기 때문이다.

# 6. Limitations

1. 원문 abstract만으로는 exact evaluation harness가 충분히 드러나지 않는다.
   - browser tool, search engine, time limit, token budget, retry policy, answer matching rule은 score 해석에 매우 중요하다.
   - 이 부분은 본문 table과 appendix에서 반드시 재확인해야 한다.

2. 400문항은 고품질 benchmark로는 의미 있지만, domain coverage를 완전히 보장하기에는 작을 수 있다.
   - 한국 맥락은 행정, 문화, 법, 교육, 의료, 지역, 상업, 뉴스 등으로 넓다.
   - 어떤 domain이 많이 포함되었는지에 따라 model ranking이 달라질 수 있다.

3. Web benchmark는 time drift에 취약하다.
   - 검색 결과와 web page 내용은 시간이 지나며 변한다.
   - 정답이 바뀌거나 source가 사라지면 benchmark reproducibility가 약해질 수 있다.

4. Synthetic diagnostic split은 main benchmark와 같은 의미로 해석하면 안 된다.
   - failure-mode-targeted generation은 stress test에는 좋지만, real user distribution을 대표한다고 보기 어렵다.
   - 저자들이 별도로 보고하는 이유도 이 점 때문이라고 보는 편이 자연스럽다.

5. Model별 score만으로는 실패 원인을 알기 어렵다.
   - agent가 검색을 못 한 것인지, 좋은 source를 찾고도 답을 잘못 정규화한 것인지, browsing trajectory가 필요하다.
   - trajectory-level 공개가 있으면 benchmark의 연구 가치가 크게 올라갈 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

K-BrowseComp는 한국어 LLM 평가에서 꽤 중요한 기준점을 만든다. 그동안 한국어 benchmark는 번역, 상식, 독해, 수학, instruction following처럼 model 내부 capability 중심으로 많이 논의됐다. 하지만 실제 agent product에서는 model이 모르는 정보를 찾아야 한다.

K-BrowseComp가 중요한 이유는 바로 여기에 있다. 이 benchmark는 한국어 모델이 한국어를 잘하는가보다, 한국어 web에서 필요한 정보를 **agentically 찾을 수 있는가** 를 묻는다.

이 차이는 크다. 한국어 customer support, public information assistant, legal/administrative search assistant, research assistant를 만들려면 model memory보다 browsing policy가 중요해질 때가 많다. K-BrowseComp는 그 지점을 평가하는 출발점으로 쓸 수 있다.

## 7-2. Reuse potential

실무나 연구에서의 재사용 가능성은 아래와 같다.

1. Korean search agent evaluation
   - web browsing agent의 regression test로 쓸 수 있다.
   - model upgrade, search backend 변경, prompt 변경의 영향을 측정할 수 있다.

2. Error taxonomy construction
   - 실패 case를 query failure, source failure, reasoning failure, answer formatting failure로 나눌 수 있다.
   - 이 분해가 agent improvement recipe로 이어진다.

3. RL or SFT data candidate
   - trajectory가 있으면 search trajectory supervision에 사용할 수 있다.
   - 다만 benchmark leakage를 피하려면 training용 derivative data와 evaluation split을 분리해야 한다.

4. Korean-context benchmark design reference
   - K-MetBench, VideoKR, K-BrowseComp처럼 local context를 중심에 둔 benchmark가 늘고 있다.
   - 앞으로는 한국어 benchmark도 단순 language test에서 domain and tool-use test로 넘어갈 가능성이 크다.

## 7-3. Follow-up papers

- BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents
- GrepSeek: Training Search Agents for Direct Corpus Interaction
- LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards
- VideoKR: Towards Knowledge- and Reasoning-Intensive Video Understanding
- K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology

# 8. Summary

- K-BrowseComp는 한국어와 한국 맥락에 grounded된 web browsing agent benchmark다.
- 전체 400문항 중 300개는 native Korean speaker가 수동 구성 및 검증한 K-BrowseComp-Verified subset이다.
- 100개 synthetic diagnostic split은 hard few-shot exemplar와 failure-mode-targeted generation으로 만든 stress test다.
- frontier model도 verified subset에서 30.00-45.67 percent에 머물고, Korean LLM은 0.00-10.33 percent 범위로 보고된다.
- 이 논문의 핵심 가치는 한국어 유창성과 한국어 web agent capability를 분리해서 봐야 한다는 점을 명확히 만든 데 있다.
