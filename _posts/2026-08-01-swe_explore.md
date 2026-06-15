---
layout: single
title: "SWE-Explore Review"
categories: Study-concept
tag: [CodingAgent, SoftwareEngineering, Benchmark]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.07297)

[Code link](https://github.com/Qiushao-E/SWE-Explore-Bench)

[Dataset link](https://huggingface.co/datasets/SWE-Explore-Bench/SWE-Explore-Bench)

SWE-Explore는 coding agent benchmark를 보는 관점을 꽤 깔끔하게 바꾸는 논문이다. 기존 SWE-bench 계열 평가는 보통 issue를 주고 최종 patch가 test를 통과했는지를 본다. 이 방식은 실용적이지만, agent가 왜 실패했는지는 잘 쪼개지지 않는다. 문제를 잘못 읽은 것인지, repo에서 관련 파일을 못 찾은 것인지, 관련 파일은 찾았지만 결정적인 line span을 놓친 것인지, 아니면 patch synthesis가 약한 것인지가 하나의 resolved/unresolved label 안에 섞인다.

SWE-Explore는 이 중에서 repository exploration만 따로 떼어낸다. Agent에게 patch를 쓰라고 하지 않는다. 대신 issue와 repository snapshot을 주고, fixed line budget 안에서 relevant code region의 ranked list를 반환하게 한다. 그런 다음 성공한 독립 repair trajectory들이 실제로 읽었던 line-level context를 ground truth로 삼아 coverage, ranking, context efficiency를 측정한다.

> 한 줄 요약: SWE-Explore는 coding agent의 최종 patch 성공률을 직접 재는 대신, issue 해결 전에 agent가 repo 안에서 필요한 file과 line span을 얼마나 빨리, compact하게, 충분히 찾아내는지를 평가하는 trajectory-grounded repository exploration benchmark다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- coding agent 성능 병목을 patch generation 하나로 뭉개지 않고, exploration, localization, context selection으로 분해한다.
- 성공한 agent trajectory에서 line-level ground truth를 만들기 때문에, 단순 file-level localization보다 더 날카로운 진단이 가능하다.
- 실험 결과가 꽤 직관적이면서도 중요하다. 현대 agent는 correct file은 자주 찾지만, 실제 repair에 필요한 line-level evidence는 여전히 많이 놓친다.
- benchmark와 code, dataset이 공개되어 있어 retrieval, IDE agent, long-context selector, localizer를 같은 contract로 비교하기 좋다.

이 논문의 핵심 메시지는 단순히 새로운 benchmark를 하나 추가했다는 것이 아니다. Software engineering agent를 개선하려면 patch를 더 잘 쓰는 모델만 볼 것이 아니라, agent가 어떤 evidence를 읽고 patch를 쓰는지부터 별도 target으로 최적화해야 한다는 주장에 가깝다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 repository-level coding benchmark가 너무 holistic하다는 점이다.

SWE-bench류 benchmark는 실제 GitHub issue와 repo snapshot을 주고, agent가 patch를 만들어 test를 통과하는지 본다. 이 평가는 현실적인 장점이 크다. 결국 developer agent는 patch를 내야 하고, patch가 test를 통과해야 한다. 하지만 이 방식에는 진단상 한계가 있다.

같은 실패라도 원인은 여러 가지일 수 있다.

1. Issue 이해 실패
   - 자연어 issue에서 요구사항이나 bug condition을 잘못 해석한다.
2. Repository exploration 실패
   - 관련 file이나 module을 찾지 못한다.
3. Line-level context selection 실패
   - file은 맞게 찾았지만, 실제 수정에 필요한 function, config, test span을 놓친다.
4. Patch synthesis 실패
   - evidence는 충분히 봤지만, code edit를 잘못 만든다.
5. Validation 실패
   - patch는 plausible하지만 test나 hidden behavior를 만족하지 못한다.

기존 resolved/unresolved score는 이 원인들을 분리하지 않는다. 그래서 agent가 실패했을 때 더 강한 model이 필요한지, 더 좋은 search policy가 필요한지, 더 좋은 edit policy가 필요한지 판단하기 어렵다.

SWE-Explore는 이 중 두 번째와 세 번째에 집중한다. 즉, agent가 issue를 해결하기 전에 repo에서 어떤 code region을 찾아야 하는가를 별도 문제로 만든다.

## 1-2. Why previous approaches are insufficient

기존 접근은 크게 세 부류로 볼 수 있다.

첫째, final repair benchmark다. SWE-bench, SWE-bench Verified, SWE-bench Multilingual, SWE-bench-Pro 같은 benchmark는 patch success를 중심으로 한다. 이 방식은 end-to-end agent 성능을 잘 보여주지만, exploration 품질을 직접 평가하지 않는다.

둘째, file-level localization benchmark다. Bug localization이나 일부 code search benchmark는 relevant file 또는 function ranking을 본다. 하지만 실제 agent가 patch를 만들 때 중요한 것은 file name만이 아니다. 같은 file 안에서도 어떤 line span을 읽었는지가 중요하다. 큰 file 하나를 통째로 맞혔다고 해서 충분한 evidence를 찾았다고 말하기 어렵다.

셋째, retrieval 기반 context benchmark다. BM25, TF-IDF, dense retriever, reranker, long-context selector를 비교할 수는 있지만, 이들이 실제 repair trajectory에서 쓰인 evidence와 어떻게 연결되는지는 약하다. Query-snippet relevance와 issue repair에 필요한 line-level evidence는 다른 문제다.

SWE-Explore의 문제 설정은 이 공백을 찌른다. 질문은 다음처럼 바뀐다.

> Given an issue and a repository, can an explorer return a compact ranked list of code regions that overlaps the evidence used by successful repair trajectories?

여기서 중요한 점은 patch를 만들지 않는다는 것이다. SWE-Explore는 agent가 고친 결과가 아니라, 고치기 전에 무엇을 읽었는지를 평가한다.

# 2. Core Idea

## 2-1. Main contribution

SWE-Explore의 기여는 크게 4가지다.

1. Exploration을 standalone task로 정의한다.
   - 입력은 issue와 repository snapshot이다.
   - 출력은 ranked list of code regions다.
   - 각 region은 file path와 line range로 구성된다.
   - explorer는 patch를 만들 필요가 없다.

2. Trajectory-grounded line-level ground truth를 만든다.
   - 성공한 독립 repair trajectory들이 실제로 읽은 code region을 수집한다.
   - 여러 successful trajectory에서 반복적으로 등장한 region을 core context 후보로 본다.
   - optional context와 core context를 구분하고, LLM refinement와 human audit를 거친다.

3. Coverage, ranking, efficiency를 함께 평가한다.
   - 단순히 맞는 file을 찾았는지만 보지 않는다.
   - 맞는 line을 얼마나 많이 덮었는지, 얼마나 앞쪽에 배치했는지, 얼마나 compact하게 반환했는지를 본다.

4. Restricted-context repair validation으로 metric validity를 확인한다.
   - explorer가 반환한 context만 patching agent에게 보여준다.
   - 그 상태에서 patch가 test를 통과하는지를 본다.
   - 이 validation은 standard evaluation loop가 아니라, exploration metric이 실제 repair success와 연결되는지 확인하는 sanity check다.

## 2-2. Design intuition

설계 직관은 꽤 명확하다.

Coding agent의 실력은 patch를 쓰는 능력만으로 구성되지 않는다. 큰 repo에서 bug를 고치려면 먼저 issue와 연결된 source, test, config, helper function을 찾아야 한다. 이 단계가 실패하면 아무리 patch generator가 강해도 보이는 context가 부족해서 틀린 수정을 하게 된다.

또 하나 중요한 점은 repository exploration은 단순 retrieval 문제가 아니라는 것이다. BM25나 dense retrieval은 issue text와 비슷한 chunk를 찾을 수는 있지만, software repository는 symbol relation, call path, test relation, config relation이 얽혀 있다. 실제 agent는 grep, file view, test inspection, stack trace reading, code graph traversal 같은 multi-step action으로 evidence를 모은다.

SWE-Explore는 바로 이 behavior를 평가 가능한 형태로 압축한다. Agent가 어떤 tool을 썼는지는 다양할 수 있지만, 최종적으로 반환해야 하는 artifact는 동일하다.

- file path
- start line
- end line
- rank order

이 contract 덕분에 sparse retriever, dense retriever, IDE agent, CLI coding agent, academic localizer를 같은 scoreboard에 올릴 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | repository exploration을 patch generation과 분리해서 평가 |
| Input | issue text, repository snapshot |
| Output | ranked list of file path and line range |
| Ground truth | successful repair trajectories에서 추출한 line-level core context |
| Dataset scale | 848 issues, 203 repositories, 10 programming languages |
| Source benchmarks | SWE-bench Verified, SWE-bench-Pro, SWE-bench Multilingual |
| Main metrics | precision, recall, hit file, hit region, nDCG@500, first useful hit, context efficiency, noise rate |
| Validation | selected context만 보여주는 restricted-context repair protocol |
| Release | paper, code, dataset 공개 |

## 3-2. Module breakdown

### 1) Task formulation

SWE-Explore의 task는 다음처럼 볼 수 있다.

1. Issue와 repository snapshot이 주어진다.
2. Explorer는 repo를 읽거나, 검색하거나, long-context selection을 수행한다.
3. Explorer는 최대 K개의 ranked region을 반환한다.
4. 각 region은 repository-relative path와 line interval이다.
5. Evaluator는 반환 region을 line-level ground truth와 비교한다.

논문 실험에서는 K=5를 사용한다. 이유는 refined ground truth 기준으로 instance당 core region 수가 평균 약 4.7개이기 때문이다. 즉 agent에게 대략 ground truth 크기와 비슷한 수의 region을 고르게 해서, 너무 많이 읽는 방식으로 점수를 올리는 것을 막는다.

### 2) Ground-truth construction

Ground truth는 사람이 처음부터 annotation하지 않는다. 이 부분이 SWE-Explore의 가장 중요한 engineering choice다.

Pipeline은 대략 다음과 같다.

1. 기존 repository-level benchmark에서 issue와 repo snapshot을 가져온다.
2. strong LLM agent들이 실제로 issue를 해결한 successful trajectory를 모은다.
3. 각 trajectory에서 read action을 추출한다.
4. read action을 file path와 line interval로 normalize한다.
5. 여러 successful trajectory가 공통으로 읽은 region을 core context 후보로 잡는다.
6. 일부 model-specific optional context를 LLM refinement로 조정한다.
7. 최종 refined ground truth를 human audit로 검증한다.

이 방식은 완전한 oracle annotation은 아니다. 하지만 실제로 issue를 성공적으로 해결한 agent들이 읽은 evidence를 기반으로 하므로, 단순 static file label보다 repair behavior에 더 가깝다.

### 3) Core and optional context

SWE-Explore는 모든 read region을 동일하게 다루지 않는다.

- Core context: 여러 successful trajectory에서 반복적으로 등장하거나, 해결에 직접적으로 필요한 것으로 정제된 region이다.
- Optional context: 특정 model이나 특정 route에서 도움을 준 보조 evidence다.

Scoring target은 core context다. Context efficiency를 계산할 때는 core와 optional을 함께 고려해, 완전히 off-target인 context와 보조적으로 타당한 context를 구분한다.

이 구분이 중요하다. Software debugging에는 여러 valid route가 있다. 어떤 agent는 test부터 보고, 어떤 agent는 implementation부터 볼 수 있다. 특정 route에만 등장한 region을 모두 wrong으로 보면 평가가 지나치게 좁아진다. 반대로 모든 read를 core로 넣으면 ground truth가 너무 넓어진다. SWE-Explore는 이 둘 사이에서 core와 optional을 나누는 절충을 택한다.

### 4) Unified explorer interface

SWE-Explore의 실용적 장점은 output contract가 단순하다는 점이다.

| Explorer type | Example | Converted output |
| --- | --- | --- |
| Sparse retrieval | BM25, TF-IDF | ranked chunks mapped to line ranges |
| Dense retrieval | Potion, RAG | retrieved chunks mapped to line ranges |
| General coding agent | Claude Code, Codex, OpenHands, Mini-SWE-Agent, AweAgent | searched/read regions normalized to line ranges |
| Academic localizer | AutoCodeRover, LocAgent, OrcaLoca, CoSIL | file, function, or graph-search output normalized to line ranges |

이렇게 하면 method 내부가 서로 달라도 평가 단위는 동일해진다. 어떤 system은 shell로 grep을 하고, 어떤 system은 IDE tool을 쓰고, 어떤 system은 static retrieval을 할 수 있다. 하지만 마지막에는 ranked region list를 내야 한다.

### 5) Restricted-context repair bridge

SWE-Explore의 primary score는 exploration metric이다. 하지만 논문은 이 metric이 실제 repair success와 연결되는지 확인하기 위해 restricted-context validation을 추가한다.

절차는 다음과 같다.

1. Explorer가 top-K region을 반환한다.
2. Repository에서 선택된 line interval만 patching agent에게 보이게 한다.
3. 선택되지 않은 file이나 line은 숨기거나 blank placeholder로 처리한다.
4. 고정된 Mini-SWE-Agent patcher가 patch를 생성한다.
5. 원래 benchmark harness로 patch가 resolved되는지 확인한다.

이 protocol에서 중요한 것은 patcher가 고정된다는 점이다. 변수는 explorer가 제공한 context뿐이다. 그래서 resolve rate 차이를 patching 능력보다 context selection quality로 해석할 수 있다.

# 4. Training / Data / Recipe

## 4-1. Data

SWE-Explore는 training recipe 논문이라기보다 benchmark construction 논문이다. 따라서 핵심은 model training data가 아니라 evaluation dataset 구성이다.

| Component | Detail |
| --- | --- |
| Source benchmarks | SWE-bench Verified, SWE-bench-Pro, SWE-bench Multilingual |
| Retained issues | 848 |
| Repositories | 203 open-source repositories |
| Programming languages | 10 |
| Mean issue length | 191.2 words |
| Mean ground-truth files | 4.3 |
| Mean context regions | 4.7 |
| Mean ground-truth lines | 1,578 |
| Mean source trajectories | 2.9 |
| Mean non-test codebase files | 759 |
| Mean non-test codebase lines | 179.6K |

Dataset card 기준으로 Hugging Face에는 default split train 848 rows가 공개되어 있다. 각 instance는 instance_id, repo path, repo snapshot metadata, ground_truth, read_step_info, meta 등을 포함한다. 즉 단순 issue list가 아니라, trajectory provenance와 line-level target을 포함한 benchmark artifact다.

## 4-2. Evaluation recipe

SWE-Explore의 evaluation recipe는 다음처럼 요약할 수 있다.

1. Benchmark instance를 로드한다.
2. Repo snapshot을 local path에 준비한다.
3. Explorer를 실행해 top-K ranked regions를 만든다.
4. Output을 path와 closed line interval 형식으로 normalize한다.
5. Invalid path, empty interval, repo 바깥 region을 제거한다.
6. Ground truth와 비교해 per-instance metric을 계산한다.
7. Instance average로 aggregate score를 만든다.

논문과 repo 기준으로 사용할 수 있는 explorer family는 다음과 같다.

- Local retrieval: bm25, tfidf, potion, rag, embed, swerank
- Simple baselines: oracle, random, simple_rule
- Agentic CLIs: claude_code, cursor
- Academic agents: autocr, cosil, locagent, orcaloca, mini_swe_agent, awe_agent

실험 table에서는 Claude Code, Codex, OpenHands, Mini-SWE-Agent, AweAgent 같은 general-purpose coding agent와 AutoCodeRover, LocAgent, OrcaLoca, CoSIL 같은 localizer를 비교한다.

## 4-3. Metrics

SWE-Explore metric은 크게 4개 축으로 볼 수 있다.

| Axis | Metric | Meaning |
| --- | --- | --- |
| Coverage | precision, recall, F1 | predicted line과 core line의 overlap |
| Coarse hit | HitFile, HitRegion | correct file 또는 region neighborhood에 도달했는지 |
| Ranking | nDCG@500, Rec@100, Rec@500, FUH | useful evidence가 budget 안에서 얼마나 앞에 나오는지 |
| Efficiency | Context Efficiency, NoiseRate | context가 compact하고 off-target region이 적은지 |

여기서 특히 중요한 것은 line budget이다. nDCG@500은 단순 rank cutoff가 아니라 line budget을 고려한다. 너무 큰 region을 앞에 두면 budget을 빨리 소진하므로, 뒤에 더 좋은 evidence가 있어도 score가 낮아질 수 있다. 이 설계는 실제 agent context window와 잘 맞는다. Repository exploration은 찾았느냐만큼, 몇 줄 안에 넣었느냐가 중요하기 때문이다.

# 5. Evaluation

## 5-1. Main results

가장 큰 결과는 네 가지다.

### 1) Agentic explorer는 classical retrieval보다 뚜렷하게 위에 있다

BM25, TF-IDF, Potion 같은 retrieval baseline은 대부분 metric에서 Random에 가까운 낮은 점수를 보인다. 반면 Claude Code, Codex, OpenHands, Mini-SWE-Agent, AweAgent 같은 agentic explorer는 HitFile, nDCG@500, FUH, Context Efficiency에서 훨씬 높은 operating point를 만든다.

예를 들어 Table 6 기준 BM25는 HitFile 0.079, nDCG@500 0.132, Context Efficiency 0.087에 그친다. 반면 Claude Code는 HitFile 0.667, nDCG@500 0.938, Context Efficiency 0.829를 보인다. 이 차이는 issue text와 lexical similarity만으로 repo exploration을 해결하기 어렵다는 것을 보여준다.

### 2) 현대 agent도 line-level recall은 낮다

더 흥미로운 결과는 agentic explorer가 좋은데도 완전하지 않다는 점이다. General-purpose coding agent들은 file-level hit와 ranking은 높지만, line-level recall은 낮다.

대표 수치를 보면 다음과 같다.

| Explorer | HitFile | Rec_line | nDCG@500 | CtxEff |
| --- | ---: | ---: | ---: | ---: |
| OpenHands | 0.645 | 0.179 | 0.867 | 0.737 |
| Mini-SWE-Agent | 0.640 | 0.151 | 0.885 | 0.754 |
| AweAgent | 0.682 | 0.140 | 0.954 | 0.829 |
| Claude Code | 0.667 | 0.154 | 0.938 | 0.829 |
| Codex | 0.649 | 0.194 | 0.901 | 0.762 |

이 패턴은 이 논문의 핵심 메시지다. Agent들은 맞는 file 근처에는 잘 간다. 하지만 성공한 repair trajectory가 실제로 읽은 line-level evidence를 충분히 덮지는 못한다. 즉 "맞는 동네"까지 가는 능력과 "고칠 때 필요한 줄"을 찾는 능력은 다르다.

### 3) CoSIL은 recall frontier를 다르게 만든다

Academic localizer 결과도 흥미롭다. AutoCodeRover는 precision 0.680으로 높지만 Rec_line 0.233, HitFile 0.280으로 conservative하다. OrcaLoca는 NoiseReg 0.003으로 거의 noise가 없지만, Rec_line 0.033으로 relevant span을 많이 놓친다.

반면 CoSIL은 Rec_line 0.788, F1 0.602로 non-oracle 중 가장 높은 line-level recall을 보인다. 다만 NoiseReg 0.471도 높다. 즉 CoSIL은 더 넓게 읽어서 recall을 얻는 대신, context noise를 더 많이 포함한다.

이 결과는 중요한 시사점을 준다. High-recall exploration은 단순히 더 강한 base LLM을 붙인다고 해결되는 문제가 아닐 수 있다. Iterative code-graph search처럼 search mechanism 자체를 바꾸는 접근이 필요할 수 있다.

### 4) Exploration metric은 downstream repair와 연결된다

Restricted-context validation에서는 explorer가 반환한 context만 patcher에게 보여준다. Table 3 기준 resolve rate는 다음과 같다.

| Explorer | Resolve Rate |
| --- | ---: |
| Oracle | 59.7 |
| CoSIL | 59.3 |
| Codex | 50.3 |
| Mini-SWE-Agent | 50.0 |
| Claude Code | 48.0 |
| OpenHands | 47.7 |
| AutoCodeRover | 44.7 |
| LocAgent | 44.7 |
| BM25 | 12.7 |
| Random | 4.7 |

또 Table 4에서는 Context Efficiency가 downstream resolve rate와 Pearson r=0.950, Rec@100이 Spearman rho=0.845 수준으로 강하게 연결된다고 보고한다. 이것은 exploration metric이 단순 proxy가 아니라 실제 repair behavior와 꽤 강하게 맞물린다는 근거다.

## 5-2. What really matters in the experiments

### 1) File hit만 보면 병목을 놓친다

HitFile은 여전히 유용하다. 관련 file에 도달하지 못하면 repair는 거의 불가능하다. 하지만 HitFile만 보면 line-level evidence를 충분히 찾았는지 알 수 없다. Claude Code와 Codex는 HitFile이 각각 0.667, 0.649로 높지만, Rec_line은 각각 0.154, 0.194다. 이 간극이 SWE-Explore가 line-level target을 둔 이유다.

### 2) Compact context가 중요하다

Context Efficiency가 downstream resolve rate와 강하게 연결된다는 점은 실무적으로 중요하다. Agent가 많이 읽는 것만으로는 부족하다. 적은 context 안에 핵심 evidence를 넣어야 patcher가 제대로 쓸 수 있다. 실제 coding workflow에서도 너무 많은 irrelevant file을 context에 넣으면 model이 길을 잃는다.

### 3) Missing core evidence가 noise보다 더 아프다

Controlled context degradation 실험은 patcher가 redundant irrelevant context보다 missing relevant context에 더 민감하다는 쪽으로 해석된다. 핵심 evidence가 충분히 보이면 moderate noise는 견딜 수 있지만, 핵심 evidence가 빠지면 patch success가 크게 떨어진다.

이 결과는 context selection 연구에서 중요한 trade-off를 준다. Precision을 조금 더 높이는 것보다, core evidence coverage를 올리는 것이 더 중요할 수 있다. 물론 context window가 제한적이면 둘 다 중요하지만, 이 논문 결과만 보면 현재 agent의 병목은 over-reading보다 under-covering에 더 가깝다.

### 4) Base model 교체만으로 bottleneck이 사라지지 않는다

Table 5는 같은 Mini-SWE-Agent scaffold에 다른 LLM을 넣어 비교한다. GPT-5.4와 GPT-5.4-mini가 강한 tier를 형성하지만, 모든 LLM에서 file-level hit가 line-level recall보다 훨씬 높다는 큰 패턴은 유지된다. 즉 더 강한 model을 넣으면 operating point는 움직이지만, exploration mechanism 자체의 한계는 남는다.

# 6. Limitations

1. Trajectory-derived ground truth는 완전한 necessity proof가 아니다.
   - 성공한 agent들이 읽은 region은 유용한 evidence일 가능성이 높다.
   - 하지만 다른 valid solution path가 다른 evidence를 사용할 수 있다.
   - 따라서 ground truth는 empirical approximation으로 읽어야 한다.

2. Solved instances 중심의 selection bias가 있다.
   - SWE-Explore는 최소한 일부 strong agent가 성공한 trajectory를 필요로 한다.
   - 완전히 unsolved issue나 agent가 아직 풀지 못하는 distribution은 덜 대표될 수 있다.

3. Successful trajectory 품질에 evaluation이 의존한다.
   - trajectory가 특정 agent family의 search habit을 반영할 수 있다.
   - 예를 들어 어떤 agent는 test를 많이 읽고, 어떤 agent는 implementation을 많이 읽는다.
   - 이 bias가 core/optional context 구성에 영향을 줄 수 있다.

4. Restricted-context validation은 standard benchmark가 아니라 sanity check다.
   - patcher, prompt, tool budget, context materialization 방식에 따라 resolve rate가 달라질 수 있다.
   - 따라서 Table 3 resolve rate를 일반 SWE-bench leaderboard처럼 읽으면 안 된다.

5. Line-level metric은 useful context의 semantics를 완전히 이해하지 않는다.
   - 어떤 line은 읽지 않아도 infer할 수 있고, 어떤 line은 주변 context와 함께 봐야 의미가 있다.
   - line interval overlap은 좋은 operational target이지만, semantic sufficiency를 완벽히 측정하지는 않는다.

6. Cost가 작지 않다.
   - Agentic explorer 실행과 restricted-context validation은 LLM call과 executable harness run을 요구한다.
   - benchmark 연구에는 좋지만, 모든 실험을 빠르게 반복하기에는 비용이 부담될 수 있다.

7. 아직 training target으로 바로 쓰기에는 조심할 부분이 있다.
   - SWE-Explore score를 reward로 쓰면 agent가 benchmark의 region contract에 과적합할 수 있다.
   - 예를 들어 whole-file emission으로 Rec_line을 올리거나, specific benchmark source의 pattern을 외우는 식의 shortcut이 생길 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 좋은 이유는 coding agent를 "patch generator"가 아니라 "read before edit system"으로 본다는 점이다. 실제 개발자는 코드를 바로 고치지 않는다. issue를 읽고, stack trace를 보고, 관련 test를 찾고, implementation을 따라가고, 주변 API contract를 확인한 뒤 patch를 만든다. Coding agent도 결국 이 과정을 해야 한다.

지금까지 많은 benchmark는 이 전체 loop를 final pass rate 하나로 봤다. SWE-Explore는 그중 reading과 evidence selection을 평가 대상으로 만든다. 이건 agent 연구에서 꽤 중요한 방향이다. Agent가 실패했을 때 patch model을 키울지, retriever를 바꿀지, code graph search를 넣을지, memory를 바꿀지 판단하려면 exploration score가 필요하기 때문이다.

특히 실무적으로는 아래 질문에 바로 연결된다.

- IDE agent가 실제로 충분한 context를 읽고 있는가?
- RAG 기반 code assistant가 correct file은 찾지만 decisive span은 놓치고 있지 않은가?
- Long-context model에 repo를 많이 넣는 것이 실제로 patch success를 올리는가, 아니면 compact evidence selection이 더 중요한가?
- SWE-bench score가 낮은 이유가 edit policy 때문인가, context policy 때문인가?

SWE-Explore는 이 질문들을 별도 실험으로 분해할 수 있게 해준다.

## 7-2. Reuse potential

재사용해볼 만한 포인트는 다음과 같다.

1. Agent pipeline을 explorer와 patcher로 분리해 평가
   - 현재 agent의 end-to-end score만 보지 말고, context selection output을 따로 저장한다.
   - 같은 patcher에 다른 explorer output을 넣어 비교하면 병목을 더 잘 찾을 수 있다.

2. Internal codebase benchmark에 line-level target 추가
   - 사내 repo에서는 human developer의 debug trace, PR review trace, test failure investigation log를 이용해 유사한 ground truth를 만들 수 있다.
   - production bug triage에서도 issue -> relevant spans ranking을 별도 metric으로 둘 수 있다.

3. Code RAG 평가에 Context Efficiency 도입
   - 단순 recall@K보다 "맞는 줄을 얼마나 compact하게 넣었는가"가 중요하다.
   - context window cost가 큰 code assistant에서는 이 metric이 직접적인 product metric에 가깝다.

4. Whole-file retrieval과 span retrieval의 trade-off 분석
   - whole-file retrieval은 recall을 올릴 수 있지만 context cost가 커진다.
   - function-level 또는 line-level span selection은 compact하지만 evidence를 놓칠 수 있다.
   - SWE-Explore는 이 trade-off를 수치로 보기 좋은 framework다.

5. Search mechanism ablation
   - BM25, dense retrieval, AST graph, call graph, test relation, grep, LLM planning을 각각 켜고 끄며 같은 ranked-region contract로 비교할 수 있다.

## 7-3. Follow-up papers

- SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
- SWE-bench Verified
- SWE-bench Multilingual
- SWE-bench-Pro
- ContextBench: A Benchmark for Context Retrieval in Coding Agents
- LocAgent: Graph-guided LLM Agents for Code Localization
- AutoCodeRover
- CoSIL: Issue Localization via LLM-driven Iterative Code Graph Searching
- CodeScout

# 8. Summary

- SWE-Explore는 coding agent의 repository exploration 능력을 patch generation과 분리해 평가하는 benchmark다.
- 848 issues, 203 repositories, 10 languages를 포함하고, successful repair trajectory에서 line-level ground truth를 만든다.
- 현대 agent들은 relevant file은 꽤 잘 찾지만, line-level evidence coverage는 여전히 낮다.
- Context Efficiency, Rec@100, HitFile 같은 metric은 restricted-context downstream repair와 강하게 연결된다.
- 핵심 시사점은 coding agent 개선의 병목이 patch synthesis만이 아니라, repo를 어떻게 읽고 evidence를 어떻게 compact하게 고르는가에 있다는 점이다.
