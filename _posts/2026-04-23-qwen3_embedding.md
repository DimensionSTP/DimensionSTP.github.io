---
layout: single
title: "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models Review"
categories: Study-concept
tag: [Qwen3, Embedding, Reranking]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2506.05176)

Qwen3 Embedding은 "임베딩 모델이 또 하나 좋아졌다" 정도로 읽으면 아까운 기술 보고서다. 이 리포트의 진짜 흥미로운 지점은 **text embedding과 reranking을 각각 별도 작은 모델 문제로 보지 않고, foundation LLM에서 파생된 두 개의 task interface**로 재정의한다는 데 있다. 즉, "좋은 encoder를 새로 설계하자"가 아니라 **Qwen3라는 foundation model을 최대한 유지한 채, 어떤 입력 형식과 어떤 stage-wise recipe를 붙이면 강한 embedder와 reranker가 되는가**를 묻는다.

특히 요즘 RAG, agent search, code retrieval에서는 생성 모델만큼이나 retrieval stack이 중요해졌는데, 실제로는 여전히 retriever/reranker가 블랙박스 API이거나 작은 전용 encoder로만 취급되는 경우가 많다. 그런데 실무에서는 오히려 반대로, **multilingual understanding, instruction following, long-context handling을 이미 잘하는 foundation model을 검색용 표현 모델로 어떻게 전환할 것인가**가 더 중요한 문제일 수 있다.

Qwen3 Embedding은 그 점에서 꽤 인상적이다. embedding 모델은 causal LLM의 **마지막 `[EOS]` hidden state**를 그대로 표현으로 쓰고, reranker는 **같은 backbone을 yes/no relevance scorer**로 바꾼다. 그리고 성능 차이는 새 pooling 블록이나 복잡한 bidirectional encoder 변형보다, **synthetic weak supervision -> high-quality supervised tuning -> checkpoint merging** 같은 recipe에서 끌어낸다. 그래서 이 논문은 leaderboard 발표라기보다 **foundation-model-based retrieval recipe report**로 읽는 편이 더 맞다.

> 한 줄 요약: Qwen3 Embedding은 Qwen3 dense foundation model을 기반으로, **instruction-aware EOS embedding + yes/no point-wise reranking + synthetic-data-driven multi-stage training + model merging**을 결합해 multilingual/code retrieval과 reranking 성능을 끌어올린 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- embedding과 reranking을 **같은 foundation model family에서 파생하는 방식**을 꽤 선명하게 보여준다.
- 성능 향상의 핵심을 architecture novelty보다 **synthetic data, supervision staging, model merging** 쪽에서 설명한다.
- multilingual / Chinese / code / retrieval / reranking을 함께 다루기 때문에, **실서비스 검색 파이프라인 관점**에서 참고할 포인트가 많다.

내가 보기엔 이 논문의 핵심 메시지는 단순하다. **좋은 embedding model은 꼭 encoder-like 구조 변형에서만 나오는 것이 아니고, foundation model을 어떤 task interface와 어떤 data factory 위에 올려놓느냐로도 크게 달라질 수 있다.** Qwen3 Embedding은 바로 그 "interface + recipe" 쪽을 강하게 밀어붙인 사례다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 **text embedding과 reranking을 multilingual, task-aware, instruction-aware하게 동시에 강하게 만들기 어렵다**는 점이다.
- 최근 retrieval은 단순 semantic search를 넘어, RAG, tool-using agent, multilingual retrieval, code retrieval, long-document retrieval까지 포괄하게 됐다.
- 따라서 embedding 모델은 retrieval만 잘해서는 부족하고, **classification, clustering, STS, bitext mining**까지 어느 정도 범용적으로 커버해야 한다.
- reranking도 마찬가지다. 1차 dense retrieval 뒤에서 더 정밀한 relevance 판정을 해야 하므로, 단순 유사도 계산보다 **instruction-conditioned relevance judgment**가 중요해진다.
- 결국 이 논문은 "좋은 임베딩 모델 하나 만들기"가 아니라, **foundation LLM 위에서 embedding과 reranking을 각각 어떻게 task-aware하게 구현할 것인가**를 다룬다.

## 1-2. Why previous approaches are insufficient

- 기존 embedding의 전통적 기반은 BERT 계열 encoder였다. 이런 구조는 문장 표현 학습에는 강하지만, **LLM이 갖는 instruction following, multilingual transfer, reasoning prior**를 그대로 활용하기 어렵다.
- 반대로 LLM 기반 embedding 연구들은 강한 결과를 보이더라도, 종종 **proprietary API, 불투명한 data recipe, 혹은 backbone 변경**에 의존하는 경우가 있다.
- weak supervision 데이터도 기존에는 Q&A 포럼, 논문, open corpus에서 pair를 수집하는 방식이 많았는데, 이 경우 **task / language / difficulty / length를 세밀하게 제어하기 어렵다.**
- reranking 역시 prompt-only zero-shot과 supervised reranker가 섞여 발전했지만, **foundation model 계열의 임베딩 모델과 같은 철학으로 함께 설계된 end-to-end recipe**는 상대적으로 덜 정리되어 있었다.
- 결국 기존 접근의 한계는 한 가지 블록이 부족해서라기보다, **backbone 활용 방식 / pair data 생성 방식 / supervision stage / reranking interface**가 하나의 시스템으로 묶여 있지 않았다는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- Qwen3 Embedding의 핵심 기여는 새로운 거대 구조를 제안하는 것이 아니라, **Qwen3 foundation model을 검색용 representation model과 reranker로 변환하는 recipe를 정리한 것**이다.
- 첫째, embedding 모델은 **causal attention을 유지한 채** 입력 끝의 `[EOS]` hidden state를 최종 embedding으로 사용한다.
- 둘째, query 쪽에는 `{Instruction} {Query}<|endoftext|>` 형식을 써서 **instruction-aware retrieval**를 직접 지원한다.
- 셋째, reranker는 query / document / instruction을 한 context 안에 넣고, 다음 토큰이 `"yes"`일지 `"no"`일지의 확률로 relevance score를 계산한다.
- 넷째, embedding 모델은 **3-stage recipe**를 쓴다.
  1. large-scale synthetic weak supervision
  2. high-quality supervised fine-tuning
  3. slerp 기반 model merging
- 다섯째, reranking 모델은 **2-stage recipe**를 쓴다.
  1. high-quality supervised fine-tuning
  2. model merging
- 여섯째, Qwen3-32B를 data synthesizer로 활용해 **retrieval, bitext mining, classification, STS**를 아우르는 대규모 pair data를 만든다.

## 2-2. Design intuition

- 이 논문의 설계 직관은 꽤 명확하다. **foundation model이 이미 가지고 있는 언어 이해와 instruction following을 최대한 보존하고, 검색용 적응은 입력 인터페이스와 데이터로 해결하자**는 것이다.
- 그래서 embedding 쪽은 bidirectional encoder로 아예 바꾸지 않고, **Qwen3의 기본 causal LLM 구조를 그대로 유지**한다.
- reranking 쪽도 별도 classification head를 붙이는 대신, **LLM이 가장 잘하는 next-token prediction을 relevance judgment로 재활용**한다.
- weak supervision은 generalization을 넓히고, supervised stage는 품질을 다듬고, model merging은 분포별 편차를 줄인다. 즉 이 논문은 성능 향상을 한 번의 fine-tuning으로 해결하지 않고, **서로 다른 역할을 가진 stage들의 조합**으로 본다.
- 또 하나 흥미로운 점은 foundation model이 **backbone이면서 동시에 data generator**라는 점이다. 이건 단순 self-training이라기보다, LLM을 통해 retrieval pair의 형태 자체를 더 정교하게 통제할 수 있다는 발상에 가깝다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Qwen3 dense foundation model을 embedding / reranking 전용 모델로 전환하는 것 |
| Key module | instruction-aware EOS embedding + yes/no point-wise reranking + multi-stage training |
| Core design principle | backbone은 크게 바꾸지 않고, task interface와 data recipe를 통해 성능을 끌어올림 |
| Difference from prior work | foundation model을 backbone과 synthetic data engine으로 동시에 활용하고, model merging까지 포함한 end-to-end recipe를 제시 |

## 3-2. Module breakdown

### 1) Instruction-aware EOS embedding

- embedding 모델은 **Qwen3 dense base model** 위에 올라간다.
- 구조적으로 가장 눈에 띄는 점은 의외로 단순하다는 것이다.
- query 입력 형식은 아래와 같다.

```text
{Instruction} {Query}<|endoftext|>
```

- document는 별도 instruction 없이 그대로 넣고, 최종 embedding은 **마지막 layer의 `[EOS]` hidden state**에서 읽어낸다.
- 즉 이 논문은 embedding quality를 위해 새로운 pooling module이나 bidirectional attention을 핵심으로 밀지 않는다.
- 오히려 **Qwen3의 기존 표현 능력을 instruction-aware asymmetric retrieval 형식으로 잘 꺼내 쓰는 것**이 더 중요하다고 본다.
- model spec 상으로는 embedding 모델이 **custom dimension support(MRL support)**도 제공한다. 다만 본문은 이 기능 자체를 깊게 분석하기보다, deployment-friendly option으로만 제시한다.

### 2) Yes/No next-token reranking

- reranker는 query, document, instruction을 **하나의 context** 안에 넣는다.
- 논문은 이를 point-wise reranking으로 설명하고, 입력을 chat template으로 감싼다.
- 핵심은 relevance score를 별도 classifier로 만들지 않고, **다음 토큰이 `"yes"`인지 `"no"`인지의 확률**로 계산한다는 점이다.

```text
system:
Judge whether the Document meets the requirements based on the Query and the Instruct provided.
Note that the answer can only be "yes" or "no".

user:
<Instruct>: {Instruction}
<Query>: {Query}
<Document>: {Document}

assistant:
<think>...</think>
```

- score는 `P(yes)`와 `P(no)`를 softmax처럼 정규화한 형태로 계산된다.
- 즉 reranker도 본질적으로는 **LLM language modeling head를 재활용한 relevance scorer**다.
- 이 방식은 깔끔하다. foundation chat model이 이미 가진 instruction-following prior를 그대로 relevance 판단으로 연결할 수 있기 때문이다.

### 3) Improved contrastive loss with false-negative control

- embedding 모델 학습에는 **InfoNCE 기반의 개선된 contrastive loss**를 사용한다.
- positive pair만 보는 단순한 형태가 아니라,
  - hard negatives
  - in-batch queries
  - in-batch documents
  - query vs other documents  
  를 함께 negative pool에 넣는다.
- 그리고 false negative 문제를 줄이기 위해, 특정 쌍의 similarity가 positive score보다 너무 높거나 같은 positive document일 경우 **mask factor**를 둬서 손실 계산에서 제외한다.
- 즉 이 논문은 "synthetic data를 많이 만들었다"에만 의존하지 않고, **negative construction과 false-negative suppression**까지 함께 고려한다.

### 4) 3-stage embedding recipe / 2-stage reranking recipe

- embedding 모델의 training pipeline은 세 단계다.
  1. **Weakly supervised pre-training** with large-scale synthetic pair data
  2. **Supervised fine-tuning** with high-quality synthetic + labeled data
  3. **Model merging** with sampled checkpoints from stage 2
- 반면 reranking 모델은 1단계 weak supervision 없이,
  1. **High-quality supervised fine-tuning**
  2. **Model merging**
  만 수행한다.
- 이 차이는 중요하다. 저자들은 embedding 쪽에서는 대규모 noisy-but-diverse synthetic pair가 generalization에 크게 기여한다고 보고, reranking 쪽은 오히려 **고품질 supervised signal로 바로 들어가는 편**이 낫다고 본다.

# 4. Training / Data / Recipe

## 4-1. Data

Qwen3 Embedding에서 가장 인상적인 부분은 아키텍처보다도 **synthetic dataset 설계**다.

- synthetic pair data는 다음 네 가지 큰 task category를 포괄한다.
  - retrieval
  - bitext mining
  - classification
  - semantic textual similarity(STS)
- 데이터 생성에는 **Qwen3-32B**가 사용된다.
- retrieval data를 만들 때는 단순히 문서에서 질의를 생성하는 것이 아니라, 각 문서에 대해 **잠재적 사용자 역할(role)** 을 부여한다.
- 구체적으로는 retrieval model을 이용해 role library에서 **top-5 role candidates**를 찾고, 이 role 후보와 문서를 함께 prompt에 넣어 적절한 query를 생성하게 한다.
- 그리고 prompt 안에서 아래 차원들을 제어한다.
  - query type: keyword / factual / summary / judgment
  - query length
  - difficulty
  - language
- 이건 꽤 중요하다. 단순 open web pair 수집보다, **원하는 데이터 분포를 더 직접적으로 설계**할 수 있기 때문이다.

최종 규모도 크다.

- 1단계 weak supervision용 synthetic data는 **약 150M pairs**다.
- 이후 cosine similarity 기반 filtering을 통해 **약 12M high-quality pairs**를 2단계 supervised training용으로 선별한다.
- filtering 기준은 무작위 샘플에서 cosine similarity가 **0.7 초과**인 pair를 유지하는 방식이다.

다만 여기서 주의할 점도 있다.

- 본문 main text는 synthetic pipeline은 비교적 자세히 설명하지만, 2단계 supervised training에 들어가는 **모든 labeled dataset 구성과 mixing ratio를 조목조목 열거하지는 않는다.**
- 그래서 실무에서 완전 재현을 목표로 한다면, technical report 본문만이 아니라 **repo / training script / supplementary material**까지 함께 보는 편이 낫다.

## 4-2. Training strategy

모델 스펙은 다음처럼 정리할 수 있다.

| Model | Size | Layers | Context | Embedding Dim | Notes |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | 0.6B | 28 | 32K | 1024 | MRL support, instruction-aware |
| Qwen3-Embedding-4B | 4B | 36 | 32K | 2560 | MRL support, instruction-aware |
| Qwen3-Embedding-8B | 8B | 36 | 32K | 4096 | MRL support, instruction-aware |
| Qwen3-Reranker-0.6B | 0.6B | 28 | 32K | - | instruction-aware |
| Qwen3-Reranker-4B | 4B | 36 | 32K | - | instruction-aware |
| Qwen3-Reranker-8B | 8B | 36 | 32K | - | instruction-aware |

학습 단계는 embedding과 reranking이 다르다.

### Embedding
- **Stage 1**: synthetic weak supervision으로 broad generalization 확보
- **Stage 2**: high-quality data로 supervised tuning
- **Stage 3**: stage 2 checkpoints를 **slerp(spherical linear interpolation)** 로 merge

### Reranking
- weak supervision 단계 없이 **supervised fine-tuning -> model merging**

이 구조를 보면 Qwen3 Embedding은 결국 "foundation model adaptation"을 아래처럼 나눈 셈이다.

- broad pair prior는 synthetic weak supervision에서
- task sharpening은 supervised tuning에서
- robustness는 merging에서

즉 성능 향상을 단일 objective 하나로 설명하는 것이 아니라, **data quality와 checkpoint diversity까지 포함한 staged optimization**으로 본다.

## 4-3. Engineering notes

- embedding loss는 hard negatives뿐 아니라 in-batch negatives를 풍부하게 쓰면서도, **false negative masking**으로 그 부작용을 줄인다.
- reranker score는 LLM의 next-token score를 그대로 활용하기 때문에, 별도 head 설계보다 **prompt/interface 설계의 영향**이 크다.
- 모델 전반이 **32K context**를 지원한다는 점도 실무적으로 의미가 있다. long retrieval, long instruction, code/document chunking과 연결되기 때문이다.
- 또 spec 상으로는 MRL support와 instruction-aware customization이 제공되지만, 본문은 이 둘을 **핵심 contribution으로 깊게 분석하지는 않는다.**  
  따라서 배포 시 flexible dimension이나 instruction engineering을 적극 쓰려면 release repo와 사용 예시를 추가로 보는 편이 좋다.
- reranking evaluation은 **Qwen3-Embedding-0.6B가 먼저 top-100 candidates를 뽑고**, 그 위에 여러 reranker를 공정 비교하는 방식으로 진행된다. 이건 reranker 자체 비교에는 깔끔하지만, production에서는 1차 retriever 품질과 candidate set 크기에 따라 결과가 달라질 수 있다.

# 5. Evaluation

## 5-1. Main results

이 논문의 결과는 크게 **embedding**과 **reranking** 두 축으로 읽으면 된다.

### 1) Embedding 성능

가장 대표적인 결과는 다음과 같다.

- **Qwen3-Embedding-8B**
  - MTEB Multilingual: **70.58**
  - MTEB English v2: **75.22**
  - CMTEB: **73.83**
  - MTEB Code: **80.68**
- **Qwen3-Embedding-4B**
  - MTEB Multilingual: **69.45**
  - MTEB English v2: **74.60**
  - CMTEB: **72.26**
  - MTEB Code: **80.06**
- **Qwen3-Embedding-0.6B**
  - MTEB Multilingual: **64.33**
  - MTEB English v2: **70.70**
  - CMTEB: **66.33**
  - MTEB Code: **75.41**

이 결과를 어떻게 읽는 게 좋을까?

- 4B와 8B는 multilingual / English / Chinese / code 전반에서 강하다.
- 특히 0.6B 모델도 꽤 의미 있다. 논문 표현대로 **Gemini Embedding 바로 뒤**에 붙는 구간이 있고, 일부 open 7B 계열과도 경쟁력이 있다.
- 즉 이 논문은 "큰 모델만 강하다"보다 **작은 스케일에서도 foundation-model-derived embedder가 꽤 강하다**는 점을 보여준다.

### 2) Reranking 성능

reranking은 0.6B embedding retrieval baseline 위에 얹어서 평가한다.

- baseline retrieval-only인 **Qwen3-Embedding-0.6B**는
  - MTEB-R: 61.82
  - CMTEB-R: 71.02
  - MMTEB-R: 64.64
  - MLDR: 50.26
  - MTEB-Code: 75.41
  - FollowIR: 5.09
- **Qwen3-Reranker-4B**는
  - MTEB-R: **69.76**
  - CMTEB-R: **75.94**
  - MMTEB-R: **72.74**
  - MLDR: **69.97**
  - MTEB-Code: **81.20**
  - FollowIR: **14.84**
- **Qwen3-Reranker-8B**는
  - MTEB-R: 69.02
  - CMTEB-R: **77.45**
  - MMTEB-R: **72.94**
  - MLDR: **70.19**
  - MTEB-Code: **81.22**
  - FollowIR: 8.05

여기서 포인트는 단순하다.

- 모든 Qwen3 reranker가 retrieval-only baseline보다 확실히 좋아진다.
- 8B가 대부분의 retrieval 계열 지표에서 가장 높지만,
- **4B가 MTEB-R와 FollowIR에서는 더 높다.**

즉 이 논문을 "8B가 다 이겼다"로 단순화하면 안 되고, **reranking은 task마다 최적 규모가 다를 수 있다**는 점도 함께 읽는 편이 좋다.

## 5-2. What really matters in the experiments

내가 보기엔 이 논문에서 진짜 중요한 실험은 최종 leaderboard보다 **Table 5 ablation**이다.

### 1) Synthetic weak supervision은 실제로 의미가 있는가?

그렇다.

- only synthetic data 모델의 MMTEB는 **58.49**
- final 0.6B 모델은 **64.33**

즉 synthetic-only로도 꽤 강한 출발점을 만들 수 있다.

그런데 더 중요한 건 synthetic stage를 아예 빼면 final이 내려간다는 점이다.

- w/o synthetic data: **61.21**
- full model: **64.33**

즉 1단계 large-scale weak supervision은 "있으면 좋음"이 아니라, **final quality에 실질적으로 기여하는 stage**로 보인다.

### 2) Model merging은 정말 필요한가?

이것도 그렇다.

- w/o model merge: **62.56**
- full model: **64.33**

English / Chinese / Code 쪽도 비슷하게 full model이 더 높다.

즉 이 논문은 model merging을 부가 trick으로 다루지 않는다. 오히려 **강한 embedding model을 만들기 위한 핵심 레버**로 본다.

### 3) 이 논문이 남기는 더 큰 메시지

Qwen3 Embedding은 architecture를 많이 바꾸지 않았는데도 강한 결과를 낸다.  
그래서 이 논문이 남기는 더 큰 메시지는 다음에 가깝다.

- foundation LLM이 충분히 강하면,
- embedding / reranking 품질은
  - 입력 인터페이스
  - synthetic data controllability
  - supervision stage 설계
  - checkpoint merging  
  같은 recipe 요소에서 크게 갈릴 수 있다.

내가 보기엔 이게 이 논문의 가장 재사용 가치가 높은 부분이다.

# 6. Limitations

1. 이 논문은 recipe를 잘 보여주지만, **2단계 supervised data 구성과 mixing details**를 main text에서 완전히 다 풀어주지는 않는다. 완전 재현을 하려면 보고서만으로는 부족할 수 있다.
2. 비교 결과 중 일부는 **2025년 6월 4일 leaderboard snapshot**을 바탕으로 한다. 따라서 "현재도 같은 순위인가"는 별개 문제다.
3. embedding architecture는 intentionally simple하다. 그래서 이 논문은 "왜 EOS causal embedding이 모든 대안보다 구조적으로 우수한가"를 증명하기보다, **Qwen3 위에서 이 recipe가 실제로 잘 작동한다**는 사실을 보여주는 데 더 가깝다.
4. 150M synthetic pairs를 만들고, Qwen3-32B를 data synthesizer로 쓰는 파이프라인은 분명 강력하지만, **작은 팀이 그대로 재현하기에는 비용과 운영 복잡도**가 있다.

# 7. My Take

## 7-1. Why this matters for my work

- 이 논문은 retrieval stack을 "작은 encoder 하나 고르는 문제"로 보지 않고, **foundation model에서 검색 인터페이스를 파생시키는 문제**로 본다는 점에서 가치가 크다.
- 특히 multilingual search, code retrieval, instruction-aware retrieval처럼 요구사항이 복합적인 환경에서는, backbone 자체의 세계지식과 instruction prior가 중요해진다.
- 내 관점에서 Qwen3 Embedding의 가장 큰 의미는, **embedding과 reranking을 같은 foundation model ecosystem 안에서 묶어 설계할 수 있다**는 점이다.
- 즉 생성 모델은 따로, retrieval 모델은 완전히 다른 계열로 따로 가져가는 대신, foundation family를 중심으로 representation stack을 통합하는 방향을 보여준다.

## 7-2. Reuse potential

내가 보기엔 실무에서 바로 재사용 가치가 높은 포인트는 아래 4가지다.

1. **Instruction-aware asymmetric embedding interface**  
   query에만 instruction을 붙이고 document는 그대로 두는 설계는 task-specific retrieval에서 바로 응용 가능하다.

2. **Synthetic weak supervision의 controllability**  
   query type / length / difficulty / language를 직접 설계하는 방식은 domain retrieval pair 생성에도 유용하다.

3. **Embedding과 reranking의 stage 분리**  
   둘 다 retrieval stack이지만, weak supervision이 필요한 정도와 supervision granularity가 다르다는 점을 명확히 보여준다.

4. **Model merging**  
   checkpoint selection을 감으로 하지 않고, merge를 정식 recipe로 넣는 것은 생각보다 실용적이다.

반대로 당장 그대로 가져가기 어려운 부분도 있다.

- Qwen3-32B 기반 대규모 synthetic data factory
- full-scale multilingual pair generation
- 완전한 foundation-family 수준의 end-to-end release

그래서 현실적으로는 **instruction-aware formatting + synthetic pair design + model merge**부터 가져가는 편이 더 실용적이라고 본다.

## 7-3. Follow-up papers

- **NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models**  
  decoder-only LLM을 embedder로 바꾸는 또 다른 철학이다. Qwen3 Embedding과 비교하면, architecture/interface를 어디까지 바꿀지에 대한 관점 차이가 잘 보인다.
- **Gemini Embedding: Generalizable Embeddings from Gemini**  
  proprietary foundation model 계열의 embedding 방향과 비교하기 좋다.
- **Improving General Text Embedding Model: Tackling Task Conflict and Data Imbalance through Model Merging**  
  Qwen3 Embedding의 merge 전략이 어디서 왔는지 추적하기 좋은 후속 읽기다.

# 8. Summary

- Qwen3 Embedding은 embedding과 reranking을 **Qwen3 foundation model의 두 가지 파생 인터페이스**로 다룬다.
- embedding은 **instruction-aware EOS representation**, reranker는 **yes/no next-token scoring**으로 구현된다.
- 성능의 핵심은 architecture 대수술보다 **synthetic weak supervision, high-quality supervised tuning, model merging**에 있다.
- 4B / 8B는 multilingual / English / Chinese / code에서 강하고, 0.6B도 상당히 경쟁력 있다.
- 실무적으로는 "좋은 embedder는 어떤 backbone인가"보다 **foundation model을 어떤 data recipe와 task interface로 적응시키는가**가 더 중요하다는 메시지를 남긴다.
