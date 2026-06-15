---
layout: single
title: "Multi-Vector Index Compression in Any Modality Review"
categories: Study-concept
tag: [MultimodalRetrieval, IndexCompression, LateInteraction]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2602.21202)

[Code link](https://github.com/HanxiangQin/omni-col-press)

> 한 줄 요약: 이 논문은 text, visual document, video, audio-visual retrieval에서 late interaction multi-vector index의 길이 의존 storage and compute cost를 줄이기 위해 query-agnostic fixed-budget document compression을 다루고, attention-guided clustering, AGC를 통해 compressed index에서도 full index에 가까운 retrieval quality를 유지하려는 논문이다.

이 논문을 그냥 retrieval compression 논문으로만 보면 조금 아쉽다. 요즘 RAG와 multimodal search에서 점점 중요한 병목은 encoder 성능만이 아니라 index를 얼마나 저장하고, 얼마나 빠르게 score할 수 있는가다. ColBERT 계열 late interaction은 query token과 document token 사이의 fine-grained MaxSim matching을 유지하기 때문에 single-vector retrieval보다 표현력이 좋다. 하지만 document가 길어질수록 저장해야 할 vector 수도 같이 늘어난다.

Text document에서는 이 비용이 아직 감당 가능할 때가 많다. 하지만 visual document, long video, audio-visual corpus에서는 이야기가 다르다. 한 document가 수백에서 수천 token 혹은 frame/audio-derived representation을 만들 수 있고, 이를 그대로 multi-vector index로 저장하면 retrieval system의 병목이 model이 아니라 index가 된다.

이 논문의 핵심 질문은 아래와 같다.

> Query를 모르는 indexing time에 document representation을 고정된 token budget으로 줄이면서도, late interaction retrieval의 장점을 얼마나 유지할 수 있는가?

이 질문이 중요한 이유는 retrieval system을 실제 서비스에 넣을 때 거의 항상 index cost가 운영 변수로 들어오기 때문이다. 모델이 아무리 좋아도, document마다 1000개 이상의 vector를 저장하고 score해야 한다면 large-scale multimodal corpus에서 바로 문제가 생긴다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Multi-vector retrieval을 text retrieval의 특수 기법이 아니라 multimodal retrieval의 common interface로 본다.
- Compression을 post-hoc quantization이 아니라 sequence dimension의 representation design 문제로 다룬다.
- SeqResize, MemTok, H-Pool, AGC를 같은 setting에서 비교해 어떤 compression failure가 생기는지 보여준다.
- BEIR, ViDoRe, MSR-VTT, MultiVENT 2.0까지 text, visual document, video, audio-visual setting을 함께 본다.
- 공개 repo인 OmniColPress가 있어서 index building, training, evaluation workflow를 재사용할 수 있다.

이 논문의 메시지는 단순히 "AGC가 좋다"가 아니다. 더 중요한 메시지는 multimodal late interaction index에는 엄청난 redundancy가 있고, 좋은 compression은 단순히 vector 수를 줄이는 것이 아니라 남은 vector들이 retrieval matching에서 고르게 쓰이도록 만드는 문제라는 점이다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 multi-vector late interaction retrieval의 index length scaling이다.

Late interaction retrieval은 query와 document를 각각 여러 개의 vector로 표현한 뒤, query token마다 document token 중 가장 잘 맞는 vector를 찾는다. ColBERT-style MaxSim scoring은 개념적으로 아래처럼 볼 수 있다.

$$
score(q, d) = sum_i max_j dot(q_i, d_j)
$$

여기서 $m$은 query vector 수, $n$은 document vector 수다. Query 쪽은 보통 짧기 때문에 큰 문제가 아니다. 병목은 document 쪽이다. Document가 길어질수록 $n$이 커지고, index storage와 scoring cost가 같이 증가한다.

이 논문의 compression 문제는 다음처럼 볼 수 있다.

$$
C_{\phi}(D) = Z, Z \in \mathbb{R}^{K x h}, K < n
$$

여기서 $D$는 원래 document token representation이고, $Z$는 고정 budget $K$개의 compressed document vector다. 핵심 제약은 $C_phi$가 query-agnostic이어야 한다는 점이다. Retrieval index는 query가 들어오기 전에 미리 만들어져야 하므로, compression은 특정 query에 맞춰 token을 고를 수 없다.

이 조건이 생각보다 빡빡하다. Long-context generation의 token pruning은 현재 prompt나 attention score를 보고 query-aware하게 token을 줄일 수 있다. 반면 retrieval index compression은 future query를 모르는 상태에서 document 안의 어떤 정보가 나중에 discriminative할지 예측해야 한다.

## 1-2. Why previous approaches are insufficient

기존 방식은 크게 세 계열로 볼 수 있다.

1. Sequence resizing
   - Full document representation을 만든 뒤 sequence dimension을 MLP로 줄인다.
   - 직관적이고 parameterized training이 가능하다.
   - 하지만 compressed token이 retrieval matching에서 고르게 쓰이지 않고 일부 token만 과하게 쓰이는 failure가 생길 수 있다.

2. Memory tokens
   - Document context 뒤에 learnable memory tokens를 붙이고, memory token hidden state만 compressed representation으로 사용한다.
   - 전체 document를 memory token이 흡수하게 만드는 방식이다.
   - 하지만 서로 다른 local detail이 memory token에 섞이며 information collapse나 smoothing이 생길 수 있다.

3. Hierarchical pooling
   - Similar vector를 greedy하게 merge해 token 수를 줄인다.
   - 학습이 필요 없고 modality-agnostic하다.
   - 하지만 noisy outlier나 multimodal redundancy를 semantic importance 기준으로 다루지는 못한다.

이 논문은 특히 multimodal data의 redundancy에 주목한다. Video에는 거의 변하지 않는 frame이 많고, audio에는 silence나 의미 없는 background가 있으며, visual document에는 margin, repeated layout, decorative element가 많다. 이 정보들은 token 수를 늘리지만 retrieval에 항상 도움이 되지는 않는다.

따라서 좋은 compression은 두 가지를 동시에 해야 한다.

- redundant or noisy token은 과감히 줄인다.
- future query에서 discriminative할 수 있는 semantic detail은 유지한다.

이 둘을 같이 만족시키려면 단순 averaging도, 단순 projection도 충분하지 않다. AGC는 이 지점에서 attention으로 salient centroid를 고르고, clustering으로 redundancy를 줄이며, weighted aggregation으로 hard clustering의 optimization 문제를 완화한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 four compression methods를 같은 multi-modality retrieval setting에서 비교하고, 그중 AGC라는 hybrid compression method를 제안한다는 점이다.

전체 비교 대상은 아래와 같다.

| Method | Type | Main idea | Main risk |
| --- | --- | --- | --- |
| SeqResize | Parameterized projection | Sequence dimension을 MLP로 고정 길이로 줄임 | 일부 compressed token만 쓰이는 under-utilization |
| MemTok | Parameterized token pooling | Learnable memory token이 document를 흡수 | Distinct feature가 섞이는 information collapse |
| H-Pool | Non-parametric clustering | Similar vector를 hierarchical merge | Greedy merge와 noisy outlier에 취약 |
| AGC | Attention-guided clustering | Universal query token으로 centroid를 고르고 weighted clustering | Training과 implementation이 가장 복잡함 |

AGC의 핵심은 fixed budget 안에서 token utilization을 높이는 것이다. Multi-vector retrieval의 품질은 vector 수 자체보다, 그 vector들이 query matching에서 얼마나 의미 있게 쓰이는가에 달려 있다. Full index가 있어도 실제 query matching이 처음 몇 token에만 몰린다면, 남은 token은 storage cost만 만들고 retrieval utility는 낮다.

## 2-2. Design intuition

AGC의 설계 직관은 세 가지다.

첫째, future query를 모르는 상태에서도 salient region을 찾아야 한다. 그래서 AGC는 learned universal query tokens를 사용한다. 실제 user query가 아니라, document 안에서 정보가 많은 영역을 probe하는 trainable query token이라고 보면 된다.

둘째, redundancy는 clustering으로 줄여야 한다. Multimodal document에는 비슷한 frame, 반복 layout, silence segment처럼 유사 token이 많다. 이들을 모두 별도 vector로 저장하면 budget이 낭비된다. AGC는 selected centroid 주변으로 hard assignment를 수행해 비슷한 token을 group으로 묶는다.

셋째, hard assignment만 쓰면 optimization이 거칠어진다. 그래서 AGC는 attention-derived saliency weight로 cluster aggregation을 수행한다. 단순 mean pooling이 아니라 token별 중요도를 반영한 weighted average를 만들고, 이로써 salient token이 compressed vector에 더 크게 반영된다.

이 논문에서 AGC보다 더 중요한 키워드는 "index utilization"이다. 좋은 compressed index는 단지 작은 index가 아니라, query-document MaxSim matching에서 남은 vector들이 낭비 없이 쓰이는 index다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Problem | Late interaction multi-vector document index의 storage and query-time cost를 줄이는 것 |
| Constraint | Document compression은 query-agnostic이어야 함 |
| Input | Text, image, video, audio-visual document representation |
| Output | 고정 budget $K$개의 compressed document vector |
| Main method | Attention-Guided Clustering, AGC |
| Baselines | SeqResize, MemTok, H-Pool, full uncompressed index |
| Benchmarks | BEIR, ViDoRe v2, MSR-VTT, MultiVENT 2.0 |
| Code artifact | OmniColPress |

## 3-2. Module breakdown

### 1) Late interaction scoring

Late interaction은 query와 document를 각각 token-level vector set으로 두고, query token마다 가장 잘 맞는 document token을 찾는다.

Single-vector retrieval은 document 하나를 vector 하나로 압축한다. 이는 빠르지만 fine-grained evidence matching이 약하다. 반대로 late interaction은 여러 document vector를 남기므로 local evidence matching이 가능하다. Visual document나 video retrieval에서는 이 점이 중요하다. Query가 특정 figure, table, object, scene, audio cue를 물을 수 있기 때문이다.

하지만 이 장점이 바로 비용이 된다. Document length가 1000이면 document 하나에 vector가 1000개 생긴다. Query마다 이 vector들과 MaxSim을 해야 하고, index도 그만큼 커진다.

### 2) SeqResize

SeqResize는 encoder output을 만든 뒤, sequence dimension을 MLP로 줄인다. 예를 들어 document hidden state가 $n x h$라면, sequence length $n$을 고정 budget $K$로 projection한다.

장점은 단순하다. Encoder가 전체 document를 contextualize한 뒤 별도 projection으로 길이를 줄이면 된다. 하지만 논문은 SeqResize가 budget을 늘려도 retrieval matching에서 일부 token만 쓰는 under-utilization을 보인다고 해석한다. 즉 vector 수가 늘어도 실제로는 model이 그 vector를 다 쓰지 못할 수 있다.

### 3) MemTok

MemTok은 document sequence에 learnable memory token을 붙인다. Encoder self-attention 이후 memory token hidden state만 document representation으로 사용한다.

이 방식은 LLM이나 encoder에서 special token을 global representation으로 쓰는 방식과 닮았다. 하지만 multi-vector retrieval에서는 문제가 생길 수 있다. Retrieval은 global gist만 필요한 것이 아니라, hard negative와 positive document를 구분할 local discriminative detail이 필요하다. Memory token이 document 정보를 너무 부드럽게 섞으면 서로 다른 evidence가 collapse될 수 있다.

### 4) H-Pool

H-Pool은 non-parametric method다. Document token vector의 cosine distance를 계산하고, Ward linkage 기반 hierarchical pooling으로 비슷한 vector를 merge한다. 정확히는 cluster 수가 budget $K$가 될 때까지 merge를 반복하고, 각 cluster의 평균 vector를 compressed representation으로 사용한다.

장점은 학습이 필요 없다는 것이다. 어떤 modality에도 적용 가능하고, training recipe가 없어도 바로 실험할 수 있다. 실제로 논문에서도 H-Pool은 non-text benchmark에서 꽤 강한 baseline으로 나타난다.

단점은 merge 기준이 semantic saliency를 직접 보지 않는다는 것이다. 비슷한 vector를 합치는 것은 redundancy reduction에는 도움이 되지만, future query에서 중요한 evidence를 중심으로 budget을 배분한다고 보장하지는 않는다.

### 5) AGC: Attention-based centroid selection

AGC는 learned universal query tokens를 document sequence에 붙인다. 이 universal query는 실제 retrieval query가 아니라, document 안의 semantically salient region을 찾기 위한 trainable probes다.

Encoder를 통과한 뒤, universal query token과 document token 사이의 attention score를 사용해 centroid 후보를 고른다. 이 과정은 query-agnostic이다. 실제 user query가 없어도, model은 training을 통해 retrieval에 자주 도움이 되는 document region을 찾도록 학습된다.

### 6) AGC: Hard clustering

Centroid가 정해지면 document token을 centroid에 hard assignment한다. 비슷한 token을 같은 cluster에 넣고, 각 cluster가 하나의 compressed vector가 된다.

Hard clustering의 장점은 distinct semantic concept을 분리하기 쉽다는 점이다. MemTok처럼 모든 정보를 soft하게 섞으면 detail이 blur될 수 있다. AGC는 cluster boundary를 만들어 서로 다른 evidence가 같은 memory에 과하게 섞이는 것을 줄이려 한다.

### 7) AGC: Weighted aggregation

마지막으로 각 cluster 안의 token을 saliency weight로 평균낸다. Naive average는 모든 token을 같은 중요도로 보지만, multimodal data에서는 정보 밀도가 균일하지 않다. Video의 핵심 frame과 반복 background, document의 본문 표와 margin, audio의 speech segment와 silence가 같은 weight를 받으면 안 된다.

AGC는 attention-derived saliency를 aggregation weight로 사용해 중요한 token이 compressed vector에 더 크게 들어가도록 만든다. 이 구조 덕분에 hard clustering의 discrete structure를 유지하면서도, gradient와 optimization 측면에서는 조금 더 smooth한 경로를 확보한다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 text, visual document, video, audio-visual retrieval을 함께 본다.

| Benchmark | Modality | Task summary | Notable detail |
| --- | --- | --- | --- |
| BEIR | Text | Text retrieval | 1M document 미만 subset 사용, Quora 제외 |
| ViDoRe v2 | Visual document | Visually rich PDF retrieval | Insurance, biomedical, economics, ESG domain 포함 |
| MSR-VTT | Video | Text-to-video retrieval | Test에 1000 query-video pairs, query당 relevant video 1개 |
| MultiVENT 2.0 | Audio-visual video | Text-to-video retrieval | 2546 queries, 109800 videos, query당 relevant videos 10개 |

이 benchmark 구성이 중요한 이유는 compression failure가 modality마다 다르게 나타나기 때문이다. Text는 정보 밀도가 비교적 높고 sequence order가 중요하다. Visual document는 layout과 OCR, table, figure가 중요하다. Video는 temporal redundancy가 크다. Audio-visual video는 visual signal뿐 아니라 audio sampling cost까지 들어간다.

## 4-2. Training strategy

논문이 제시한 experimental setup을 요약하면 다음과 같다.

| Setting | Backbone / setup | Training detail | Index / eval note |
| --- | --- | --- | --- |
| BEIR | gte-modernbert-base 기반 | 10000 steps, MSMARCO hard negatives, batch size 20 | FastPlaid with 4-bit residuals |
| ViDoRe v2 | Qwen2.5-VL-3B | ColPali train set, 2 epochs, global batch size 112 | Full uncompressed는 flat index brute-force로 비교 |
| MSR-VTT | Qwen2.5-VL-3B, Qwen2.5-VL-7B, Qwen3-VL-4B | Train 9k split, 2 epochs, 24 frames, global batch size 28 | Matching position and strength 분석 포함 |
| MultiVENT 2.0 | Qwen2.5-Omni-3B | Human-written + synthetic queries, 2 epochs, batch size 8 | Full uncompressed index를 만들 수 없어 compressed method만 FastPlaid 사용 |

여기서 실무적으로 눈에 띄는 점은 두 가지다.

첫째, compression method가 indexing and evaluation stack과 강하게 연결되어 있다. ViDoRe처럼 hidden dimension과 vector 수가 큰 경우 full uncompressed index는 FastPlaid에 바로 올리기 어렵고, flat brute-force로 비교해야 한다. MultiVENT 2.0에서는 full model index 자체를 만들 수 없었다고 보고한다.

둘째, audio-visual setting은 model architecture보다 preprocessing cost가 먼저 병목이 될 수 있다. MultiVENT 2.0에서는 batch size 8을 맞추기 위해 Qwen-Omni training rate인 16KHz가 아니라 4KHz audio sampling을 사용했다. 이 점은 audio-visual retrieval의 index compression이 단순 storage 문제만이 아니라 input processing and memory budget 문제와도 연결된다는 뜻이다.

## 4-3. Engineering notes

공개 repo인 OmniColPress는 이 논문을 재현하고 확장하기 위한 modular framework로 공개되어 있다. README 기준으로 text, image, video, audio query/document를 role별로 control할 수 있고, AGC, memory tokens, hierarchical pooling, sequence resizing을 지원한다. 또한 LoRA, DeepSpeed ZeRO, gradient checkpointing, half precision, distributed contrastive loss, mixed-precision similarity computation 같은 training option을 포함한다.

실무 관점에서 재사용할 만한 포인트는 아래다.

1. Compression method를 pooling argument로 바꿀 수 있게 설계
   - 같은 retrieval pipeline에서 single-vector, multi-vector, compressed multi-vector를 비교하기 좋다.

2. Index type을 분리
   - `flat`, `multivec`, `fast-plaid`를 구분해 실험할 수 있다.

3. Data schema가 modality-aware
   - Corpus row에 `text`, `title`, `image`, `video`, `audio` field를 둘 수 있다.
   - Query에도 optional media field를 둘 수 있다.

4. Query/document role control
   - Text query to video document, text query to visual document, text query to audio-visual document 같은 setting을 하나의 framework로 다룰 수 있다.

5. Training과 evaluation의 argument consistency가 중요
   - README는 model and pooling arguments를 training, index building, evaluation에서 동일하게 맞춰야 한다고 강조한다.

이 논문은 method paper이지만, 실제로는 retrieval system paper로 읽는 것이 더 맞다. Compression objective만 보는 것이 아니라, training, index building, retrieval evaluation, data schema가 하나의 system으로 연결되어야 한다.

# 5. Evaluation

## 5-1. Main results

Table 1의 핵심 결과를 압축하면 아래와 같다. 수치는 논문 HTML 기준이며, benchmark마다 metric이 다르므로 row 간 직접 평균 비교는 조심해야 한다.

| Method | BEIR R@10 | BEIR nDCG@10 | ViDoRe R@1 | ViDoRe nDCG@5 | MSR-VTT R@1 | MSR-VTT nDCG@10 | MultiVENT R@10 | MultiVENT nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 37.1 | 46.2 | 27.7 | 60.0 | 55.7 | 71.9 | - | - |
| SeqResize | 35.8 | 43.9 | 23.5 | 51.7 | 53.3 | 69.9 | 41.1 | 38.5 |
| MemTok | 36.3 | 45.0 | 25.0 | 54.4 | 54.2 | 69.9 | 48.7 | 44.8 |
| H-Pool | 35.5 | 41.2 | 26.0 | 56.4 | 54.1 | 70.4 | 49.2 | 46.5 |
| AGC | 37.0 | 45.0 | 26.3 | 56.7 | 56.9 | 71.5 | 49.6 | 46.3 |

논문이 강조하는 핵심은 AGC가 모든 metric에서 절대 1등이라는 단순한 얘기가 아니다. 더 정확히는 아래처럼 읽어야 한다.

- AGC는 learned compression method인 SeqResize와 MemTok보다 일관되게 강하다.
- H-Pool은 non-parametric method인데도 강력한 baseline이다.
- MSR-VTT에서는 AGC가 budget 32일 때 R@1 56.9로 full baseline 55.7보다 높다.
- MultiVENT 2.0에서는 full uncompressed index를 만들 수 없어 compression이 선택이 아니라 필요 조건이 된다.
- BEIR text retrieval에서는 MemTok과 AGC가 비슷하고, text-only setting에서는 modality redundancy가 multimodal보다 덜 극적이다.

## 5-2. ViDoRe result interpretation

ViDoRe v2는 visual document retrieval benchmark다. 보험, biomedical, economics, ESG report처럼 layout과 visual element가 중요한 document를 다룬다.

논문은 ViDoRe에서 AGC와 H-Pool이 SeqResize와 MemTok보다 유리하다고 해석한다. Table 3 기준 average nDCG@5는 다음과 같다.

| Method | Token budget | Avg nDCG@5 |
| --- | ---: | ---: |
| Base | 1297 | 60.0 |
| ColPali | - | 53.3 |
| ColQwenOmni | - | 56.5 |
| MetaEmbed | 64 | 58.8 |
| SeqResize | 64 | 51.7 |
| MemTok | 64 | 54.3 |
| H-Pool | 64 | 56.4 |
| AGC | 64 | 56.7 |

여기서 흥미로운 점은 AGC가 full base를 이겼다는 것이 아니라, compressed method 중에서는 AGC와 H-Pool이 강하게 나온다는 점이다. Visual document에서는 layout, figure, table, OCR region처럼 sparse but important region이 많다. 이 상황에서는 memory token이 전체를 부드럽게 요약하는 것보다, salient region을 유지하거나 비슷한 region을 잘 merge하는 방식이 유리할 수 있다.

## 5-3. MSR-VTT result interpretation

MSR-VTT는 이 논문에서 가장 메시지가 강하게 나오는 benchmark다. Table 4 기준 주요 수치는 아래와 같다.

| Token budget | Method | R@1 | R@10 | nDCG@10 |
| ---: | --- | ---: | ---: | ---: |
| 1 | OmniEmbed-7B | 51.5 | 83.2 | 67.1 |
| 26 | Video-ColBERT | 51.5 | 85.5 | 67.7 |
| 1318 | Baseline 3B | 55.7 | 88.3 | 71.9 |
| 5 | AGC | 53.9 | 85.8 | 69.2 |
| 32 | AGC | 56.9 | 87.0 | 71.5 |
| 128 | AGC | 56.4 | 87.6 | 71.6 |

Budget 32 AGC는 full baseline보다 R@1이 높고, nDCG@10도 거의 유지한다. 이 결과는 매우 흥미롭다. Compression은 보통 성능을 조금 희생하고 비용을 줄이는 trade-off로 생각된다. 그런데 multimodal video retrieval에서는 compression objective가 redundancy와 noise를 줄여 오히려 R@1을 개선할 수 있다.

다만 이 결과를 과해석하면 안 된다. R@10과 nDCG@10은 full baseline이 여전히 강하다. 따라서 "compressed index가 항상 full index보다 낫다"가 아니라, "video retrieval에서는 full index의 많은 vector가 실질적으로 낭비될 수 있고, 잘 설계된 compression은 일부 metric에서 full index를 넘을 수 있다" 정도로 읽는 것이 안전하다.

## 5-4. Compression range and transferability

AGC는 budget 5, 32, 128에서 모두 실험된다. 논문은 budget이 늘면 전반적으로 성능이 좋아지는 경향이 있지만, 단순히 token 수를 늘린다고 항상 선형적으로 좋아지지는 않는다고 본다.

Table 6의 transferability 결과도 중요하다.

| Method | Train budget | Test budget | R@1 | R@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1318 | 1318 | 55.7 | 88.3 | 71.9 |
| AGC | 32 | 5 | 53.6 | 87.4 | 70.1 |
| AGC | 32 | 32 | 56.9 | 87.0 | 71.5 |
| AGC | 32 | 128 | 56.4 | 87.5 | 71.7 |
| H-Pool | 1318 | 5 | 52.6 | 86.1 | 68.9 |
| H-Pool | 1318 | 32 | 54.1 | 87.3 | 70.4 |
| H-Pool | 1318 | 128 | 54.4 | 87.2 | 70.9 |

AGC는 train budget 32로 학습해도 test budget 5와 128에서 꽤 안정적으로 동작한다. 이 점은 운영 관점에서 중요하다. 실제 서비스에서는 index cost budget이 corpus, tier, user segment, hardware에 따라 달라질 수 있다. Compression method가 budget 변화에 유연하면, 같은 model을 여러 serving profile에서 쓸 수 있다.

## 5-5. Backbone scaling

논문은 AGC가 backbone size와 generation에 따라 어떻게 변하는지도 본다.

| Model variant | R@1 | R@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Qwen2.5-VL-3B | 56.9 | 87.0 | 71.5 |
| Qwen2.5-VL-7B | 58.0 | 89.0 | 73.0 |
| Qwen3-VL-4B | 58.5 | 88.4 | 73.0 |

이 결과는 AGC가 특정 backbone trick만은 아니라는 근거다. 더 큰 모델이나 더 최신 backbone으로 가면 representation 자체가 좋아지고, compression도 더 나은 출발점에서 작동한다. 다만 scale이 커질수록 compression training cost도 커지므로, service setting에서는 retrieval quality, training cost, index cost를 같이 봐야 한다.

## 5-6. Index utilization analysis

이 논문의 실험에서 가장 마음에 드는 부분은 index utilization 분석이다. 논문은 MSR-VTT에서 MaxSim matching이 어떤 document token position에 몰리는지 보고, compressed token이 얼마나 고르게 쓰이는지 시각화한다.

핵심 관찰은 다음과 같다.

- Full baseline은 document representation을 많이 만들지만, 실제 matching은 앞쪽 token에 강하게 몰릴 수 있다.
- SeqResize는 일부 token만 선택되는 under-utilization pattern을 보인다.
- MemTok은 memory token 구조상 앞쪽 token bias가 생길 수 있다.
- AGC와 H-Pool은 document-derived representation을 사용하기 때문에 compressed representation 활용이 더 균형적이다.
- Retrieval performance와 MaxSim match distribution evenness 사이에 높은 상관이 관찰된다.

논문은 Pearson correlation이 0.959에서 0.996 범위라고 보고하지만, 이 correlation은 sample 수가 5로 작기 때문에 강한 결론보다 diagnostic signal로 보는 편이 맞다. 그래도 개발 관점에서는 유용하다. Multimodal retrieval에서는 full evaluation index를 매번 만드는 비용이 크기 때문에, small query set에서 token utilization을 먼저 보는 것이 compression method debugging에 도움이 될 수 있다.

## 5-7. Ablation

AGC의 구성 요소 ablation은 MSR-VTT에서 수행된다.

| Method | R@1 | R@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| AGC | 56.9 | 87.0 | 71.5 |
| w/o Attn Weight | 55.7 | 86.5 | 71.0 |
| w/o Attn Select | 54.1 | 86.8 | 70.0 |
| w/o Cluster | 52.9 | 87.3 | 69.8 |

이 ablation은 설계 의도를 꽤 잘 뒷받침한다.

- Attention weight를 빼면 saliency 기반 aggregation이 약해진다.
- Attention selection을 random selection으로 바꾸면 signal과 noise를 구분하기 어렵다.
- Clustering을 빼면 redundancy reduction과 representation diversity가 약해진다.

특히 w/o Cluster에서 R@10은 크게 나쁘지 않지만 R@1과 nDCG@10이 떨어지는 점이 흥미롭다. Retrieval system에서는 top candidate를 제대로 잡는 능력과 ranking quality가 중요하기 때문에, hard clustering이 단순 compression이 아니라 discriminative representation에도 기여한다고 볼 수 있다.

# 6. Limitations

1. Query-agnostic compression의 근본 한계
   - Indexing time에는 future query를 모른다.
   - AGC의 universal query token은 query distribution에서 평균적으로 중요한 region을 찾는 방식이다.
   - 특정 rare query가 요구하는 detail을 compression이 버릴 가능성은 남는다.

2. Static budget 문제
   - AGC는 fixed token budget을 사용한다.
   - 하지만 실제 document의 information density는 다르다.
   - 짧고 dense한 document와 길지만 repetitive한 video가 같은 budget을 받아야 하는지는 별도 문제다.
   - 논문도 future work로 document content에 비례해 budget을 조절하는 방향을 언급한다.

3. Dataset coverage 한계
   - BEIR, ViDoRe, MSR-VTT, MultiVENT 2.0은 폭이 넓지만 모든 modality를 대표하지는 않는다.
   - OCR-heavy enterprise document, surveillance video, long lecture video, meeting audio, multimodal web page 같은 실제 corpus에서는 redundancy pattern이 다를 수 있다.

4. Baseline and index condition 차이
   - ViDoRe에서는 full uncompressed index를 FastPlaid에 fit하지 못해 flat brute-force를 사용한다.
   - MultiVENT 2.0에서는 full uncompressed index를 만들 수 없어 compressed method만 평가한다.
   - 이는 compression의 필요성을 보여주는 동시에, 모든 row가 완전히 같은 serving condition은 아니라는 점을 의미한다.

5. Audio-visual preprocessing bottleneck
   - MultiVENT 2.0에서 batch size 8을 맞추기 위해 audio sampling을 16KHz에서 4KHz로 낮춘다.
   - Audio cue가 중요한 retrieval task에서는 sampling choice가 retrieval quality에 영향을 줄 수 있다.
   - 따라서 audio-visual retrieval에서는 index compression뿐 아니라 audio representation design도 같이 봐야 한다.

6. Training complexity
   - H-Pool은 non-parametric이라 간단하지만, AGC는 universal query tokens, attention scoring, clustering, weighted aggregation이 필요하다.
   - Production system에 넣으려면 training/evaluation/indexing code path가 잘 관리되어야 한다.

7. Correlation analysis의 sample size
   - Token utilization evenness와 retrieval performance의 correlation은 흥미롭지만, 논문 자체도 작은 sample 수를 언급한다.
   - 따라서 이 결과는 design heuristic으로는 유용하지만, general law처럼 읽으면 안 된다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 retrieval model을 볼 때 "encoder가 좋은가"에서 "index가 실제로 쓰이는가"로 시야를 넓혀준다. 특히 document AI, OCR retrieval, image-heavy report search, video search, meeting/audio retrieval 같은 실무 시스템에서는 model accuracy와 index cost를 분리해서 볼 수 없다.

Multi-vector retrieval은 local evidence matching이 강력하지만, document마다 vector를 많이 저장해야 한다. 이 비용은 corpus scale이 커질수록 바로 서비스 비용이 된다. 따라서 retrieval model을 평가할 때 아래 세 가지를 같이 봐야 한다.

- Retrieval quality
- Index size
- Token utilization

이 논문은 multimodal RAG pipeline에서 compressed multi-vector retrieval을 꽤 현실적인 option으로 만든다. Single-vector retrieval은 빠르지만 detailed evidence matching이 약하고, full multi-vector retrieval은 강하지만 비싸다. AGC 같은 method는 그 사이에서 operational sweet spot을 찾는 시도다.

## 7-2. Reuse potential

재사용해볼 만한 포인트는 다음과 같다.

1. Visual document retrieval index compression
   - OCR + layout + figure가 섞인 문서에서 full multi-vector index는 금방 커질 수 있다.
   - AGC나 H-Pool을 써서 document당 vector budget을 제한하면, retrieval quality와 index cost를 함께 조정할 수 있다.

2. Video RAG memory compression
   - Video corpus에서는 frame redundancy가 크다.
   - 모든 frame token을 index로 남기기보다 salient frame/region 중심으로 compressed vector를 만들면 storage cost를 줄일 수 있다.

3. Retrieval debugging metric으로 token utilization 사용
   - nDCG나 recall만 보면 왜 compression이 실패했는지 알기 어렵다.
   - MaxSim matching이 특정 compressed token에 몰리는지 보면 representation collapse나 under-utilization을 더 빨리 찾을 수 있다.

4. Budget-aware serving tier
   - Free tier는 budget 5 또는 16, paid or internal tier는 budget 32 또는 64처럼 retrieval quality/cost tier를 나눌 수 있다.
   - AGC가 budget transferability를 보인다는 점은 이런 운영 시나리오에 잘 맞는다.

5. H-Pool as a strong baseline
   - AGC를 바로 training하기 어렵다면 H-Pool을 먼저 baseline으로 두는 것이 좋다.
   - Non-parametric이면서도 multimodal redundancy reduction에는 꽤 강하게 나온다.

## 7-3. Follow-up papers

후속으로 같이 읽으면 좋은 논문은 아래다.

- ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
- ColBERTv2: Efficient and Effective Retrieval via Lightweight Late Interaction
- ColPali: Efficient Document Retrieval with Vision Language Models
- MetaEmbed: Efficient Constant-Space Multi-vector Retrieval
- Video-ColBERT and ColQwen-Omni 계열 multimodal late interaction retrieval 논문
- MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings

# 8. Summary

- 이 논문은 late interaction multi-vector retrieval의 document-length-dependent index cost를 fixed-budget compression 문제로 다룬다.
- AGC는 universal query token으로 salient centroid를 고르고, hard clustering과 weighted aggregation으로 redundancy를 줄인다.
- BEIR, ViDoRe, MSR-VTT, MultiVENT 2.0에서 text, visual document, video, audio-visual retrieval을 함께 평가한다.
- MSR-VTT에서는 budget 32 AGC가 R@1 기준 full baseline을 넘고, nDCG@10도 거의 유지한다.
- 실무적으로는 retrieval quality만큼 index size와 token utilization을 함께 보는 것이 중요하다는 메시지가 크다.
