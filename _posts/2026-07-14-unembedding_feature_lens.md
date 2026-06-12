---
layout: single
title: "Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings Review"
categories: Study-concept
tag: [Embedding, LLM, Retrieval]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.07502)

[Code link](https://github.com/CentreChen/EmbFilter)

이 논문은 LLM을 text embedding model로 바로 쓸 때 왜 생각보다 성능이 애매한가를 꽤 다른 각도에서 본다. 보통 이 문제를 보면 pooling, contrastive learning, instruction tuning, hard negative, bidirectional attention 같은 쪽을 먼저 떠올린다. 그런데 이 논문은 마지막 hidden state 자체를 vocabulary space로 투영해보고, 그 안에 너무 자주 나오는 token 성향이 과하게 드러나는 현상에 주목한다.

> 한 줄 요약: 이 논문은 LLM hidden state가 high-frequency token subspace에 과하게 끌리는 현상을 unembedding matrix로 진단하고, 그 subspace를 선형적으로 걸러내는 EmbedFilter를 통해 LLM 기반 zero-shot text embedding을 개선하려는 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM embedding 성능을 단순히 pooling 문제로 보지 않고, **unembedding matrix가 만드는 feature basis** 문제로 재해석한다.
- 학습 없이 적용 가능한 **post-hoc linear filter**를 제안하기 때문에, 기존 LLM 기반 embedding pipeline에 붙여보기 쉽다.
- embedding quality 개선과 dimension reduction을 같은 메커니즘 안에서 다룬다는 점이 retrieval serving 관점에서 흥미롭다.

이 논문을 한 문장으로 다시 쓰면 이렇다.

"LLM의 unembedding matrix는 단순 출력 head가 아니라, hidden state 안에 어떤 token-frequency feature가 섞였는지 읽어내는 lens처럼 쓸 수 있다."

이 논문의 재미있는 부분은 성능 향상 자체보다도, LLM embedding failure를 분석하는 좌표계를 하나 더 준다는 점이다. 좋은 embedder를 만들려면 대규모 contrastive training만 필요한 것이 아니라, causal LM이 next-token prediction을 위해 유지하던 feature 중 어떤 것이 sentence representation에는 방해가 되는지도 봐야 한다는 메시지로 읽힌다.

# 1. Problem Setting

## 1-1. Problem definition

- LLM은 generation, instruction following, zero-shot reasoning에서는 강하지만, 같은 모델의 hidden state를 그대로 text embedding으로 쓰면 전용 embedding model보다 약한 경우가 많다.
- 특히 decoder-only LLM의 마지막 token hidden state를 pooling해서 cosine similarity에 쓰는 방식은 단순하고 편하지만, representation space가 retrieval이나 STS에 최적으로 맞춰졌다고 보기는 어렵다.
- 이 논문이 겨냥하는 문제는 아래 질문에 가깝다.

"LLM hidden state는 왜 semantic representation으로 바로 쓰기에 noisy한가?"

- 저자들은 embedding vector를 vocabulary space에 투영했을 때, frequent but uninformative token과의 alignment가 강하게 나타난다고 본다.
- 즉 sentence embedding 안에 semantic signal만 있는 것이 아니라, next-token prediction 과정에서 자주 쓰이는 frequency-biased token feature가 섞여 있고, 이 성분이 nuance를 흐린다는 것이다.

## 1-2. Why previous approaches are insufficient

- 기존 접근은 대체로 embedding 전용 학습을 추가한다.
  - contrastive learning
  - instruction tuning
  - hard negative mining
  - pooling layer 추가
  - causal mask 제거 또는 encoder-style conversion
- 이런 방식은 성능은 올릴 수 있지만, 왜 raw LLM embedding이 약한지에 대한 mechanistic explanation은 충분하지 않을 수 있다.
- 또한 학습 기반 접근은 data, compute, negative mining recipe, benchmark coverage에 크게 의존한다.
- 반대로 이 논문은 학습을 새로 하기 전에, 이미 존재하는 **unembedding matrix**를 이용해 hidden state 내부의 feature 성분을 읽고 제거해보자고 제안한다.
- 그래서 문제 설정이 조금 다르다. 이 논문은 더 좋은 embedding model을 처음부터 훈련하는 논문이라기보다, LLM이 이미 가진 representation에서 무엇을 빼야 하는가를 묻는 논문에 가깝다.

# 2. Core Idea

## 2-1. Main contribution

- 핵심 기여는 EmbedFilter라는 단순한 linear transformation이다.
- EmbedFilter는 LLM의 `lm_head`, 즉 unembedding matrix를 이용해 hidden representation을 새 좌표계로 보고, high-frequency token influence를 줄이는 방향으로 embedding을 정제한다.
- 논문 abstract 기준으로 저자들은 unembedding matrix가 frequent token을 embedding space에 active하게 쓰는 latent space를 encode한다고 주장한다.
- 따라서 이 subspace를 filtering하면 semantic representation이 더 잘 살아나고, 동시에 낮은 차원 embedding으로도 품질을 유지할 수 있다고 본다.

이 아이디어는 아래처럼 볼 수 있다.

$$
s = h W_U^T
$$

여기서 $h$는 text embedding으로 쓰려는 LLM hidden state이고, $W_U$는 unembedding matrix다. $s$는 vocabulary logit처럼 해석할 수 있다. 만약 semantic text embedding이어야 하는 $h$가 항상 frequent token 쪽으로 강하게 투영된다면, 그 embedding은 의미 차이를 보는 대신 frequency direction을 많이 들고 있을 가능성이 있다.

EmbedFilter는 이 문제를 fixed linear projection으로 다룬다.

$$
z = Fh
$$

여기서 $F$는 unembedding matrix에서 유도한 filter다. 구현 코드 기준으로는 `lm_head.weight`에 대해 SVD를 수행하고, `Vh`의 일부 row를 projection matrix로 사용한다.

$$
W_U = U \Sigma V^T
$$

중요한 점은 이 filter가 별도 학습된 MLP가 아니라는 것이다. 즉 이 논문은 representation을 새로 학습하기보다, output head에 이미 들어 있는 feature basis를 재활용한다.

## 2-2. Design intuition

- LLM의 unembedding matrix는 다음 token을 맞히기 위한 출력 공간이다.
- 하지만 hidden state를 unembedding matrix로 보면, hidden state가 어떤 token feature를 강하게 표현하는지 읽을 수 있다.
- 이 논문은 그 lens를 text embedding diagnostic으로 가져온다.
- frequent token은 language modeling에서는 중요하지만, sentence embedding에서는 자주 nuisance feature가 될 수 있다.
- 그래서 semantic matching에 필요 없는 frequency-heavy direction을 줄이면 retrieval, STS, clustering 같은 embedding task에서 더 나은 representation이 나올 수 있다.

이 설계 직관은 꽤 실용적이다. 기존 LLM을 embedding model로 쓰는 팀이라면 모델을 다시 학습하지 않고도, hidden state 뒤에 아주 작은 fixed projection을 넣어 embedding을 재정렬할 수 있기 때문이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | LLM hidden state를 zero-shot text embedding으로 더 잘 쓰도록 정제 |
| Key observation | raw embedding이 frequent but uninformative tokens와 강하게 align됨 |
| Key module | unembedding SVD 기반 EmbedFilter |
| Training | 추가 학습 없이 fixed linear transformation 적용 |
| Evaluation focus | MTEB style zero-shot embedding benchmark와 dimension reduction |
| Difference from prior work | contrastive tuning 대신 output head geometry를 이용해 post-hoc filtering |

## 3-2. Module breakdown

### 1) Vocabulary-space probing

- 첫 단계는 hidden state를 vocabulary space로 투영해보는 것이다.
- 일반적인 embedding evaluation에서는 hidden state끼리 cosine similarity를 바로 계산한다.
- 이 논문은 그 전에 hidden state가 unembedding matrix를 통과하면 어떤 token 성향을 보이는지 본다.
- 여기서 frequent but low-information token이 상위에 반복적으로 등장한다면, embedding space 안에 frequency-driven component가 과도하게 들어 있다고 해석할 수 있다.

이 관점의 장점은 debug가 쉽다는 점이다. embedding vector 자체는 사람이 보기 어렵지만, vocabulary projection의 top token은 사람이 어느 정도 해석할 수 있다.

### 2) Unembedding matrix as feature lens

- unembedding matrix는 보통 generation output head로만 본다.
- 하지만 이 논문에서는 unembedding matrix를 hidden state의 feature lens로 본다.
- 즉 `lm_head.weight`는 token logit을 만들기 위한 matrix인 동시에, hidden state가 어떤 token feature 방향에 민감한지 보여주는 basis가 된다.
- 공식 구현에서도 핵심은 `lm_head.weight.float()`에 대해 `torch.linalg.svd`를 수행하는 부분이다.
- 그 다음 `Vh`에서 선택한 row들을 새로운 fixed linear layer의 weight로 넣는다.

이 부분은 논문의 제목과 잘 맞는다. unembedding matrix는 단순한 classifier head가 아니라, text embedding을 읽어내는 feature lens처럼 작동한다.

### 3) EmbedFilter

- EmbedFilter는 hidden state 뒤에 붙는 선형 변환이다.
- 구현상 모델의 최종 norm 이후 hidden state에 `lm_embed` projection을 적용하고, 그 결과를 last-token pooling으로 읽는다.
- 즉 전체 pipeline은 아래처럼 볼 수 있다.

$$
input \rightarrow LLM \rightarrow h_{last} \rightarrow EmbedFilter \rightarrow z
$$

- `filter_ratio`는 보존할 차원의 비율을 결정한다.
- GitHub README 예시 기준으로 `filter_ratio=2`이면 전체 차원의 절반을 저장한다.
- `filter_type=edge`가 기본 실행 예시에 사용되며, 구현 코드는 `head`, `tail`, `mid`, `rand`, `ideal` 같은 variant도 포함한다.

여기서 중요한 점은 projection이 compression과 denoising을 동시에 한다는 것이다. 차원을 줄인다는 것은 단순 저장 공간 절감처럼 보이지만, 이 논문에서는 오히려 nuisance subspace를 버리는 의미도 갖는다.

### 4) Prompt interface

- official repo에는 Qwen, Llama, Mistral 계열 실행 스크립트가 포함되어 있다.
- README 기준 실행 예시는 `python run4qwen_prompteol.py --filter_ratio 2`다.
- embedding은 마지막 token hidden state에서 읽는 방식으로 해석된다.
- PromptEOL과 Echo style script가 따로 제공되는 점을 보면, EmbedFilter 자체와 prompt interface를 분리해서 볼 필요가 있다.

즉 이 논문을 구현 관점에서 보면, 핵심 알고리즘은 매우 작지만 evaluation wrapper는 task별로 꽤 민감할 수 있다.

# 4. Training / Data / Recipe

## 4-1. Data

- 이 논문은 새로운 embedding training dataset을 제안하는 논문은 아니다.
- 핵심은 기존 LLM backbones의 hidden representation을 post-hoc으로 정제하는 것이다.
- 논문 abstract는 multiple LLM backbones에서 실험했다고 설명한다.
- 공식 repo에는 Qwen, Llama, Mistral 계열 실행 파일이 포함되어 있다.

따라서 이 논문을 읽을 때 data recipe보다 중요한 것은 아래 두 가지다.

- 어떤 backbone에서 같은 frequency-token 문제가 반복되는가.
- 어떤 filter ratio와 filter type이 task별로 안정적인가.

## 4-2. Training strategy

- 추가 training objective는 없다.
- EmbedFilter는 frozen LLM 위에서 동작한다.
- 구현의 핵심 절차는 다음과 같다.

1. pretrained causal LM을 로드한다.
2. `lm_head.weight`를 float로 변환한다.
3. SVD를 수행한다.
4. 선택한 `Vh` row를 fixed linear projection으로 넣는다.
5. 기존 hidden state 대신 filtered hidden state를 embedding으로 평가한다.

이 과정을 수식으로 단순화하면 아래와 같다.

$$
W_U = U \Sigma V^T
$$

$$
F = Select(V^T)
$$

$$
z = Fh
$$

여기서 `Select`는 filter ratio와 filter type에 따라 일부 direction을 고르는 연산이다.

## 4-3. Engineering notes

- official repo는 `python run4qwen_prompteol.py --filter_ratio 2` 형태의 실행 예시를 제공한다.
- README 기준 권장 환경은 Python 3.10, torch 2.6.0, mteb 1.4.0, transformers 4.52.3이다.
- `filter_ratio=1`은 전체 차원을 보존하고, `filter_ratio=2`는 절반 차원을 저장하는 식으로 해석된다.
- 즉 retrieval index 관점에서는 storage와 search latency를 줄일 수 있는 여지가 있다.
- 다만 실제 latency gain은 vector DB, ANN index type, batch size, CPU/GPU placement, normalization 방식에 따라 달라진다.

**실무적으로 중요한 점**

- 이 방식은 embedding model을 새로 train하지 않아도 된다.
- 기존 embedding extraction code에 작은 projection만 추가하면 된다.
- 하지만 backbone별 `lm_head` geometry가 다르므로, filter ratio와 filter type은 그대로 고정하기보다 검증해야 한다.

# 5. Evaluation

## 5-1. Main results

논문 abstract가 강조하는 결과는 크게 세 가지다.

1. EmbedFilter를 붙인 LLM이 multiple LLM backbones에서 더 나은 zero-shot downstream performance를 보인다.
2. dimension을 크게 줄여도 refined embedding quality가 유지된다.
3. 따라서 retrieval index storage를 낮추고 retrieval speed를 높일 수 있는 가능성이 있다.

정확한 table별 수치는 원문 PDF와 appendix를 다시 확인해야 한다. 다만 abstract와 code를 기준으로 보면, 이 논문의 평가 포인트는 단순히 평균 점수 하나가 아니다.

- raw LLM embedding 대비 개선이 있는가.
- PromptEOL / Echo 같은 prompt interface와 함께 써도 일관성이 있는가.
- Qwen, Llama, Mistral 계열에서 같은 경향이 보이는가.
- filter ratio를 높여 차원을 줄여도 성능이 유지되는가.
- Retrieval, STS, Classification, Clustering, Reranking 등 task category별로 어떤 trade-off가 있는가.

## 5-2. What really matters in the experiments

이 논문에서 진짜 봐야 할 실험은 두 가지다.

### 1) Frequency feature 제거가 정말 semantic quality를 올리는가?

- 만약 EmbedFilter가 단순히 차원을 줄이는 PCA-like trick이라면, 성능 개선의 의미는 제한적이다.
- 반대로 frequent token subspace를 제거할수록 semantic benchmark가 좋아진다면, 논문의 mechanism claim이 힘을 얻는다.
- 따라서 raw embedding, random projection, head/tail/mid/edge filter, ideal filter 사이 비교가 중요하다.

### 2) Dimension reduction이 품질 저하 없이 가능한가?

- embedding serving에서 dimension은 곧 비용이다.
- dimension이 줄면 index memory, distance computation, network transfer cost가 줄어든다.
- 하지만 semantic quality가 무너지면 의미가 없다.
- 이 논문은 EmbedFilter가 filtering과 compression을 동시에 수행할 수 있다고 주장한다.

여기서 주의할 점은, lower dimension 자체가 항상 좋은 것은 아니라는 점이다. 도메인 검색에서는 lexical cue, entity cue, rare token cue가 중요할 수 있고, filter가 이런 신호까지 줄이면 성능이 떨어질 수 있다. 따라서 실제 서비스에서는 task-specific validation set이 필요하다.

# 6. Limitations

1. 논문 abstract와 공개 repo만으로는 모든 table의 exact score와 task별 분산을 바로 검증하기 어렵다. 최종 PDF 기준 table 확인이 필요하다.
2. frequent token direction을 줄이는 것이 항상 좋은 것은 아니다. domain-specific retrieval에서는 자주 등장하는 표현이라도 중요한 구분 feature일 수 있다.
3. EmbedFilter는 post-hoc filter이므로, embedding model을 instruction-tuned contrastive objective로 재학습한 방법과는 성격이 다르다.
4. dimension reduction의 실질적인 serving gain은 index implementation과 hardware에 따라 달라진다.
5. unembedding SVD 기반 projection은 backbone-specific이다. 한 모델에서 좋은 filter type이나 ratio가 다른 모델에서 그대로 최적이라고 단정하면 안 된다.

# 7. My Take

## 7-1. Why this matters for my work

- 이 논문은 LLM embedding을 볼 때 generation head를 버리지 말고, 오히려 representation diagnostic 도구로 쓰라는 메시지를 준다.
- RAG나 agent search pipeline에서 causal LLM을 embedding extractor로 재사용하고 있다면, EmbedFilter는 매우 싼 ablation으로 넣어볼 수 있다.
- 특히 이미 보유한 LLM checkpoint가 있고 embedding-specific training data가 부족한 상황에서는, post-hoc filter가 first-pass baseline으로 유용할 수 있다.

**중요하게 볼 포인트**

- 이 방식은 retrieval model을 새로 훈련하는 방법이 아니다.
- 대신 기존 representation에서 어떤 성분이 semantic matching을 방해하는지 찾는 방법이다.
- 그래서 성능 개선이 작더라도, 분석 도구로서 가치가 있다.

## 7-2. Reuse potential

- **Embedding debug**: sentence embedding을 vocab projection으로 보고, 상위 token이 의미 있는지 확인한다.
- **Post-hoc compression**: ANN index를 만들기 전에 unembedding 기반 projection으로 차원을 줄여본다.
- **Backbone comparison**: Qwen, Llama, Mistral 계열에서 같은 filter가 작동하는지 비교한다.
- **Domain validation**: 일반 MTEB가 아니라 사내 FAQ, code search, legal search 같은 domain retrieval set에서 filter type을 고른다.
- **Training signal design**: 나중에 embedding fine-tuning을 할 때 frequency-heavy token subspace를 regularization target으로 삼을 수 있다.

## 7-3. Follow-up papers

- Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models
- NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models
- PromptEOL: Prompting Language Models for Text Embeddings
- Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings 후속 code release

# 8. Summary

- 이 논문은 LLM text embedding의 약점을 high-frequency token alignment 문제로 해석한다.
- unembedding matrix를 output head가 아니라 hidden state를 읽는 **feature lens**로 사용한다.
- EmbedFilter는 `lm_head.weight`의 SVD에서 유도한 fixed linear projection으로 hidden state를 정제한다.
- 핵심 장점은 학습 없이 적용 가능하고, dimension reduction까지 같이 얻을 수 있다는 점이다.
- 다만 실제 효과는 backbone, prompt interface, domain retrieval set, filter ratio에 따라 검증해야 한다.
