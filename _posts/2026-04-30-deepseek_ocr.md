---
layout: single
title: "DeepSeek-OCR: Contexts Optical Compression Review"
categories: Study-concept
tag: [DeepSeekOCR, OCR, LongContext]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2510.18234)

DeepSeek-OCR이 던지는 흥미로운 질문은 정확히 **문서를 얼마나 잘 읽느냐**가 아니라, **긴 텍스트를 2D 시각 표현으로 압축한 뒤 다시 복원할 수 있느냐**다. 즉 OCR을 단순 인식 문제가 아니라, **LLM long-context 비용 문제를 검증하기 위한 compression-decompression testbed**로 재정의한다.

이 프레이밍이 흥미로운 이유는 명확하다. 요즘 long-context 논의는 대개 attention 효율화나 memory architecture 쪽으로 흘러가는데, 이 논문은 아예 "텍스트를 계속 텍스트로 들고 있을 필요가 있는가?"를 묻는다. 문서를 이미지로 렌더링하면, 원래 수백~수천 개의 text token이 필요하던 내용을 훨씬 적은 수의 vision token으로 들고 갈 수 있다. 그리고 그 압축이 어느 정도까지 유효한지를 **OCR precision**으로 계량한다.

또 한 가지 인상적인 점은, 이 논문이 아이디어 수준의 speculative essay에 그치지 않는다는 것이다. DeepSeek-OCR은 실제 문서 OCR/파싱 모델로도 돌아가고, OmniDocBench 기준 practical performance도 제시하며, 대규모 LLM/VLM용 pretraining data production 도구로서의 가치도 함께 강조한다. 그래서 이 논문은 "OCR 성능 보고서"보다 **token budget-aware document VLM 설계 문서**로 읽는 편이 맞다.

> 한 줄 요약: DeepSeek-OCR은 OCR을 long-context optical compression의 proof-of-concept로 재해석하고, SAM + 16x compressor + CLIP + 3B MoE decoder 조합의 DeepEncoder/decoder 구조를 통해 **적은 vision token으로 문서 정보를 얼마나 복원할 수 있는지**를 정량화한 기술 리포트다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- long-context 문제를 **attention 최적화가 아니라 modality 변환**으로 푼다는 점에서 발상이 다르다.
- 문서 AI 관점에서는 OCR 정확도보다 더 중요한 **vision token budget, activation memory, multi-resolution support**를 같이 다룬다.
- 실제로는 OCR 리포트지만, 연구 관점에서는 **VLM encoder 설계와 LLM memory/forgetting 메커니즘** 쪽으로 이어질 수 있는 아이디어가 많다.

내가 보기엔 이 논문의 핵심은 "DeepSeek-OCR가 강하다"보다, **문서를 읽는 모델과 긴 문맥을 다루는 모델을 분리해서 생각할 필요가 없을 수도 있다**는 데 있다. OCR은 여기서 끝이 아니라, optical compression이라는 더 큰 문제를 검증하는 가장 측정 가능한 시작점이다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 **LLM이 긴 텍스트를 처리할 때 드는 비용이 너무 크다**는 점이다.
- 텍스트 시퀀스는 길어질수록 attention 비용이 커지고, 실제 시스템에서는 prefill latency와 memory pressure가 빠르게 증가한다.
- 저자들은 여기서 다른 질문을 던진다. "문서를 텍스트로 직접 넣지 말고 이미지로 압축하면 어떨까?" 즉, **text token을 vision token으로 치환해 더 작은 표현으로 저장한 뒤, 필요할 때 다시 텍스트로 decode**할 수 있는지를 본다.
- OCR은 이 문제를 다루기에 좋은 테스트베드다. 이미지와 텍스트 사이에 자연스러운 compression-decompression mapping이 있고, 결과도 precision/edit distance 같은 지표로 비교적 명확하게 측정할 수 있기 때문이다.
- 따라서 이 논문의 문제 설정은 "좋은 OCR 모델 만들기"보다, **문서 텍스트를 optical form으로 압축했을 때 어디까지 정보가 유지되는가**를 묻는 데 더 가깝다.

## 1-2. Why previous approaches are insufficient

- 기존 open-source VLM encoder들은 이 목적에 딱 맞지 않는다.
- dual-tower 계열은 고해상도 처리에 유리할 수 있지만, **이중 전처리와 encoder pipeline parallelism의 어려움**이 있다.
- tile-based 계열은 activation memory를 줄이지만, **native resolution이 낮고 이미지가 과도하게 잘게 쪼개져 vision token 수가 많아지기 쉽다**.
- NaViT/adaptive resolution 계열은 유연하지만, **큰 이미지에서 activation memory가 크게 늘고 시퀀스 길이가 길어져 학습/추론 비용이 커진다**.
- 기존 end-to-end OCR 연구도 주로 "정확도와 효율의 trade-off"를 다뤘지, **N개의 text token을 복원하는 데 최소 몇 개의 vision token이 필요한가**를 정면으로 묻지는 않았다.
- 결국 기존 접근의 한계는 OCR 성능 자체보다, **high-resolution 처리 + low activation + few vision tokens + multi-resolution support**를 한 시스템 안에서 동시에 최적화하지 않았다는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- DeepSeek-OCR의 핵심 기여는 OCR을 **contexts optical compression**이라는 관점으로 다시 정의한 데 있다.
- 첫째, OCR을 단순 인식이 아니라 **vision-token compression study**로 다룬다. 즉, 텍스트를 시각적으로 압축했을 때 compression ratio와 decoding precision이 어떻게 바뀌는지를 정량적으로 본다.
- 둘째, 이를 위해 **DeepEncoder**라는 새 encoder를 설계한다. window attention 중심의 perception stage와 dense global attention 중심의 knowledge stage를 직렬로 연결하고, 그 사이에 **16x token compressor**를 둔다.
- 셋째, decoder는 **DeepSeek-3B-MoE**를 사용해 압축된 latent vision token에서 다시 텍스트를 복원한다.
- 넷째, 모델을 하나의 고정 해상도로만 운용하지 않고 **Tiny / Small / Base / Large / Gundam / Gundam-M** 식의 multi-resolution regime으로 설계해, 연구용 compression study와 실제 문서 파싱을 동시에 지원한다.
- 다섯째, OCR 1.0 / OCR 2.0 / general vision / text-only 데이터를 함께 써서, 단순 free OCR뿐 아니라 **layout-aware OCR, chart parsing, formula parsing, geometry parsing, multilingual OCR, 제한된 general vision understanding**까지 묶는다.

## 2-2. Design intuition

- 이 논문의 설계 직관은 꽤 선명하다. 고해상도 문서를 다루려면 초반에는 로컬한 시각 인식이 많이 필요하지만, 그 상태 그대로 dense global attention으로 넘기면 token 수도 activation도 너무 커진다.
- 그래서 DeepSeek-OCR은 **먼저 많이 보고, 그다음 강하게 압축한 뒤, 마지막에 전역적으로 이해한다**는 순서를 택한다.
- SAM 기반 local/window attention이 먼저 세밀한 시각 패턴을 받아들이고,
- convolutional compressor가 token 수를 16x 줄인 뒤,
- CLIP 기반 global attention이 더 압축된 표현 위에서 전역적 지식을 붙인다.
- 이 구조는 "좋은 encoder 하나"보다, **어느 stage에서 token을 줄일 것인가**를 먼저 생각한 설계다.
- 그래서 DeepSeek-OCR을 OCR 모델로만 보기보다, **token-budget-aware VLM construction manual**로 읽는 편이 더 유익하다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | 긴 문서 정보를 적은 vision token으로 표현하면서도 OCR/문서 파싱 품질을 유지하는 것 |
| Key module | DeepEncoder (SAM + 16x compressor + CLIP) + DeepSeek-3B-MoE decoder |
| Core design principle | high-resolution local perception 이후 token을 강하게 압축하고, 그 뒤에 dense global attention을 적용 |
| Difference from prior work | OCR 정확도뿐 아니라 compression ratio, activation memory, multi-resolution support를 동시에 최적화 |

## 3-2. Module breakdown

### 1) OCR as optical compression

- DeepSeek-OCR의 encoder는 단순 feature extractor가 아니다.
- 이 encoder의 역할은 **이미지를 tokenizing하는 동시에, 텍스트를 담고 있는 시각 표현을 강하게 압축하는 것**이다.
- decoder는 이 압축된 latent vision token을 받아, 다시 텍스트 표현으로 복원한다.
- 즉 전체 시스템은 image-to-text OCR이지만, 개념적으로는 **text -> image-like storage -> text reconstruction**에 더 가깝다.
- 이 점이 중요하다. 그래서 이 논문의 실험은 accuracy leaderboard보다 **압축률 대비 정보 보존 곡선**에 더 큰 의미가 있다.

### 2) DeepEncoder

- DeepEncoder는 DeepSeek-OCR의 핵심이다.
- 전체 encoder는 약 **380M parameters** 규모이고, 대략 **SAM-base 80M + CLIP-large 300M**을 직렬로 연결한 구조다.
- 앞단의 SAM은 window/local attention 중심으로 고해상도 perception을 담당한다.
- 뒷단의 CLIP은 dense global attention으로 더 압축된 표현 위에서 지식과 전역 관계를 붙인다.
- CLIP은 원래 이미지를 직접 받는 patch embedding layer를 제거하고, 앞단에서 나온 token을 입력으로 받도록 바뀐다.
- 두 모듈 사이에는 **2-layer convolutional compressor**가 들어가며, vision token을 16x downsample한다.
- 예를 들어 1024x1024 이미지는 patch 기준으로 4096 token이 되지만, compressor 뒤에서는 **256 token** 수준으로 줄어 global attention에 들어간다.
- 이 구조 덕분에 "큰 이미지를 보고도 activation을 감당할 수 있고, 동시에 적은 수의 vision token만 decoder로 넘길 수 있는" operating point가 만들어진다.

### 3) Multiple resolution support

- 이 논문의 또 다른 핵심은 **한 모델이 여러 vision-token budget을 지원하도록 설계**했다는 점이다.
- native resolution 모드는 다음과 같다.
  - Tiny: 512x512, 64 tokens
  - Small: 640x640, 100 tokens
  - Base: 1024x1024, 256 tokens
  - Large: 1280x1280, 400 tokens
- 여기에 dynamic resolution 모드로 **Gundam**과 **Gundam-M**이 추가된다.
- Gundam은 local tiles + global view를 결합해 **nx100 + 256** token 형태를 만든다. 특히 신문처럼 해상도는 높고 텍스트 밀도도 높은 문서에 대응하기 위한 모드다.
- Gundam-M은 더 큰 local/global 조합을 쓰는 continued training 버전이다.
- 즉 DeepSeek-OCR은 단순히 "모델 하나"가 아니라, **같은 architecture 위에서 token budget을 가변적으로 운영하는 실험 프레임워크**이기도 하다.

### 4) The MoE decoder

- decoder는 **DeepSeek-3B-MoE**를 쓴다.
- 추론 시에는 **64 routed experts 중 6개 + 2 shared experts**를 활성화하며, 활성 파라미터 수는 약 **570M** 수준이다.
- 저자들의 해석대로 보면, 이 decoder는 **3B model의 표현력**과 **500M급 small model에 가까운 inference 효율** 사이의 타협점이다.
- 설계상 포인트는, 복잡한 reconstruction을 아주 큰 dense decoder로 미는 대신, **domain-centric OCR/VLM에 맞는 작은 MoE decoder**를 붙였다는 점이다.
- 결과적으로 encoder가 information bottleneck을 잘 만들면, decoder는 그 압축 표현을 텍스트로 되돌리는 역할에 집중할 수 있다.

### 5) Promptable OCR and deep parsing

- DeepSeek-OCR은 단일 free OCR 모델로만 동작하지 않는다.
- prompt를 통해 **layout-aware output / non-layout OCR / figure parsing**을 제어할 수 있다.
- 특히 저자들이 "deep parsing"이라고 부르는 기능이 흥미롭다. 문서 안의 차트, 화학식, 기하 이미지, 자연 이미지에 대해 **2차 model call**로 더 깊은 구조화 결과를 뽑는다.
- 이 부분은 그냥 부가 기능이라기보다, OCR 1.0을 넘어 **OCR 2.0 / document figure understanding**까지 한 interface 안에 넣으려는 시도라고 보는 편이 맞다.

# 4. Training / Data / Recipe

## 4-1. Data

- 데이터 엔진은 크게 **OCR 1.0 / OCR 2.0 / general vision / text-only** 네 축으로 구성된다.

- **OCR 1.0 data**는 전통적인 문서 OCR과 scene OCR 중심이다.
  - 인터넷에서 수집한 PDF 문서 **30M pages**를 사용하며, 약 **100 languages**를 포함한다.
  - 이 중 중국어/영어가 약 25M pages, 그 외 언어가 5M pages다.
  - 문서 데이터는 **coarse annotation**과 **fine annotation** 두 종류로 구성된다.
  - fine annotation은 중국어/영어 각각 2M pages 규모이며, layout model과 OCR model을 이용해 **layout + text가 interleaved된 sequence**로 만든다.
  - minority language 쪽은 small patch data로 GOT-OCR2.0 스타일 recognition 모델을 학습해 **600K data flywheel**을 돌린다.
  - 추가로 **3M Word data**를 수집해 layout 없는 고품질 image-text pair도 만든다. 이 데이터는 formula나 HTML table 쪽에 특히 도움이 된다고 적는다.
  - 자연 장면 OCR은 중국어/영어 각각 **10M** 샘플을 사용하며, LAION/Wukong 이미지에 PaddleOCR 라벨을 붙인다.

- **OCR 2.0 data**는 더 구조적인 artificial image parsing을 겨냥한다.
  - chart parsing은 pyecharts와 matplotlib로 **10M images**를 렌더링한다.
  - 이때 OneChart의 dictionary format 대신 **HTML table format**을 라벨로 써 token을 절약한다.
  - chemical formula parsing은 PubChem의 **SMILES**를 RDKit으로 렌더링해 **5M image-text pairs**를 만든다.
  - plane geometry parsing은 Slow Perception 방식을 따라 **1M samples**를 만들고, translation-invariant augmentation도 넣는다.

- **general vision data**는 caption, detection, grounding 같은 작업을 위해 들어간다.
  - 다만 중요한 점은, 저자들이 직접 **DeepSeek-OCR is not a general VLM model**이라고 선을 긋는다는 것이다.
  - 이 데이터는 전체의 **20%**만 차지하며, 목적도 general vision SOTA가 아니라 **general vision interface를 보존하는 것**이다.

- **text-only data**도 별도로 들어간다.
  - 전체의 **10%**를 차지하고, 길이는 **8192 tokens**로 맞춘다.
  - 결국 최종 DeepSeek-OCR 학습에서 데이터 비율은 **OCR 70% / general vision 20% / text-only 10%**다.

## 4-2. Training strategy

- 학습 파이프라인은 크게 두 단계다.
  1. **DeepEncoder를 독립적으로 학습**
  2. **완성된 DeepEncoder 위에 DeepSeek-OCR 전체를 학습**

- DeepEncoder 학습은 Vary를 따라 **compact language model + next-token prediction** 프레임워크를 사용한다.
- 여기에는 OCR 1.0, OCR 2.0, 그리고 LAION에서 샘플한 **100M general data**가 들어간다.
- 이 단계는 **2 epochs**, **batch size 1280**, **AdamW + cosine annealing**, **learning rate 5e-5**, **sequence length 4096**로 진행된다.

- 이후 DeepSeek-OCR 전체 학습은 HAI-LLM 플랫폼에서 수행된다.
- 전체 모델은 **4-way pipeline parallelism**으로 나뉜다.
  - PP0: SAM + compressor (vision tokenizer, frozen)
  - PP1: CLIP part (unfrozen)
  - PP2 / PP3: DeepSeek-3B-MoE decoder 12 layers를 6층씩 분할
- 학습 자원은 **20 nodes x 8 A100-40G**, data parallelism은 **40**, global batch size는 **640**이다.
- optimizer는 AdamW, scheduler는 step-based, initial learning rate는 **3e-5**다.
- 보고된 처리량은 **text-only 90B tokens/day**, **multimodal 70B tokens/day**다.

- 연구용 확장 모드인 **Gundam-M**은 별도 구조가 아니라, 이미 학습된 DeepSeek-OCR 위에서 **6M sampled data**로 continued training한 버전이다.
- 이 점도 실용적이다. 해상도 모드를 전부 한 번에 키우는 대신, **load balancing이 가능한 범위에서 기본 모델을 만들고, 더 무거운 모드는 후속 적응**으로 해결한다.

## 4-3. Engineering notes

- 이 논문은 architecture novelty 못지않게 **training/deployment practicality**를 강하게 의식한다.
- DeepEncoder 앞단의 SAM과 compressor를 **frozen vision tokenizer**처럼 쓰고, CLIP 이후부터 학습하는 방식은 꽤 실용적이다. 계산과 안정성을 동시에 잡으려는 선택으로 보인다.
- coarse label과 fine label, layout output과 non-layout output을 **prompt로 구분**하는 방식도 중요하다. annotation schema 차이를 별도 모델이 아니라 interface 차원에서 흡수한다.
- multi-resolution 지원도 연구용 gimmick이 아니라, 실제 문서 종류별 token budget 조정이라는 practical need와 이어져 있다.
- production 관점의 메시지도 강하다. abstract에서는 **single A100-40G로 200k+ pages/day**의 data generation 능력을 언급하고, 본문에서는 **20 nodes 기준 33M pages/day** 규모의 LLM/VLM pretraining data production 가능성을 말한다. 즉 이 모델은 online OCR engine인 동시에 **data factory** 역할도 의식한 설계다.

# 5. Evaluation

## 5-1. Main results

| Setting | What the paper reports | Why it matters |
| --- | --- | --- |
| Fox compression study | 10x 이내 압축에서는 decoding precision이 약 97% 수준까지 가능하고, 20x 근처 압축에서도 약 60%를 유지 | 단순 OCR 정확도보다 **compression boundary**를 계량화했다는 점이 핵심 |
| OmniDocBench practical OCR | 100 vision tokens로 GOT-OCR2.0(256 tokens)을 넘고, 400 tokens로 강한 비교군에 근접하며, 800 미만 token으로 MinerU2.0(약 7000 tokens)을 앞선다고 보고 | 문서 OCR에서도 **token efficiency**가 실제 성능과 함께 중요하다는 점을 보여줌 |
| Document-type analysis | 슬라이드는 64 tokens, 책/리포트는 100 tokens로도 괜찮지만, 신문은 Gundam/Gundam-M 수준이 필요 | 문서 종류에 따라 필요한 token budget이 크게 달라짐 |

- Fox benchmark 실험은 이 논문의 핵심 메시지를 가장 직접적으로 보여준다.
- 저자들은 Fox의 **English documents 100 pages**를 골라, ground-truth text를 tokenizer로 다시 토큰화한 뒤 text token 수가 **600-1300**인 구간만 따로 평가한다.
- 여기서 Tiny(64 tokens)와 Small(100 tokens) 모드의 precision/compression 곡선을 측정한다.
- 논문이 내리는 결론은 단순하다.
  - **10x 전후 압축까지는 상당히 높은 precision**을 기대할 수 있고,
  - 그 이상에서는 성능이 떨어지지만,
  - **20x 가까이 압축해도 완전히 붕괴하지는 않는다**.
- 이건 long-context 연구 관점에서 꽤 중요한 신호다. 완전 무손실이 아니어도, 과거 context를 "읽을 수는 있지만 조금 흐린 형태"로 저장할 수 있다는 뜻이기 때문이다.

- OmniDocBench 결과는 practical value를 보여준다.
- 논문 기준으로 보면,
  - **Small / 100 tokens**는 GOT-OCR2.0의 256-token 설정보다 낫고,
  - **Large / 400 tokens (285 valid)**는 강한 end-to-end 비교군에 근접하며,
  - **Gundam / 795 tokens**는 약 6790 tokens를 쓰는 MinerU2.0보다 낮은 edit distance를 보고한다.
- 여기서 중요한 건 headline SOTA보다, **비슷하거나 더 나은 품질을 얼마나 적은 vision token으로 냈는가**다.

- qualitative study도 의외로 중요하다.
- DeepSeek-OCR은 차트, 화학식, 기하 이미지, 자연 이미지를 **deep parsing** 모드로 처리할 수 있고,
- multilingual 쪽에서는 **nearly 100 languages**를 다룬다고 적는다.
- 또 general vision understanding도 일부 지원한다.
- 이건 "범용 VLM을 대체했다"는 뜻은 아니지만, 최소한 OCR 전용 박스 안에 갇히지 않게 설계했다는 뜻이다.

## 5-2. What really matters in the experiments

- 이 논문에서 진짜 중요한 지표는 edit distance 자체보다 **compression ratio 대비 recoverability**다.
- 보통 OCR 모델은 "얼마나 정확한가"만 보면 되지만, DeepSeek-OCR은 거기에 **몇 개의 vision token으로 그 정확도를 냈는가**가 추가된다.
- 그래서 동일한 OmniDocBench 점수라도, 수천 token을 쓰는 모델과 수백 token을 쓰는 모델은 해석이 다르다.
- 또 Fox 실험은 단순 accuracy benchmark가 아니라, **vision token budget을 바꿨을 때 precision이 어떻게 무너지는지**를 보여주는 boundary study다.
- 이 점에서 DeepSeek-OCR은 OCR 논문인 동시에, **VLM token allocation study**이기도 하다.
- 다만 practical OCR benchmark와 long-context compression benchmark는 아직 동일하지 않다. 저자들 스스로도 OCR은 proof-of-concept일 뿐이고, 진짜 context compression 검증은 future work라고 인정한다.

# 6. Limitations

1. **OCR만으로 true context compression을 다 검증한 것은 아니다.** 저자들도 OCR alone is insufficient하다고 적고, 향후 digital-optical text interleaved pretraining이나 needle-in-a-haystack 평가가 필요하다고 말한다.
2. **Fox compression study의 범위가 좁다.** English document subset, 100 pages, 600-1300 tokens 구간에 한정된 실험이라서, 이것만으로 일반적인 long-context 기억 곡선을 단정하긴 어렵다.
3. **"좋은 OCR"과 "좋은 long-context memory"는 아직 같은 문제가 아니다.** 문서를 이미지로 렌더링해 저장하는 방식이 multi-turn dialogue, code context, tool traces 같은 비문서형 이력에도 잘 작동하는지는 아직 검증되지 않았다.
4. **general vision capability는 제한적이다.** 저자들도 이 모델을 general VLM이 아니라고 분명히 적는다. general vision data는 interface preservation용 20% 정도일 뿐이다.
5. **데이터 엔진의 재현 비용이 높다.** 30M document pages, OCR 2.0 synthetic data, multilingual flywheel, layout/OCR teacher model 활용 등은 아이디어는 공개돼도 실제로 재구현하려면 상당한 데이터/엔지니어링 비용이 든다.
6. **benchmark headline은 비교 조건을 조심해서 읽어야 한다.** 논문은 주로 end-to-end 모델 안에서의 경쟁력을 강조하므로, classical pipeline OCR이나 proprietary OCR 전체를 한 줄로 정리해 해석하면 과장이 될 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

- 내가 이 논문을 흥미롭게 본 이유는, OCR을 다시 **LLM systems problem**으로 돌려놨기 때문이다.
- 요즘 long-context는 attention kernel, state-space model, retrieval memory 쪽으로 많이 논의되는데, 이 논문은 훨씬 다른 축에서 접근한다. **어차피 오래된 문맥은 완전한 텍스트 fidelity가 꼭 필요하지 않을 수도 있다**는 가정이다.
- Document AI 관점에서도 의미가 있다. 보통 OCR/VLM 논문은 읽기 성능만 강조하지만, 실제 문서 시스템은 **token budget, latency, activation memory, multi-resolution serving**이 더 중요할 때가 많다.
- 그래서 이 논문은 OCR 논문이면서도, **VLM encoder design과 memory system design을 동시에 생각하게 만드는 문서**다.

## 7-2. Reuse potential

- 바로 재사용 가능한 포인트가 꽤 많다.
  - **window attention -> early compression -> global attention**이라는 encoder 설계 원리
  - **single model, multiple token budgets**라는 multi-resolution regime
  - layout-aware / non-layout / figure parsing을 prompt interface로 분리한 방식
  - OCR를 data generation engine으로 보는 관점
- 특히 encoder 설계는 OCR 외에도 document-heavy MLLM에 바로 응용할 수 있다. 고해상도 perception을 먼저 하고, global reasoning은 압축 뒤에 하라는 메시지는 꽤 보편적이다.
- 더 흥미로운 건 long-context memory 쪽이다. 이 논문 마지막의 "forgetting mechanism" 해석은 아직 speculative하지만, **오래된 context를 progressively blurred image로 바꾸는 메모리 계층**은 agent 시스템에서도 실험해볼 가치가 있다.
- 다만 전체 recipe를 그대로 복제하기보다는, **optical compression이라는 개념을 자기 시스템에 어디까지 이식할 수 있는지**를 따져 읽는 편이 낫다.

## 7-3. Follow-up papers

- GOT-OCR 2.0
- OLMOCR
- Qwen2.5-VL Technical Report
- PaddleOCR 3.0 Technical Report

# 8. Summary

- DeepSeek-OCR은 OCR을 **contexts optical compression**의 proof-of-concept로 재해석한 논문이다.
- 핵심은 SAM + 16x compressor + CLIP으로 이루어진 DeepEncoder와 DeepSeek-3B-MoE decoder 조합이다.
- 이 구조는 high-resolution 입력을 감당하면서도 vision token 수를 강하게 줄여, compression ratio와 OCR 품질 사이의 경계를 계량한다.
- practical OCR 관점에서도 적은 vision token으로 강한 OmniDocBench 성능을 보이며, 문서 종류별 token budget 차이도 보여준다.
- 다만 아직은 OCR 기반 초기 탐색에 가깝고, true long-context memory나 general multimodal context compression으로 일반화하려면 후속 검증이 더 필요하다.
