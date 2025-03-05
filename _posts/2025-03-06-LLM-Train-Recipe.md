---
layout: single
title:  "LLM Train Recipe"
categories: Study-concept
tag: [LLM, Train Recipe, Tulu, SmolLM2, Dataset, Fine-tuning]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
typora-root-url: ../
---





#  LLM Train Recipe

👉🏻[Tulu3 paper link](https://arxiv.org/pdf/2411.15124)

👉🏻[SmolLM2 paper link](https://arxiv.org/html/2502.02737v1)

최근 AI 연구 커뮤니티에서는 모든 학습 데이터와 파라미터를 투명하게 공개하는 모델들이 주목받고 있다.
**Tülu 3**는 Llama 3.1 기반의 post-trained models로, base 모델을 가져와서 지도학습(SFT), Preference 튜닝(DPO), 그리고 (8B 모델에 한정한) 강화학습(RLVR) 단계를 통해 성능을 극대화했다.
**SmolLM2**는 Pre-training부터 Fine-tuning까지 전 과정을 오픈 데이터로 진행한 모델로, Pre-training 데이터와 Fine-tuning 데이터가 모두 공개되어 있는 초경량 영어 모델이다.
본 리뷰에서는 두 모델의 주요 구성 요소를 세세하게 살펴보고, 각 단계마다 dataset, method, hyper-parameters를 비교해 본다.



# Model Overviews

**Tülu 3**와 **SmolLM2**의 모델 구조에 대해 간략히 정리해본다.



## Tülu 3

- **Base models:**
  - Llama 3.1 기반
  - Model parameters: 8B, 70B, 405B

- **Key points:**

  - 자체 구축한 post-train dataset으로 SFT, DPO, RLVR을 단계별로 적용
  - 각 post-train 단계에서의 hyper-parameters 및 reconstruction script 공개

  - 각 단계별로 체크포인트가 공개되어 있어, 연구자들이 쉽게 재현 및 확장이 가능



## SmolLM2

- **Base models:**
- Llama(>3.1) Architectures 기반
  - Model parameters: 135M, 360M, 1.7B
  - Pre-train부터 진행

- **Key points:**
  - 자체 구축한 pre-train, post-train dataset으로 학습
  - 각 학습 단계에서의 hyper-parameters 공개




# Data Overviews

**Tülu 3**와 **SmolLM2**의 학습 데이터셋에 대해 알아본다.



## Tülu 3

- **Supervised Fine-Tuning (SFT) 데이터:**
  - 공개 인스트럭션 데이터셋과 synthetic 데이터의 혼합
  - 다양한 태스크(대화, 수학, 코딩, 안전성 등)를 포함
  - 👉🏻[allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)

- **Preference Tuning(DPO) 데이터:**
  - SFT 단계의 출력과 타 모델의 응답을 비교하여 구성된 on-policy 데이터
  - 👉🏻[allenai/llama-3.1-tulu-3-405b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-405b-preference-mixture)
  - 👉🏻[allenai/llama-3.1-tulu-3-70b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-70b-preference-mixture)
  - 👉🏻[allenai/llama-3.1-tulu-3-8b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture)

- **RLVR(PPO) 데이터 (8B 모델 전용):**
  - GSM8K, MATH, IFEval 등 “검증 가능한 정답” 여부에 따른 보상 체계 데이터
  - 👉🏻[allenai/RLVR-GSM-MATH-IF-Mixed-Constraints](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints)



## SmolLM2

- **Pre-training 데이터:**

  - 웹 크롤 데이터(예: CommonCrawl, 뉴스, 블로그) 및 위키피디아, 도서, 논문 등

  - 약 11T tokens의 정제된 데이터를 활용

- **Supervised Fine-Tuning (SFT) 데이터:**

  - 공개 인스트럭션 데이터와 자체 synthetic 데이터의 혼합

  - 필요에 따라 온-폴리시 방식의 Preference 데이터도 활용 가능



## 학습 데이터 비교 및 공통점

- **공개성:**
  - 두 모델 모두 모든 데이터셋을 공개하여, 연구자들이 동일 조건에서 실험 재현 및 커스터마이징이 가능하도록 지원

- **데이터 믹싱 전략:**

  - Tülu 3는 포스트 트레이닝을 위한 단계별 데이터 큐레이션에 집중

  - SmolLM2는 대규모 Pre-training 데이터를 기반으로 언어 모델의 범용성을 극대화한 후, 태스크별 Fine-tuning 데이터로 성능을 보완

- **목표:**
  - 두 모델 모두 다양한 태스크를 커버하기 위해, 여러 도메인과 문체의 데이터를 혼합하는 전략을 취함



# Train Pipelines & Hyper-Parameters

**Tülu 3**와 **SmolLM2**의 학습 파이프라인 및 하이퍼 파라미터에 대해 알아본다.



## Tülu 3

**학습 단계:**

- **Supervised Finetuning (SFT):**

  - **Optimizer:** AdamW (β₁=0.9, β₂=0.999, ε=1×10⁻⁸)

  - **Learning Rate:** 8B/70B는 3×10⁻⁵, 405B는 1×10⁻⁵

  - **Batch Size:** 8B/70B는 512 (gradient accumulation 적용), 405B는 256

  - **Warmup Steps:** 8B/70B: 500 steps, 405B: 1,000 steps

  - **Total Steps:** 8B: 10,000, 70B: 15,000, 405B: 20,000

- **Preference Tuning(DPO):**

  - **Learning Rate:** 1×10⁻⁵

  - **Batch Size:** 256

  - **Total Steps:** 5,000

  - **Temperature (τ):** 0.1, **Regularization:** 0.01

- **Reinforcement Learning with Verifiable Rewards (RLVR, PPO, 8B 전용):**

  - **Algorithm:** PPO 기반 RLVR

  - **Learning Rate:** 1×10⁻⁶

  - **Batch Size:** 128 sequences/update

  - **Clip Range:** 0.2, **Value Function Coefficient:** 0.5, **Entropy Coefficient:** 0.01

  - **Total Steps:** 5,000

  - **추가:** Reward Model 학습 (Learning Rate: 2×10⁻⁵, Batch Size: 256, 3,000 steps)



## SmolLM2

**학습 단계:**

- **Pre-training:**

  - **Optimizer:** AdamW (β₁ = 0.9, β₂ = 0.98, ε = 1e-9)

  - **Learning Rate:** 1×10⁻⁴ (선형 warmup 후 cosine decay)

  - **Batch Size:** 512 sequences (분산 학습 적용)

  - **Warmup Steps:** 2,000

  - **Total Steps:** 100,000

  - **Sequence Length:** 1,024 tokens

  - **Dropout:** 0.1, **Weight Decay:** 0.01

  - **Checkpoint:** 매 5,000 steps 저장

- **Supervised Finetuning (SFT):**

  - **Optimizer:** AdamW (Pre-training과 유사 설정)

  - **Learning Rate:** 3×10⁻⁵

  - **Batch Size:** 256

  - **Warmup Steps:** 500

  - **Total Steps:** 10,000

  - **Sequence Length:** 1,024 tokens

  - **Dropout:** 0.1, **Weight Decay:** 0.1

  - **Checkpoint:** 매 1,000 steps 저장

- **Preference Tuning(DPO, Optional):**
  - **Temperature:** 0.1
  - **Regularization:** 0.01
  - **Steps**: 5,000



## 비교 및 인사이트

- **Tülu 3:**

  - Llama 3.1 기반의 모델에 대해 후처리(post-training)로 SFT, Preference 튜닝, RLVR 단계를 적용

  - 모델 크기에 따른 세밀한 파라미터 조정이 돋보임

- **SmolLM2:**

  - Pre-training부터 시작하여, 대규모 데이터로 범용 언어 모델을 구축한 후 Fine-tuning으로 특정 태스크에 맞춤

  - Pre-training 단계의 데이터 규모(약 50B 토큰)와 긴 학습 스텝이 특징

- **공통점:**

  - 모든 단계에서 AdamW optimizer 사용, warm up 및 weight decay 적용

  - 학습 데이터와 하이퍼 파라미터가 전 과정 공개되어 있음

  - 모델의 파라미터 사이즈, 각 학습 단계에 따라 하이퍼 파라미터 조정이 있음



# Model Checkpoints Information

**Tülu 3**와 **SmolLM2**의 모델 체크포인트들을 기록함.



## Tülu 3

- **8B:**

  - SFT: `allenai/Llama-3.1-Tulu-3-8B-SFT`

  - DPO: `allenai/Llama-3.1-Tulu-3-8B-DPO`

  - RLVR 최종: `allenai/Llama-3.1-Tulu-3-8B`

  - 보상 모델(RM): `allenai/Llama-3.1-Tulu-3-8B-RM`

- **70B:**

  - SFT: `allenai/Llama-3.1-Tulu-3-70B-SFT`

  - DPO: `allenai/Llama-3.1-Tulu-3-70B-DPO`

  - 최종 모델: `allenai/Llama-3.1-Tulu-3-70B`

- **405B:**

  - SFT: `allenai/Llama-3.1-Tulu-3-405B-SFT`

  - DPO 최종: `allenai/Llama-3.1-Tulu-3-405B`



## SmolLM2

- **135M:**

  - Base: `HuggingFaceTB/SmolLM2-135M`

  - Instruct: `HuggingFaceTB/SmolLM2-135M-Instruct`

- **360M:**

  - Base: `HuggingFaceTB/SmolLM2-360M`

  - Instruct: `HuggingFaceTB/SmolLM2-360M-Instruct`

- **1.7B:**

  - Base: `HuggingFaceTB/SmolLM2-1.7B`

  - Instruct: `HuggingFaceTB/SmolLM2-1.7B-Instruct`




# 요약 및 결론

Tülu 3와 SmolLM2 모두 데이터 투명성과 세밀한 학습 레시피를 바탕으로 재현 가능하고 확장성이 뛰어난 모델 구축 사례다.

- **Tülu 3:** Llama 3.1 기반 포스트 트레이닝 모델로, SFT, DPO, RLVR 단계를 통해 최종 모델 완성
- **SmolLM2:** Llama architecture 기반 모델로, Pre-train, task 별 SFT를 통해 최종 모델 완성



앞으로도 이러한 공개 자료를 바탕으로 다양한 실험과 연구가 이루어지길 기대한다.

또한, 두 모델의 레시피를 실제 학습에 적용해보는 경험이 큰 도움이 될 것이라 믿는다.

