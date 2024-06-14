---
layout: single
title:  "UpStage NLP competition 회고"
categories: Competition
tag: [UpStage, FastCampus, Competition, NLP, Text-Summarization, HuggingFace, PyTorch, PyTorch-Lightning, Hydra-core, WandB, Optuna]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
typora-root-url: ../
---





#  UpStage AI Lab 과정 NLP Competition 회고록

👉🏻[NLP competition 참여 개인 깃허브](https://github.com/DimensionSTP/upstage-nlp "UpStage NLP competition personal github")



FastCampus에서 진행하는 UpStage AI Lab 2기를 진행하는 동안 참여했던 NLP 내부 competition에 대한 회고록이다.

ML , CV 내부 competition에 이어서 NLP  competition이 진행되었다.

aistages라는 플랫폼에서 진행되었으며, 세부 주제는 text summarization이다.



__대회에서의 개인 목표__

+ 높은 등수 달성하기
+ 팀원들을 리딩하며 각자에 알맞은 역할 및 업무 배분하기
+ 팀 내, 그리고 수강생 전체에 대해 파이프라인을 공유하여 오픈 소스 정신 잇기, 수강생들이 반복되는 boilerplate를 줄이고 ideation에 집중하여 조금 더 선의의 경쟁을 펼칠 수 있게 하기
+ Decoder-only transformer 기반(causal LM) 모델의 학습 방법론 공부하기
+ open source causal LM(Llama 2, 3, SOLAR 등)을 full-fine tuning, LoRA, QLoRA를 모두 진행해보기
  + 해당 과정에 대해 자동화 파이프라인 만들기
  + 해당 과정에 대해 실험 및 리서치를 통하여 최적의 hyper-parametes 조합 정리하기
  + 해당 자료 공유하기

+ Multi-GPU에서 SLM을 학습하며 겪을 수 있는 모든 엔지니어링 이슈 해결 및 정리하기



# 대회 세부

대회의 내용은 text summarization으로, dialogue를 학습 시켜서 golden summary를 추론하는 summarization task이다.



+ 원본 데이터가 영어인 Dialogsum dataset
+ SOLAR API를 이용하여 dialogue, summary 모두 번역됨



## 대회 이슈 사항

+ 번역투, 번역체, 번역된 고유명사등의 이슈를 극복해야 함
+ 대회 기간이 2주로 굉장히 짧음
+ 제공 서버의 자원이 굉장히 한정적임(storage 100GB, GPU RTX 3090 1장, storage 용량 초과 시 자동으로 서버 터지고 이전 결과 복구 불과)



## 대회 접근 방법

직관적인 판단 및 1기 대회 피드백을 이미 했던 패스트캠퍼스 측 강사님과 대화를 통해 짧은 생각으로 대회를 구성했단 것을 알 수 있었다.

Summarization task를 주면서 대회 기간은 고작 2주, 서버는 3090 1대, 제공되는 베이스라인은 BART 기반이다.

1기에서 1등 솔루션이 거의 토크나이저에 몰빵했단 것을 들었다.

역시나 업스테이지 측 대회를 구성하는데 참여했던 멘토의 멘토링 시간에서 많은 실망을 했다.

너무 기본적인 모델링, 토크나이저를 다루는 방법만 알려주려했다.

나의 경우 A100 40GB 4개가 달린 서버 1, H100 80GB 1개 서버 2를 자유롭게 운용가능한 상태라, 제공되는 서버에서는 BART, T5 등의 기본적인 모델만 돌려보고 서버 1, 2에서 SOLAR, Llama 3 기반의 한국어 모델을 돌렸다.

이미 full-fine tuning 및 LoRA, QLoRA를 진행 중이라고 하니, 추가적인 멘토의 답변이 RAG나 DPO를 시도해보라고해서 읭? 했었다.

그냥 LLM에 대해 여기저기서 듣기만하고 깊이가 없다는걸 느껴서 그냥 넘겼다.



그럼에도 불구하고 대회는 진행해야 하니, summarization 분야의 다양한 방법론과 모델들을 적용하고 실제 적용에서 겪는 이슈들을 해결하면서 개인적인 성장을 이루는 것에 중점을 뒀다.



## Pipeline 사전 공유

CV competition에 이어서 NLP competition에서도 pipeline을 공유했다.



다른 수강생들의 개인적으로 많은 요청이 있어서, 수요 조사를 위한 글을 올리니 모두가 참여한다고 했다.

패스트캠퍼스 측 매니저님께서 도와주셔서 따로 공유회 시간을 마련해 코드 설명 및 실습을 녹화하면서 공유회를 진행했다.



지난번, CV 대회 때 대회가 시작되고 나서 pipeline을 공유하고 공유회를 진행하니 수강생들 대부분이 짧은 대회 기간으로 인해 대부분 활용하지 못하고 끝나는걸 보고, 이번에는 대회 전에 대회 개요를 보고 미리 비슷한 데이터셋을 구해서 대회 데이터의 형식에 맞게 전처리 했고, 전처리한 데이터를 대상으로 파이프라인을 만들어서 데이터 전처리 코드와 데이터 소스, 그리고 파이프라인을 공유했다.

데이터의 경우 kaggle에서 얻은 BBC News Summarization 데이터셋이었으며, 대회 시작 후 대회 데이터에 맞게 업데이트하여 재공유했다.



## Engineering Issues

Multi, Single-GPU 서버에서 SLM을 fine-tuning하며 GPU 단에서 많은 engineering issue들을 겪고 해결했다.



### LoRA, QLoRA

총 8B, 10.7B, 13B, 23B, 30B의 CasualLM을 fine-tuning 해봤다.

Full fine-tuning 역시 진행했었지만, 성능에 차이가 거의 없으면서 시간적으로 효율적인 방법은 LoRA, QLoRA이다.



LoRA, QLoRA의 개념 및 각각의 기본적인 적용법, 그리고 기타 라이브러리 및 적용되는 기법들에 대한 설명은 이미 충분하니 무엇을 어떤 순서로 적용해야하는지만 간략히 정리해본다.

LoRA, 혹은 QLoRA를 적용하기 위해, 다음을 순서대로 적용해야 한다.

+ Model load 시 BitsAndBytesConfig 적용(QLoRA 시)
+ Model loat 시 device map을 auto로 지정하지 말 것(Multi-GPU 사용 시)
+ gradient_checkpointing_enable 적용(GPU VRAM 부하 감소)
+ prepare_model_for_kbit_training 적용(QLoRA 시)
+ enable_input_require_grads 적용
+ get_peft_model을 통해 LoRA config 적용



LoRA의 경우 HuggingFace Hub의 다른 모델들의 정보, 리포트 및 논문들을 참고 및 현재 데이터셋 사이즈와 모델 크기를 고려하여 다음과 같이 지정했다.

+ LoRA r : 64
+ LoRA a : 16
+ LoRA dropout : 0.1
+ LR : 1e-4 ~ 2e-5
+ Cosine Anealing LR Scheduler 적용
+ ETA min : LR의 1/10
+ Period : 2
+ T max : Period * total steps per epoch



Total steps per epoch의 경우 LightningModule의 Trainer Flag API의 self.trainer.num_training_batches를 통해 구할 수 있다.



### Model Parallel

모델의 사이즈가 워낙 크기 때문에, 13B 짜리 모델을 FP16으로만 올려도 이미 26GB이다.

그렇게 되면 충분한 배치 사이즈를 확보할 수 없기 때문에, A100 40GB * 4 서버에서는 model parallel을 사용했고, DeepSpeed stage 2 or 3 offload를 활용했다.



+ MoE Architecture problem in DeepSpeed stage 3
  + Model states를 full sharding하는 DeepSpeed stage 3 상, 결과에 따라 선택적으로 top k feed-forward 네트워크를 선택하는 MoE architecture를 가지는 모델을 QLoRA fine-tuning 시 에러 발생
+ Batch size in DeepSpeed stage 3
  + Model states를 full sharding하는 DeepSpeed stage 3 상, batch size가 GPU 수의 제곱 수가 되게 해야 속도 최적화가 일어남
  + 왜 LLM 학습에서 server node 구성, batch size 구성을 8의 제곱 수로 하는 지 알 수 있는 부분

+ Speed
  + 이론 상 TFLops 32, 16에서 A100 3대 성능 = H100 1대
  + 그러나, H100 1대가 A100 4대 보다 학습 및 추론에서 빠름
  + Model parallel의 특성 상 model states가 GPU들에 분할되어 올라가서 해당 model state가 쓰일 때 마다 모든 GPU에 전파해주는 통신 비용이 발생
  + A100의 통신 속도는 빠른 편이나, 그럼에도 불구하고 GPU간 통신 병목 발생
  + 일반적인 소형 모델 DDP의 경우 A100 * 4가 H100 * 1보다 훨씬 빠름
  + A100 80GB 였다면...



### Multi GPU Batch Prediction & Generation

일반적인 서비스에서는 GPU 1대에서 1개의 데이터를 실시간으로 추론하는 것이 보통이지만, 나의 경우 Multi-GPU 서버를 갖고 있고, 추론 시간을 줄이기 위해서는 Multi-GPU에서 batch prediction 및 generation을 할 필요가 있었다.



그러나, 상기 작업 진행 시 multi-gpu에서 predict 혹은 generate한 결과를 predict step마다 GPU 간에 all_gather 후 epoch end 시점에 stack해서 CPU로 보내야 한다.

하지만, text generation의 경우 generate 자체가 오래걸리는 작업이어서 멀티스레드 상에 작업 순서가 꼬여서 모두 generate 되지 않은 상태에서 중간에 CPU로 보내버림으로 결과가 잘린다.

Prediction의 경우도 용량이 매우 커서 병목 현상이 심하게 일어나다가 결과가 손실된다.



그래서 아래와 같은 방법으로 해결했다.



+ LightningModule의 batch_idx, self.device.index를 이용하여 중복되지 않게 GPU에서 step마다 바로 결과 저장
+ Prediction이 끝난 후 위 결과를 병합



평가 Score는 가중치 없이 ROUGE 1, ROUGE 2, ROUGE N의 F1 score의 단순 합인데, ROUGE 스코어를 학습 혹은 validation 과정에서 계산하는 burden이 너무 커서 단순 loss로만 tracking 및 logging 했다.

제대로 된 데이터셋이었다면 가중치가 동일하다는 점에서 ROUGE 1을 최대로 높이는 식의 치팅도 가능할 듯 하다.



## Modeling Method

모델의 경우 다음 목록들을 실험했다.



+ BART, T5 류의 Encoder-Decoder model(제공되는 서버에서 진행)
+ Llama-3, SOLAR 등의 CausalLM full fine-tuning(개인 서버 1 or 2에서 진행)
+ Llama-3, SOLAR 등의 CausalLM LoRA or QLoRA(개인 서버 1 or 2에서 진행)



결과적으로는 이준범님이 올려주신 SOLAR 기반 모델을 full fine-tuning한 것이 한국어 모델링 중에서는 가장 성능이 좋았다.

Prediction에서 각 generation step에서의 logit 값을 저장했다가 soft voting을 이용한 ensemble도 시도했지만, vocab size가 워낙에 크고 단순 greedy search generation이 아니었기 때문에 ensemble 시 매우 이상한 토큰들이 튀어나와서 해당 방법은 폐기했다.



CausalLM과 Encoder-Decoder model의 경우 학습 방법도, generation 시 설정도 다르다.

학습 시 CausalLM은 system prompt, dialogue, summary를 묶어서 prompt로 만들고, target도 prompt가 되어 next token에 대한 categorical cross entopy가 loss function이 된다

학습 시 Encoder-Decoder model은 dialogue가 데이터 summary가 target이 된다.

추론 시 CausalLM은 전체 prompt를 재생산하기 때문에 max_token_length로 summary를 포함한 예상 prompt 전체 길이, min_token_length로 summary를 제외한 예상 prompt 전체 길이를 설정해줘야하나, Encoder-Decoder model은 max_length로 summary 예상 길이를 설정해주면 된다.



## Data Engineering Method

데이터의 경우 다음 목록들을 실험했다.



### Data engineering

다음은 대회 때 적용한 기법들의 목록이다.



+ Text Cleansing : 의미 없는 특수 문자, 공백 제거
+ Back Translation
+ Dialogue : Summarization 길이 상관관계 분석
  + 높은 r 값으로 DIalogue 길이의 약 0.14가 Summarization 길이였음
  + 이후 batch prediction이 아닌, Dialogue 길이의 0.14 정도를 max length로 하여 generate 함

+ Add tokens
  + EDA를 통해 반복적으로 등장하는 단어들을 토크나이저 토큰 목록에 추가
  + Sentencepiece를 이용, 대회 데이터셋의 토큰 목록 생성(약 1100여개)
  + 상기 토큰 목록에서 EDA 결과와 겹치거나 EDA 결과 토큰을 방해할만한 유사 토큰을 제거하고 토큰 목록을 토크나이저에 추가




상기 기법들 중 Add tokens에서 유의미한 성능 향상을, 길이 상관관계 분석에서 약간의 성능 향상을 보였다.



### Predict dataset cleansing

다음은 추론 시 적용한 방법들의 목록이다.



+ 불필요한 공백 제거

+ 생성된 번역 고유 명사 오탈자 수정

+ 번역투 수정

+ 문장 구성 수정

  

사실 어떤 방법을 진행해도 모든 조가 특정 점수 대에서 머무는 유리천장을 경험했다.

대회 플랫폼이 shuffle을 하지 않고 상위 절반은 public, 하위 절반은 private이고 데이터의 절대 수가 많지가 않아서 위 방법대로 수정하면서 해당 방법이 유의미한지 확인할 수 있었다.



## Winning Method

결과적으로 마지막에 다른 조들에 비해 압도적인 스코어 차이로 1위를 할 수 있었는데, 그 방법이 허무하다.



### Restoration to Original dataset

대회 데이터셋은 근본적으로 문제가 있다.

원본이 영어인 데이터셋을 그대로 번역만 해서 사용했고, 게다가 dialogue만 번역했으면 모르겠는데, golden summaries도 번역한 것을 그대로 썼다는 것이다.

Golden summaries는 영어 데이터셋을 영어 원어민이 보고 영어로 작성한 것인데 이것을 그대로 한국어로 번역한 것을 번역한 대화 목록을 모델이 학습하여 잘 맞출 수 없이 않겠는가?

x로 y를 추론할 수 있다 할 때, f(한국어 번역 API)를 적용한 f(x)로 f(y)를 추론할 수 있다는 것은 f가 단순한 일대일 대응 함수도 아닌데, 어불성설이다.

그러기 위해서는 데이터라도 많아야하는데 그렇지도 않다.



대회 규정에는 원본 데이터셋인 DialogSum을 학습 및 추론에는 사용할 수 없지만, 분석 혹은 기타 활용에는 제약이 없었다.

따라서 한국어 번역 데이터셋을 원본 영어 데이터셋으로 복구하여 영어 모델을 돌리는 것을 시도해봤다.



+ SamSum 데이터셋 이용
  + 같은 task의 SamSum 데이터셋을 SOLAR API로 번역하여 원본, 번역본 쌍을 만듦

+ ChatGPT API 이용
  + 위 SamSum 원본 번역본 쌍, 대회 데이터셋인 Dialogsum 번역본, 그리고 Dialogsum 데이터셋의 메타 정보를 이용
  + 위 재료들을 활용하여 prompt를 수정해가며 Dialogsum 원본 데이터로 복구 시도
  + dev 데이터셋 기준 거의 정확하게 복구 해냄
  + train 데이터의 경우 수가 많아 일일이 확인하지 못했지만, dev 데이터가 완벽히 복구되었다면, train 및 test 데이터도 그럴 것이라고 생각하고 진행



### Modeling Method

이후 모델 학습은 기존 한국어 모델 과정과 같다.



+ Llama-3 영어 pretrained 모델 full-fine tuning
+ 기존 한국어에서의 데이터 엔지니어링 및 기타 모델링 기법들 모두 적용



## 결과

총 8조 중 압도적인 스코어 차이로 1등으로 마무리 했다.

Winning method가 밤을 세고, 대회 마감 당일 몇시간 전에 마무리한 것이다.

내가 직관적으로 확신한 방향, 나의 가설이 마지막에 틀렸으면 어쩌지 불안함이 엄습하여 제출하기가 갑자기 두려워졌다.

그래서 원본 데이터셋을 SOLAR API로 번역한 것을 제출하여 확인해보니 말도 안되는 스코어를 기록한 것을 보고 그동안 어떤 기법을 적용해도 모든 조의 스코어가 유리 천장 밑에서 머물렀던 것이 단순 번역, 특히 golden summaries 마저 단순 번역한 한계로 인한 내 가설이 맞음을 확신했다.

이후 winning method의 결과들을 제출해보니 과연 월등한 기존 우리 팀 내 제출물들 그리고 다른 조들의 public score를 압도하는 것을 확인했고, 대회 종료 후 private score도 마찬가지였다.





# 후기

대회 회고 시간에 1등이기에 가장 마지막 순서에 발표했다.

1팀 당 약 15분 내의 발표 시간이었지만, 대회 이전시간부터 준비하여 이런 저런 방법들을 적용해가며 연속해서 밤을 새가며 디버깅 및 적용하고, 서버 3대를 쉼없이 돌렸던 경험을 발표하는데 약 30분이라는 시간이 걸렸다.

다행이 앞 조들의 발표가 일찍 끝나고, 뒤에 수강생들이 우리 조의 발표를 기대하고 있어서 모두 경청해주었다.

놀라운건 내 총 발표 시간이 약 1시간이라는 점인데, 이후 30분 정도를 공개적으로 대회에 대한 전반적인 문제점들을 비판하는데 썼다.



비판 목록은 다음과 같다.

+ 생각이 짧은 대회 데이터 구성
+ 자사에서는 SOLAR를 홍보하나 베이스라인은 시대에 뒤처진 BART라는 점
+ 모듈화 하기 힘든 주피터 노트북에 manually args를 때려박은 것을 베이스라인으로 제공한 점(이래서 모듈화하여 확장 가능한 파이프라인을 이전 대회부터 공유해옴)
+ 멘토링에서도 다른 조들에게 한계에 맞춰 BART나 T5 정도만을 돌려보라고 추천한 점
+ 멘토링에서 우리 조에게 수준이 얕은걸 떠나서 제대로 알지 못하고 멘토링을 한 점
+ 대회 기간이 요약 task인데도 불구하고 2주로 매우 짧은 점
+ 서버의 GPU 스펙이 매우 아쉬운 점
+ 서버의 스토리지 스펙이 매우 아쉬운 점
  + 100GB 한계
  + 용량 lock을 걸어둔 것이 아니라 용량 초과 시 자동으로 서버가 터지고 이전 결과들은 로컬에 저장해두지 않으면 소실됨
    + 본인은 제공 서버에서는 기본적인 BART, T5만 돌렸음에도 5번이나 서버가 터지고 새로 발급 받고 세팅함
    + BART T5의 경우도 체크포인트 1개당 1GB는 넘음
    + Causal LM의 경우 FP 16이라 할지라도 체크포인트 1개당 10GB 이상
    + DeepSpeed stage 3로 CasualLM 학습 시, 추론을 위해 model states를 합칠 때 따로 구현하지 않고 기본 API 사용 시 FP 32로 합치므로 용량이 약 10B 모델 기준 거의 40GB에 육박
    + 모델 학습 시 top 1 checkpoint만 저장하지 않음



위 비판 목록들을 업스테이지 관계자들이 진행하는 대회 wrap-up 시간 때도 이전 그룹 회고의 비판 내용 및 우리들의 발표 내용이 거의 피드백 되지 않은 것을 보고 정식으로 질문 시간에 다시 항의했고, wrap-up 시간 이후 업스테이지측 매니저와 따로 시간을 내서 정식으로 재 비판 및 피드백을 받는 시간을 가졌다.



부디 유명 kaggler들이 모인 UpStage라는 이름에 걸맞게, 또 해당 교육 과정에서 대회를 많이 홍보하는 만큼 이후에는 개선이 있길 바란다.