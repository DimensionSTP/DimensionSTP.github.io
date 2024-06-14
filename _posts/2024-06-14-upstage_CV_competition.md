---
layout: single
title:  "UpStage CV competition 회고"
categories: Competition
tag: [UpStage, FastCampus, Competition, CV, Computer-Vision, Document-Classification, PyTorch, PyTorch-Lightning, Hydra-core, WandB, Optuna]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
typora-root-url: ../
---





#  UpStage AI Lab 과정 Computer Vision Competition 회고록

👉🏻[CV competition 참여 개인 깃허브](https://github.com/DimensionSTP/upstage-cv "UpStage CV competition personal github")



FastCampus에서 진행하는 UpStage AI Lab 2기를 진행하는 동안 참여했던 CV 내부 competition에 대한 회고록이다.

이전 현장 강의들이 끝나고, 온라인 강의로 넘어오면서 팀 대회 프로젝트, ML 내부 competition에 이어서 CV  competition이 진행되었다.

aistages라는 플랫폼에서 진행되었으며, 세부 주제는 document classification이다.



__대회에서의 개인 목표__

+ 높은 등수 달성하기
+ 팀원들을 리딩하며 각자에 알맞은 역할 및 업무 배분하기
+ 팀 내, 그리고 수강생 전체에 대해 파이프라인을 공유하여 오픈 소스 정신 잇기, 수강생들이 반복되는 boilerplate를 줄이고 ideation에 집중하여 조금 더 선의의 경쟁을 펼칠 수 있게 하기
+ 모델링에 있어서 이전 내가 공부 및 연구했던 것들 녹여내기

 

# 대회 세부

대회의 내용은 document classification으로, 다양한 문서 이미지를 각 label에 맞게 분류하는 image classification task이다.



+ 차량 계기판, 차량 번호판, 병원 진료 문서, 병원 입원 문서 등 다양한 문서 이미지(문서라고 하기에 애매한 이미지 포함) 존재

+ Train, Predict 데이터의 수가 비슷하고, 데이터의 절대량이 적은 Data Augmentation이 키워드가 되는 대회

+ CutMix, Blur 등의 현실적이지 않은 augmentation 존재(대회 설명에서 실제 비즈니스에서 데이터 수집 시 퀄리티를 반영하려 했다고 함)



## 대회 이슈 사항

+ 문서 이미지가 아닌 이미지들 존재
+ 억지로 데이터를 늘려놓은 듯한 augmentation 존재(CutMix, Blur, Gaussian Noise 등)
+ 병원 관련 문서에서 의도적으로 label을 세분화 함(병원비 납입 증명서, 진단서, 입원 확인서 등 양식이 거의 같고 텍스트로 구분할 수 있는 문서들)
+ Miss label 존재
+ File submission인데도 불구하고, shuffle 없이 Predict dataset의 상위 절반은 public, 하위 절반은 private score로 반영
+ 1기에서 Predict dataset에 대해 human labeling한 것이 1등을 함



## 대회 접근 방법

1기 대회 피드백을 이미 했던 패스트캠퍼스 측 강사님과 대화를 통해 이미 대회 구성부터 어설프단 것을 알게 됐다.

1기에서 한 조가 5명의 조원이 human labeling 후 교차 검증을 한 것이 1등을 했다고 들었고(데이터가 작고, shuffle이 없고, label 수가 많지 않아서 가능함), 모델링 쪽을 건드리는 것이 스코어 향상에 도움이 되지 않을 것이란 내용을 전해들었다.

또, 업스테이지 측 대회를 구성하는데 참여했던 멘토의 멘토링 시간에서 많은 실망을 했다.

너무 기본적인 augmentation 수준 정도만 알려주려고 준비해온 것이 느껴졌고, document classification에서 논문, papers with code에서 찾은 데이터셋과 각 데이터셋에 SOTA를 달성 중인 OCR method, Layout LM 등의 접근 방법들에 대해 현재 진행 중인 것을 보고하고 세부적인 피드백을 요청하니 수강생들이 이 정도 수준이나 되냐며 얼타는 것을 보고 멘토링에 기대는 하지 않기로 했다.



그럼에도 불구하고 대회는 진행해야 하니, 스코어에 중점을 두기보다 document classification 분야의 다양한 방법론들을 적용하고 실제 적용에서 겪는 이슈들을 해결하면서 개인적인 성장을 이루는 것에 중점을 뒀다.



## Modeling Method

모델의 경우 크게 3가지를 실험 및 앙상블했다.



+ CNN 혹은 ViT 기반 image classifier
+ OCR API로 추출한 text를 BERT 계열의 모델로 classification
+ Layout LM(document의 layout - OCR 후 text의 위치 정보를 포함)



Layout LM의 경우  HuggingFace Hub에서 한글 기반 사전 학습 모델이 없고, Tesseract OCR만을 사용할 수 있다는 점, Tesseract OCR에서 한글로 추출해도 tokenizer가 영어 기반이라는 문제, 한글 tokenizer로 교체한다 할지라도 embedding이 제대로 될 수 없다는 문제들로 인해 성과가 좋지 않았다.



따라서 image classifier와 text classifier 2개의 모델을 앙상블 했고, 2개의 모델에 들어가는 데이터의 퀄리티를 높이는데 신경썼다.



## Data Augmentation Method

Data Augmentation의 경우 image와 text, 2가지 측면에서 이루어졌다.



### Image Augmentation

다음은 대회 때 적용한 이미지 증강 기법들의 목록이다.



+ TTA (Test Time Augmentation) : Predict 데이터에 대해 TTA 진행 후 각 이미지의  logit 값을 soft voting 후 argmax

+ General Augmentation : 대회 데이터 셋에서 쓰인 augmentation 기법들(rotation, flip, noise, cutmix 등)을 분석하여 같은 방법으로 augmentation

+ Image Alignment : OCR text 추출이 더 잘되게 하기 위함, train 및 predict에 모두 적용 시 image classifier도 성능 향상

+ Super resolution : OCR text 추출이 더 잘되게 하기 위함, train 및 predict에 모두 적용 시 image classifier도 성능 향상

  

해당 augmentation 모두 약간의 성능 향상을 보였다.



### Text Augmentation

다음은 대회 때 적용한 텍스트 증강 기법들의 목록이다.



+ Back Translation
+ Synonym Replacement
+ Sentence Structure Modification
+ Random Deletion
+ TF-IDF를 이용한 반복 키워드 추출, 용어 통일



Back Translation 정도만 의미 있었고, 나머지 증강 기법들은 성능이 거의 동일했다(데이터 절대 수 부족).

TF-IDF를 이용해 용어를 통일 시키는 것을 Predict dataset에 적용했을 때 의미있는 성능 향상을 보였다(특정 키워드 용어를 특정 target에 매핑하는 것이 유효함).



## Training, Prediction Method

학습 및 추론에서도 여러가지 방법을 시도해봤다.



### Training Method

대회 종료 이후, 더 실험하고싶은 아쉬움이 남아 결과에 반영 및 제출은 하지못하더라도 밑의 방법들을 시도해봤다.



+ Cross-modal transformer 적용
  + Image classifier와 text classifier의 logit 결과들을 단순 ensemble하는 것이 아니라, 각 image model과 text model를 feature extractor로 사용하고, 각 feature를 transformer encoder layer에서 cross - attention 시킴

+ 학위 연구 때 사용했던 Curriculum Guided Dynamic Loss Function 적용
  + 위 cross-modal transformer에서 image model에서의 classification loss, text model에서의 classification loss, multi-modal model에서의 classification loss, 총 3개의 classification loss에 대해 학습 진행 정도(epoch)에 따라 각각의 가중치를 다르게 설정함
  + 각 single-modality의 이론적 지식 및 실제 실험 결과에서 image가 text에 비해 늦게 학습되는 것을 확인, multi-modal loss는 고정 가중치로 두고 text 가중치를 초기에 높게 두고 점점 줄여나가고, 반대로 image의 경우 초기 가중치를 낮게 두고 점점 높여 나감
  + LightningModule에서 Trainer Flag API를 이용, total epoch와 current epoch를 구할 수 있음



### Prediction Method

후처리의 경우 다른 조들과 이전 기수 수강생들이 적용했던 것처럼 사람이 수작업으로 수정하는 것을 나와 팀원들이 같이 했다.



+ 텍스트에서 특정 키워드 검출 시 특정 label로 매핑
+ Image alignment 후 사람이 확인하여 수정



아쉽게도 위의 수작업과 rule-base에 해당하는 행동들이 성능에 제일 큰 영향을 미쳤다.



## 결과

총 8조 중 6등으로 마무리했다.

모든 수강생들이 우리가 높은 등수를 기록할 것이라고 예상했으나, 의미없는 방법론, 치팅에 가까운 행동들을 하며 순위를 높이느니 의미있는 방법을 공부하고 최신 동향을 리서치하는 것으로 개인 성장을 도모하는 것이 더 낫다고 팀 내에서 합의하고 그렇게 진행했다.





# 후기

대회 그룹 회고 시간에 상위 1, 2등 팀의 발표를 보니 모델링은 아예 신경쓰지도 못하고 timm에 올라온 기본 모델, 고정 hyper-parameters만 사용했고, 대부분 데이터를 manually 수작업으로 수정하는데 대부분의 시간을 보냈더라.



이후 wrap-up 시간 때 대회 기획자의 발표를 들어보니 이래저래 좋은 말로 포장해놨지만, 결국 외주를 줘서 데이터를 받아왔는데 데이터의 전체 수가 적어서 강제로 augmentation을 했고, 이 과정에서 다른 외주가 들어가서 좀 비현실적인 augmentation 방법들이 진행되었으며, 데이터 label 종류도 적어서 병원 문서 쪽의 label을 세분화하다보니 miss-label도 생기고, 사람이 봐도 이게 무슨 label에 해당하는지 헷갈리는 것도 생긴 것 같다.



유명한 kaggler들이 모인 업스테이지가 구성한 대회가 이정도 수준이라는 것에 솔직히 많이 실망했다.

AI 보다는 사람이 수작업하는게 더 의미있는 것이 과연 AI 경진대회라고 할 수 있을까?

멘토링의 경우도 멘토들의 준비가 너무 미흡하고, 대회 기간도 2주, 실질적으론 1주로 매우 짧고 촉박한 시간이었다.