---
layout: single
title:  "DataParallel 사용 시 TorchMetrics 동기화 문제 해결"
categories: Code
tag: [multi-gpu, DataParallel, DP, TorchMetrics, Pytorch_lightning, PyTorch]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
typora-root-url: ../
---



# 문제 많은 DataParallel





약 한 달 전, 예전에 진행했던 프로젝트의 코드들을 구조화하여 재사용하기 위해 리팩토링을 진행함.

당시에는 colab을 SSH로 연결해서 local VS code로 사용하다가, 이제는 서버가 생겨서 (windows, ubuntu 각 1대씩) 서버에 맞게 코드를 수정함.

windows server에서 사용할 예정이라 DDP가 불가하여 어쩔 수 없이 DP에 맞춰서 사용했는데, metric 계산과 self.log 부분에서 에러가 남.



##### TorchMetrics의 DP mode 사용

TorchMetrics의 DP mode 사용 시 유의사항

![dp_metric](/images/2022-11-18-dp_problems/dp_metric.png)



+ 위 설명에 따르면 single forward pass 동안 metric objects의 복제를 생성하고 정리함.

+ 이렇게 되면 동기화 전에 복제본의 metric states들이 삭제되므로, dist_sync_on_step=True로 설정할 것을 권장

+ Additionaly, pytorch lightning에서는 그래서 step_end를 따로 구현해서 metric 계산과 logging을 해주는 것이 좋다.