---
layout: single
title:  "ChatGPT API와 Streamlit을 이용한 간단한 음성 챗봇 개발"
categories: Project
tag: [ChatGPT, ChatGPT API, Streamlit, Chatbot, Voice Chatbot, Illustration]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
typora-root-url: ../
---



#  Upstage AI Lab 교육 과정 파이썬 프로젝트

👉🏻[프로젝트 깃허브](https://github.com/DimensionSTP/voillust-chatbot "ChatGPT API와 Streamlit을 이용한 음성 챗봇")

현재 수강 중인 Upstage AI Lab의 파이썬 교육 후 첫번째 파이썬 프로젝트의 목록 중 데이터 시각화, 웹 크롤링 등 여러가지 프로젝트가 있었지만, "STT / TTS를 이용한 음성 챗봇"이라는 주제를 선정했다.

__선정 이유__

+ 제일 재밌을 것 같아서
+ 프로젝트 목록을 보자마자 가장 먼저 구조가 머리에 떠올라서
+ 대학원 수업 때 했던 프로젝트를 접목시킬 수 있을 것 같아서
+ 모델 개발 뿐만 아니라, 다양한 인공지능 서비스의 API를 이용한 개발 또한 중요하다고 생각해서
+ 단순한 음성 챗봇 외에 추가적인 차별점을 줄 수 있을 것 같아서



__구체적인 챗봇의 기획은 "일러스트를 도와주는 음성 챗봇"이다.__

+ 이전 대학원 수업 프로젝트에서 디자이너 혹은 기획자들이 본업에 집중하면서, attention resource를 최대한 뺐기지 않는 선에서 ideation을 돕고, 일러스트 예시를 보여주는 챗봇에 대한 수요가 있다는 것을 조사를 통해 확인함
+ 최근 전참시에서 자이언티가 ChatGPT 앱을 이용해 음성으로 가사를 작사하는 등의 사용 예시를 보게 됨
+ ChatGPT의 STT, TTS 모듈인 whispher가 한국어 정확도가 높아서 ChatGPT API만으로 별다른 전처리 없이 간단한 챗봇 구현을 할 수 있겠다는 각을 봄

![chatgpt_voice_example](/images/2024-01-04-voice_chatbot/chatgpt_voice_example.webp){: .align-center}

 

# 챗봇 구성

STT, TTS, 대화, image generation 모두 ChatGPT API 이용(Whispher와 Dall-e 사용을 위해 5$ 결제)

웹페이지 구성을 위해 Streamlit 사용



## 주요 라이브러리

python>=3.8

streamlit

audio-recorder-streamlit

streamlit-float

openai

python-dotenv



## 챗봇 대화 예시

음성을 통해 챗봇과 대화하고, 어떤 그림을 "그려줘" 혹은 "그려줄래"라고 사용자가 발화하면 해당 그림을 그려준다.

챗봇은 최대한 일러스트 ideation을 도와줄 수 있도록 영어로 content 프롬프트를 작성했다.

![chatbot_example](/images/2024-01-04-voice_chatbot/chatbot_example.gif){: .align-center}



# 후기

약 1주일 정도의 프로젝트 기간이 있었지만 졸업 논문 수정, 수술 회복 등으로 인해 거의 진행을 하지 못하다가 제출 당일 아침에서 점심까지 약 반나절만에 만들었다.

Streamlit 사용 방법, ChatGPT API 사용 방법 등을 익히고 바로 바로 적용했다.

Streamlit에 바로 deploy할 수 있는 기능이 있어서, 사이드 바에 사용자가 password 형식으로 자신의 API 키를 입력하면 작동되게끔 최종 수정하려했으나 실패했다. 

문자열로 받거나 혹은 session state로 받아서 적용하는 것 모두 실패했고, API 키 입력이 먼저 되지 않으면 앱이 실행이 되지 않게끔 예외 처리 해봤으나, 계속 API Connection Error를 반복했다.

관련된 예시도 없고, 처음 Streamlit과 ChatGPT API를 사용해보는 것이라 시간 관계상 env 파일에 본인 키를 사전에 넣고 실행할 수 있도록 하는 단계에서 제출했다.

상황적으로 여유가 됐다면, 오류를 해결하거나 혹은 Flutter를 배워서 크로스 플랫폼 앱을 구축했으면 더 재밌었을 것 같았는데 여기서 마무리했다.