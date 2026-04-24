---
layout: single
title: "TradingAgents: Multi-Agents LLM Financial Trading Framework Review"
categories: Study-concept
tag: [Finance-AI, Multi-Agent, LLM-Agent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2412.20138)

TradingAgents는 LLM agent를 금융 trading에 적용하는 논문이지만, 단순히 LLM에게 매수와 매도 신호를 묻는 방식은 아니다. 이 논문은 실제 trading firm의 조직 구조를 모사해 analyst, researcher, trader, risk manager, fund manager를 분리하고, 각 agent가 맡은 역할에 맞춰 시장 정보를 수집하고 논쟁하고 의사결정하도록 만든다.

흥미로운 지점은 이 논문이 finance LLM의 핵심 병목을 model 하나의 forecasting accuracy가 아니라 **decision workflow**에서 찾는다는 점이다. 금융 시장에서 하나의 가격 신호만으로 매수와 매도를 결정하기 어렵다. 기업 fundamentals, 뉴스, social sentiment, insider transaction, technical indicator, portfolio risk가 서로 다른 시간 스케일과 신뢰도를 가진다. TradingAgents는 이 heterogeneous signal을 한 prompt에 몰아넣지 않고, 역할이 다른 agent들이 단계적으로 정리하게 만든다.

> 한 줄 요약: TradingAgents는 금융 trading을 단일 LLM prediction 문제가 아니라 **specialized agents, structured reports, debate, risk review**로 구성된 multi-agent decision pipeline으로 재정의한 framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM agent를 실제 domain workflow에 맞게 어떻게 분해할지 보여주는 좋은 case study다.
- financial decision making에서 explainability, risk control, multi-source evidence integration을 agent architecture로 다룬다.
- multi-agent debate가 단순한 role-play가 아니라 decision state와 risk approval process 안에서 어떻게 쓰이는지 볼 수 있다.

TradingAgents의 핵심은 다음과 같다. 금융 trading에서 LLM을 쓰려면 smarter predictor 하나를 만드는 것보다, **정보 수집, 반대 의견 생성, 위험 검토, 최종 승인**을 분리한 의사결정 프로토콜을 먼저 설계해야 한다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 문제는 LLM 기반 trading agent를 실제 trading organization과 유사한 구조로 만드는 것이다.
- 기존 finance LLM은 sentiment classification, report summarization, stock movement prediction, alpha factor generation처럼 특정 task에 집중하는 경우가 많다.
- 하지만 trading decision은 여러 evidence source를 종합하고, 반대 논리를 검토하고, risk limit 안에서 실행 여부를 판단해야 한다.
- 따라서 문제는 price direction을 맞히는 model 하나를 만드는 것이 아니라, 시장 정보를 decision-ready state로 변환하는 **agentic trading workflow**를 설계하는 것이다.

## 1-2. Why previous approaches are insufficient

- Single-agent trading system은 많은 정보를 한 prompt나 memory에 넣고 decision을 생성하기 쉽다. 이 경우 어떤 정보가 왜 중요한지 추적하기 어렵다.
- 기존 multi-agent finance framework도 agent를 나누기는 하지만, 실제 trading firm처럼 analyst, debate, risk control, approval workflow가 명확히 분리되지 않는 경우가 많다.
- 대부분의 agent framework는 긴 natural language message history에 의존한다. 대화가 길어질수록 중요한 정보가 희석되고, 이전 state가 왜곡되며, irrelevant text가 decision context를 오염시킬 수 있다.
- 금융 domain에서는 wrong decision의 비용이 크기 때문에 explainability와 risk review가 중요하다. 단순한 black-box signal은 실무에서 그대로 신뢰하기 어렵다.

결국 TradingAgents는 LLM의 reasoning 능력을 trading에 붙이는 논문이라기보다, **LLM reasoning을 금융 조직의 workflow로 제한하고 구조화하는 논문**에 가깝다.

# 2. Core Idea

## 2-1. Main contribution

TradingAgents의 핵심 기여는 세 가지로 정리할 수 있다.

- 첫째, trading firm을 모사하는 multi-agent organization을 만든다. Analyst Team, Researcher Team, Trader, Risk Management Team, Fund Manager가 순차적으로 decision state를 갱신한다.
- 둘째, agent 간 communication을 long chat history가 아니라 **structured report and global state** 중심으로 구성한다.
- 셋째, bull and bear debate, risky neutral safe risk debate를 통해 investment thesis와 risk control을 분리해 검토한다.

이 구조에서 중요한 것은 모든 agent가 같은 질문을 반복해서 답하지 않는다는 점이다. 각 agent는 자기 역할에 필요한 tool과 observation만 보고, 결과를 report로 남긴다. 다음 agent는 전체 대화 history를 읽는 대신 필요한 report state를 조회한다.

## 2-2. Design intuition

TradingAgents의 설계 직관은 실제 trading firm의 의사결정 process와 닮아 있다.

- Analyst는 market data를 수집하고 해석한다.
- Researcher는 bullish thesis와 bearish thesis를 충돌시킨다.
- Trader는 논쟁 결과와 market evidence를 바탕으로 buy, sell, hold를 결정한다.
- Risk manager는 trader decision을 risk profile 관점에서 재검토한다.
- Fund manager는 최종적으로 execution 여부를 승인한다.

내가 보기엔 이 구조의 핵심은 **structured disagreement**다. 단일 agent가 스스로 pros and cons를 쓰는 것과, 역할이 분리된 bull/bear agent가 서로의 argument를 공격하는 것은 다르다. 후자는 decision 전에 일부러 conflict를 만든다. TradingAgents는 이 conflict를 random discussion으로 두지 않고, stateful report와 facilitator를 통해 decision artifact로 남긴다.

이 점이 finance domain에서 특히 중요하다. 시장에서는 bullish evidence만 모아도 설득력 있는 narrative가 나오고, bearish evidence만 모아도 그럴듯한 narrative가 나온다. 좋은 trading system은 둘 중 하나를 빨리 고르는 시스템이 아니라, 두 narrative가 충돌할 때 어떤 risk를 감수할지 명시하는 시스템이어야 한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | LLM 기반 multi-agent trading firm simulation |
| Input signals | price, news, social media, insider transaction, financial statements, technical indicators |
| Key roles | Analyst, Researcher, Trader, Risk Manager, Fund Manager |
| Communication | structured reports, global state, limited natural language debate |
| Output | buy, sell, hold decision and trading rationale |
| Difference from single-agent trading | evidence gathering, debate, risk review, approval을 역할별로 분리 |

TradingAgents의 pipeline은 한 번의 LLM call로 끝나지 않는다. Analyst Team이 source별 report를 만들고, Researcher Team이 bull and bear debate를 수행한다. Trader는 이 결과를 바탕으로 transaction proposal을 만들고, Risk Management Team은 risky, neutral, safe 관점에서 다시 토론한다. 마지막으로 Fund Manager가 실행 여부를 판단한다.

## 3-2. Module breakdown

### 1) Analyst Team

Analyst Team은 네 종류의 specialist로 구성된다.

- **Fundamental Analyst**는 financial statements, earnings reports, insider transactions 등을 보고 기업의 intrinsic value와 long-term risk를 판단한다.
- **Sentiment Analyst**는 social media, public sentiment, sentiment score를 활용해 market mood와 short-term pressure를 본다.
- **News Analyst**는 news article, government announcement, macroeconomic indicator를 통해 sudden market shift 가능성을 분석한다.
- **Technical Analyst**는 MACD, RSI, Bollinger Bands 같은 technical indicator와 trading volume을 분석해 entry and exit timing에 가까운 signal을 만든다.

이 설계는 signal source별 inductive bias를 agent role로 고정한다는 점에서 실용적이다. 모든 데이터를 하나의 agent에게 주면 prompt가 복잡해지고 reasoning path가 흐려진다. 반대로 source별 analyst가 report를 만들면 이후 단계에서 어떤 evidence가 어느 source에서 왔는지 추적하기 쉽다.

### 2) Researcher Team

Researcher Team은 bullish researcher와 bearish researcher로 구성된다.

- Bullish researcher는 positive indicator, growth potential, favorable market condition을 강조한다.
- Bearish researcher는 downside risk, unfavorable signal, valuation concern, liquidity issue 등을 강조한다.
- 두 researcher는 여러 round의 debate를 수행하고, facilitator가 debate history를 검토해 prevailing perspective를 structured state로 기록한다.

여기서 debate는 단순히 흥미로운 output을 만들기 위한 장치가 아니다. 금융 의사결정에서 confirmation bias를 줄이기 위한 구조적 장치다. 특정 주식에 대해 좋은 이유만 찾으면 대부분의 asset은 그럴듯해 보인다. TradingAgents는 의도적으로 반대 agent를 두어 investment thesis가 공격받도록 만든다.

### 3) Trader Agent

Trader Agent는 analyst report와 researcher debate 결과를 종합해 trading decision을 만든다.

- analyst와 researcher의 recommendation을 평가한다.
- trade timing과 position size를 결정한다.
- buy 또는 sell order를 제안한다.
- market change와 new information에 따라 portfolio allocation 조정을 고려한다.

논문에서 trader의 역할은 price prediction model이라기보다 **decision synthesizer**에 가깝다. 여러 source에서 올라온 report를 받아 최종 transaction proposal로 바꾸는 agent다.

### 4) Risk Management Team and Fund Manager

Risk Management Team은 trader decision을 다시 세 관점에서 검토한다.

- Risky agent는 high-reward, high-risk 전략을 주장한다.
- Neutral agent는 balanced perspective를 제공한다.
- Safe agent는 conservative strategy와 risk mitigation을 강조한다.

이후 Fund Manager가 risk discussion을 검토하고, final trade approval을 수행한다. 이 구조가 중요한 이유는 trading signal과 risk tolerance를 분리한다는 점이다. 어떤 asset에 대해 positive signal이 있어도 position size, volatility, liquidity, drawdown risk에 따라 실행 여부는 달라질 수 있다.

### 5) Communication Protocol

TradingAgents의 중요한 engineering point는 communication 방식이다.

- 대부분의 agent output은 long chat history가 아니라 structured report로 저장된다.
- agent는 global state에서 필요한 report를 조회한다.
- natural language dialogue는 researcher debate와 risk debate처럼 논쟁이 필요한 순간에 제한적으로 사용된다.
- debate 결과도 다시 structured entry로 기록된다.

이 부분이 논문의 가장 실무적인 contribution이다. Multi-agent system은 agent 수를 늘리면 좋아지는 것이 아니라, state가 어떻게 흐르는지 통제할 때 좋아진다. TradingAgents는 agent 간 message passing을 줄이고, decision artifact를 report 형태로 남기는 쪽을 선택한다.

### 6) Backbone LLM choice

논문은 agent별 task complexity에 따라 backbone LLM을 다르게 사용할 수 있다고 설명한다.

- gpt-4o-mini와 gpt-4o 같은 quick-thinking model은 summarization, data retrieval, table-to-text conversion 같은 low-depth task에 사용된다.
- o1-preview 같은 deep-thinking model은 decision making, evidence-based report writing, data analysis처럼 reasoning-heavy task에 사용된다.
- analyst, researcher, trader는 깊은 reasoning이 필요한 단계로 분류된다.
- data retrieval과 API call 주변 작업은 빠른 model로 처리해 비용과 latency를 줄인다.

즉 TradingAgents는 특정 LLM 하나에 묶인 model architecture라기보다, **role-specific model routing**을 전제로 한 agent framework다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 새로운 LLM을 pretrain하거나 fine-tune하는 논문이 아니다. 핵심은 backtesting environment와 multi-modal financial data integration이다.

논문은 2024-01-01부터 2024-03-29까지의 backtesting simulation을 사용한다. 대상은 Apple, Nvidia, Microsoft, Meta, Google 등 major technology stocks이며, main result table은 AAPL, GOOGL, AMZN을 중심으로 제시된다.

사용된 data source는 다음과 같다.

| Data type | Description |
| --- | --- |
| Historical price | open, high, low, close, volume, adjusted close |
| News | Bloomberg, Yahoo, EODHD, FinnHub, Reddit 등의 daily news |
| Social media | Reddit, X/Twitter 등 public post와 sentiment score |
| Insider data | public filing과 insider transaction 기반 signal |
| Financial statements | quarterly and annual report |
| Company profile | company description, industry, financial history |
| Technical indicators | asset별 60개 standard technical indicators |

이 setup의 장점은 LLM이 처리하기 좋은 textual data와 quant signal을 함께 넣는다는 점이다. 단점은 source별 latency, data availability, point-in-time correctness를 매우 엄격하게 관리해야 한다는 점이다.

## 4-2. Training strategy

TradingAgents는 model training paper가 아니라 **agent workflow paper**에 가깝다. 따라서 training strategy는 parameter update보다 prompt, tool, role, state 설계에 해당한다.

- 모든 agent는 ReAct prompting framework를 따른다.
- Agent는 shared environment state를 보고 필요한 action을 수행한다.
- Analyst는 source별 tool을 사용해 report를 만든다.
- Researcher와 risk manager는 natural language debate를 수행하되, 결과는 structured communication protocol에 기록된다.
- Trader와 fund manager는 이전 report를 query하고 decision state를 갱신한다.

실무적으로 보면 이 논문을 재현할 때 가장 중요한 것은 LLM choice보다 state schema다. 각 report가 어떤 field를 갖고, 어떤 agent가 어느 report를 읽을 수 있으며, debate 결과가 어떤 structured state로 남는지 정의해야 한다.

## 4-3. Metrics and formulas

논문은 네 가지 evaluation metric을 사용한다.

- Cumulative Return (CR)
- Annualized Return (AR)
- Sharpe Ratio (SR)
- Maximum Drawdown (MDD)

수식은 다음처럼 이해할 수 있다.

$$
CR = \frac{V_{end} - V_{start}}{V_{start}} * 100
$$

$$
AR = \left(\left(\frac{V_{end}}{V_{start}}\right)^{1/N} - 1\right) * 100
$$

$$
SR = \frac{\bar{R} - R_f}{\sigma}
$$

$$
MDD = max_t \frac{Peak_t - Trough_t}{Peak_t} * 100
$$

금융 benchmark에서는 return만 보면 위험하다. CR과 AR이 높아도 MDD가 크면 실전 운용에서는 버티기 어렵다. 반대로 MDD가 낮아도 return capture가 약하면 trading system으로서 의미가 줄어든다. 이 논문은 SR과 MDD를 함께 보고 risk-adjusted performance를 강조한다.

## 4-4. Engineering notes

TradingAgents를 실무적으로 구현할 때 주의할 점은 다음과 같다.

- 모든 data source는 point-in-time 기준으로 조회되어야 한다. 미래 데이터가 prompt에 들어가면 backtest가 깨진다.
- Multi-agent call 수가 많기 때문에 cost budget을 먼저 산정해야 한다.
- 논문 footnote 기준으로 prediction당 11 LLM calls와 20+ tool calls가 사용된다.
- Report schema가 흔들리면 downstream agent가 이전 state를 잘못 해석할 수 있다.
- Finance domain에서는 hallucination보다 더 위험한 것이 stale data와 misaligned timestamp다.
- Trading system으로 쓰려면 execution cost, slippage, liquidity constraint, transaction fee, tax를 별도 modeling해야 한다.

내가 보기엔 TradingAgents의 구현 난도는 LLM prompting보다 **financial data plumbing**에 있다. Agent architecture가 좋아도 data timestamp와 execution assumption이 잘못되면 결과는 쉽게 과대평가된다.

# 5. Evaluation

## 5-1. Main results

논문은 Buy and Hold, MACD, KDJ+RSI, ZMR, SMA와 비교한다. Main table의 핵심 수치는 다음과 같다.

| Asset | Method | CR% | AR% | SR | MDD% |
| --- | --- | ---: | ---: | ---: | ---: |
| AAPL | TradingAgents | 26.62 | 30.50 | 8.21 | 0.91 |
| GOOGL | TradingAgents | 24.36 | 27.58 | 6.39 | 1.69 |
| AMZN | TradingAgents | 23.21 | 24.90 | 5.60 | 2.11 |

논문 기준으로 TradingAgents는 세 stock 모두에서 CR과 AR이 높게 나온다. AAPL에서는 CR 26.62, AR 30.50을 기록하고, GOOGL에서는 CR 24.36, AR 27.58을 기록한다. AMZN에서는 CR 23.21, AR 24.90을 기록한다.

하지만 이 수치는 무조건적으로 일반화하면 안 된다. Backtest 기간은 2024-01-01부터 2024-03-29까지 약 3개월이고, 논문도 LLM과 tool 사용 비용 때문에 더 긴 backtest가 필요하다는 점을 언급한다. 특히 Sharpe Ratio가 매우 높게 나온 부분은 기간이 짧고 pullback이 적었던 구간의 영향을 받을 수 있다.

## 5-2. What really matters in the experiments

이 논문에서 가장 중요한 실험 포인트는 score 자체보다 다음 세 가지다.

### 1) Agent workflow가 return signal보다 explainability를 만든다

TradingAgents는 deep learning trading model처럼 hidden feature로 decision을 내리지 않는다. 각 agent는 report와 reasoning trace를 남긴다. 이 점은 금융 시스템에서 debugging 가능성을 높인다. 어떤 decision이 sentiment 때문인지, fundamentals 때문인지, risk debate 때문인지 추적할 수 있다.

### 2) Debate가 risk review와 분리되어 있다

Researcher debate는 bullish thesis와 bearish thesis를 검토한다. Risk debate는 trader decision의 risk profile을 검토한다. 이 둘은 비슷해 보이지만 역할이 다르다. 전자는 투자 근거의 방향성을 검증하고, 후자는 position execution의 위험을 검증한다.

### 3) Short backtest의 한계를 정직하게 봐야 한다

논문은 prediction당 11 LLM calls와 20+ tool calls가 사용되었고, 이로 인해 3개월 backtest를 수행했다고 설명한다. 또한 매우 높은 Sharpe Ratio가 경험적으로 기대되는 범위를 넘는다는 점도 footnote에서 언급한다. 이 부분은 반드시 같이 읽어야 한다.

따라서 TradingAgents의 결과는 robust trading strategy가 완성되었다는 증거라기보다, multi-agent decision protocol이 financial decision task에서 의미 있는 실험 방향이라는 증거로 보는 편이 안전하다.

# 6. Limitations

1. **Backtest period가 짧다.** Main experiment는 약 3개월 구간이다. Market regime이 바뀌거나 volatility가 커지는 구간에서도 같은 결과가 유지되는지 확인이 필요하다.
2. **LLM and tool cost가 크다.** 논문은 prediction당 11 LLM calls와 20+ tool calls를 언급한다. 실시간 trading이나 large universe backtest에서는 비용이 빠르게 증가한다.
3. **Sharpe Ratio 해석에 주의가 필요하다.** 논문 footnote에서도 SR이 경험적 기대 범위를 넘는다고 설명한다. 짧은 기간과 적은 pullback의 영향을 분리해서 봐야 한다.
4. **Execution assumption이 제한적이다.** Real market에서는 spread, slippage, partial fill, liquidity, borrow cost, transaction fee, tax, market impact가 성능을 크게 바꿀 수 있다.
5. **Data leakage 관리가 어렵다.** Agent가 news, social media, financial statement, insider data를 모두 조회하기 때문에 point-in-time correctness가 중요하다.
6. **Natural language explainability가 correctness를 보장하지 않는다.** Reasoning trace가 그럴듯해도 decision이 맞다는 뜻은 아니다. LLM은 사후 합리화를 생성할 수 있다.
7. **Investment advice로 사용하면 안 된다.** GitHub README도 research purpose를 강조한다. 이 글 역시 논문 리뷰이며 금융 조언이 아니다.

# 7. My Take

## 7-1. Why this matters for my work

- 이 논문은 agent system을 설계할 때 role decomposition이 왜 중요한지 잘 보여준다.
- LLM agent의 성능을 올리는 방법이 항상 더 강한 model을 쓰는 것은 아니다. 어떤 state를 누구에게 보여줄지, 어떤 disagreement를 의도적으로 만들지, 어떤 decision artifact를 남길지가 더 중요할 수 있다.
- TradingAgents는 financial trading이라는 high-stakes domain에서 **agent architecture, tool use, state management, explainability**가 어떻게 엮이는지 보여주는 좋은 예시다.
- RAG나 document AI 시스템에서도 비슷한 교훈을 얻을 수 있다. 하나의 answer agent보다 evidence collector, critic, risk checker, final synthesizer를 분리하는 구조가 더 안정적일 수 있다.

## 7-2. Reuse potential

재사용하고 싶은 포인트는 네 가지다.

1. **Role-specific evidence collection**
   - Source별 specialist agent가 report를 만들고, downstream agent는 report만 읽게 한다.
2. **Structured communication protocol**
   - Long conversation history 대신 typed report와 global state를 사용한다.
3. **Opposing-perspective debate**
   - Bull and bear처럼 의도적으로 반대 관점을 만들고, facilitator가 decision artifact로 정리한다.
4. **Risk review before execution**
   - Final answer나 action을 바로 실행하지 않고, risk profile을 별도 agent team이 검토한다.

이 네 가지는 금융 trading뿐 아니라 enterprise decision support, legal review, medical triage, security incident analysis 같은 domain에도 적용 가능하다. 핵심은 answer generation이 아니라 decision process를 설계하는 것이다.

## 7-3. Follow-up papers

- FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design
- TradingGPT: Multi-agent System with Layered Memory and Distinct Characters for Enhanced Financial Trading Performance
- QuantAgent: Seeking Holy Grail in Trading by Self-Improving Large Language Model
- FinRobot: An Open-Source AI Agent Platform for Financial Applications using Large Language Models
- Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment
- Kronos: A Foundation Model for the Language of Financial Markets

# 8. Summary

- TradingAgents는 trading decision을 single-agent prediction이 아니라 multi-agent firm workflow로 구성한다.
- Analyst, researcher, trader, risk manager, fund manager가 역할별로 state를 갱신한다.
- 핵심 설계는 **structured reports, global state, debate, risk review**다.
- Main backtest에서는 AAPL, GOOGL, AMZN에서 높은 CR, AR, SR을 보고하지만, 기간이 짧고 LLM/tool cost가 크다.
- 실무 적용 전에는 point-in-time data, execution cost, slippage, liquidity, transaction fee, risk constraint를 반드시 별도 검증해야 한다.
