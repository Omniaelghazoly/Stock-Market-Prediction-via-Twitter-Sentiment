Stock Market Prediction Using ARIMA and CNN-LSTM with Twitter Sentiment Analysis

⚠️ Work in Progress
This project is currently under active development. Enhancements, refactoring, and additional experiments are in progress.
The updated and finalized version will be uploaded once ready.

Overview

The stock market is a major focus for investors aiming to maximize potential profits. Consequently, interest in stock market prediction from both technical and financial perspectives continues to grow.

Despite this interest, stock market prediction remains a challenging task due to its dependence on multiple unpredictable factors such as political events and public sentiment reflected on social media platforms like Twitter.

This project proposes a hybrid forecasting approach by combining:

Traditional time series models (ARIMA)

Deep learning architectures (CNN + LSTM)

Twitter sentiment analysis

to predict stock prices more accurately.

Background
ARIMA (Autoregressive Integrated Moving Average)

ARIMA is a statistical model used to analyze and forecast time series data.

It consists of three components:

Auto-Regressive (AR)

The current value depends on previous values in the time series:
Y(t)=β1​+ϕ1​Y(t−1)+ϕ2​Y(t−2)+⋯+ϕp​Y(t−p)

p: lag order

Moving Average (MA)

Models the effect of past errors on current values:
Y(t)=β2​+θ1​ε(t−1)+θ2​ε(t−2)+⋯+θq​ε(t−q)

q: moving average order

Integrated (I)

Applies differencing to remove trends and make the data stationary:

d: differencing order

SARIMA Extension

ARIMA cannot handle seasonality, so SARIMA extends it with seasonal components.

SARIMA Parameters

p, d, q: non-seasonal parameters

P, D, Q: seasonal parameters

S: length of seasonal cycle

Model identification techniques include:

ACF and PACF plots

Grid search

AIC & BIC criteria

Augmented Dickey-Fuller (ADF) test

Note: Auto-ARIMA is used to automatically select optimal parameters.

Why CNN for Time Series?

1D Convolutional Neural Networks (CNNs) are effective for time series forecasting because they:

Remove noise

Extract local temporal patterns

Perform smoothing through convolution and pooling

Learn features automatically without manual tuning

Why LSTM for Time Series?

Long Short-Term Memory (LSTM) networks excel at learning long-term dependencies in sequential data.

Advantages:

Handles vanishing gradients

Captures temporal dependencies efficiently

Bidirectional LSTM (BiLSTM):

Learns from both forward and backward directions

Uses complete sequence information

## Methodology
### Project Pipeline

The project consists of the following stages:

- Importing libraries

- Loading stock and Twitter data

- Data preprocessing

- Exploratory Data Analysis (EDA)

- Twitter sentiment analysis

- Merging sentiment with stock data

- Time series modeling

- ARIMA baseline model

- CNN–BiLSTM deep learning model

- Model comparison and evaluation

- Real-time evaluation

- Clustering analysis

- Classification tasks

## Dataset Collection
Stock Market Data

Companies: FAANG (Meta, Apple, Netflix, Amazon, Google)

Source: Yahoo Finance

Period: Sep 30, 2021 – Sep 30, 2022

Twitter Data

~350 tweets per day per company

Used for sentiment analysis

Data Preprocessing
Text Cleaning Steps

Convert text to lowercase

Remove URLs, hashtags, symbols, and numbers

Translate emojis to text

Tokenize tweets

Remove stop words and punctuation

Apply lemmatization

Sentiment Analysis

Due to the large volume of tweets, a pre-trained sentiment model was used:

Model: twitter-xlm-roberta-base-sentiment

Trained on 198M tweets

Sentiment labels:

1: Positive

0: Neutral

-1: Negative

Daily sentiment was aggregated using the average polarity score and merged with stock market data.

Exploratory Data Analysis (EDA)

EDA was conducted to:

Understand stock price movements

Analyze sentiment trends

Identify correlations between sentiment and stock prices

Clustering & Classification
Clustering

Feature extraction: TF-IDF, Bag of Words (BoW)

Algorithms:

K-Means

Agglomerative Clustering

Classification

Deep Neural Networks (DNN)

Support Vector Machine (SVM)

Random Forest

ARIMA Model Preparation

y: target variable (next-day Open & Close)

X: features

Features used:

Open, High, Low, Close, Adj Close

Volume

Sentiment mean (P_mean)

Sentiment sum (P_sum)

Tweet count

CNN–BiLSTM Model Preparation

Total samples: 1128

Look-back window: 5 days

Features:

With sentiment: 7

Without sentiment: 6

Data Shapes

Features: (1118, 5, 7)

Targets: (1118, 1, 2)

Architecture:

1D CNN layers

Pooling layers

Bidirectional LSTM layers

Results

Twitter sentiment improved prediction accuracy

CNN–BiLSTM achieved ARIMA-level accuracy in fewer epochs

Faster learning with minimal tuning

Conclusion

Social media sentiment positively impacts stock prediction performance

CNN–BiLSTM outperforms traditional ARIMA in learning efficiency

Performance is expected to improve with more data and tuning

Future Work

Extend the dataset time range

Add macroeconomic indicators

Experiment with Transformer-based models

Deploy real-time prediction pipelines
