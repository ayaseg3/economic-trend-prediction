# economic-trend-prediction

## Overview
This data is a program code that simplifies the results of my graduation research.

For an overview of your graduation research, please read below.

[森田彩星_卒論.pdf](https://github.com/ayaseg3/economic-trend-prediction/blob/master/%E6%A3%AE%E7%94%B0%E5%BD%A9%E6%98%9F_%E5%8D%92%E8%AB%96.pdf)

## The title of the graduation research
en : Analysis of economic trends using small data sets using deep learning

ja : 深層学習を用いた少量データセットによる景気動向分析

## Requirement
* torch 1.8.1+cu111
* pytorch_lightning 1.2.7
* transformers 4.8.2
* neologdn 0.5.1
* mojimoji 0.0.11

## Installation
```bash
pip install torch==1.8.1+cu111
pip install pytorch_lightning==1.2.7
pip install transformers==4.8.2
pip install neologdn==0.5.1
pip install mojimoji==0.0.11
```

## Usage
```bash
git clone https://github.com/ayaseg3/economic-trend-prediction.git
cd economic-trend-prediction
python test.py
```

## Note
After implementing test.py, please enter the economic text.

Then, the economic situation is displayed in 5 stages.

If you want to enter a lot of text instead of command input, comment out lines 183 and 184 of test.py and uncomment lines 180.
