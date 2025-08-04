# Coughvid-COVID-Predictor
Detect COVID-19 from cough audio recordings


# Overview
This project uses the COUGHVID dataset to detect COVID-19 from cough audio recordings. The dataset consists of 25,000+ crowdsourced cough audio samples collected between April 1st, 2020 and December 1st, 2020 through a web application. Each sample is accompanied by self-reported metadata, including COVID-19 status, presence of respiratory symptoms, and demographic information. Melspectograms were used to transform the raw audio signals into an image that is suitable input for convolutional neural networks (CNN). The goal is to develop a model capable of identifying COVID-19 infections from cough audio alone, enabling a low-cost, accessible, and non-invasive method for preliminary screening that could be deployed through mobile devices or web-based platforms.

# Pre processing 
12745 valid audio files were used initially. Audi were unsilenced and splitted in multiple segments if multiple coughs were detected. A Sound-to-noise ratio based threshold was used to eliminiate non-coughs sound (sound saturated, people talking...). A pitch shift of 4 units were applied as an augmentation step for the non-healthy patient (minority class). Mel-spectograms masking was also generated to increase the sample size in our training set (~62,000 samples in training set of healthy vs non-healthy and ~15,000 in the covid vs symptomatic ). 

# Models
ConvNet: A convolutional neural network with three convolutional layers, batch normalization, ReLU activations, average pooling, and dropout, designed for basic spatial feature extraction and classification.

CNN_LSTM: Combines convolutional layers for spatial feature extraction with an LSTM to capture sequential dependencies across one spatial dimension, enabling the model to learn temporal or ordered patterns in the data.

CNN_LSTM_attention: Extends the CNN_LSTM model by incorporating a self-attention mechanism over LSTM outputs, allowing the model to dynamically weight important sequence steps for improved focus and classification performance.

# Results 
The 3 models were tested in a first task of classifying healthy patient vs non-healty (symptomatic + COVID-19). The performances were great (AUC of 0.81, 0.97 and 0.98). The metrics we choose to focus on was the F1-score because class imbalance is common in health settings and the precision vs. Recall trade-off matters. CNN_LSTM and CNN_LSTM_attention both achived a F1-score of 98%. 

However the most interesting and complex tasks was to actually clssify between symptomatic and COVID-19 patients. We only evaluated the CNN_LSTM_attention model. The performance is not incredible (AUC : 0.64 and max F1-score of 45%).  

# References 
This project was inspired by the paper "Attention-based hybrid CNN-LSTM and spectral data augmentation for COVID-19 diagnosis
from cough sound"  by S. Hamdi et al.