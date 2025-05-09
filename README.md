# Coughvid-COVID-Predictor
 detect COVID-19 from cough audio recordings


# Overview
This project uses the COUGHVID dataset to detect COVID-19 from cough audio recordings. The dataset consists of 25,000+ crowdsourced cough audio samples collected between April 1st, 2020 and December 1st, 2020 through a web application. Each sample is accompanied by self-reported metadata, including COVID-19 status, presence of respiratory symptoms, and demographic information. Melspectograms were used to transform the raw audio signals into an image that is suitable input for convolutional neural networks (CNN). The goal is to develop a model capable of identifying COVID-19 infections from cough audio alone, enabling a low-cost, accessible, and non-invasive method for preliminary screening that could be deployed through mobile devices or web-based platforms.

# Models
CNN.py : CNN model trained on melspectogram images for COVID-19 classification

# Results 
CNN: model performance was  good - Test Accuracy: 83.48 %, Precision: 69.46 %, Recall: 71.48 %, F1-Score: 70.46 %, AUC: 80.51 %


