# Bangkit Capstone Project 2023 - ML Module
## Team Code: C23-PS338

There are 2 main features build using Machine Learning Approach

# Sentiment Analysis
Sentiment analysis to determine whether a comment / review is considered POSITIVE or NEGATIVE. 
There are 3 approaches to building this model:
1. Using Sklearn
- In this approach, vectorization is done using TfidfVectorizer. A collection of raw text documents is converted into a matrix of numerical features.
- Selected model: MultinomialNB<br>
Avg Val_acc: 0.82<br>
Avg Val_loss: -

2. Using a neural network model
- The model is built using 3 layers. <br>
-- Dense layer, units = 0.39 * input_data_dimension, activation='tanh' <br>
-- Dense layer, units = 5, activation='tanh' <br>
-- Dense layer, units=1, activation='sigmoid' <br>
Loss: binary crossentropy<br>
Optimizer: sgd<br>
Avg Val_acc: 0.82<br>
Avg Val_loss: 0.44

3. Using Indo-BERT transfer learning
- Tokenizer using BertTokenizer<br>
- Models use TFBertForSequenceClassification.<br>
Avg val_acc: 0.91<br>
Avg val_loss: 0.26<br>

Of the 3 approaches, we chose the model built using Indo-BERT
Models that have been deployed cannot be pushed to this github repository because the file size is too large. Therefore, we created another alternative to access this model via the link below <br>
https://drive.google.com/file/d/1GdiHcF_grUFWcP-XsyGOxqokGG-M7Jaw/view?usp=sharing

# Recommender System using Dot-Product of 2 Neural Network Model
Owner's recommender profile is built using owner's reviews and respective influencer's features. Using both owner's recommender profile and influencer's profile, we feed 2 different neural network that outputs 2 l2-normalized vectors. We get the recommender score using Dot-Product between the vectors.

# Usage
Sentiment model training can be seen in SentimentAnalysis-IndoBERT.ipynb

Recommender system model training can be seen in FINAL-Recommender-MAL-SMOTE_better_pricing.ipynb
