# Sentiment-Classification

Sentiment-Classification is a project for Sentiment Classification.(Take Yelp reviews as training input)

## Preparing Dependencies
- conda env create -f freeze.yml
- Get models and mv it to model-the-folder by following cmd (on linux)
 ```
wget "https://drive.google.com/u/0/uc?id=1Fu9IUtS96L9L7gmQNO3LLUsa0Ec2YzJj&export=download" -O "mlp_best0"
wget "https://drive.google.com/u/0/uc?id=15FqMOWCt5kJsEEvbNx8F1uZ3VR_QMlC_&export=download" -O "tfidf.pickle"
mkdir model
mv tfidf.pickle model
mv mlp_best0 model
```
## Usage
Get the prediction
```
import SentimentClassifier
text = "bed experience"
SentimentClassifier.predict(text) 
#will return 1 for positive experience, 0 for negative experience
```
## Data Resource
* [reviews download](https://www.yelp.com/dataset)


## What's New

### 3.1

* Explore other numerical features (rather than only text)
* Experiments `weighted samples` by leverage the "useful" information (a attribute provided by yelp)
* Use 'mean' to deal missing value 

### 2.4
* Bert transfer learning
* Establish & tuning `bert` model.
* Visualization data distribution

### 2.3
* Changing the way to represent the sentence vector
* Establish & tuning `LSTM` model.

### 2.2
* Establish & tuning `LinearSVC` model.
* Establish & tuning `BernoulliNB` model.
* Establish & tuning `MLPClassifier` model.
* Establish & tuning `LogisticRegression` model.
* Establish & tuning `DecisionTree` model.

### 2.1
* Create sentiment classification using w2f
* Training `word representation` model
* Explore word representation using TSNE and PCA

### 1.1
* Create sentiment classification using tf-idf

* Establish & tuning `LinearSVC` model.
* Establish & tuning `BernoulliNB` model.
* Establish & tuning `MLPClassifier` model.
* Establish & tuning `LogisticRegression` model.

## Observation

- Generate descriptive statistics
<div align="center">
	<img src="https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/GS.png" alt="Editor" width="500">
</div>

- Word Cloud - get a  glimpse of the data
<div align="center">
	<img src="https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/WC.png" alt="Editor" width="500">
</div>



- Top influence features which extract from Logistic Regression model 

<div align="center">
	<img src="https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/word%20observation_tf-idf.png" alt="Editor" width="600">
</div>


- Word frequency distribution (Black contour is normal distribution. Bule contour is actual distribution. The average of each review has 75 words in our corpus and the actually distribution is quite skew)
<div align="center">
	<img src="https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/distribution_word_freq.png" alt="Editor" width="500">
</div>


In this experiment, I use k-fold + auc_roc as the evaluation mertic to determine the best paramters of GridSearchCV. However, using k-fold will reduce variance which may make this distribution more skew. Accroding to this anlysis, we can determine proper embedding size for word representation.

- Explore word representation using PCA + cos similarity
<div align="center">
	<img src="https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/PCA_w2v.png" alt="Editor" width="500">
</div>

Words near 'delightful' are the following words - superb, wornderful and fantasti and lovely, which show the ability of capture the semtatic in langauge. 

- Sentence Length
<div align="center">
	<img src="https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/sentence%20length.png" alt="Editor" width="500">
</div>
Here, we explore the sentence length and found - the words in reviews are quit long.

- Sentence Lenght per star ranking (displying values are medium per stars)
<div align="center">
	<img src="https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/sentence%20length%20per%20star%20ranking.png" alt="Editor" width="500">
</div>


## Evaluation metrics

### 2.4

Using bert and 25% data to classify. F1 (weighted avg) got 95%.
```
>> training set 

              precision    recall  f1-score   support

         0.0       1.00      0.97      0.99      2320
         1.0       0.99      1.00      0.99      4384

    accuracy                           0.99      6704
   macro avg       0.99      0.99      0.99      6704
weighted avg       0.99      0.99      0.99      6704


>> testing set 

              precision    recall  f1-score   support

         0.0       0.97      0.88      0.92       588
         1.0       0.94      0.98      0.96      1088

    accuracy                           0.95      1676
   macro avg       0.95      0.93      0.94      1676
weighted avg       0.95      0.95      0.95      1676
```

### 2.3

Using the last state to represent a sentence (instead of using avgerage vector from  vector of tokens). The result if better than avg method like 2.2 did.

```
              precision    recall  f1-score   support

         0.0       0.91      0.94      0.93      7957
         1.0       0.94      0.91      0.92      8043

    accuracy                           0.93     16000
   macro avg       0.93      0.93      0.93     16000
weighted avg       0.93      0.93      0.93     16000

```
### 2.1 & 2.2
Using word representaion as features. The perfomance was not better than tf-idf, that because we use average vector to compute sentence vector. By this approach, word representation lose strucutre information. From this point of view, Tf-idf provided more inform than word representation did.

```
MODEL: LinearSVC(C=20, class_weight={1: 2}, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=64, tol=0.0001,
          verbose=0) 
>> training set 

              precision    recall  f1-score   support

         0.0       0.92      0.89      0.90     32043
         1.0       0.89      0.92      0.91     31957

    accuracy                           0.91     64000
   macro avg       0.91      0.91      0.91     64000
weighted avg       0.91      0.91      0.91     64000

>> testing set 

              precision    recall  f1-score   support

         0.0       0.92      0.89      0.90      7957
         1.0       0.89      0.92      0.91      8043

    accuracy                           0.90     16000
   macro avg       0.91      0.90      0.90     16000
weighted avg       0.91      0.90      0.90     16000

Training set- roc_auc score:	0.91
Testing set - roc_auc score:	0.90



MODEL: BernoulliNB(alpha=0.01, binarize=0.004, class_prior=None, fit_prior=True) 
>> training set 

              precision    recall  f1-score   support

         0.0       0.72      0.73      0.73     32043
         1.0       0.73      0.72      0.72     31957

    accuracy                           0.73     64000
   macro avg       0.73      0.73      0.73     64000
weighted avg       0.73      0.73      0.73     64000

>> testing set 

              precision    recall  f1-score   support

         0.0       0.72      0.73      0.73      7957
         1.0       0.73      0.73      0.73      8043

    accuracy                           0.73     16000
   macro avg       0.73      0.73      0.73     16000
weighted avg       0.73      0.73      0.73     16000

Training set- roc_auc score:	0.73
Testing set - roc_auc score:	0.73



MODEL: LogisticRegression(C=100, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=32, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False) 
>> training set 

              precision    recall  f1-score   support

         0.0       0.91      0.92      0.92     32043
         1.0       0.92      0.91      0.91     31957

    accuracy                           0.92     64000
   macro avg       0.92      0.92      0.92     64000
weighted avg       0.92      0.92      0.92     64000

>> testing set 

              precision    recall  f1-score   support

         0.0       0.91      0.92      0.91      7957
         1.0       0.92      0.91      0.91      8043

    accuracy                           0.91     16000
   macro avg       0.91      0.91      0.91     16000
weighted avg       0.91      0.91      0.91     16000

Training set- roc_auc score:	0.92
Testing set - roc_auc score:	0.91



MODEL: MLPClassifier(activation='relu', alpha=0.02, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=5, learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=1000,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=32, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False) 
>> training set 

              precision    recall  f1-score   support

         0.0       0.93      0.92      0.92     32043
         1.0       0.92      0.93      0.92     31957

    accuracy                           0.92     64000
   macro avg       0.92      0.92      0.92     64000
weighted avg       0.92      0.92      0.92     64000

>> testing set 

              precision    recall  f1-score   support

         0.0       0.92      0.91      0.92      7957
         1.0       0.91      0.92      0.92      8043

    accuracy                           0.92     16000
   macro avg       0.92      0.92      0.92     16000
weighted avg       0.92      0.92      0.92     16000

Training set- roc_auc score:	0.92
Testing set - roc_auc score:	0.92



MODEL: DecisionTreeClassifier(ccp_alpha=0.01, class_weight={1: 1}, criterion='entropy',
                       max_depth=3, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=64, splitter='best') 
>> training set 

              precision    recall  f1-score   support

         0.0       0.76      0.75      0.76     32043
         1.0       0.76      0.76      0.76     31957

    accuracy                           0.76     64000
   macro avg       0.76      0.76      0.76     64000
weighted avg       0.76      0.76      0.76     64000

>> testing set 

              precision    recall  f1-score   support

         0.0       0.76      0.75      0.75      7957
         1.0       0.75      0.77      0.76      8043

    accuracy                           0.76     16000
   macro avg       0.76      0.76      0.76     16000
weighted avg       0.76      0.76      0.76     16000

Training set- roc_auc score:	0.76
Testing set - roc_auc score:	0.76


```

### 1.1 

Text data created by tf-idf is high dim and spares, using simple models or linear models such as SVM, LR, NN would be better choice than tree models:
```
MODEL: LinearSVC(C=0.3, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=2018, tol=0.0001,
          verbose=0) 
>> training set 

              precision    recall  f1-score   support

         0.0       0.98      0.98      0.98     32043
         1.0       0.98      0.98      0.98     31957

    accuracy                           0.98     64000
   macro avg       0.98      0.98      0.98     64000
weighted avg       0.98      0.98      0.98     64000

>> testing set 

              precision    recall  f1-score   support

         0.0       0.94      0.95      0.94      7957
         1.0       0.95      0.94      0.94      8043

    accuracy                           0.94     16000
   macro avg       0.94      0.94      0.94     16000
weighted avg       0.94      0.94      0.94     16000

Training set- roc_auc score:	0.98
Testing set - roc_auc score:	0.94



MODEL: BernoulliNB(alpha=0.001, binarize=0.001, class_prior=None, fit_prior=True) 
>> training set 

              precision    recall  f1-score   support

         0.0       0.92      0.81      0.86     32043
         1.0       0.83      0.93      0.88     31957

    accuracy                           0.87     64000
   macro avg       0.88      0.87      0.87     64000
weighted avg       0.88      0.87      0.87     64000

>> testing set 

              precision    recall  f1-score   support

         0.0       0.90      0.79      0.84      7957
         1.0       0.82      0.92      0.86      8043

    accuracy                           0.85     16000
   macro avg       0.86      0.85      0.85     16000
weighted avg       0.86      0.85      0.85     16000

Training set- roc_auc score:	0.87
Testing set - roc_auc score:	0.85



MODEL: MLPClassifier(activation='relu', alpha=0.3, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=5, learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=1000,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=2018, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False) 
>> training set 

              precision    recall  f1-score   support

         0.0       0.95      0.96      0.96     32043
         1.0       0.96      0.95      0.96     31957

    accuracy                           0.96     64000
   macro avg       0.96      0.96      0.96     64000
weighted avg       0.96      0.96      0.96     64000

>> testing set 

              precision    recall  f1-score   support

         0.0       0.94      0.94      0.94      7957
         1.0       0.94      0.94      0.94      8043

    accuracy                           0.94     16000
   macro avg       0.94      0.94      0.94     16000
weighted avg       0.94      0.94      0.94     16000

Training set- roc_auc score:	0.96
Testing set - roc_auc score:	0.94



MODEL: LogisticRegression(C=6, class_weight={1: 1}, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=2018, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False) 
>> training set 

              precision    recall  f1-score   support

         0.0       0.98      0.98      0.98     32043
         1.0       0.98      0.98      0.98     31957

    accuracy                           0.98     64000
   macro avg       0.98      0.98      0.98     64000
weighted avg       0.98      0.98      0.98     64000

>> testing set 

              precision    recall  f1-score   support

         0.0       0.94      0.94      0.94      7957
         1.0       0.94      0.94      0.94      8043

    accuracy                           0.94     16000
   macro avg       0.94      0.94      0.94     16000
weighted avg       0.94      0.94      0.94     16000

Training set- roc_auc score:	0.98
Testing set - roc_auc score:	0.94
```
