# Sentiment-Classification

Sentiment-Classification is a project about Sentiment Classification using unstructured text.(Take Yelp reviews as input)

## Data Resource
* [reviews download](https://www.yelp.com/dataset)

Sentiment Classification using unstructured text

## What's New
### 1.1
* create sentiment classification using tf-idf

* establish & tuning `LinearSVC` model.
* establish & tuning `BernoulliNB` model.
* establish & tuning `MLPClassifier` model.
* establish & tuning `LogisticRegression` model.

## Observation
* top influence features (from LogisticRegression model)
![image](https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/word%20observation_tf-idf.png)


## Evaluation metrics
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
```
