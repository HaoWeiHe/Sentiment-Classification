# Sentiment-Classification

Sentiment-Classification is a project about Sentiment Classification using unstructured text.(Take Yelp reviews as input)

## Data Resource
* [reviews download](https://www.yelp.com/dataset)

Sentiment Classification using unstructured text

## What's New

### 2.2
* establish & tuning `LinearSVC` model.
* establish & tuning `BernoulliNB` model.
* establish & tuning `MLPClassifier` model.
* establish & tuning `LogisticRegression` model.
* establish & tuning `DecisionTree` model.

### 2.1
* create sentiment classification using w2f
* training `word representation` model
* explore word representation using TSNE and PCA


### 1.1
* create sentiment classification using tf-idf

* establish & tuning `LinearSVC` model.
* establish & tuning `BernoulliNB` model.
* establish & tuning `MLPClassifier` model.
* establish & tuning `LogisticRegression` model.

## Observation
- Top influence features (from LogisticRegression model)
![image](https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/word%20observation_tf-idf.png)


- Word frequency distribution (Black contour is normal distribution. Bule contour is actual distribution. The average of each review has 75 words in our corpus and the actually distribution is quite skew)
![image](https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/distribution_word_freq.png)

In this experiment, I use k-fold + auc_roc as the evaluation mertic to determine the best paramters of GridSearchCV. However, using k-fold will reduce variance which may make this distribution more skew.

- Explore word representation using PCA + cos similarity
![image](https://github.com/HaoWeiHe/Sentiment-Classification/blob/main/Imgs/PCA_w2v.png)
Words near 'delightful' are the following words - superb, wornderful and fantasti and lovely, which show the ability of capture the semtatic in langauge. 

## Evaluation metrics
### 2.1
Using word representaion as features. The perfomance do not better than tf-idf, that because we use average vector to compute sentence vector. By this approach, word representation lose strucutre information. From this point of view, Tf-idf provided more inform than word representation did.

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
