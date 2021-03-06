# -*- coding: utf-8 -*-
"""yelp_review_sentiment_analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wPM7XWViblEMPvECQaWWA1YvQ58DsmO_
"""

!pip install scikit-learn==0.24.1

# Upload text from google drive
# !pip install PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

#Authenticate & create google drive clinet
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

#Get donwload id from google drive
downloaded = drive.CreateFile({'id':"19vqs637KK5rNgxlW-p8JUKUZi2m1zBe1"})  
downloaded.GetContentFile('yelp_reviews.csv')

def download(file_name):
  from google.colab import files
  files.download(file_name) 
# download("yelp_reviews.csv")

import pandas as pd
def load_yelp_review():
  DATA_PATH = "./yelp_reviews.csv"
  df_review = pd.read_csv("yelp_reviews.csv")
  return df_review
df_review = load_yelp_review()

# We make score 4,5 to  positive (1) and score 1,2 to  negtive(0)
def sentiment_assign(x):
  if x ==3:
    return None
  return 1 if x > 2 else 0
df_review['sentiment'] = df_review["stars"].apply( sentiment_assign )

from sklearn.preprocessing import StandardScaler
X = df_review[['useful', 'funny','cool','stars']]
scaler = StandardScaler()
Z_sk = scaler.fit_transform(X)  

Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

print(df_review[df_review.sentiment == 1].useful.median())
print(df_review[df_review.stars == 1].useful.mean())

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

n_components = 2
random_state = 64

pca = PCA(n_components = n_components, 
          random_state = random_state)

L = pca.fit_transform(Z)  # (n_samples, n_components)


plt.scatter(L[:, 0], L[:, 1])
plt.axis('equal');

import numpy as np

pcs = np.array(pca.components_) # (n_comp, n_features)

df_pc = pd.DataFrame(pcs, columns=X.columns[:])
df_pc.index = [f"No.{c} PC" for c in range(1,3)]
df_pc.style\
    .background_gradient(cmap='bwr_r', axis=None)\
    .format("{:.2}")

# Observe data, here we find out the data is not balance
empty_text = (df_review['text'].isnull() \
            | df_review['sentiment'].isnull())
df_review = df_review[~empty_text]
print(df_review['sentiment'].value_counts())

#Umbalence data preprocess - unsample data with 40,000 data for each sentiment level 
def sample_data(n):
    return pd.concat([df_review[df_review['sentiment'] == i].head(n) for i in range(0,2)])
df_resample = sample_data(40000)

df_resample['sentiment'].value_counts()

# Remove stop words (TF-IDF can make the effect of stop word lightly) 
# from gensim.parsing.preprocessing import remove_stopwords
# df_resample['text'].apply(remove_stopwords)

#Tokenize and numberic data will be removed in the stage

from gensim.utils import simple_preprocess 
df_resample['prep'] = df_resample['text'].apply(simple_preprocess)
print("Finish Tokenize")

#Stemming words and case lower
from gensim.parsing.porter import PorterStemmer
def stem(ws):
  return [PorterStemmer().stem(w) for w in ws]
df_resample['prep'] = df_resample['prep'].apply(stem)
print("Finish Stem")

#join word list to sentence
def toSentence(x):
  return " ".join(x)

df_resample['clean'] = df_resample['prep'].apply(toSentence)

df_resample['clean'].head(10)
# import matplotlib.pyplot as plt
# import seaborn as sns
# df_resample.head()

df_review.iloc[0]

from sklearn.model_selection import train_test_split

def split_train_test(data, test_size=0.2, shuffle_state = True):
    FEATURES = ['clean',"stars","useful","cool"]
    X_train, X_test, Y_train, Y_test = train_test_split(
                                                        data[FEATURES],
                                                        data['sentiment'], 
                                                        shuffle = shuffle_state,
                                                        test_size = test_size, 
                                                        random_state = 32)

    print("Term frequency (training)")
    print(Y_train.value_counts())
    print("Term frequency (testing)")
    print(Y_test.value_counts())
    
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.to_frame()    
    Y_train = Y_train.reset_index()
    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
  
    
    return X_train, X_test, Y_train, Y_test

# Call the train_test_split
X_train, X_test, Y_train, Y_test = split_train_test(df_resample)

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import words
params = {
          "ngram_range" : (1,3),
          "max_features" : 30000,
          "stop_words" :"english"
}
tfidf = TfidfVectorizer(**params)

# print(X_train['stem_tokens'])
train_tv = tfidf.fit_transform(X_train['clean'])
test_tv = tfidf.transform(X_test['clean'])
vocab = tfidf.get_feature_names()

# import pickle
# with open('vectorizer.pk', 'wb') as f:
#   pickle.dump(tfidf, f)

# def download(file_name):
#   from google.colab import files
#   files.download(file_name) 

# download("vectorizer.pk")

import numpy as np

dist = np.sum(train_tv, axis=0)
checking = pd.DataFrame(dist, columns = vocab)
checking

df_resample.iloc[0]
print(df_resample['text'].iloc[1])

# Commented out IPython magic to ensure Python compatibility.
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# %matplotlib inline
def cloud(data,backgroundcolor = 'white', width = 800, height = 600):
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

cloud(' '.join(X_train['clean']))

#Observe distribution

X_train['freq_word'] = X_train['clean'].apply(lambda x: len(str(x).split()))
X_train['unique_freq_word'] = X_train['clean'].apply(lambda x: len(set(str(x).split())))
                                                 
X_test['freq_word'] = X_test['clean'].apply(lambda x: len(str(x).split()))
X_test['unique_freq_word'] = X_test['clean'].apply(lambda x: len(set(str(x).split())))                  

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(10,5)

sns.distplot(X_train['freq_word'], bins = 90, ax=axes[0], fit = stats.norm)
(mu0, sigma0) = stats.norm.fit(X_train['freq_word'])
axes[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
axes[0].set_title("Distribution Word Frequency")
axes[0].axvline(X_train['freq_word'].median(), linestyle='dashed')
print("median of word frequency: ", X_train['freq_word'].median())


sns.distplot(X_train['unique_freq_word'], bins = 90, ax=axes[1], color = 'r', fit = stats.norm)
(mu1, sigma1) = stats.norm.fit(X_train['unique_freq_word'])
axes[1].set_title("Distribution Unique Word Frequency")
axes[1].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1)],loc='best')
axes[1].axvline(X_train['unique_freq_word'].median(), linestyle='dashed')
print("median of uniuqe word frequency: ", X_train['unique_freq_word'].median())

# # Create the visualizer and draw the vectors
# from yellowbrick.text import TSNEVisualizer

# plt.figure(figsize = [15,9])
# tsne = TSNEVisualizer()
# n = 20000

# tsne.fit(train_tv[:n], Y_train['sentiment'][:n])
# tsne.poof()

#Modeling - TF-iDF data is high dim and spare data. linear model like  SVM, NN, Bernouli Naive Byes (for binary catelogry) would be the better chioces rather than tree method

from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

kfold = StratifiedKFold( n_splits = 10 , random_state = 98,shuffle=True )

# LinearSVC

lsv = LinearSVC(random_state = 64)
param_grid2 = {}


gs_lsv = GridSearchCV(lsv, param_grid = [param_grid2], verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc')
gs_lsv.fit(train_tv, Y_train['sentiment'])#, sample_weight=weighted_sample)
gs_lsv_best = gs_lsv.best_estimator_
print(gs_lsv.best_params_)

# best params: {'C': 0.3, 'loss': 'squared_hinge'} #0.9818(with weighted) #0.9856

print(gs_lsv.best_score_)

print(df_review[df_review.sentiment == 1].useful.median())
print(df_review[df_review.useful!=0].useful.mean())

rivsed_Value = df_review[df_review.useful!=0].useful.mean()
def f(x):
  return rivsed_Value if x!=0 else 1

X_train['rivised_useful'] = X_train['useful'].apply(f)  #Z.useful.tolist()
weighted_sample = X_train['rivised_useful'].tolist()

# LinearSVC

lsv = LinearSVC(random_state = 64)


gs_lsv = GridSearchCV(lsv, param_grid = [param_grid2], verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc')
gs_lsv.fit(train_tv, Y_train['sentiment'], sample_weight=weighted_sample)
gs_lsv_best = gs_lsv.best_estimator_
print(gs_lsv.best_params_)

# best params: {'C': 0.3, 'loss': 'squared_hinge'}  #0.9845037900214797 mean: 1

print(gs_lsv.best_score_)

#Predict sentiment using SVM model
import gensim
stn = "I think I've found the worst place on google maps. "
tokens = simple_preprocess(stn)
stemm = stem(tokens)
stn_clean = " ".join(stemm)
stn_v = tfidf.transform([stn_clean])
print(gs_lsv.predict(stn_v))

bnb = BernoulliNB()
gs_bnb = GridSearchCV(bnb, param_grid = {'alpha':[0.001],# [0.001 ,0.01,  0.1]
                                         'binarize': [0.001]}, verbose = 1, cv = kfold, n_jobs = 1, scoring = "roc_auc")
gs_bnb.fit(train_tv, Y_train['sentiment'])
gs_bnb_best = gs_bnb.best_estimator_
print(gs_bnb.best_params_)

# {'alpha': 0.001, 'binarize': 0.001}

print(gs_bnb.best_score_)

MLP = MLPClassifier(random_state = 56)

mlp_param_grid = {
    'hidden_layer_sizes':[(5)],
    'activation':['relu'],
    'solver':['adam'],
    'alpha':[0.3],
    'learning_rate':['constant'],
    'max_iter':[1000]
}


gsMLP = GridSearchCV(MLP, param_grid = mlp_param_grid, cv = kfold, scoring = 'roc_auc', n_jobs= 1, verbose = 1)
gsMLP.fit(train_tv,Y_train['sentiment'])
print(gsMLP.best_params_)
mlp_best0 = gsMLP.best_estimator_

print(gsMLP.best_score_)

lr = LogisticRegression(random_state = 2018)

lr2_param = {
    'penalty':['l2'],
    'C':[6],
    'class_weight':[{1:1}],
    'solver':['lbfgs'],
    }

lr_CV = GridSearchCV(lr, param_grid = [lr2_param], cv = kfold, scoring = 'roc_auc', n_jobs = 1, verbose = 1)
lr_CV.fit(train_tv, Y_train['sentiment'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_

# {'C': 6, 'class_weight': {1: 1}, 'dual': True, 'penalty': 'l2'}

print(lr_CV.best_score_)

# Extract the coefficients from the best model Logistic Regression and sort them by index.
coefficients = logi_best.coef_
index = coefficients.argsort()

# Extract the feature names.
feature_names = np.array(tfidf.get_feature_names())

feature_names[index][0][:30]

feature_names[index][0][-30::]

index_comb = list(coefficients[0][index[0][:30]]) + list(coefficients[0][index[0][-30::]])
feature_names = list(feature_names[index][0][:30]) + list(feature_names[index][0][-30::])
plt.figure(figsize=(25,10))
barlist = plt.bar(list(i for i in range(60)), index_comb)
plt.xticks(list(i for i in range(61)) , feature_names, rotation=60 , size=15)
plt.ylabel('Coefficient magnitude',size=20)
plt.xlabel('Features',size=20)

# color the first smallest 30 bars red
for i in range(30):
    barlist[i].set_color('orange')

plt.show()

models = [gs_lsv_best,gs_bnb_best,mlp_best0,logi_best]

for model in models:
  from sklearn.metrics import classification_report
  print("MODEL: {} ".format(str(model)))
  train_predict = model.predict(train_tv)
  test_predict = model.predict(test_tv)
  print(">> training set \n")
  print(classification_report(Y_train['sentiment'],train_predict))
  print(">> testing set \n")
  print(classification_report(Y_test['sentiment'],test_predict))

  from sklearn.metrics import roc_auc_score
  roc_score1 = roc_auc_score(Y_train['sentiment'],train_predict)
  roc_score = roc_auc_score(Y_test['sentiment'],test_predict)
  print("Training set- roc_auc score:\t{:.2f}".format(roc_score1))
  print("Testing set - roc_auc score:\t{:.2f}".format(roc_score))
  print("\n\n")

