# -*- coding: utf-8 -*-
"""yelp_review_sentiment_analysis_w2v.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ly_Yiyk6Rm_gkkl-uj2XsKmAEW7MrZuG
"""

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

# remove empty data
empty_text = (df_review['text'].isnull() \
            | df_review['sentiment'].isnull())
df_review = df_review[~empty_text]

NUM_FEATURES = 200

#resampling data preprocess - 40,000 data for each sentiment level 
def sample_data(n):
    return pd.concat([df_review[df_review['sentiment'] == i].head(n) for i in range(0,2)])
df_resample = sample_data(40000)

df_resample['sentiment'].value_counts()

# remove stop word - since we prep to centroid word vect to represent sentence, remove stop word can reduce the effect of noise
from gensim.parsing.preprocessing import remove_stopwords
df_resample['text'].apply(remove_stopwords)

#Tokenize and numberic data will be removed in the stage

from gensim.utils import simple_preprocess 
df_resample['tokens'] = df_resample['text'].apply(simple_preprocess)
print("Finish Tokenize")

#Stemming words and case lower
from gensim.parsing.porter import PorterStemmer
def stem(ws):
  return [PorterStemmer().stem(w) for w in ws]
df_resample['tokens'] = df_resample['tokens'].apply(stem)
print("Finish Stem")

# join word list to sentence
def toSentence(x):
  return " ".join(x)

df_resample['clean'] = df_resample['tokens'].apply(toSentence)

# df_resample['clean'].head(10)
# import matplotlib.pyplot as plt
# import seaborn as sns
# df_resample.head()

from sklearn.model_selection import train_test_split

def split_train_test(data, test_size=0.2, shuffle_state = True):
    FEATURES = ['tokens']
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

import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=1)
fig.set_size_inches(10,5)

X_train['freq_word'] = X_train['tokens'].apply(lambda x: len(x))

sns.distplot(X_train['freq_word'], bins = 90, fit = stats.norm)
(mu0, sigma0) = stats.norm.fit(X_train['freq_word'])
axes.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
axes.set_title("Distribution Word Frequency")
axes.axvline(X_train['freq_word'].median(), linestyle='dashed')
print("median of word frequency: ", X_train['freq_word'].median())

from gensim.models import Word2Vec
#default is skip-gram
tokens = pd.Series(df_resample['tokens']).values
params  = { 
            "sentences" :tokens,
            "size" : 200,
            "window" : 10,
            "min_count" : 3,
            "workers" : 1,
          }
m2v_model = Word2Vec(**params)
m2v_model_name = "w2v.model"
m2v_model.save(m2v_model_name)

def download(file_name):
  from google.colab import files
  files.download(file_name)

# from google.colab import files
# model = files.upload()

# m2v_model = Word2Vec.load("w2v.model")
m2v_model

#average the vetcor as an sentence representation
#OOV happend when the frequency of word is less than  limitation (min_count)
import numpy as np

def agv_v(review, num_features, model):
    featureVec = np.zeros((num_features,), dtype = "float32")
    c = 0
    for word in review:
        if word in model:
            featureVec = np.add(featureVec, model[word])
            c += 1
    featureVec = np.divide(featureVec, c)        
    return featureVec
    
def get_stn_vec(reviews,num_features, model):
  idx = 0
  review_vecs = np.zeros((len(reviews),num_features), dtype = "float32") 
   
  for review in reviews:
    review_vecs[idx] = agv_v(review, NUM_FEATURES, model)
    idx += 1
   
  return review_vecs

# # print(get_stn_vec([df_resample['tokens'][0]],NUM_FEATURES, m2v_model))

m2v_model.most_similar("delightful")

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt

vocab = list(m2v_model.wv.vocab)

w_vs = m2v_model[vocab]
print(w_vs)
print("Total Number of Vocab:", len(w_vs))

#Visualize 1000 words 

tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(w_vs[:100,:])

# with open('embedding1.tsv', 'a') as f:
    
#     for v in vocab:
#       line = ""
#       for e in m2v_model[v]:
#         line += str(e) + "\t"
#       line = line[:-1]
#       f.write('{}\n'.format(line))

# download("embedding1.tsv")

# with open('vcb1.tsv', 'a') as f:
#   f.write("{}\n".format("vocab"))
#   for v in vocab:
#     f.write("{}\n".format(v))

# download('vcb.tsv')

df = pd.DataFrame(X_tsne, index = vocab[:100], columns = ['X','Y'])
df.head()

fig = plt.figure()
fig.set_size_inches(100,20)

ax = fig.add_subplot(2,2,2)
ax.scatter(df['X'], df['Y'])

# Put the label on each point.
for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize = 30)
plt.show()

print(m2v_model.wv.syn0.shape)

m2v_model.wv.syn0

from sklearn.cluster import KMeans
# import time
# num_clusters = m2v_model.wv.syn0.shape[0] // 5

# start = time.time()

# kmean = KMeans(n_clusters = num_clusters)
# index = kmean.fit_predict(m2v_model.wv.syn0)

# end = time.time()
# print("Time taken for K-Means clustering: ", end - start, "seconds.")

# LinearSVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

kfold = StratifiedKFold(n_splits=5, random_state = 32)

train_v = get_stn_vec(X_train['tokens'],NUM_FEATURES,m2v_model)
train_v = np.nan_to_num(train_v)
test_v = get_stn_vec(X_test['tokens'],NUM_FEATURES,m2v_model)
test_v = np.nan_to_num(test_v)

dt = DecisionTreeClassifier(random_state= 64)

param_grid1 = {
    'max_depth':[3],#,10,20],#,20,30,40],
    'criterion' : ["gini","entropy"],
    'min_samples_leaf':[2],
    'class_weight': [{1:1}, {2:5}],
    'ccp_alpha':[0.01, 0.02, 0.03, 0.001]
}


dt_sv = GridSearchCV(dt, param_grid = [param_grid1], verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc' )
dt_sv.fit(train_v, Y_train['sentiment'])
dt_sv_best = dt_sv.best_estimator_
print(dt_sv.best_params_)
# {'ccp_alpha': 0.01, 'class_weight': {1: 1}, 'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 2}

print(dt_sv.best_score_)

from sklearn.metrics import classification_report
print(classification_report(Y_test['sentiment'],dt_sv_best.predict(test_v)))

sv = LinearSVC(random_state= 64)

param_grid1 = {
    'loss':['squared_hinge'],
    'class_weight':[{1:2}],
    'C': [20],
    'penalty':['l2']
}
gs_sv = GridSearchCV(sv, param_grid = [param_grid1], verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc' )
gs_sv.fit(train_v, Y_train['sentiment'])
gs_sv_best = gs_sv.best_estimator_
print(gs_sv.best_params_)

# {'C': 20, 'class_weight': {1: 2}, 'loss': 'squared_hinge', 'penalty': 'l2'} - 0.96569

print(gs_sv.best_score_)

bnb = BernoulliNB()
gs_bnb = GridSearchCV(bnb, param_grid = {'alpha':[0.001 ,0.01, 0.02,0.03, 0.04],
                                         'binarize': [0.001,0.002, 0.003, 0.004]}, verbose = 1, cv = kfold, n_jobs = 1, scoring = "roc_auc")
gs_bnb.fit(train_v, Y_train['sentiment'])
gs_bnb_best = gs_bnb.best_estimator_
print(gs_bnb.best_params_)
# {'alpha': 0.001, 'binarize': 0.004} 0.8087

print(gs_bnb.best_score_)

MLP = MLPClassifier(random_state = 32)

mlp_param_grid = {
    'hidden_layer_sizes':[(5)],
    'activation':['relu','tahn'],
    'solver':['adam'],
    'alpha':[0.3,0.1,0.2,0.01, 0.02],
    'learning_rate':['constant'],
    'max_iter':[1000]
}

gsMLP = GridSearchCV(MLP, param_grid = mlp_param_grid, cv = kfold, scoring = 'roc_auc', n_jobs= 1, verbose = 1)
gsMLP.fit(train_v,Y_train['sentiment'])
print(gsMLP.best_params_)
mlp_best0 = gsMLP.best_estimator_
# {'activation': 'relu', 'alpha': 0.02, 'hidden_layer_sizes': 5, 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} 0.9740315130475148

print(gsMLP.best_score_)

lr = LogisticRegression(random_state = 32)

lr_param = {
     'penalty':['l1'],
    'dual':[False],
    'C':[100],
    'class_weight':['balanced'],
    'solver':['saga']
    }

lr_CV = GridSearchCV(lr, param_grid = lr_param, cv = kfold, scoring = 'roc_auc', n_jobs = 1, verbose = 1)
lr_CV.fit(train_v, Y_train['sentiment'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_

print(lr_CV.best_score_)

models = [gs_sv_best,gs_bnb_best,logi_best,mlp_best0,dt_sv_best]

for model in models:
  from sklearn.metrics import classification_report
  print("MODEL: {} ".format(str(model)))
  train_predict = model.predict(train_vec)
  test_predict = model.predict(test_vec)
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

models = [gs_sv_best,gs_bnb_best,logi_best,mlp_best0,dt_sv_best]

for model in models:
  from sklearn.metrics import classification_report
  print("MODEL: {} ".format(str(model)))
  train_predict = model.predict(train_v)
  test_predict = model.predict(test_v)
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

