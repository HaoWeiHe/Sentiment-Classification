

import pickle
import joblib
class SentimentClassification():
  def __init__(self):

    self.mlp = joblib.load('model/mlp_best0')
    self.tfidf = pickle.load(open('model/tfidf.pickle', 'rb'))

  def preprocss(self, text):
    from gensim.utils import simple_preprocess 
    from gensim.parsing.porter import PorterStemmer

    def stem(ws):
      return [PorterStemmer().stem(w) for w in ws]
    def toSentence(x):
      return " ".join(x)

    text = simple_preprocess(text)
    text = stem(text)
    clean =toSentence(text)
    return clean

  def predict(self, text = ""):
    
    text = self.preprocss(text)
    text_tv = self.tfidf.transform([text])
    return self.mlp.predict(text_tv)[0]
    
if __name__ == '__main__':
  text = "Bad experience"
  sc = SentimentClassification()
  print(sc.predict(text))
