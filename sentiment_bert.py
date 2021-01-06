# -*- coding: utf-8 -*-
"""sentiment_bert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GqqqRgA_qAxCyzfAR5bTDNtakwuGMqsB
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

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# pip install transformers tqdm boto3 requests regex -q

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

#resampling data preprocess - 40,000 data for each sentiment level 
def sample_data(n):
    return pd.concat([df_review[df_review['sentiment'] == i].head(n) for i in range(0,2)])
df_resample = sample_data(40000)
MAX_LENGTH = 150
df_resample = df_resample[~(df_resample.text.apply(lambda x : len(x)) > MAX_LENGTH)]

import torch
from transformers import BertTokenizer
from IPython.display import clear_output

PRETRAINED_MODEL_NAME = "bert-base-cased"

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

clear_output()
print("PyTorch version：", torch.__version__)

vocab = tokenizer.vocab
print("vocab size：", len(vocab))

import random
random_tks = random.sample(list(vocab), 10)
randon_ids = [vocab[t] for t in random_tks]
print("{:20}{:20}".format("token","index"))
print("-"*30)
for t, id in zip(random_tks,randon_ids):
  print("{:20}{:20}".format(t,id))

import sys
!test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo
if not 'bertviz_repo' in sys.path:
  sys.path += ['bertviz_repo']

# import packages
from transformers import BertTokenizer, BertModel
from bertviz import head_view


def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))

clear_output()

model_version = 'bert-base-cased'
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version)


sentence_a = "There is a bird outside my window"
sentence_b = "it is singing"


inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
token_type_ids = inputs['token_type_ids']
input_ids = inputs['input_ids']
attention = model(input_ids, token_type_ids=token_type_ids)[-1]
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)
call_html()


head_view(attention, tokens)

inputs

from sklearn.model_selection import train_test_split

def split_train_test(data, test_size=0.2, shuffle_state = True):
    FEATURES = ['text','sentiment']
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

df_train = X_train.reset_index()

df_train['stn_b'] = df_train.apply(lambda x: None)
df_train = df_train.loc[:, ['text','stn_b','sentiment']]
df_train.columns = ['stn_a', 'stn_b','label']

print("#sample:", len(df_train))

df_train.to_csv("train.tsv", sep="\t", index=False)

#apply the same process for test data

df_test = X_test.reset_index()
df_test['stn_b'] = df_test.apply(lambda x: None)
df_test = df_test.loc[:, ['text','stn_b','sentiment']]
df_test.columns = ['stn_a', 'stn_b','label']

print("#sample:", len(df_test))
df_test.to_csv("test.tsv", sep="\t", index=False)

print(df_train.label.value_counts() / len(df_train))

!pip install pysnooper -q

import pysnooper
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
  def __init__(self, mode, tokenizer):
    assert mode in ["train", "test"]
    self.mode = mode
    self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
    self.len = len(self.df)
    self.tokenizer = tokenizer  
  
  # @pysnooper.snoop() 
  def __getitem__(self, idx):
    if self.mode == "test":
      text_a, text_b = self.df.iloc[idx,:2].values
      label_tensor = None
    else:
      text_a, text_b, label= self.df.iloc[idx, :].values
       
      label_tensor = torch.tensor(int(label))
    #bert tokens
    word_pieces = ["[CLS]"]
    tokens = self.tokenizer.tokenize(text_a)
    word_pieces += tokens + ["[SEP]"]
    len_a = len(word_pieces)

    tokens_b = self.tokenizer.tokenize(text_b)
    word_pieces += tokens_b + ["[SEP]"]
    len_b = len(word_pieces) - len_a
    # tokens to ids
    ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
    tokens_tensor = torch.tensor(ids)

    segments_tensor = torch.tensor([0] * len_a + [1] * len_b,  dtype=torch.long)
    return (tokens_tensor, segments_tensor, label_tensor)

  def __len__(self):
        return self.len

trainset = ReviewDataset("train", tokenizer=tokenizer)

"""
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_sequence

# samples: list[ele_0, ele_1, ele_2, ... ]
#ele_i = trainset[i]
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    #testing dataset have labels
    if samples[0][2] is not None: 
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero padding
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    masks_tensors = masks_tensors.type(torch.LongTensor)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids


BATCH_SIZE = 64
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, 
                         collate_fn = create_mini_batch)

data = next(iter(trainloader))

tokens_tensors, segments_tensors, \
    masks_tensors, label_ids = data

print(f"""
tokens_tensors.shape   = {tokens_tensors.shape} 
{tokens_tensors}
------------------------
segments_tensors.shape = {segments_tensors.shape}
{segments_tensors}
------------------------
masks_tensors.shape    = {masks_tensors.shape}
{masks_tensors}
------------------------
label_ids.shape        = {label_ids.shape}
{label_ids}
""")

from transformers import BertForSequenceClassification

NUM_LABELS = 2

model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

clear_output()

# high-level 顯示此模型裡的 modules
print("""
name            module
----------------------""")
for name, module in model.named_children():
    if name == "bert":
        for n, _ in module.named_children():
            print(f"{name}:{n}")
    else:
        print("{:15} {}".format(name, module))

data = next(iter(trainloader))

for e in data:
  print(e.shape)

def get_predictions(model, dataloader, compute_acc = False):
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        for data in dataloader:
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids = tokens_tensors, 
                            token_type_ids = segments_tensors, 
                            attention_mask = masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)

_, acc = get_predictions(model, trainloader, compute_acc = True)
print("classification acc:", acc)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


EPOCHS = 6  
for epoch in range(EPOCHS):
    
    running_loss = 0.0
    for data in trainloader:
        
        tokens_tensors, segments_tensors, \
        masks_tensors, labels = [t.to(device) for t in data]

        optimizer.zero_grad()
        
        # forward pass
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)

        loss = outputs[0]
        # backward
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    
    _, acc = get_predictions(model, trainloader, compute_acc=True)

    print('[epoch %d] loss: %.3f, acc: %.3f' %
          (epoch + 1, running_loss, acc))

from sklearn.metrics import classification_report
testset = ReviewDataset("test", tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=256, 
                        collate_fn=create_mini_batch)
test_pred = get_predictions(model, testloader)
train_pred = get_predictions(model, trainloader)


print(">> training set \n")
print(classification_report(Y_train['sentiment'],train_pred.tolist()))


print(">> testing set \n")
print(classification_report(Y_test['sentiment'],test_pred.tolist()))