from flask import Flask, request
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from simpletransformers.classification import ClassificationModel
import re
import string
import nltk
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

data = pd.read_csv('templates/UpdatedResumeDataSet.csv')

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = ''.join([word for word in text if not word.isdigit()])
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = "".join([word for word in text if word not in string.punctuation])
    text = re.sub("\W", " ", str(text))
    ext = [word for word in text.split() if word not in stopwords]
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    return text

data['cleaned_text'] = data['Resume'].apply(lambda x: clean_text(x))
labelencoder = LabelEncoder()
data["Category_N"] = labelencoder.fit_transform(data["Category"])
df = data[['cleaned_text','Category_N']]
model = ClassificationModel('roberta', 'roberta-base', use_cuda=True,num_labels=25, args={
                                                                         'train_batch_size':16,
                                                                         'reprocess_input_data': True,
                                                                         'overwrite_output_dir': True,
                                                                         'fp16': False,
                                                                         'do_lower_case': False,
                                                                         'num_train_epochs': 4,
                                                                         'max_seq_length': 128,
                                                                         'regression': False,
                                                                         'manual_seed': 1997,
                                                                         "learning_rate":2e-5,
                                                                         'weight_decay':0,
                                                                         "save_eval_checkpoints": True,
                                                                         "save_model_every_epoch": False,
                                                                         "silent": False})
model.train_model(df)

sent_lens = []
for i in data.cleaned_text:
    length = len(i.split())
    sent_lens.append(length)

print(len(sent_lens))
print(max(sent_lens))

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print("data:", request.form['Cv'])
        info = request.form['Cv']
        place = " " * len(sent_lens)
        if len(info) > 962:
            final = info[:962]
        else:
            final = info + place[len(info):]
        ans = model.predict(final)
        return (data[data.Category_N == ans[0]].Category.values[0])
    else:
        return ("Use post request with Cv load")

if __name__ == '__main__':
    app.run()
