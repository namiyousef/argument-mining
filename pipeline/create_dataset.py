import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pdb
import torch
from torch import cuda
import time
print('Yousef test')
s = time.time()
BASE_PATH = '../data/kaggle/feedback-prize-2021/'
path_from_base = lambda x: os.path.join(BASE_PATH, x)
train_df = pd.read_csv(path_from_base('train.csv'))
test_names, train_texts = [], []
for f in list(os.listdir(path_from_base('train'))):
    test_names.append(f.replace('.txt', ''))
    train_texts.append(open(path_from_base(f'train/{f}'), 'r').read())
    
doc_df = pd.DataFrame({'id': test_names, 'text': train_texts})
#Create entities for each document
entities = []
for index,row in doc_df.iterrows():
    length_text = row['text'].split().__len__()
    ent = ["O" for i in range(length_text)]

    for idx,r in train_df[train_df['id'] == row['id']].iterrows():
        
        pred_idx = r['predictionstring'].split()
        ent[int(pred_idx[0])] = f"B-{r['discourse_type']}"

        for i in pred_idx[1:]:
            ent[int(i)] = f"I-{r['discourse_type']}"
        

    entities.append(ent)

doc_df['elements'] = entities

#Match the labels to entities
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}

doc_df['labels'] = doc_df['elements'].apply(lambda x: [labels_to_ids[i] for i in x])
print(doc_df.head())
for i, (text, label) in doc_df[['text', 'labels']].iterrows():
    assert len(label) == len(text.split())
print(time.time() - s)