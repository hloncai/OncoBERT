#!/usr/bin/env python
# coding: utf-8

# In[45]:


import csv
import logging
import argparse
import random
from tqdm import trange, tqdm
import matplotlib as mpl
import pandas as pd
mpl.use('Agg')

import matplotlib.pyplot as plt

from scipy import interp
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
#from sklearn.utils.fixes import signature
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.nn.functional as F

import py_compile
py_compile.compile('modeling_readmission.py')


# In[46]:


from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
#important
from modeling_readmission import BertGetCLSOutput, BertForSequenceClassification
print('in the modeling class')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


# In[48]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.cuda.is_available(), torch.cuda.device_count()


# In[49]:


def read_training(experiment_number):
    train_path = "/data/users/linh/USF_Practicum/glioma_tokenized/glioma_train_180_" + str(experiment_number) + "_tokens.pkl"
    data = pd.read_pickle(train_path)
    return data


# In[50]:


def read_testing(experiment_number):
    test_path = "/data/users/linh/USF_Practicum/glioma_tokenized/glioma_test_180_" + str(experiment_number) + "_tokens.pkl"
    data_test = pd.read_pickle(test_path)
    return data_test


# In[51]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_seq_length = 512
features = []
batch_size = 2


# In[52]:


# 4
no_cuda = False
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu") # defult setting
n_gpu = torch.cuda.device_count() # n_gpu = 0
logger.info("device %s n_gpu %d distributed %r", device, n_gpu, False)


# In[53]:


# 5
bert_model = '/home/linh/nlp/UCSF_Clinical_Notebook/pretrained_ClinicalBERT'
model = BertForSequenceClassification.from_pretrained(bert_model, 1)


# In[54]:


logit = LogisticRegression(C=5e1, solver='lbfgs', random_state=17, n_jobs=4)


# # 2. Convert to BERT vector and aggregate

# In[55]:


# 1
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


# In[56]:


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for i in range(len(examples)):
        tokens_import = examples['Token_trunc'][i]
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_import:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens) 
        # BertTokenizer.from_pretrained('bert-base-uncased').convert_tokens_to_ids
        # convert tokens to number (max 512)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

#         if i < 5:
#             logger.info("*** Example ***")
#             logger.info("tokens: %s" % " ".join(
#                     [str(x) for x in tokens]))
#             logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#             logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#             logger.info(
#                     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


# In[57]:


roc_auc, f1, accuracy, precision, recall = ([] for i in range(5))


# In[ ]:


for experiment_number in range(21):
    df_train = read_training(experiment_number)
    df_train['ID'] = df_train['ID'].apply(lambda x: int(x))
    df_test = read_testing(experiment_number)
    df_test['ptId'] = df_test.index
    df_test['ID'] = df_test['ID'].apply(lambda x: int(x))
    
    train_features = convert_examples_to_features(df_train, max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    
    test_features = convert_examples_to_features(df_test, max_seq_length, tokenizer)
    all_input_ids_t = torch.tensor([t.input_ids for t in test_features], dtype=torch.long)
    all_input_mask_t = torch.tensor([t.input_mask for t in test_features], dtype=torch.long)
    all_segment_ids_t = torch.tensor([t.segment_ids for t in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids_t, all_input_mask_t, all_segment_ids_t)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    model.to(device)
    model.eval()

    outputs = []
    outputs_test = []
    dense = nn.Linear(768, 768)
    activation = nn.Tanh()
    for input_ids, input_mask, segment_ids in tqdm(train_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            _, cls_output = model(input_ids, segment_ids, input_mask)
            cls_output = cls_output.cpu().numpy()
            for v in cls_output:
                outputs.append(v)
    
    for input_ids, input_mask, segment_ids in tqdm(test_dataloader):
        input_ids_t = input_ids.to(device)
        input_mask_t = input_mask.to(device)
        segment_ids_t = segment_ids.to(device)
        with torch.no_grad():       
            _, cls_output_t = model(input_ids_t, segment_ids_t, input_mask_t)
            cls_output_t = cls_output_t.cpu().numpy()
            for v in cls_output_t:
                outputs_test.append(v)
    
    keys = np.array(df_train['ID'])
    vals = np.asarray(outputs)
    array_grouped_dict_train = {key: vals[keys == key] for key in np.unique(keys)}
    for key in array_grouped_dict_train:
        array_grouped_dict_train[key] = array_grouped_dict_train[key].max(axis=0)
    
    data_train = pd.read_pickle('/data/users/linh/USF_Practicum/glioma/glioma_train_180_' + str(experiment_number) + '.pkl')
    data_train['ptId'] = data_train.index
    df_train_agg = pd.DataFrame(data_train)
    df_train_agg['vector'] = df_train_agg['ptId'].map(array_grouped_dict_train)
    
    keys = np.array(df_test['ID'])
    vals = np.asarray(outputs_test)
    array_grouped_dict_test = {key: vals[keys == key] for key in np.unique(keys)}
    for key in array_grouped_dict_test:
        array_grouped_dict_test[key] = array_grouped_dict_test[key].max(axis=0)
    
    data_test = pd.read_pickle('/data/users/linh/USF_Practicum/glioma/glioma_test_180_' + str(experiment_number) + '.pkl')
    data_test['ptId'] = data_test.index
    df_test_agg = pd.DataFrame(data_test)
    df_test_agg['vector'] = df_test_agg['ptId'].map(array_grouped_dict_test)
    
    # training data
    X = torch.tensor(list(df_train_agg['vector'].values) )
    y = torch.tensor(list(df_train_agg['label'].values)).reshape(-1,1).float()
    X_test = torch.tensor(list(df_test_agg['vector'].values) )
    y_test = torch.tensor(list(df_test_agg['label'].values)).reshape(-1,1).float()
    logit.fit(X, y)
    y_pred = logit.predict(X_test)
    y_pred_proba = logit.predict_proba(X_test)[:,1]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)
    print ('AUROC = ', auc(fpr, tpr) )
    roc_auc.append(auc(fpr, tpr))
    print ('Accuracy = ', accuracy_score(y_test, y_pred) )
    accuracy.append(accuracy_score(y_test, y_pred) )
    print ('Precision = ', precision_score(y_test, y_pred) )
    precision.append(precision_score(y_test, y_pred) )
    print ('F1 = ', f1_score(y_test, y_pred) )
    f1.append(f1_score(y_test, y_pred) )
    print ('Recall = ', recall_score(y_test, y_pred) )
    recall.append(recall_score(y_test, y_pred) )
    


# In[ ]:


f1


# In[ ]:


roc_auc


# In[ ]:


import statistics

mean_auc = statistics.mean(roc_auc).round(3) 
std_auc = round(statistics.stdev(roc_auc), 3)
mean_accuracy = statistics.mean(accuracy).round(3) 
std_accuracy = round(statistics.stdev(accuracy), 3)
mean_precision = statistics.mean(precision).round(3) 
std_precision = round(statistics.stdev(precision), 3)
mean_f1 = statistics.mean(f1).round(3) 
std_f1 = round(statistics.stdev(f1), 3)
mean_recall = statistics.mean(recall).round(3) 
std_recall = round(statistics.stdev(recall), 3)

print ("AUROC: ", mean_auc, "+/-", std_auc)
print ("Accuracy: ", mean_accuracy, "+/-", std_accuracy)
print ("Precision: ", mean_precision, "+/-", std_precision)
print ("F1: ", mean_f1, "+/-", std_f1)
print ("Recall: ", mean_recall, "+/-", std_recall)





