# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from transformers import logging
logging.set_verbosity_error()
import warnings 
warnings.filterwarnings("ignore")
from config import *

class MSAR_NER(nn.Module):
    def __init__(self, model_type):
        super(MSAR_NER, self).__init__()

        # self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=0) # 不使用BERT时需要先手动embedding
        self.bert = BertModel.from_pretrained(model_type)
        self.bilstm = nn.LSTM(
            input_size=EMBEDDING_DIM, 
            hidden_size=HIDDEN_DIM//2,
            num_layers=2, 
            bidirectional=True, 
            batch_first=True)
        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.classifier = nn.Linear(HIDDEN_DIM, LABEL_SIZE)
        # self.classifier = nn.Linear(EMBEDDING_DIM, LABEL_SIZE) # 不用LSTM时，全连接层输入维度直接是embedding的维数
        self.crf = CRF(LABEL_SIZE, batch_first=True)
  
    def forward(self, input_ids, labels=None, mask=None, compute_loss=False):
        # with torch.no_grad(): # 锁预训练模型的参数;解锁时学习率要调小,分级调节
        embedding = self.bert(input_ids=input_ids, attention_mask=mask)[0] # output[0]即BERT输出的last_hidden_state，output[1]是cls的embedding
        # embedding = self.embed(input_ids) # 不使用BERT时需要先手动embedding
        output= self.bilstm(embedding)[0]
        output = self.dropout(output)
        # output = self.dropout(embedding) # 不用LSTM时，embedding直接进行dropout
        emissions = self.classifier(output)

        # return emissions
        if compute_loss:  # Training，return loss
            loss = -self.crf.forward(emissions, labels, mask, reduction='mean')
            return loss
        else:  # Testing，return decoding
            decode = self.crf.decode(emissions, mask)
            return decode

class NER_BILSTM_CRF(nn.Module):
    def __init__(self, model_type):
        super(MSAR_NER, self).__init__()

        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=0) # 不使用BERT时需要先手动embedding
        # self.bert = BertModel.from_pretrained(model_type)
        self.bilstm = nn.LSTM(
            input_size=EMBEDDING_DIM, 
            hidden_size=HIDDEN_DIM//2,
            num_layers=2, 
            bidirectional=True, 
            batch_first=True)
        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.classifier = nn.Linear(HIDDEN_DIM, LABEL_SIZE)
        # self.classifier = nn.Linear(EMBEDDING_DIM, LABEL_SIZE) # 不用LSTM时，全连接层输入维度直接是embedding的维数
        self.crf = CRF(LABEL_SIZE, batch_first=True)
  
    def forward(self, input_ids, labels=None, mask=None, compute_loss=False):
        # with torch.no_grad(): # 锁预训练模型的参数;解锁时学习率要调小,分级调节
        # embedding = self.bert(input_ids=input_ids, attention_mask=mask)[0] # output[0]即BERT输出的last_hidden_state，output[1]是cls的embedding
        embedding = self.embed(input_ids) # 不使用BERT时需要先手动embedding
        output= self.bilstm(embedding)[0]
        output = self.dropout(output)
        # output = self.dropout(embedding) # 不用LSTM时，embedding直接进行dropout
        emissions = self.classifier(output)

        # return emissions
        if compute_loss:  # Training，return loss
            loss = -self.crf.forward(emissions, labels, mask, reduction='mean')
            return loss
        else:  # Testing，return decoding
            decode = self.crf.decode(emissions, mask)
            return decode

if __name__ == '__main__':
    model = MSAR_NER(model_type=BERT_BASE_CHINESE)
    input_ids = torch.randint(0, 17, (BATCH_SIZE, 10))
    # print(model)
    model.eval()
    loss = model(input_ids, input_ids, compute_loss=True)
    print(loss)
    decode = model(input_ids)
    print(decode)
    print([IDX2LABEL[i] for i in decode[0]])
