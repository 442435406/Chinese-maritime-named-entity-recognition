# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from config import *
from tqdm import  *

class NERDataset(Dataset):
    def __init__(self, dataset_path, model_type):
        self.token_ids_list = []
        self.label_ids_list = []
        self.tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=True)
        data = np.load(dataset_path, allow_pickle=True)
        
        # 这里在init时就把数据集中所有句子中的token变成id，不写在getitem中，避免dataloader运转时GPU的空转
        # 先不加[cls]和[sep]；另外这个时候不用padding，留在collate里根据batch的max_len padding而不是定义模型输入的截断长度MAX_LEN
        for tokens in data['words']:
            tokens = tokens[:MAX_LEN]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.token_ids_list.append(token_ids)

        for labels in data['labels']:
            labels = labels[:MAX_LEN]
            label_ids = [LABEL2IDX[label] for label in labels]
            self.label_ids_list.append(label_ids)

    def __getitem__(self, idx):
        token_ids = self.token_ids_list[idx]
        label_ids = self.label_ids_list[idx]
        seq_len = len(token_ids)
        
        return token_ids, label_ids, seq_len  # 这里返回个seqlen，方便collate里算max_len来padding

    def __len__(self):
        return len(self.token_ids_list)

def collate_fn(batch):
    max_len = max([item[2] for item in batch]) 
    token_tensors = torch.LongTensor([item[0] + [0] * (max_len - len(item[0])) for item in batch])
    label_tensors = torch.LongTensor([item[1] + [0] * (max_len - len(item[1])) for item in batch])
    mask = (token_tensors > 0) # 这里mask是布尔型的，作attention-mask和crf-mask都可以
    return token_tensors, label_tensors, mask

if __name__ == '__main__':
    dataset = NERDataset(dataset_path=TOTAL_DATA_DIR, model_type=CURRENT_PLM)
    dataloader = DataLoader(dataset, BATCH_SIZE, collate_fn=collate_fn)
    print(len(dataloader))
    print(iter(dataloader).next())
    # for batch in tqdm(dataloader):
    #     continue