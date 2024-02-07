# -*- coding: utf-8 -*-
import logging
from log_setting import set_logger
from tqdm import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast
from models import MSAR_NER
from utils import NERDataset, collate_fn 
from config import *
from main import test
import re

tokenizer = BertTokenizerFast.from_pretrained(BERT_BASE_CHINESE, do_lower_case=True)

# BERT
model = MSAR_NER(CURRENT_PLM)
model.load_state_dict(torch.load(MODEL_DIR + 'bert\\' + 'MSAR_NER_1e-05_8_0.3_30.pth'))
model.to(DEVICE)


if __name__ == "__main__":
    # 打开文本文件
    file_path = "labeled20_3.txt"  # 替换为你的文本文件路径

    # 初始化一个空数组来存储文本行
    text_lines = []
    label_lines = []

    # 使用with语句来打开文件并按行读取文本
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # 去除行尾的换行符并将文本行添加到数组中
            text_lines.append(tokenizer.tokenize(line))
            input_ids = tokenizer(line, max_length=MAX_LEN, truncation=True, add_special_tokens=False)['input_ids']
            input_ids = torch.LongTensor([input_ids]).to(DEVICE) # 注意要把输入变成二维数组，bert模型中batch_size, seq_length = input_shape）。不然报错：not enough values to unpack (expected 2, got 1)

            model.eval()
            decode = model(input_ids)
            label_line=[IDX2LABEL[i] for i in decode[0]]
            label_lines.append(label_line)


    # 打印存储的文本行或进行其他操作
    # for line in text_lines:
    #     print(merge_adjacent_numbers_and_chars(line))


    with open("pre_self_labeld20_3.txt", 'w', encoding='utf-8') as f:
        # f.write('text label\n')
        for i in range(0,len(text_lines)):
            for j in range(0,len(text_lines[i])):
                f.write(text_lines[i][j] + ' ' + label_lines[i][j] +'\n')
            print(len(text_lines[i]))
            print(len(label_lines[i]))
            f.write('\n')

        
    print('Finished!')

    # # 自由输入预测
    # text = '1月6日2时许，正在石岛执行救助待命任务的“北海救111”轮接救助值班室电话通知：“鲁荣渔54887”轮在石岛南82海里处海域被撞翻，船上5人遇险，其中1人获救，4人失踪，令“北海救111”轮立即前往事发海域搜寻失踪渔民。'
    # input_ids = tokenizer(text, max_length=MAX_LEN, truncation=True, add_special_tokens=False)['input_ids']
    # input_ids = torch.LongTensor([input_ids]).to(DEVICE) # 注意要把输入变成二维数组，bert模型中batch_size, seq_length = input_shape）。不然报错：not enough values to unpack (expected 2, got 1)

    # model.eval()
    # decode = model(input_ids)
    # print([IDX2LABEL[i] for i in decode[0]])