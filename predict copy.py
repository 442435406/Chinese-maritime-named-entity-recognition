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

words = np.load(DEV_DATA_DIR, allow_pickle=True)
words = words['words']
test_dataset = NERDataset(DEV_DATA_DIR, CURRENT_PLM)
test_loader = data.DataLoader(dataset=test_dataset,
                            batch_size=(BATCH_SIZE)//2,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_fn)

tokenizer = BertTokenizerFast.from_pretrained(BERT_BASE_CHINESE_P, do_lower_case=True)

# BERT
model = MSAR_NER(CURRENT_PLM)
model.load_state_dict(torch.load(MODEL_DIR + 'bert\\' + '280_MSAR_NER_1e-05_8_0.3_30_2.pth'))
model.to(DEVICE)

# BiLSTM
model_1 = MSAR_NER(CURRENT_PLM) # 在model模块中再写一个用作对比的模型类，如MSAR_NER_nobert
model_1.load_state_dict(torch.load(MODEL_DIR + 'bert\\' + '280_MSAR_NER_1e-05_8_0.3_30_2.pth'))
model_1.to(DEVICE)
def merge_adjacent_numbers_and_chars(original_array):
    merged_array = []
    current_number = ""

    for item in original_array:
        if re.match(r"\d", item):  # 如果当前元素是数字
            current_number += item  # 将数字添加到当前数字字符串
        else:
            if current_number:  # 如果当前数字字符串不为空
                merged_array.append(current_number)  # 添加合并的数字元素
                current_number = ""  # 重置当前数字字符串
            merged_array.append(item)  # 添加当前非数字元素

    # 处理最后一个数字元素
    if current_number:
        merged_array.append(current_number)

    return merged_array
if __name__ == "__main__":
    #测试集实例效果查看
    Y, Y_hat = test(model,test_loader)
    _, Y_hat_1 = test(model_1,test_loader)

    if os.path.exists(COMPARE_RESULT_DIR) is True:
        print('Result Exists')

    else:
        with open("E:\\code\\海上搜救NER\\模型\\MSAR_NER\\model\\data7", 'w', encoding='utf-8') as f:
            f.write('text label bert\n')
            for i in range(0,len(words)):
                for j in range(0,len(words[i])):
                    f.write(words[i][j] + ' ' + Y[i][j] + ' ' + Y_hat[i][j] + '\n')
            f.write('\n\n')
        print('Finished!')

    # # 自由输入预测
    # text = '2022年12月22日0150时许，香港籍散货船"金旺岭"轮空载由山东日照岚山港驶往天津港途中，在石岛东南约23海里处，与张峰所属的日照张家台籍“鲁日山渔61027”轮发生碰撞,导致“鲁日山渔61027”轮沉没，5人死亡，6人失踪，构成重大等级水上交通事故。'
    # input_ids = tokenizer(text, max_length=MAX_LEN, truncation=True, add_special_tokens=False)['input_ids']
    # input_ids = torch.LongTensor([input_ids]).to(DEVICE) # 注意要把输入变成二维数组，bert模型中batch_size, seq_length = input_shape）。不然报错：not enough values to unpack (expected 2, got 1)

    # model.eval()
    # decode = model(input_ids)
    # labels=[IDX2LABEL[i] for i in decode[0]]
    # print(tokenizer.tokenize(text))
    # print((labels))