import os
import torch

#device=====================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# model_structrue=====================================
EMBEDDING_DIM=768
HIDDEN_DIM=256

# hyperparameters=====================================******core******
DROPOUT_P = 0.3
BATCH_SIZE = 8
EPOCH = 30
LR = 1e-5
WARM_UP_RATIO = 0.1
CLIP_GRAD = 5

#data_path=====================================
DATA_DIR = os.getcwd() + '\\data\\'
TOTAL_DATA_DIR = DATA_DIR + 'total_data.npz'
TRAIN_DATA_DIR = DATA_DIR  + 'train_280.npz'
DEV_DATA_DIR = DATA_DIR + 'dev_data.npz'
TEST_DATA_DIR = DATA_DIR + 'test_data.npz'

# pretrained_model path=====================================
VOCAB_SIZE = 21128
ALBERT_BASE_CHINESE = r'E:\code\海上搜救NER\模型\PLM\albert-base-chinese-cluecorpussmall'
BERT_BASE_CHINESE = r'E:\code\海上搜救NER\模型\PLM\bert-base-chinese'
BERT_BASE_CHINESE_P = r'E:\code\海上搜救NER\模型\PLM\16_10'
CHINESE_BERT_WWM_EXT = r'E:\code\海上搜救NER\模型\PLM\chinese-bert-wwm-ext'
CHINESE_ROBERTA_WWM_EXT = r'E:\code\海上搜救NER\模型\PLM\chinese-roberta-wwm-ext'
CURRENT_PLM = BERT_BASE_CHINESE_P

# trained_model and train_log path=====================================
MODEL_DIR = os.getcwd() + '\\model\\'
ALBERT_DIR = MODEL_DIR + 'albert\\'
BERT_DIR = MODEL_DIR + 'bert\\'
BERT_WWM_DIR = MODEL_DIR + 'bert_wwm\\'
ROBERTA_DIR = MODEL_DIR + 'roberta\\'
BILSTM_CRF_DIR = MODEL_DIR + 'bilstm\\'
BERT_CRF_DIR = MODEL_DIR + 'bert_crf\\'
CRF_DIR = MODEL_DIR + 'crf\\'
CURRENT_SAVE_DIR = BERT_DIR
CURRENT_MODEL_DIR = CURRENT_SAVE_DIR + 'MSAR_NER_{}_{}_{}_{}.pth'.format(LR, BATCH_SIZE, DROPOUT_P, EPOCH)
CURRENT_RESULT_DIR = CURRENT_SAVE_DIR + 'Result_{}_{}_{}_{}.txt'.format(LR, BATCH_SIZE, DROPOUT_P, EPOCH)
COMPARE_RESULT_DIR = MODEL_DIR + 'BERT_BiLSTM_COMPARE_{}_{}_{}_{}.txt'.format(LR, BATCH_SIZE, DROPOUT_P, EPOCH)

# labels=====================================
LABELS = ('O', 
          'B-AccidentType', 
          'I-AccidentType', 
          'B-Cargo', 
          'I-Cargo',
          'B-Casualty', 
          'I-Casualty',
          'B-Damage', 
          'I-Damage',          
          'B-Ship', 
          'I-Ship', 
          'B-Location', 
          'I-Location',           
          'B-ShipType', 
          'I-ShipType', 
          'B-Organization', 
          'I-Organization')


LABEL_SIZE = len(LABELS)

LABEL2IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX2LABEL = {idx: label for idx, label in enumerate(LABELS)}
MAX_LEN = 256