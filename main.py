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
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
from models import MSAR_NER
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import NERDataset, collate_fn 
from config import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(epoch, model, train_loader, optimizer, scheduler):
    model.train()
    train_losses = 0.0

    for i, batch in enumerate(tqdm(train_loader)):
        input_ids, labels, mask = batch
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        mask = mask.to(DEVICE)

        loss = model(input_ids, labels, mask, compute_loss=True)
        train_losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=CLIP_GRAD) # 梯度裁剪
        optimizer.step()
        scheduler.step()

    train_loss = float(train_losses) / len(train_loader)
    train_loss_list.append(train_loss)

    logging.info("Epoch: {}, Loss:{:.4f}".format(epoch, train_loss))


def dev(epoch, model, dev_loader):
    model.eval()
    Y, Y_hat = [], []
    dev_losses = 0

    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            input_ids, labels, mask = batch
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            mask = mask.to(DEVICE)

            loss = model(input_ids, labels, mask, compute_loss=True)
            dev_losses += loss.item()
            y_hat = model(input_ids, labels, mask)

            # seqeval包相关函数的输入是1d/2d list原始标签的形式，故要把每个batch的预测/原始标签id拆成一条一条，注意把GPU上的tensor送回CPU
            for j in y_hat:
                j = [IDX2LABEL[idx] for idx in j]
                Y_hat.append(j)
            #注意masked_select返回的是1d tensor，不是和y_hat形状相同的list[list]，因此这里只有麻烦一点，将batch中的labels-mask一对对拿出来mask然后送到Y里  
            for num in range(len(labels)):
                K = torch.masked_select(labels[num], mask[num]).cpu().numpy().tolist()
                K = [IDX2LABEL[idx] for idx in K]
                Y.append(K)            

    dev_loss = float(dev_losses) / len(dev_loader)
    dev_f1 = f1_score(Y, Y_hat, scheme=IOB2)
    dev_loss_list.append(dev_loss)
    dev_f1_list.append(dev_f1)

    logging.info("Epoch: {}, Dev Loss:{:.4f}, Dev F1:{:.3f}".format(epoch, dev_loss, dev_f1))
    return dev_loss, dev_f1


def test(model, test_loader):
    model.eval()
    Y, Y_hat = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids, labels, mask = batch
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            mask = mask.to(DEVICE)

            y_hat = model(input_ids, labels, mask)

            for j in y_hat:
                j = [IDX2LABEL[idx] for idx in j]
                Y_hat.append(j)
     
            for num in range(len(labels)):
                K = torch.masked_select(labels[num], mask[num]).cpu().numpy().tolist()
                K = [IDX2LABEL[idx] for idx in K]
                Y.append(K)          

    return Y, Y_hat


def plotting(train_loss, dev_loss, dev_f1):
    epochs = np.arange(1, EPOCH+1)
    fig,ax=plt.subplots()

    l1 = ax.plot(epochs, train_loss, label='Training Loss', color="r")
    l2 = ax.plot(epochs, dev_loss, label='Validation Loss', color="g")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    # ax.set_ylim(0, 30)
    ax.set_title("Training Loss and Validation Loss&F1")

    ax_r=ax.twinx()
    l3 = ax_r.plot(epochs, dev_f1, label='Validation F1', color="b")
    ax_r.set_ylabel("F1")
    plt.legend(handles=[l1[0],l2[0],l3[0]], loc=5)
    # plt.xticks(np.arange(1, EPOCH+1, 2))
    # plt.show()
    plt.savefig(ner.model_save + 'loss_{}_{}_{}_{}.jpg'.format(ner.lr, ner.batch_size, DROPOUT_P, ner.n_epochs),dpi=300)


if __name__ == "__main__":

    for i in range(0,20):
        # Arguments and Log Setting
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--n_epochs", type=int, default=EPOCH)
        parser.add_argument("--trainset", type=str, default=TRAIN_DATA_DIR)
        parser.add_argument("--devset", type=str, default=DEV_DATA_DIR)
        parser.add_argument("--testset", type=str, default=TEST_DATA_DIR)
        parser.add_argument("--model_type", type=str, default=CURRENT_PLM) 
        parser.add_argument("--model_save", type=str, default=CURRENT_SAVE_DIR) #后面这几个输入麻烦，都是路径，可以直接在config里面改动了
        ner = parser.parse_args()

        set_logger(ner.model_save + '20.log')
        logging.info('--------Start training a NER model based on {}--------'.format(ner.model_type))
        logging.info("--------device: {}--------".format(DEVICE))
        logging.info("--------lr: {} batch_size: {} dropout: {} epoch: {}--------".format(ner.lr, ner.batch_size, DROPOUT_P, ner.n_epochs))

        # Data Loading
        train_dataset = NERDataset(ner.trainset, ner.model_type)
        dev_dataset = NERDataset(ner.devset, ner.model_type)
        test_dataset = NERDataset(ner.testset, ner.model_type)

        train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=ner.batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    collate_fn=collate_fn)

        dev_loader = data.DataLoader(dataset=dev_dataset,
                                    batch_size=(ner.batch_size)//2,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=collate_fn)

        test_loader = data.DataLoader(dataset=test_dataset,
                                    batch_size=(ner.batch_size)//2,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=collate_fn)
        logging.info('--------Data Loading Done, train:{}, dev:{}, test:{}--------'.format(len(train_dataset), 
                                                                                        len(dev_dataset), len(test_dataset)))

        # Model Initialization
        model = MSAR_NER(ner.model_type)
        model.to(DEVICE)
        logging.info('--------Model Initialization Done--------')

        # Optimizer and Scheduler(Warmup) Setting
        optimizer_grouped_parameters = [
            # {'params': model.bert.parameters()},
            {'params': model.bilstm.parameters(), 'lr': 0.01},
            {'params': model.classifier.parameters(), 'lr': 0.01},
            {'params': model.crf.parameters(), 'lr': 0.01},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=ner.lr) 
        len_dataset = len(train_dataset) 
        epoch = ner.n_epochs
        batch_size = ner.batch_size
        total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = WARM_UP_RATIO * total_steps, 
                                                    num_training_steps = total_steps) 

        # Training
        logging.info('--------Start Training--------')
        best_model = None
        best_dev_loss = 1e6
        # best_dev_acc = 0.0
        best_dev_f1 = 0.0
        train_loss_list = []
        dev_loss_list = []
        dev_f1_list = []

        for epoch in range(1, ner.n_epochs+1):

            train(epoch, model, train_loader, optimizer, scheduler)
            dev_loss, dev_f1 = dev(epoch, model, dev_loader)

            if dev_loss < best_dev_loss and dev_f1 > best_dev_f1:
                best_model = model
                best_dev_loss = dev_loss
                best_dev_f1 = dev_f1

        logging.info("--------Training Finished--------")

        # Testing
        y_true, y_pred = test(best_model, test_loader)
        logging.info('\n'+ classification_report(y_true, y_pred, scheme=IOB2, digits=3))
        logging.info("--------Testing Finished--------")

        # Save Losses and F1s
        plotting(train_loss_list, dev_loss_list, dev_f1_list)
        np.savez_compressed(ner.model_save + 'record_{}_{}_{}_{}_{}.npz'.format(ner.lr, ner.batch_size, DROPOUT_P, ner.n_epochs,i), 
                            train_loss=np.array(train_loss_list), 
                            dev_loss=np.array(dev_loss_list), 
                            dev_f1=np.array(dev_f1_list))

        # Save model
        torch.save(best_model.state_dict(), ner.model_save + '280_MSAR_NER_{}_{}_{}_{}_{}.pth'.format(ner.lr, ner.batch_size, DROPOUT_P, ner.n_epochs,i))
        logging.info("====================================================================================")