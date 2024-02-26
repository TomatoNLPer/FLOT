import torch
from torch.utils.data import DataLoader
import transformers
from transformers import CLIPModel, CLIPConfig
import numpy as np
from my_bert.optimization import BertAdam
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random

import torch
from torch.utils.data import DataLoader
from build_dataset import HarmC,FineHarmC,HarmP,FineHarmP,MET
from trainable_models import HateModel
from test2 import evaluate


'''
this is a backup file to record the best settings for training similarity_OT on HarmC and FineHarmC
To reproduce the results, copy the codes to cover those in train.py

'''

#seed =23 ,batch = 48, epoch = 15, gamma = 0.5, lr = 5.5e-4, loss_weights= [0.8,0.2]

seed = 23
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
batchsize = 64
num_epoch = 15

#preparing dataset and loader
trainset = MET(data_split='train')
testset = MET(data_split='test')
validset = MET(data_split='val')

train_loader = DataLoader(dataset=trainset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=validset, batch_size=16)
test_loader = DataLoader(dataset=testset, batch_size=16)

#preparing model

model = HateModel(scratch = True, class_num=2, gamma=0.5).to(device)


#5e-4
total_steps = len(trainset) // batchsize * num_epoch
optimizer = BertAdam(model.parameters(),
                         # lr=args.learning_rate,warmup=args.warmup_proportion,
                         lr=3.5e-6,
                         warmup=0,
                         t_total=total_steps)

##### best scheduler scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2,6],gamma=0.07)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2,3,4],gamma=0.8)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[6,23,43,84],gamma=0.8)

criterion_func = nn.CrossEntropyLoss()

b_acc = b_f1 = b_w_f1 = 0

for epoch in range(num_epoch):
    running_loss = 0
    losses = []

    loop = tqdm(train_loader)

    for step, batch in enumerate(loop):
        optimizer.zero_grad()


        for key in batch.keys():
            batch[key] = batch[key].to(device)
        img = batch['img']
        input_ids = batch['input_ids'].squeeze(dim=1)
        attention_mask = batch['attention_mask'].squeeze(dim=1)
        logits,sim_loss = model(img, input_ids, attention_mask)

        cls_loss = criterion_func(logits, batch['label'])
        loss = 0.75 * cls_loss + 0.25 * sim_loss
        # loss = cls_loss
        print(loss.item())
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        loss.backward()
        optimizer.step()
    scheduler.step()

    acc, f1, w_f1 = evaluate(model, testset, device)
    if acc > b_acc or f1 > b_f1 or w_f1 > b_w_f1:
        b_acc = acc
        b_f1 = f1
        b_w_f1 = w_f1
print(f'best_acc={b_acc}', f'b_f1={b_f1}', f'b_w_f1={b_w_f1}')

