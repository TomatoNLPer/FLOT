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

import torch
from torch.utils.data import DataLoader
from build_dataset import HarmC,FineHarmC,HarmP,FineHarmP
from models import HateModel
from test import evaluate


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
batchsize =64
num_epoch = 7

#preparing dataset and loader
trainset = HarmP(data_split='train')
testset = HarmP(data_split='test')
validset = HarmP(data_split='val')

train_loader = DataLoader(dataset=trainset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=validset, batch_size=16)
test_loader = DataLoader(dataset=testset, batch_size=16)

#preparing model

model = HateModel(scratch = True, class_num=2, gamma=0.85).to(device)

total_steps = len(trainset) // batchsize * num_epoch
optimizer = BertAdam(model.parameters(),
                         # lr=args.learning_rate,warmup=args.warmup_proportion,
                         lr=5e-5,
                         warmup=0.5,
                         t_total=total_steps)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2,3],gamma=1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[0,2],gamma=0.09)
criterion_func = nn.CrossEntropyLoss()

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
        logits = model(img, input_ids, attention_mask)
        loss = criterion_func(logits, batch['label'])
        print(loss.item())
        # if epoch == 1:
        #     evaluate(model, 'test')
        loss.backward()
        optimizer.step()
        # scheduler.step()

    evaluate(model, 'test')


