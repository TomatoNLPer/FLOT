import torch
from build_dataset import HarmC, FineHarmC, HarmP, FineHarmP
from models import HateModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support




def get_accuracy(pred, true):
    correct = int(torch.sum(pred == true))
    return correct/len(pred)

def get_p_r_f(pred,true):
    p_macro, r_macro, f_macro, support_macro  = \
        precision_recall_fscore_support(pred, true, average='weighted',zero_division=1)
    return p_macro,r_macro,f_macro



def evaluate(model, data_split,device):

    data_set = FineHarmC(data_split=data_split)
    data_loader = DataLoader(dataset=data_set, batch_size=16)
    with torch.no_grad():
        loop = tqdm(data_loader)
        y_true = []
        y_predicted = []
        for step, batch in enumerate(loop):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            img = batch['img']
            input_ids = batch['input_ids'].squeeze(dim=1)
            attention_mask = batch['attention_mask'].squeeze(dim=1)
            logits,aa = model(img, input_ids, attention_mask)
            predicted_h = torch.argmax(logits, dim = 1)

            y_true.append(batch['label'].to(device))
            y_predicted.append(predicted_h)


        y_true = torch.cat(y_true, dim=0).long().detach().cpu()

        y_predicted = torch.cat(y_predicted, dim=0).long().detach().cpu()

        #f1_h = f1_score(y_true.detach().cpu().numpy(), y_predicted.cpu().numpy(), average='macro')
        p, r, f1 = get_p_r_f(y_predicted, y_true)
        acc = get_accuracy(y_predicted, y_true)


        # print(y_true_h.cpu().numpy(), y_predicted_h.cpu().numpy())
        # print(y_true_s.cpu().numpy(), y_predicted_s.cpu().numpy())
        # print(y_true_o.cpu().numpy(), y_predicted_o.cpu().numpy())
        # print(y_true_m.cpu().numpy(), y_predicted_m.cpu().numpy())

        print(f'acc_hate = {acc}',
              f'f1_hate={f1}',
              )
        return acc, f1

