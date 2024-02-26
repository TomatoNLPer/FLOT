import os
import json


anno_path = './data/HarmP/annotations/train.jsonl'
save_path = './data/HarmP/annotations/fine_train.jsonl'
def load_annotations(path):

    anno_path = path
    sample = []
    with open(anno_path, mode='r') as f:
        for line in f.readlines():
            sample.append(json.loads(line))
    return sample



def convert_fine_label(label):
    if label == 'not harmful':
        num = 0
    elif label == 'somewhat harmful':
        num = 1
    else :
        num = 2
    return num

def convert_label(label):
    # 0归为0， 1，2都归为1
    if label == 'not harmful':
        num = 0
    else:
        num = 1
    return num


def convert_fine_label(label):
    # 3-class
    if label == 'not harmful':
        num = 0
    elif label == 'somewhat harmful':
        num = 1
    else:
        num = 2
    return num


def convert_label_v2(label):
    # 0,1归为0， 2归为1
    if label == 'very_harmful':
        num = 1
    else:
        num = 0
    return num





annotations = load_annotations(anno_path)

for anno in annotations:
    anno['labels'] = convert_fine_label(anno['labels'][0])

with open(save_path, mode='w',encoding='utf-8') as f:
    for anno in annotations:
        json.dump(anno,f)
        f.write('\n')

