import os
import json
import codecs
import csv

anno_path = '/workspace/zlh_fourth/CDG/BLIP/data/MET-memes/test_E.csv'
save_path = './data/MET/fine_test.jsonl'
def load_annotations():

    path = anno_path
    f = codecs.open(path, encoding='utf-8')
    labels = csv.DictReader(f, skipinitialspace=True)
    label_list = []
    for row in labels:
        save_items = {}
        save_items['image'] = row['file_name']
        save_items['labels'] = int(row['offensiveness detection'][0])
        save_items['text'] = ocr_text_dict[row['file_name']]

        label_list.append(save_items)
    return label_list



def convert_fine_label(label):
    if label == '0(non-offensive)':
        num = 0
    elif label == 'somewhat harmful':
        num = 1
    else :
        num = 2
    return num

def convert_label(label):
    # 0归为0， 1，2都归为1
    if label == 0:
        num = 0
    else:
        num = 1
    return num

def convert_label_v2(label):
    # 0,1归为0， 2归为1
    if label == 'very_harmful':
        num = 1
    else:
        num = 0
    return num











f_text = codecs.open('/workspace/zlh_fourth/CDG/BLIP/data/MET-memes/E_text.csv', encoding='utf-8', errors='ignore')
text_labels = csv.DictReader(f_text, skipinitialspace=True)
ocr_text_dict = {}
for text_row in text_labels:
    ocr_text_dict[text_row['file_name']] = text_row['text']



annotations = load_annotations()




# for anno in annotations:
#     anno['labels'] = convert_label(anno['labels'])

with open(save_path, mode='w',encoding='utf-8') as f:
    for anno in annotations:
        json.dump(anno,f)
        f.write('\n')

