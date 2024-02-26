from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision
from transformers import BertTokenizer
import torch

class HarmC(Dataset):


    def __init__(self, data_split = 'val'):
        super(HarmC, self).__init__()
        self.root_path = './data/HarmC'
        self.image_path = os.path.join(self.root_path,'images')
        self.data_split = data_split
        self.max_txt_length = 77
        self.tokenizer = BertTokenizer.from_pretrained('../pretrain_model/vl-bert/bert-base-uncased')
        self.data_base = self.load_annotations()
        self.transforms = torchvision.transforms.Compose(
            [
                #torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )


    def load_annotations(self):
        split = str('new_') + self.data_split + str('.jsonl')
        anno_path = os.path.join(self.root_path, 'annotations', split)
        #valid_json_path = './data/HarmC/annotations/val.jsonl'
        sample = []
        with open(anno_path, mode='r') as f:
            for line in f.readlines():
                sample.append(json.loads(line))
        return sample


    def __len__(self):
        return len(self.data_base)


    def __getitem__(self, index):
        anno = self.data_base[index]
        image_path = os.path.join(self.image_path, anno['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        image = torch.Tensor(image)
        label = anno['labels']
        input_text_features = self.tokenizer.encode_plus(
            anno['text'],
            max_length=self.max_txt_length,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        # text_features = self.tokenizer(anno['text'], max_length=100, truncation=True, padding=True)
        return {'input_ids':input_text_features['input_ids'],\
                'token_type_ids':input_text_features['token_type_ids'],\
                'attention_mask':input_text_features['attention_mask'],\
                'img':image,
                'label':label
                }


class FineHarmC(Dataset):


    def __init__(self, data_split = 'val'):
        super(FineHarmC, self).__init__()
        self.root_path = './data/HarmC'
        self.image_path = os.path.join(self.root_path,'images')
        self.data_split = data_split
        self.max_txt_length = 77
        self.tokenizer = BertTokenizer.from_pretrained('../pretrain_model/vl-bert/bert-base-uncased')
        self.data_base = self.load_annotations()
        self.transforms = torchvision.transforms.Compose(
            [
                #torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )


    def load_annotations(self):
        split = str('fine_') + self.data_split + str('.jsonl')
        anno_path = os.path.join(self.root_path, 'annotations', split)
        #valid_json_path = './data/HarmC/annotations/val.jsonl'
        sample = []
        with open(anno_path, mode='r') as f:
            for line in f.readlines():
                sample.append(json.loads(line))
        return sample


    def __len__(self):
        return len(self.data_base)


    def __getitem__(self, index):
        anno = self.data_base[index]
        image_path = os.path.join(self.image_path, anno['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        image = torch.Tensor(image)
        label = anno['labels']
        input_text_features = self.tokenizer.encode_plus(
            anno['text'],
            max_length=self.max_txt_length,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        # text_features = self.tokenizer(anno['text'], max_length=100, truncation=True, padding=True)
        return {'input_ids':input_text_features['input_ids'],\
                'token_type_ids':input_text_features['token_type_ids'],\
                'attention_mask':input_text_features['attention_mask'],\
                'img':image,
                'label':label
                }


class HarmP(Dataset):


    def __init__(self, data_split = 'val'):
        super(HarmP, self).__init__()
        self.root_path = './data/HarmP'
        self.image_path = os.path.join(self.root_path,'images')
        self.data_split = data_split
        self.max_txt_length = 77
        self.tokenizer = BertTokenizer.from_pretrained('../pretrain_model/vl-bert/bert-base-uncased')
        self.data_base = self.load_annotations()
        self.transforms = torchvision.transforms.Compose(
            [
                #torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )


    def load_annotations(self):
        split = str('new_') + self.data_split + str('.jsonl')
        anno_path = os.path.join(self.root_path, 'annotations', split)
        #valid_json_path = './data/HarmC/annotations/val.jsonl'
        sample = []
        with open(anno_path, mode='r') as f:
            for line in f.readlines():
                sample.append(json.loads(line))
        return sample


    def __len__(self):
        return len(self.data_base)


    def __getitem__(self, index):
        anno = self.data_base[index]
        image_path = os.path.join(self.image_path, anno['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        image = torch.Tensor(image)
        label = anno['labels']
        input_text_features = self.tokenizer.encode_plus(
            anno['text'],
            max_length=self.max_txt_length,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        # text_features = self.tokenizer(anno['text'], max_length=100, truncation=True, padding=True)
        return {'input_ids':input_text_features['input_ids'],\
                'token_type_ids':input_text_features['token_type_ids'],\
                'attention_mask':input_text_features['attention_mask'],\
                'img':image,
                'label':label
                }


class FineHarmP(Dataset):


    def __init__(self, data_split = 'val'):
        super(FineHarmP, self).__init__()
        self.root_path = './data/HarmP'
        self.image_path = os.path.join(self.root_path,'images')
        self.data_split = data_split
        self.max_txt_length = 77
        self.tokenizer = BertTokenizer.from_pretrained('../pretrain_model/vl-bert/bert-base-uncased')
        self.data_base = self.load_annotations()
        self.transforms = torchvision.transforms.Compose(
            [
                #torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )


    def load_annotations(self):
        split = str('fine_') + self.data_split + str('.jsonl')
        anno_path = os.path.join(self.root_path, 'annotations', split)
        #valid_json_path = './data/HarmC/annotations/val.jsonl'
        sample = []
        with open(anno_path, mode='r') as f:
            for line in f.readlines():
                sample.append(json.loads(line))
        return sample


    def __len__(self):
        return len(self.data_base)


    def __getitem__(self, index):
        anno = self.data_base[index]
        image_path = os.path.join(self.image_path, anno['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        image = torch.Tensor(image)
        label = anno['labels']
        input_text_features = self.tokenizer.encode_plus(
            anno['text'],
            max_length=self.max_txt_length,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        # text_features = self.tokenizer(anno['text'], max_length=100, truncation=True, padding=True)
        return {'input_ids':input_text_features['input_ids'],\
                'token_type_ids':input_text_features['token_type_ids'],\
                'attention_mask':input_text_features['attention_mask'],\
                'img':image,
                'label':label
                }




class MET(Dataset):


    def __init__(self, data_split = 'val'):
        super(MET, self).__init__()
        self.root_path = './data/MET'
        self.root_image_path = '/workspace/zlh_fourth/CDG/BLIP/data/MET-memes/Cimages/Eimages/Eimages'
        self.image_path = '/workspace/zlh_fourth/CDG/BLIP/data/MET-memes/Cimages/Eimages/Eimages'
        self.data_split = data_split
        self.max_txt_length = 77
        self.tokenizer = BertTokenizer.from_pretrained('../pretrain_model/vl-bert/bert-base-uncased')
        self.data_base = self.load_annotations()
        self.transforms = torchvision.transforms.Compose(
            [
                #torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )


    def load_annotations(self):
        split = str('new_') + self.data_split + str('.jsonl')
        anno_path = os.path.join(self.root_path, 'annotations', split)
        #valid_json_path = './data/HarmC/annotations/val.jsonl'
        sample = []
        with open(anno_path, mode='r') as f:
            for line in f.readlines():
                sample.append(json.loads(line))
        return sample


    def __len__(self):
        return len(self.data_base)


    def __getitem__(self, index):
        anno = self.data_base[index]
        image_path = os.path.join(self.image_path, anno['image'])

        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        image = torch.Tensor(image)


        label = anno['labels']
        input_text_features = self.tokenizer.encode_plus(
            anno['text'],
            max_length=self.max_txt_length,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        # text_features = self.tokenizer(anno['text'], max_length=100, truncation=True, padding=True)
        return {'input_ids':input_text_features['input_ids'],\
                'token_type_ids':input_text_features['token_type_ids'],\
                'attention_mask':input_text_features['attention_mask'],\
                'img':image,
                'label':label
                }



class fineMET(Dataset):


    def __init__(self, data_split = 'val'):
        super(MET, self).__init__()
        self.root_path = './data/MET'
        self.root_image_path = '/workspace/zlh_fourth/CDG/BLIP/data/MET-memes/Cimages/Eimages/Eimages'
        self.image_path = '/workspace/zlh_fourth/CDG/BLIP/data/MET-memes/Cimages/Eimages/Eimages'
        self.data_split = data_split
        self.max_txt_length = 77
        self.tokenizer = BertTokenizer.from_pretrained('../pretrain_model/vl-bert/bert-base-uncased')
        self.data_base = self.load_annotations()
        self.transforms = torchvision.transforms.Compose(
            [
                #torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )


    def load_annotations(self):
        split = str('fine_') + self.data_split + str('.jsonl')
        anno_path = os.path.join(self.root_path, 'annotations', split)
        #valid_json_path = './data/HarmC/annotations/val.jsonl'
        sample = []
        with open(anno_path, mode='r') as f:
            for line in f.readlines():
                sample.append(json.loads(line))
        return sample


    def __len__(self):
        return len(self.data_base)


    def __getitem__(self, index):
        anno = self.data_base[index]
        image_path = os.path.join(self.image_path, anno['image'])

        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        image = torch.Tensor(image)


        label = anno['labels']
        input_text_features = self.tokenizer.encode_plus(
            anno['text'],
            max_length=self.max_txt_length,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        # text_features = self.tokenizer(anno['text'], max_length=100, truncation=True, padding=True)
        return {'input_ids':input_text_features['input_ids'],\
                'token_type_ids':input_text_features['token_type_ids'],\
                'attention_mask':input_text_features['attention_mask'],\
                'img':image,
                'label':label
                }