from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BertTokenizerFast
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
# from sacrebleu import corpus_bleu
import argparse
import json
from torch import cuda
# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console


from network import BERTModel, BERT_CRF_Model
from scorer import evaluate_ner

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_model_name', type=str, default="bert-base-uncased", help='pretrain Model type')
parser.add_argument('--train_batch_size', type=int, default=32) #original is 16
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--num_train_epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_ckpt', type=int, default=2)
parser.add_argument('--model', type=str, default="bert_crf", help='Model type') #bert_crf
parser.add_argument('--ckpt_path', type=str, default="", help='trained model path') #ckpt_path
args = parser.parse_args()

# LABEL_MAP_PATH = "/home/bwang/project/nlpcc2022/ner/data/ner_label.json"
LABEL_MAP_PATH = "./data/ner_label.json"
MAX_SRC_LENGTH =400

console = Console(record=True)
console.print(args)

device = torch.device('cuda' if cuda.is_available() else 'cpu')
# print('Connected to GPU:', torch.cuda.get_device_name(0))
tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_model_name, do_lower_case=True)

with open(LABEL_MAP_PATH, "r", encoding= "utf-8") as f:
    label_list = json.load(f) # e.g.,I-CH:0
id2label = {}
for k, v in label_list.items():
    id2label[v] = k # e.g.,0:I-CH

class NER_dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
#
def tokenize_and_align_labels(texts, tags):
    '''
        return the tags of the tokens to id list
    '''
    tokenized_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(label_list["<pad>"])
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_list[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_list["<pad>"])
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs  


all_data = {"train":{"texts": [], "tags": []}, \
            "val" : {"texts": [], "tags": []}, \
            "test" : {"texts": [], "tags": []}}
for mode in ['train', 'val', 'test']:
    with open(f"./data/fan/{mode}.txt", 'r',encoding='UTF-8') as reader:
        sentence = [] # combine tokens and labels back to labels
        label = []
        for token_label in reader.readlines():  #token_label,e.g. : In O\n
            if not token_label.startswith("\n"):
                t, l = token_label.strip("\n").split(" ") # t: token,l:label
                sentence.append(t.lower())
                label.append(l)
            elif token_label.startswith("\n"):
                all_data[mode]["texts"].append(sentence)
                all_data[mode]["tags"].append(label)

                sentence = []
                label = []

## 建立数据集
train_inputs_and_labels = tokenize_and_align_labels(all_data["train"]["texts"], all_data["train"]["tags"])
val_inputs_and_labels = tokenize_and_align_labels(all_data["val"]["texts"], all_data["val"]["tags"])
test_inputs_and_labels = tokenize_and_align_labels(all_data["test"]["texts"], all_data["test"]["tags"])

train_dataset = NER_dataset(train_inputs_and_labels, train_inputs_and_labels["labels"])
val_dataset = NER_dataset(val_inputs_and_labels, val_inputs_and_labels["labels"])
test_dataset = NER_dataset(test_inputs_and_labels, test_inputs_and_labels["labels"])


def train_step(
    epoch,
    tokenizer,
    model,
    device,
    loader,
    optimizer
):
    model.train()
    pbar = tqdm(enumerate(loader), total=len(loader)) #这个写法也挺有意思啊
    for _, data in pbar:
        labels = data['labels'].squeeze().to(device) #(Batch,290),batch=16
        input_ids = data['input_ids'].squeeze().to(device) #(Batch,290)
        attention_mask = data['attention_mask'].to(device) #(Batch,290)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        ) #(loss,logits), logits:(batch,seqlen,num_tags)
        loss = outputs[0]
        pbar.set_postfix(loss=float(loss.detach().cpu()), refresh=True) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_step(
    model,
    device,
    loader,
    args
):
    model.eval()
    model.to(device)
    with torch.no_grad():
        preds = []
        label = []
        pbar = tqdm(enumerate(loader), desc='eval', total=len(loader))
        for _, data in pbar:
            labels = data['labels'].squeeze().to(device)
            input_ids = data['input_ids'].squeeze().to(device)
            attention_mask = data['attention_mask'].to(device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = output[1]
            if args.model == "bert_crf":
                pred = model.crf.decode(logits, attention_mask).squeeze()
                # print(pred.shape)
            elif args.model == "bert":
                pred = torch.argmax(logits, dim= -1) 
                # print(pred.shape)

            # exit()           
            preds.append(pred)             
            label.append(labels)
        y_preds = torch.cat(preds, axis = 0).detach().cpu().numpy().tolist()
        y_labels = torch.cat(label, axis = 0).detach().cpu().numpy().tolist()
        true_predictions = [
                    [id2label[p] for (p, l) in zip(prediction, label) if l != label_list["<pad>"]]
                        for prediction, label in zip(y_preds, y_labels)
                            ]
        true_labels = [[id2label[l] for l in label if l != label_list["<pad>"]] for label in y_labels]
        print("test sentence number: ", len(true_labels))
        print("show 5 example label")
        for i, j in zip(true_predictions[:5], true_labels[:5]):
            print(i)
            print(j)
            print()


        acc, recall, f1 = evaluate_ner(true_predictions, true_labels)
        
    return acc, recall, f1

def Bert_Trainer(
    args,
    output_dir='./checkpoints/'
):
    NAME = args.model
    os.makedirs(output_dir + NAME, exist_ok=True)
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    
    
    train_params = {
        "batch_size": args.train_batch_size,
        "shuffle": True,
        "num_workers": 0,
    }
    
    eval_params = {
        "batch_size": args.eval_batch_size,
        "shuffle": False,
        "num_workers": 0,
    }
    
    train_loader = DataLoader(train_dataset, **train_params) #这个赋值方法挺好，学会了，构造一个字典，然后再解包
    eval_loader = DataLoader(val_dataset, **eval_params)
    test_loader = DataLoader(test_dataset, **eval_params)

    if args.ckpt_path == "":
        if args.model == "bert_crf":
            # 使用Bert_CRF,from_pretrained是什么操作？
            model = BERT_CRF_Model.from_pretrained(args.pretrain_model_name, num_labels = 16)
        elif args.model == "bert":
            model = BERTModel.from_pretrained(args.pretrain_model_name, num_labels = 16)
        else:
            raise f"no match type for model type --- {args.model}"
        model = model.to(device)
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.learning_rate
        )
        best_epoch = 0
        best_f1 = 0
        for epoch in range(args.num_train_epochs):
            train_step(epoch, tokenizer, model, device, train_loader, optimizer)
            console.log(f"[Initiating Validation]...\n")
            acc, recall, f1 = evaluate_step(model, device, eval_loader, args)
            print(f"acc: {acc:.3f}      recall: {recall:.3f}     f1: {f1:.3f}")
            if f1 > best_f1:
                path = os.path.join(output_dir + NAME, f"checkpoints-epoch-{epoch}-F1-{f1:.3f}")
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                path_list = os.listdir(output_dir + NAME)
                if len(path_list) > args.max_ckpt:
                    path_sorted = sorted(path_list, key= lambda x : float(x.split("-")[-1]))
                    last_path = os.path.join(output_dir + NAME, path_sorted[0])
                    os.system(f"rm -rf {last_path}")
    else:
        if args.model == "bert_crf":
            model = BERT_CRF_Model.from_pretrained(args.ckpt_path)
        elif args.model == "bert":
            model = BERTModel.from_pretrained(args.ckpt_path)
        else:
            raise f"no match type for model type --- {args.model}"
        # model = BERTModel.from_pretrained(args.ckpt_path)
        acc, recall, f1 = evaluate_step(model, device, test_loader, args)
        print(f"acc: {acc:.3f}      recall: {recall:.3f}     f1: {f1:.3f}")
        
Bert_Trainer(
    args
)
