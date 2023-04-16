import pandas as pd
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts #? 필요할까?
from transformers import AutoConfig, AutoTokenizer, AdamW, EvalPrediction, AutoModelForSequenceClassification,BertModel, BertTokenizer

from pytorch_lightning import seed_everything

from tqdm.auto import tqdm

from typing import List, Optional, Union
import logging
import torchinfo
import wandb

from table_dataset import data_prep ,DataSetCatCon
from nlp_dataset import DARTProcessor, FewShotDataset
from nlp_models import BertForPromptFinetuning
from transformers import DataProcessor
from augmentations import embed_data_mask
from models import SAINT
from tqdm.auto import tqdm
import torch.functional as F

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ! 추가된 Reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class args:
    processor: DataProcessor
    prompt: bool
    mapping: str
    model_name_or_path: str = "kpfbert"
    truncate_head: bool = False
    first_sent_limit: int = 496
    max_seq_length: int = 512    
    data_dir: str = 'nlp_data'
    use_demo: bool = False
    template: str = '*cls*비재무적_요소를_고려한_기업의_환경_사회_투명_경영_등급은_*mask*_이다.*sent_0**sep*'
    batch_size: int = 24
    lr: int = 5e-5
    epoch: int= 13 #! epoch 많이 주지마

    # seed: int= 42

    run_name: str = "saint+prompt"
    save_model_root: str = './best_multi_models_seeds'
    active_log: bool = True
    num_labels = 6
    
    ##SAINT
    dtask = 'clf'
    task = 'multiclass'
    cont_embeddings = 'MLP'
    embedding_size = 16
    transformer_depth = 2
    attention_heads = 8
    attention_dropout = 0.8
    ff_dropout = 0.8
    attentiontype = 'colrow'
    dset_id = 'table_data'
    final_mlp_style = 'sep'
    
    # ?
    optimizer = 'AdamW'
    scheduler = 'cosine'

class MultimodalDataset(Dataset):
    def __init__(self, nlp_data, table_data):
        self.nlp_data = nlp_data
        self.table_data = table_data

    def __getitem__(self, index):
        x1 = self.nlp_data[index]
        x2 = self.table_data[index]

        return x1, x2

    def __len__(self):
        return len(self.nlp_data)


class BertModelModified(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert= bert_model.bert
        self.cls = bert_model.cls.predictions.transform
        self.cls.decoder = nn.Sequential(nn.Linear(768,512),nn.GELU(), nn.Linear(512,256))
        
    def forward(self, input_ids, attention_mask ,token_type_ids, mask_pos, labels):
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
            
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output , _ = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos] # 마스크 부분의 representation만을 뽑아냄.
        mask_score = self.cls(sequence_mask_output)
        score = self.cls.decoder(mask_score)
        return score


class MultimodalModel(nn.Module):
    def __init__(self, bert_model, tabular_model):
        super().__init__()
        self.bert = bert_model
        self.tabular = tabular_model
        self.linear = nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,6)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, mask_pos, labels, 
                x_categ_enc, x_cont_enc):
        
        nlp_logit = self.bert(input_ids, attention_mask, token_type_ids, mask_pos, labels)
        reps = self.tabular.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:,0,:]
        tabular_logit = self.tabular.mlpfory(y_reps)

        f_logit = torch.concat([nlp_logit, tabular_logit], dim=1)
        logit = self.linear(f_logit)
        return logit

def classification_score(model, dloader, device, tabular_model):
    model.eval()
    y_true = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for data, tabular_data in dloader:
            y_true_data = data['labels'].to(device)
            y_true = torch.cat([y_true, y_true_data.squeeze().float()], dim=0) # ground truth

            data = {k: v.to(device) for k, v in data.items()}

            x_categ, x_cont, y_gts, cat_mask, con_mask = tabular_data[0].to(device), tabular_data[1].to(device),tabular_data[2].to(device),tabular_data[3].to(device),tabular_data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, tabular_model, False) 

            data['x_categ_enc'] = x_categ_enc
            data['x_cont_enc'] = x_cont_enc
            logit = model(**data)
            y_pred = torch.cat([y_pred, torch.argmax(logit, dim=1).float()], dim=0) # model prediction
    
    correct_results_sum = (y_pred == y_true).sum().float()
    acc = correct_results_sum/y_true.shape[0] * 100

    return acc.cpu().numpy()        

def main():

    model_name_or_path = "kpfbert"

    bert_model = BertModel.from_pretrained(model_name_or_path, add_pooling_layer=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, return_tensors="pt")

    label_mapping_dict = '{"5": "최상급", "4": "상급", "3": "중상", "2": "중급", "1": "하급", "0": "최하"}'

    # ! parameters
    opt = args(
        processor = DARTProcessor(),
        prompt = True,
        mapping= label_mapping_dict
    )

    # ! WandB
    if opt.active_log:
        default_config = opt
        wandb.init(config=default_config, group=opt.run_name)
        w_config = wandb.config
        wandb.config.update(opt)

    # ! 저장공간
    os.makedirs(opt.save_model_root+ '/' + str(w_config.seed), exist_ok=True)

    # ! SEED and root
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")
    set_seed(w_config.seed)
    
    #! NLP DATA
    train_nlp = FewShotDataset(opt, tokenizer=tokenizer, mode='train')
    valid_nlp = FewShotDataset(opt, tokenizer=tokenizer, mode='dev')
    test_nlp = FewShotDataset(opt, tokenizer=tokenizer, mode='test')

    #! TABULAR DATA
    cat_dims, cat_idxs, con_idxs, \
        X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep('./table_data', 42 ,'clf')

    y_dim = len(np.unique(y_train['data'][:,0]))
    cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.

    continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 
    train_table = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask , continuous_mean_std)
    valid_table = DataSetCatCon(X_valid, y_valid, cat_idxs,opt.dtask, continuous_mean_std)
    test_table = DataSetCatCon(X_test, y_test, cat_idxs,opt.dtask, continuous_mean_std)

    train_set = MultimodalDataset(train_nlp, train_table)
    trainloader = DataLoader(train_set, batch_size= 24, shuffle=True, num_workers=4)

    valid_set = MultimodalDataset(valid_nlp, valid_table)
    validloader = DataLoader(valid_set, batch_size= 24, shuffle=True, num_workers=4)

    test_set = MultimodalDataset(test_nlp, test_table)
    testloader = DataLoader(test_set, batch_size= 24, shuffle=True, num_workers=4)
    config = AutoConfig.from_pretrained(
        opt.model_name_or_path,
        num_labels=opt.num_labels,
    )

    # ! Models
    bert_model = BertForPromptFinetuning.from_pretrained(opt.model_name_or_path, config=config)
    tabular_model = SAINT(
        categories = tuple(cat_dims), 
        num_continuous = len(con_idxs),                
        dim = opt.embedding_size,                           
        dim_out = 1,                       
        depth = opt.transformer_depth,                       
        heads = opt.attention_heads,                         
        attn_dropout = opt.attention_dropout,             
        ff_dropout = opt.ff_dropout,                  
        mlp_hidden_mults = (4, 2),       
        cont_embeddings = opt.cont_embeddings,
        attentiontype = opt.attentiontype,
        final_mlp_style = opt.final_mlp_style,
        y_dim = y_dim
    )

    bert_model.label_word_list = torch.tensor(train_nlp.label_word_list).long().to(device) 

    bert_model.load_state_dict(torch.load('./pretrained_bestmodel/nlp/bestmodel.pth'))
    tabular_model.load_state_dict(torch.load('./pretrained_bestmodel/tabular/bestmodel.pth'))

    bert_model.to(device)
    tabular_model.to(device)

    modified_bert = BertModelModified(bert_model)
    tabular_model.mlpfory = nn.Sequential(nn.Linear(16,512),nn.GELU(), nn.Linear(512,256))
    modified_bert.to(device)

    multi_model = MultimodalModel(modified_bert, tabular_model)
    multi_model.to(device)

    # modified_bert + tabular_model
    optimizer = AdamW(multi_model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # ! Training

    num_epochs = opt.epoch
    num_training_steps = num_epochs * len(trainloader)

    progress_bar = tqdm(range(num_training_steps))

    losses = []
    accuracies = []

    best_valid_accuracy = 0
    best_test_accuracy = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        batches = 0
        runnning_loss = 0.0
        multi_model.train()

        for batch, batch_tabular in tqdm(trainloader):
            optimizer.zero_grad()
            
            #! NLP part
            batch = {k: v.to(device) for k, v in batch.items()}
            #! Tabular part
            x_categ, x_cont, y_gts, cat_mask, con_mask = batch_tabular[0].to(device), batch_tabular[1].to(device),batch_tabular[2].to(device),batch_tabular[3].to(device),batch_tabular[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,tabular_model, False) 
            y_true = batch['labels']
            batch['x_categ_enc'] =x_categ_enc
            batch['x_cont_enc'] =x_cont_enc
            logit = multi_model(**batch) # here we do not need loss of nlp

            loss = criterion(logit, y_true.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(logit, 1)
            correct += (predicted == y_true).sum()
            total += len(y_true)

            progress_bar.update(1)

        if opt.active_log:
            wandb.log({'epoch': epoch ,'train_epoch_loss': total_loss, 'loss': loss.item()})
            
        losses.append(total_loss)
        accuracies.append(correct.float() / total)
        print("Train Loss:", total_loss, "Accuracy:", correct.float() / total)

        multi_model.eval()
        with torch.no_grad():
            if opt.task in ['binary','multiclass']:
                accuracy = classification_score(multi_model, validloader, device, tabular_model)
                test_accuracy = classification_score(multi_model, testloader, device, tabular_model)
                print('[EPOCH %d] VALID ACCURACY: %.4f' %(epoch + 1, accuracy))
                print('[EPOCH %d] TEST ACCURACY: %.4f' %(epoch + 1, test_accuracy ))

                if opt.active_log:
                    wandb.log({'valid_accuracy': accuracy })     
                    wandb.log({'test_accuracy': test_accuracy })

                if accuracy > best_valid_accuracy:
                    best_valid_accuracy = accuracy
                    best_test_accuracy = test_accuracy
                    torch.save(multi_model.state_dict(),f'%s/{w_config.seed}/best_model.pth' % (opt.save_model_root))
                    
        multi_model.train()
    
    print('Accuracy on best model:  %.3f' %(best_test_accuracy))
    wandb.log({'test_accuracy_bestep': best_test_accuracy})

if __name__ == "__main__":

    sweep_config = dict(
        name = 'Multimodal-ft-seed',
        method = 'grid',
        metric = {'name':'loss', 'goal':'minimize'},
        parameters = {
            'seed': {'values':[30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]},
        }
    )

    sweep_id = wandb.sweep(sweep_config, project= 'Multimodal-sweep')
    wandb.agent(sweep_id, main, count=100)
