import pandas as pd
import numpy as np
from transformers import BertForMaskedLM, BertModel, AutoTokenizer
import torch
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, plot_confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Callable, List, Optional, Tuple
from datasets import Dataset
from torch.utils.data import DataLoader
#? data collator를 안주면 안되네..?
from transformers import default_data_collator
from tqdm.auto import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
from dataclasses import dataclass
import wandb
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import List, Optional, Dict
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
import re


def set_seed(seed:int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

def values_cls(cls):
    k = cls.split(' ')
    while("" in k):
        k.remove("")
    d_precision, d_recall, d_f1, d_support = float(k[5]), float(k[6]), float(k[7]), float(k[8].replace('\n', ''))
    c_precision, c_recall, c_f1, c_support = float(k[10]), float(k[11]), float(k[12]), float(k[13].replace('\n', ''))
    b_precision, b_recall, b_f1, b_support = float(k[15]), float(k[16]), float(k[17]), float(k[18].replace('\n', ''))
    bp_precision, bp_recall, bp_f1, bp_support = float(k[20]), float(k[21]), float(k[22]), float(k[23].replace('\n', ''))
    a_precision, a_recall, a_f1, a_support = float(k[25]), float(k[26]), float(k[27]), float(k[28].replace('\n', ''))
    ap_precision, ap_recall, ap_f1, ap_support = float(k[30]), float(k[31]), float(k[32]), float(k[33].replace('\n', ''))
    return [ap_precision, ap_recall, ap_f1,  a_precision, a_recall, a_f1,  \
        bp_precision, bp_recall, bp_f1,   b_precision, b_recall, b_f1, \
        c_precision, c_recall, c_f1,  d_precision, d_recall, d_f1]

def model_evaluation(y_test, pred):
    # confusion = confusion_matrix(y_test,pred)
    cls_report = classification_report(y_test, pred, digits=4)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average = 'macro')
    recall = recall_score(y_test, pred, average = 'macro')
    f1 = f1_score(y_test, pred, average = 'macro')
    # print('오차행렬\n', confusion)
    #f1 score print 추가
    print('정확도: {0:.4f}\n 정밀도: {1:.4f}\n 재현율: {2:.4}\n F1:{3:.4f}'.format(accuracy, precision, 
                                                                                     recall, f1 ))
    return accuracy, precision, recall, f1, cls_report

class BertTransformer(BaseEstimator, TransformerMixin):
    # ! 빠른 멀티스레딩 개선
    def __init__(self,tokenizer,model, max_length: int = 512, gpu = True):

        self.tokenizer = tokenizer
        self.model = model
        if hasattr(self.model, 'cls'):
            self.model = self.model.bert

        self.model.eval()
        if gpu:
            self.model.to("cuda")
            self.gpu = gpu
        else:
            self.gpu = False
        self.max_length = max_length
        self.embedding_func = lambda x: torch.mean(x[0], axis=1).squeeze()

    def tokenize_function(self, examples):
        result = self.tokenizer(examples["TEXT"], padding=True, truncation=True, max_length=512, return_tensors="pt")
        return result

    def tokenize_embed(self, X):        
        #! Huggingface 🤗
        X.reset_index(drop=True, inplace=True)
        dataset = Dataset.from_pandas(X)
        self.tokenized_dataset = dataset.map(self.tokenize_function, batched=True, remove_columns=["TEXT", "label"])
        if self.gpu:
            self.tokenized_dataset = self.tokenized_dataset.with_format("torch", device="cuda")

        dataloader = DataLoader(self.tokenized_dataset, 
                                batch_size=24, 
                                shuffle=False, 
                                drop_last=False, 
                                collate_fn=default_data_collator)
        all_data = []
        for batch in tqdm(dataloader):
            embed = self.model(**batch)
            embed = torch.mean(embed[0], axis=1)
            embed = embed.detach().cpu()
            all_data.append(embed)

        return torch.cat(all_data)

    def transform(self, text):
        with torch.no_grad():
            return self.tokenize_embed(text)

    def fit(self, X, y=None):
        return self

class MergedExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, type:str): # mapping: Optional[Dict] = None
        self.type = type

        #! Very Optional <- transform에서 0과 1을 수정하기 귀찮다.
        # if mapping is not None:
        #     self.text_key = mapping['text']
        #     self.table_key = mapping['table']

    def transform(self, X):
        if self.type == 'text':
            return X[:,:768]
        elif self.type == 'table':
            return X[:,768:]
        else:
            raise NameError("WTF")

    def fit(self, X, y= None):
        return self


def Table_preprocess(train_data, valid_data, test_set):
    train_set = pd.concat([train_data, valid_data])
    train_set.reset_index(drop = True, inplace = True)
    
    #! X,y
    y_train = train_set['label']
    train_set.drop(columns='label', inplace=True)
    y_test = test_set['label']
    test_set.drop(columns = 'label', inplace=True)

    #! Replace NaN values
    cont_columns = train_set.columns[2:]
    for col in cont_columns:
        mean_value = train_set[col].mean()
        train_set[col].fillna(mean_value, inplace=True)
        test_set[col].fillna(mean_value, inplace =True)
    
    # * One-Hot Encoding
    train_test = pd.get_dummies(pd.concat([train_set, test_set], axis=0))
    x_train = train_test[:len(train_set)]
    x_test = train_test[len(train_set):]

    # Data Scaling
    scaler = MinMaxScaler()
    x_train_lr = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_lr, columns= x_train.columns)
    x_test_lr = scaler.transform(x_test)
    x_test = pd.DataFrame(x_test_lr, columns=x_test.columns)

    return x_train, x_test, y_train, y_test


@dataclass
class args:
    # seed: int= 42 -> hmm....
    run_name: str = "sklearn-multimodal"
    save_model_root: str = './best_sklearn-multimodal'
    gpu: bool = True

    n_jobs: int = 1 # todo ... 토크나이즈 병렬화를 꺼야 되는가?
    verbose: int =  1
    cv: int = 3
    # classifier = 'LGB'

def main():
    opt = args()
    tokenizer = AutoTokenizer.from_pretrained("../kpfbert")
    bert_model = BertForMaskedLM.from_pretrained('../futher_pretrain.pt')
    bert_transformer = BertTransformer(tokenizer, bert_model, gpu=opt.gpu)

    default_config = opt
    wandb.init(config = default_config, group = opt.run_name)
    w_config = wandb.config
    set_seed(w_config.seed)
    wandb.run.name = str(w_config.classifier) + str(w_config.seed)
    wandb.run.save()
    wandb.config.update(opt)

    train_df_text = pd.read_csv('../nlp_data/nlp_train.tsv', sep='\t')
    valid_df_text = pd.read_csv('../nlp_data/nlp_valid.tsv', sep='\t')
    test_df_text = pd.read_csv('../nlp_data/nlp_test.tsv', sep='\t')

    # * merge train & valid data
    train_df_text = pd.concat([train_df_text, valid_df_text], axis=0)
    train_df_text.reset_index(drop=True, inplace=True)
    
    x_train_text = bert_transformer.transform(train_df_text)
    y_train_text = train_df_text['label']

    x_test_text = bert_transformer.transform(test_df_text)
    y_test_text = test_df_text['label']

    #! TABLE DATA

    train_df_table = pd.read_csv('../table_data/table_train.csv')
    valid_df_table = pd.read_csv('../table_data/table_valid.csv')
    test_df_table = pd.read_csv('../table_data/table_test.csv')

    x_train_table, x_test_table, y_train_table, y_test_table = Table_preprocess(train_df_table, valid_df_table, test_df_table)

    x_train_table = x_train_table.rename(columns = lambda x:re.sub('[^가-힣A-Za-z0-9_]+', '', x))
    x_test_table = x_test_table.rename(columns = lambda x:re.sub('[^가-힣A-Za-z0-9_]+', '', x))

    k_fold = StratifiedKFold(n_splits=opt.cv, shuffle=True, random_state=w_config.seed)

    # ! merge
    # todo 변경
    x1_text = x_train_text.numpy()
    x2_table = np.array(x_train_table)
    merged_x_train = np.concatenate([x1_text, x2_table], axis=1)

    x1_ = x_test_text.numpy()
    x2_ = np.array(x_test_table) 
    merged_x_test  = np.concatenate([x1_, x2_], axis=1)

    #! classifier settings
    if w_config.classifier == 'RF':

        model_text = Pipeline([
            ('text_extractor', MergedExtractor(type='text')),
            ("classifier", RandomForestClassifier(random_state=w_config.seed))
            ])
        model_table = Pipeline([
            ('table_extractor', MergedExtractor(type='table')),
            ("classifier", RandomForestClassifier(random_state=w_config.seed))
            ])

        param_grid = { 
            'text__classifier__n_estimators': [50,200],
            'text__classifier__max_features': ['sqrt'],
            'text__classifier__max_depth' : [10,20],
            'text__classifier__criterion' :['entropy'],
            'text__classifier__ccp_alpha': [0],
            #'text__classifier__class_weight' : ["balanced", None],
            
            'table__classifier__n_estimators': [50,200],
            'table__classifier__max_features': ['sqrt'],
            'table__classifier__max_depth' : [10,20],
            'table__classifier__criterion' :['entropy'],
            'table__classifier__ccp_alpha': [0],
            #'table__classifier__class_weight' : ["balanced", None],
        }


    elif w_config.classifier == 'LR':

        model_text = Pipeline([
            ('text_extractor', MergedExtractor(type='text')),
            ("classifier", LogisticRegression(random_state=w_config.seed, multi_class='multinomial'))
            ])
        model_table = Pipeline([
            ('table_extractor', MergedExtractor(type='table')),
            ("classifier", LogisticRegression(random_state=w_config.seed, multi_class='multinomial'))
            ])

        param_grid = {
            #'text__classifier__penalty' : ['l1','l2'], 
            'text__classifier__C'       : [1, 10],
            'text__classifier__solver'  : ['newton-cg'],
            
            #'table__classifier__penalty' : ['l1','l2'], 
            'table__classifier__C'       : [1, 10],
            'table__classifier__solver'  : ['newton-cg'],
        }

    elif w_config.classifier =="XGB":
        if opt.gpu:
            text_clf = XGBClassifier(random_state= w_config.seed, objective="multiclass", tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor')
            table_clf = XGBClassifier(random_state= w_config.seed, objective="multiclass", tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor')
        else:
            text_clf = XGBClassifier(random_state= w_config.seed, objective="multiclass")
            table_clf = XGBClassifier(random_state= w_config.seed, objective="multiclass")

        model_text = Pipeline([
            ('text_extractor', MergedExtractor(type='text')),
            ("classifier", text_clf)
            ])
        model_table = Pipeline([
            ('table_extractor', MergedExtractor(type='table')),
            ("classifier", table_clf)
            ])

        param_grid = {
            'text__classifier__n_estimators' : [400,500],
            'text__classifier__max_depth' : [8,10],
            'text__classifier__gamma' : [0.5],
            'text__classifier__subsample'  : [0.8,0.9], 
            'text__classifier__colsample_bytree' : [0.8],

            'table__classifier__n_estimators' : [100,80],
            'table__classifier__max_depth' : [6,8],
            'table__classifier__gamma' : [1],
            'table__classifier__subsample'  : [0.9,1.0], 
            'table__classifier__colsample_bytree' : [0.8],
        }

    elif w_config.classifier =="LGB":

        model_text = Pipeline([
            ('text_extractor', MergedExtractor(type='text')),
            ("classifier", LGBMClassifier(random_state= w_config.seed, objective="multiclass"))
            ])
        model_table = Pipeline([
            ('table_extractor', MergedExtractor(type='table')),
            ("classifier", LGBMClassifier(random_state= w_config.seed, objective="multiclass"))
            ])

        param_grid = {
            'text__classifier__n_estimators' : [50,300,350],
            #'text__classifier__max_depth' : [ 8, 10, 12, -1],  
            #'text__classifier__num_leaves' : [80,60,100],  
            #'text__classifier__min_data_in_leaf' : [80,100],  

            'table__classifier__n_estimators' : [50,300,350],
            #'table__classifier__max_depth' : [8, 10, 12, -1],  
            #'table__classifier__num_leaves' : [140,150],  
            #'table__classifier__min_data_in_leaf' : [20,50,100],  
        }

    else:
        raise NameError('Check your Classifier')

    multi_clf = VotingClassifier(estimators=[
        ('text',model_text), 
        ('table',model_table)], 
        voting='soft')

    grid = GridSearchCV(estimator = multi_clf,
            param_grid = param_grid,
            scoring="accuracy",
            cv= k_fold,
            n_jobs= opt.n_jobs,
            verbose= opt.verbose
            )

    grid_result = grid.fit(merged_x_train, y_train_table)
    y_pred = grid.predict(merged_x_test)
    y_probas = grid.predict_proba(merged_x_test)
    
    labels = ["D","C","B","B+","A","A+"]

    # 편의성 도모
    y_train = y_train_table
    y_test= y_test_table

    accuracy, precision, recall, f1, report = model_evaluation(y_test, y_pred)
    #todo ... classification report ...
    report = values_cls(report)
    wandb.log({'accuracy': accuracy ,'precision': precision, 'recall': recall, 'f1_score': f1, 'report_values': report})
    # print(accuracy, precision, recall, f1)

    wandb.sklearn.plot_classifier(grid, 
                                merged_x_train, merged_x_test, 
                                y_train, y_test, 
                                y_pred, y_probas, 
                                labels, 
                                is_binary=False, 
                                model_name=w_config.classifier)

    best_params = grid.best_params_
    wandb.sklearn.plot_summary_metrics(grid, merged_x_train, y_train, merged_x_test, y_test)
    wandb.config.update(best_params)
    wandb.log({"best_params":best_params})
    wandb.finish()
    

if __name__ == "__main__":

    sweep_config = dict(
        name = 'sklearn-text',
        method = 'grid',
        metric = {'name':'accuracy', 'goal':'maximize'}, # ! should be changed
        parameters = {
            'seed': {'values':[31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]},
            'classifier' : {'values':['LGB','RF','LR','XGB']} # values ['RF','LR','XGB']
        }
    )

    sweep_id = wandb.sweep(sweep_config, project= 'sklearn-multimodal-re')
    wandb.agent(sweep_id, main, count=100)