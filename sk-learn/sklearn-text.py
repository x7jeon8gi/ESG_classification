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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from dataclasses import dataclass
import wandb
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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

    print('정확도: {0:.4f}\n 정밀도: {1:.4f}\n 재현율: {2:.4}\n F1:{3:.4f}'.format(accuracy, precision, recall, f1 ))

    return accuracy, precision, recall, f1, cls_report


@dataclass
class args:
    # seed: int= 42 -> hmm....
    seed: int = 42
    run_name: str = "sklearn-text"
    save_model_root: str = './best_sklearn-text'
    gpu: bool = True

    n_jobs: int = 1 # todo ... 토크나이즈 병렬화를 꺼야 되는가?
    verbose: int =  1
    cv: int = 3

class BertTransformer(BaseEstimator, TransformerMixin):
    # ! 빠른 멀티스레딩 개선
    def __init__(
        self,
        tokenizer,
        model,
        max_length: int = 512,
        gpu = True
        ):

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

    train_df = pd.read_csv('../nlp_data/nlp_train.tsv', sep='\t')
    valid_df = pd.read_csv('../nlp_data/nlp_valid.tsv', sep='\t')
    test_df = pd.read_csv('../nlp_data/nlp_test.tsv', sep='\t')

    # * merge train & valid data
    train_df = pd.concat([train_df, valid_df], axis=0)
    train_df.reset_index(drop=True, inplace=True)
    
    x_train = bert_transformer.transform(train_df)
    y_train = train_df['label']

    x_test = bert_transformer.transform(test_df)
    y_test = test_df['label']

    #! classifier settings
    if w_config.classifier == 'RF':
        classifier = RandomForestClassifier(random_state=w_config.seed)
        model = Pipeline([("classifier", classifier),])

        param_grid = { 
            'classifier__n_estimators': [100, 200],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__max_depth' : [5,10,20,50,100],
            'classifier__criterion' :['gini', 'entropy'],
            'classifier__ccp_alpha': [0.01, 0.001, 0],
            'classifier__class_weight' : ["balanced", None],
        }

    elif w_config.classifier == 'LR':
        classifier = LogisticRegression(random_state=w_config.seed, multi_class='multinomial')
        model = Pipeline([("classifier", classifier),])

        param_grid = {
            'classifier__penalty' : ['l1','l2'], 
            'classifier__C'       : [0.01, 0.1, 1, 10],
            'classifier__solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
        }

    # elif w_config.classifier == 'GB':
    #     classifier = GradientBoostingClassifier(random_state=w_config.seed)
    #     model = Pipeline([("classifier", classifier),])

    #     param_grid = { 
    #         'classifier__n_estimators' : [100,200,300,500], 
    #         'classifier__max_depth' : [6, 8, 10, 20, 30],  
    #         'classifier__min_samples_leaf' : [3, 5, 7, 10] 
    #     }

    elif w_config.classifier =="XGB":
        if opt.gpu:
            classifier = XGBClassifier(random_state= w_config.seed, objective="multiclass", tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor')
        else:
            classifier = XGBClassifier(random_state= w_config.seed, objective="multiclass")
        model = Pipeline([("classifier", classifier),])
        param_grid = {
            'classifier__n_estimators' : [100,200,300,500],
            'classifier__max_depth' : [6, 8, 10, 20, 30],
            'classifier__gamma' : [0.5, 1, 1.5],
            'classifier__subsample'  : [0.6,0.8,1.0], 
            'classifier__colsample_bytree' : [0.8,0.9],
        }

    elif w_config.classifier =="LGB":
        # if opt.gpu:
        #     device = "gpu"
        # else:
        #     device ="cpu"
        classifier = LGBMClassifier(random_state= w_config.seed, objective="multiclass") #, device=device)
        model = Pipeline([("classifier", classifier)])

        param_grid = {
            'classifier__n_estimators' : [100,200,300,500],
            'classifier__max_depth' : [6, 8, 10, 20, 30,-1],  
            'classifier__num_leaves' : [80,100,150,200],  
            'classifier__min_data_in_leaf' : [25,100,200],  
        }

    else:
        raise NameError('Check your Classifier')

    grid = GridSearchCV(estimator = model,
                param_grid = param_grid,
                scoring="accuracy",
                cv= opt.cv,
                n_jobs= opt.n_jobs,
                verbose= opt.verbose
                )
                
    grid_result = grid.fit(x_train, y_train) 
    y_pred = grid.predict(x_test)
    y_probas = grid.predict_proba(x_test)
    labels = ["D","C","B","B+","A","A+"]

    accuracy, precision, recall, f1, report = model_evaluation(y_test, y_pred)

    #todo ... classification report ...
    report = values_cls(report)
    wandb.log({'accuracy': accuracy ,'precision': precision, 'recall': recall, 'f1_score': f1, 'report_values': report})
    # print(accuracy, precision, recall, f1)

    wandb.sklearn.plot_classifier(grid, 
                                x_train, x_test, 
                                y_train, y_test, 
                                y_pred, y_probas, 
                                labels, 
                                is_binary=False, 
                                model_name=w_config.classifier)

    best_params = grid.best_params_
    wandb.sklearn.plot_summary_metrics(grid, x_train, y_train, x_test, y_test)
    wandb.config.update(best_params)
    wandb.log({"best_params":best_params})
    wandb.finish()
    

if __name__ == "__main__":

    sweep_config = dict(
        name = 'sklearn-text',
        method = 'grid',
        metric = {'name':'accuracy', 'goal':'maximize'}, # ! should be changed
        parameters = {
            'seed': {'value':42}, # [31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
            'classifier' : {'values':['RF','LR','XGB','LGB']}
        }
    )

    sweep_id = wandb.sweep(sweep_config, project= 'sklearn-text-re')
    wandb.agent(sweep_id, main, count=100)