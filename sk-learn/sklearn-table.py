import pandas as pd
import numpy as np
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
# from datasets import Dataset
# from torch.utils.data import DataLoader
#? data collator를 안주면 안되네..?
# from transformers import default_data_collator
from tqdm.auto import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
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
    seed: int = 42
    run_name: str = "sklearn-Table"
    save_model_root: str = './best_sklearn-Table'
    gpu: bool = True

    n_jobs: int = 1 # todo ... 토크나이즈 병렬화를 꺼야 되는가?
    verbose: int =  1
    cv: int = 3

def main():
    opt = args()
    # tokenizer = AutoTokenizer.from_pretrained("../kpfbert")
    # bert_model = BertForMaskedLM.from_pretrained('../futher_pretrain.pt')
    # bert_transformer = BertTransformer(tokenizer, bert_model, gpu=opt.gpu)

    default_config = opt
    wandb.init(config = default_config, group = opt.run_name)
    w_config = wandb.config
    set_seed(w_config.seed)
    wandb.run.name = str(w_config.classifier) + str(w_config.seed)
    wandb.run.save()
    wandb.config.update(opt)
    
    train_df = pd.read_csv('../table_data/table_train.csv')
    valid_df = pd.read_csv('../table_data/table_valid.csv')
    test_df = pd.read_csv('../table_data/table_test.csv')
    
    x_train, x_test, y_train, y_test = Table_preprocess(train_df, valid_df, test_df)
    
    x_train = x_train.rename(columns = lambda x:re.sub('[^가-힣A-Za-z0-9_]+', '', x))
    x_test = x_test.rename(columns = lambda x:re.sub('[^가-힣A-Za-z0-9_]+', '', x))

    k_fold = StratifiedKFold(n_splits=opt.cv, shuffle=True, random_state=w_config.seed)

    
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

        classifier = LGBMClassifier(random_state= w_config.seed, objective="multiclass" )  # , device=device
        model = Pipeline([("classifier", classifier)])

        param_grid = {
            'classifier__n_estimators' : [100,200,300,500],
            # 'classifier__max_depth' : [6, 8, 10, 20, 30, -1],  
            # 'classifier__num_leaves' : [80,100,150,200],  
            # 'classifier__min_data_in_leaf' : [25,50,100],  
        }

    else:
        raise NameError('Check your Classifier')

    grid = GridSearchCV(estimator = model,
                param_grid = param_grid,
                scoring="accuracy",
                cv= k_fold,
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
    wandb.log({'accuracy': accuracy ,'precision': precision, 'recall': recall, 'f1_score': f1, 'report_values':report})
    # print(accuracy, precision, recall, f1)

    wandb.sklearn.plot_classifier(grid, 
                                x_train, x_test, 
                                y_train, y_test, 
                                y_pred, y_probas, 
                                labels, 
                                is_binary=False, 
                                model_name=w_config.classifier)

    best_params = grid.best_params_
    wandb.sklearn.plot_summary_metrics(grid, x_train,y_train, x_test, y_test)
    wandb.log({"best_params":best_params})
    wandb.config.update(best_params)
    wandb.finish()
    

if __name__ == "__main__":

    sweep_config = dict(
        name = 'sklearn-table',
        method = 'grid',
        metric = {'name':'accuracy', 'goal':'maximize'}, # ! should be changed
        parameters = {
            'seed': {'values':[31,32,33,34]}, #todo 31,32,33,34 돌려야됨 35,36,37,38,39,40,41,42,43,44,45
            'classifier' : {'values':['LGB','LR','XGB','RF']} # values 'RF','LR','XGB',
        }
    )

    sweep_id = wandb.sweep(sweep_config, project= 'sklearn-table-re')
    wandb.agent(sweep_id, main, count=100)