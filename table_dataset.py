import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
from torch.utils.data import Dataset
import os

def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


def remove_comma(x):
    try:
        x = x.replace(',','')
    except:
        pass
    try:
        x = float(x)
    except:
        pass
    return x

#! non-string 
def non_string(x):
    if type(x) == str:
        return np.NaN
    else:
        return x

def remove_comma(x):
    try:
        x = x.replace(',','')
    except:
        pass
    try:
        x = float(x)
    except:
        pass
    return x


def data_prep(path, seed, task):

    np.random.seed(seed)

    if path is None:
        path=os.getcwd()
    else:
        pass

    #! data 불러오기
    train_ = pd.read_csv(path+'/table_train.csv', index_col=0)
    dev_ = pd.read_csv(path+'/table_valid.csv', index_col=0)
    test_ = pd.read_csv(path+'/table_test.csv', index_col=0)
    
    for_test = len(test_)
    for_dev = len(dev_)
    for_train = len(train_)
    X = pd.concat([train_, dev_, test_])

    #! reset index
    X.reset_index(drop = True, inplace = True)
    #! remove whitespace in columns
    X.rename(columns=lambda x: x.strip(), inplace=True)

    #! y target
    y = X['label']
    X.drop(['label'], axis = 1, inplace=True)

    #! categories
    #! 이부분은 후속 알고리즘에서 적절히 개선이 필요한 부분 (자동적으로 index 입력으로 바꿀것)
    #! 잘 모르겠는데? 일단 해보자
    categorical_indicator = [True, True]
    categorical_len = len(categorical_indicator)
    for _ in range(len(X.columns)-categorical_len): #Label 위해
        categorical_indicator.append(False)

    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    #! missingvalue masking
    #? 아래가 꼭 필요한건가? 안쓰는거 같은데
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)

    train_indices = X.index[ : for_train]
    valid_indices = X.index[for_train : for_train+for_dev]
    test_indices = X.index[for_train+for_dev:]

    #todo 문제는 label이네..!
    #todo 다시 코딩해야됨............................................
    cat_dims = []
    for col in categorical_columns:
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))

    for col in cont_columns:
        X.fillna(X.loc[:, col].mean(), inplace=True)
    y = y.values

    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)

    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_valid, y_valid = data_split(X,y,nan_mask,valid_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

