import os

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def onehot_encode(df,data_path_1, flag='train'):
    df = df.reset_index(drop=True)
    ## test if there is nulls in dataset
    if sum(df.isnull().any()) > 0:
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        var_numerics = df.select_dtypes(include=numerics).columns
        var_str = [i for i in df.columns if i not in var_numerics]
        ## fill nulls of numeric variables with -7777(a number that is not in the original variable)
        if len(var_numerics) > 0:
            df.loc[:,var_numerics] = df[var_numerics].fillna(-7777)
        ## fill nulls of string variables with 'NA'
        if len(var_str) > 0:
            df.loc[:,var_str] = df[var_str].fillna('NA')
    if flag == 'train':
        ## train onehot encoder with train set
        enc = OneHotEncoder().fit(df)
        ## save the encoder
        save_model = open(os.path.join(data_path_1,'onehot_encoder.pkl'), 'wb')
        pickle.dump(enc, save_model)
        save_model.close()
        df_return = pd.DataFrame(enc.transform(df).toarray())
        df_return.columns = enc.get_feature_names(df.columns)
    elif flag == 'transform':
        ## load the encoder
        load_model = open(os.path.join(data_path_1,'onehot_encoder.pkl'), 'rb')
        onehot_model = pickle.load(load_model)
        load_model.close()
        ## if there is null in test set but not in train set, delete the sample
        var_range = onehot_model.categories_ # find output of each class of each variable
        var_name = df.columns
        del_index = []
        for i in range(len(var_range)):
            if 'NA' not in var_range[i] and 'NA' in df[var_name[i]].unique():
                index = np.where(df[var_name[i]] == 'NA')
                del_index.append(index)
            elif -7777 not in var_range[i] and -7777 in df[var_name[i]].unique():
                index = np.where(df[var_name[i]] == -7777)
                del_index.append(index)
        ## delete the sample
        if len(del_index) > 0:
            def_index = np.unique(del_index)
            df = df.drop(def_index)
            print(f'The # {del_index} samples deleted in test set')
        ## encode the test set
        df_return = pd.DataFrame(enc.transform(df).toarray())
        df_return.columns = enc.get_feature_names(df.columns)
    elif flag == 'inverse_transform':
        ## load the encoder
        load_model = open(os.path.join(data_path_1,'onehot_encoder.pkl'), 'rb')
        onehot_model = pickle.load(load_model)
        load_model.close()
        ## inverse transform
        df_return = pd.DataFrame(onehot_model.inverse_transform(df))
        df_return.columns = np.unique(['_'.join(i.split('_')[:-1]) for i in df.columns])
    return df_return
        

def woe_cal_trans(x, y, target='1'):
    ## 计算总体的正负样本数
    p_total = sum(y == target) # 正样本数
    n_total = len(x) - p_total # 负样本数
    value_num = list(x.unique()) # 变量的取值
    woe_map = {}
    iv_value = 0
    y = y.reindex(x.index)
    for i in value_num:
        ## 计算每个取值箱内的正负样本数
        y1 = y[np.where(x == i)[0]]
        p_num_1 = sum(y1 == target)
        n_num_1 = len(y1) - p_num_1
        ## 计算每个取值箱内的正负样本占总体的比例
        bad_rate_1 = p_num_1 / p_total
        good_rate_1 = n_num_1 / n_total
        if bad_rate_1 == 0:
            bad_rate_1 = 1e-5
        elif good_rate_1 == 0:
            good_rate_1 = 1e-5
        woe_map[i] = np.log(bad_rate_1 / good_rate_1)
        iv_value += (bad_rate_1 - good_rate_1) * woe_map[i]
    x_woe_trans = x.map(woe_map)
    x_woe_trans.name = x.name + '_woe'

    return x_woe_trans, woe_map, iv_value


def woe_encode(df, data_path, varnames, y, filename, flag='train'):
    df = df.reset_index(drop=True)
    ## test if there is nulls in dataset
    if sum(df.isnull().any()) > 0:
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        var_numerics = df.select_dtypes(include=numerics).columns
        var_str = [i for i in df.columns if i not in var_numerics]
        ## fill nulls of numeric variables with -7777(a number that is not in the original variable)
        if len(var_numerics) > 0:
            df.loc[:,var_numerics] = df[var_numerics].fillna(-7777)
        ## fill nulls of string variables with 'NA'
        if len(var_str) > 0:
            df.loc[:,var_str] = df[var_str].fillna('NA')
    if flag == 'train':
        iv_values = {}
        woe_maps = {} # save the woe maps of each variable
        var_woe_name = []
        for var in varnames:
            x = df[var]
            ## 变量映射
            x_woe_trans, woe_map, iv = woe_cal_trans(x, y)
            var_woe_name.append(x_woe_trans.name)
            iv_values[var] = iv
            woe_maps[var] = woe_map
        ## save the woe maps
        save_woe_dicts = open(os.path.join(data_path, filename + '.pkl'), 'wb')
        pickle.dump(woe_maps, save_woe_dicts, 0)
        save_woe_dicts.close()
        return df, woe_maps, iv_values, var_woe_name

    elif flag == 'transform':
        ## transform the test set
        read_woe_dict = open(os.path.join(data_path, filename + '.pkl'), 'rb')
        woe_dict = pickle.load(read_woe_dict)
        read_woe_dict.close()
        ## if there is null in test set but not in train set, delete the sample
        woe_dict.keys()
        del_index = []
        for key, value in woe_dict.items():
            if 'NA' not in value.keys() and 'NA' in df[key].unique():
                index = np.where(df[key] == 'NA')
                del_index.append(index)
            elif -7777 not in value.keys() and -7777 in df[key].unique():
                index = np.where(df[key] == -7777)
                del_index.append(index)
        ## delete the sample
        if len(del_index) > 0:
            def_index = np.unique(del_index)
            df = df.drop(def_index)
            print(f'The # {del_index} samples deleted in test set')

        ## WOE编码映射
        var_woe_name = []
        for key, value in woe_dict.items():
            val_name = key + '_woe'
            df[val_name] = df[key].map(value)
            var_woe_name.append(val_name)
        return df, var_woe_name