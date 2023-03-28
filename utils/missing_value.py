import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


def drop_row(datas: pd.DataFrame):
    return datas.dropna(axis=0, how='any')


def filling_row(datas: pd.DataFrame):
    fill_dict = {}
    for key in datas.keys():
        if not is_numeric_dtype(datas[key]):
            fill_dict[key] = datas[key].sort_values(ascending=False)[0]
    return datas.fillna(value=fill_dict)


def similar_filling_github(datas: pd.DataFrame, dd: pd.DataFrame, epsilon=0.1):
    index = []
    for key in datas.keys():
        if is_numeric_dtype(datas[key]):
            index.append(key)

    m1 = datas.loc[:, index].values
    m1 = pd.DataFrame(m1).fillna(method='bfill').values
    m2 = dd.loc[:, index].values
    m2 = pd.DataFrame(m2).fillna(method='bfill').values

    new_df = []
    for i in tqdm(range(len(datas)), total=len(datas)):
        c = datas.iloc[i]
        if c.isnull().any():
            max_score = -1
            idx = 0
            for j in range(max(0, i - 1000), min(i + 1000, len(dd))):
                cos = cosine_similarity(np.array([m1[i]]), np.array([m2[j]]))[0][0]
                if cos > max_score:
                    idx = j
                    max_score = cos
                if max_score >= 1 - epsilon:
                    break
            fill_dict = {'primary_language': dd.iloc[idx][5], 'languages_used': dd.iloc[idx][6],
                         'licence': dd.iloc[idx][9]}
            c = c.fillna(value=fill_dict)
        new_df.append(c)
    new_df = pd.DataFrame(new_df)
    new_df.columns = datas.columns
    return new_df
