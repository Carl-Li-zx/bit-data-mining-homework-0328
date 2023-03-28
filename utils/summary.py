import pandas as pd
from pandas.api.types import is_numeric_dtype
import time
from tqdm import tqdm
import numpy as np
import math

PATH = {
    'github': "dataset/repository_data.csv",
    'stock': "dataset/full_dataset-release.csv"
}


def read_dataset(path):
    return pd.read_csv(path)


def get_counts_github(datas):
    total_value_counts = {}
    for key in datas.keys():
        if key == 'name':
            continue
        elif key == 'languages_used':
            dict_language = {'NAN': 0}
            for value in datas[key]:
                if type(value) is str:
                    for l in eval(value):
                        if l in dict_language.keys():
                            dict_language[l] += 1
                        else:
                            dict_language[l] = 1
                else:
                    dict_language['NAN'] += 1
            total_value_counts[key] = pd.Series(dict_language)
        elif key == 'primary_language' or key == 'licence':
            dict_language = {'NAN': 0}
            for value in datas[key]:
                if type(value) is str:
                    if value in dict_language.keys():
                        dict_language[value] += 1
                    else:
                        dict_language[value] = 1
                else:
                    dict_language['NAN'] += 1
            total_value_counts[key] = pd.Series(dict_language)
        elif key == 'created_at':
            time_series = {}
            for value in datas[key]:
                t = '-'.join(value.split("T")[0].split("-")[:2])
                if t in time_series.keys():
                    time_series[t] += 1
                else:
                    time_series[t] = 1
            total_value_counts[key] = pd.Series(time_series)
        else:
            total_value_counts[key] = datas[key].value_counts()
    return total_value_counts


def print_counts(total_value_counts):
    for key in total_value_counts.keys():
        v = total_value_counts[key].sort_values(ascending=False)
        output_f = []
        x_index = v.keys().to_list()[:5]
        y = v.values.tolist()[:5]
        for xx, yy in zip(x_index, y):
            output_f.append(f"({xx},{yy})")
        print(f"({key}, counts):", ', '.join(output_f), "...")


def print_nan(total_value_counts):
    lu = total_value_counts['languages_used']
    print(f"languages_used, {lu['NAN']}")
    pl = total_value_counts['primary_language']
    print(f"primary_language, {pl['NAN']}")
    l = total_value_counts['licence']
    print(f"licence, {l['NAN']}")


def summary(datas):
    limit = {}
    for key in datas.keys():
        if is_numeric_dtype(datas[key]):
            minimum, q1, median, q3, maximum, lower_limit, upper_limit = five_number(datas[key].values.tolist())
            print(f"{key}: minimum={minimum}, q1={q1}, median={median}, q3={q3}, maximum={maximum}")
            limit[key] = [lower_limit, upper_limit]
    return limit


def five_number(nums):
    minimum = min(nums)
    maximum = max(nums)
    q1 = np.nanpercentile(nums, 25)
    median = np.nanpercentile(nums, 50)
    q3 = np.nanpercentile(nums, 75)

    IQR = q3 - q1
    lower_limit = q1 - 1.5 * IQR
    upper_limit = q3 + 1.5 * IQR

    return minimum, q1, median, q3, maximum, lower_limit, upper_limit


def drop_outliers(datas: pd.DataFrame, limit):
    map_index = datas.columns.to_list()
    new_df = []

    def drop(item):
        for k in limit.keys():
            i = map_index.index(k)
            if limit[k][1] <= item[i] or item[i] <= limit[k][0]:
                return None
        return item

    for item in datas.values:
        out = drop(item)
        if out is None:
            continue
        new_df.append(out)
    new_df = pd.DataFrame(new_df)
    new_df.columns = datas.columns
    return new_df


def preprocess_stock(datas):
    out = []
    continue_idx = -1
    for index in range(len(datas)):
        if index <= continue_idx:
            continue
        s1 = datas.iloc[index]
        if math.isnan(s1['LAST_PRICE']):
            a1 = s1.values.tolist()[0:2]
            i = 1
            while True:
                tts2 = datas.iloc[index + i]
                if not math.isnan(tts2['LAST_PRICE']):
                    break
                i += 1
            ta2 = datas.iloc[index + i].values.tolist()
            if ta2[0] != ta2[0]:
                a2 = ta2[1:13]
            else:
                a2 = ta2[0:12]
            out.append(a1 + a2)
            continue_idx = index + i
        else:
            out.append(s1.values.tolist())
    out = pd.DataFrame(out)
    out.columns = datas.columns
    return out


def get_counts_stock(datas: pd.DataFrame):
    output = {
        'STOCK': datas['STOCK'].value_counts(),
        'LSTM_POLARITY': datas['LSTM_POLARITY'].value_counts(),
    }
    for key in output:
        v = output[key].sort_values(ascending=False)
        output_f = []
        x_index = v.keys().to_list()
        y = v.values.tolist()
        for xx, yy in zip(x_index, y):
            output_f.append(f"({xx},{yy})")
        print(f"({key}, counts):", ', '.join(output_f), "...")
    return output
