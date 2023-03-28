import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression



class VisualizationGithub:
    def __init__(self, datas, plt_xticks):
        self.font = {'size': 13}
        self.datas = datas
        self.plt_xticks = plt_xticks

        plt.figure(figsize=(16, 9))
        plt.subplots_adjust(wspace=0.7, hspace=0.5)
        plt.rcParams['font.family'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False

        self.stars_count(self.datas['stars_count'])
        self.forks_count(self.datas['forks_count'])
        self.watchers(self.datas['watchers'])
        self.pull_requests(self.datas['pull_requests'])
        self.commit_count(self.datas['commit_count'])
        self.created_at(self.datas['created_at'])
        self.primary_language(self.datas['primary_language'])
        self.languages_used(self.datas['languages_used'])
        self.licence(self.datas['licence'])

    def stars_count(self, stars: pd.Series):
        x_index = stars.keys().to_list()
        x = np.log10(np.array(x_index)).tolist()
        y = stars.values.tolist()
        a1 = plt.subplot(331)
        plt.title('stars count')
        a1.scatter(x, y)
        xt = self.plt_xticks['stars_count']
        xi = np.log10(np.array(xt)).tolist()
        _ = plt.xticks(xi, xt)
        a1.set_ylabel('count')
        a1.set_xlabel('stars number')

    def forks_count(self, forks: pd.Series):
        x_index = forks.keys().to_list()
        x = np.log10(np.array(x_index) + 1).tolist()
        y = forks.values.tolist()
        a1 = plt.subplot(332)
        plt.title('forks count')
        a1.scatter(x, y)
        xt = self.plt_xticks['forks_count']
        xi = np.log10(np.array(xt) + 1).tolist()
        _ = plt.xticks(xi, xt)
        a1.set_ylabel('count')
        a1.set_xlabel('forks number')

    def watchers(self, watchers: pd.Series):
        x_index = watchers.keys().to_list()
        x = np.log10(np.array(x_index) + 1).tolist()
        y = watchers.values.tolist()
        a1 = plt.subplot(333)
        plt.title('watchers')
        a1.scatter(x, y)
        xt = self.plt_xticks['watchers']
        xi = np.log10(np.array(xt) + 1).tolist()
        _ = plt.xticks(xi, xt)
        a1.set_ylabel('count')
        a1.set_xlabel('watchers number')

    def pull_requests(self, pull_requests: pd.Series):
        x_index = pull_requests.keys().to_list()
        x = np.log10(np.array(x_index) + 1).tolist()
        y = pull_requests.values.tolist()
        a1 = plt.subplot(334)
        plt.title('pull requests')
        a1.scatter(x, y)
        xt = self.plt_xticks['pull_requests']
        xi = np.log10(np.array(xt) + 1).tolist()
        _ = plt.xticks(xi, xt)
        a1.set_ylabel('count')
        a1.set_xlabel('pull requests number')

    def commit_count(self, commit_count: pd.Series):
        x_index = commit_count.keys().to_list()
        x = np.log10(np.array(x_index)).tolist()
        y = commit_count.values.tolist()
        a1 = plt.subplot(335)
        plt.title('commit count')
        a1.scatter(x, y)
        xt = self.plt_xticks['commit_count']
        xi = np.log10(np.array(xt)).tolist()
        _ = plt.xticks(xi, xt)
        a1.set_ylabel('count')
        a1.set_xlabel('commit count number')

    def created_at(self, created_at: pd.Series):
        created_at = created_at.sort_index()
        x = created_at.keys()
        y = created_at.values
        a1 = plt.subplot(336)
        plt.title('created at')
        a1.plot(x, y)
        xi = ['2009-01', '2010-01', '2011-01', '2012-01', '2013-01', '2014-01', '2015-01', '2016-01', '2017-01',
              '2018-01', '2019-01', '2020-01', '2021-01', '2022-01', '2023-01']
        _ = plt.xticks(xi, rotation=60)
        a1.set_ylabel('count')
        a1.set_xlabel('time')

    def primary_language(self, primary_language: pd.Series):
        primary_language = primary_language.sort_values(ascending=False)
        x = primary_language.keys().to_list()[:10]
        x.append('others')
        y = primary_language.values.tolist()
        yy = y[:10]
        yy.append(sum(y[10:]))
        a1 = plt.subplot(337)
        plt.pie(yy, labels=x)
        # plt.bar(range(len(yy)), yy)
        # plt.xticks(range(len(yy)), x, rotation=30)
        plt.title('primary language')
        # a1.set_ylabel('count')
        # a1.set_xlabel('language')

    def languages_used(self, languages_used: pd.Series):
        primary_language = languages_used.sort_values(ascending=False)
        x = primary_language.keys().to_list()[:10]
        x.append('others')
        y = primary_language.values.tolist()
        yy = y[:10]
        yy.append(sum(y[10:]))
        a1 = plt.subplot(338)
        plt.pie(yy, labels=x)
        # plt.bar(range(len(yy)), yy)
        # plt.xticks(range(len(yy)), x, rotation=30)
        plt.title('languages used')
        # a1.set_ylabel('count')
        # a1.set_xlabel('language')

    def licence(self, licence):
        licence = licence.sort_values(ascending=False)
        x = licence.keys().to_list()[:6]
        x.append('others')
        y = licence.values.tolist()
        yy = y[:6]
        yy.append(sum(y[6:]))
        a1 = plt.subplot(339)
        plt.bar(range(len(yy)), yy)
        plt.xticks(range(len(yy)), x, rotation=90)
        plt.title('licence')
        a1.set_ylabel('count')
        a1.set_xlabel('licence')


class BoxGithub:
    def __init__(self, datas: pd.DataFrame):
        self.datas = datas

        self.new_df = {}
        for key in datas.keys():
            if is_numeric_dtype(datas[key]):
                self.new_df[key] = np.log10(np.array(datas[key]) + 1)
        self.new_df = pd.DataFrame(self.new_df)
        self.draw_box(self.new_df)
        plt.show()

    def draw_box(self, data):
        data.plot.box(title="Box figure")


def draw_raw(total_value_counts):
    plt_xticks = {'stars_count': [2, 3, 5, 10, 50, 100, 500, 300000],
                  'forks_count': [0, 1, 2, 3, 5, 10, 50, 100, 500, 300000],
                  'watchers': [0, 1, 2, 3, 5, 10, 50, 100, 500, 8000],
                  'pull_requests': [0, 1, 2, 3, 5, 10, 50, 100, 200000],
                  'commit_count': [2, 3, 5, 10, 50, 100, 500, 8000, 300000]
                  }
    g = VisualizationGithub(total_value_counts, plt_xticks)
    plt.show()


def draw_without_outliers(total_value_counts):
    plt_xticks = {'stars_count': [2, 3, 5, 6, 7, 8, 9, 10, 20, 50, 75, 100],
                  'forks_count': [0, 1, 2, 3, 5, 10, 20, 30, 40, 50],
                  'watchers': [0, 1, 2, 3, 5, 7, 10],
                  'pull_requests': [0, 1, 2, 3, 5, 7, 10],
                  'commit_count': [2, 3, 5, 10, 50, 100, 300, 500]
                  }
    g = VisualizationGithub(total_value_counts, plt_xticks)
    plt.show()


class BoxStock:
    def __init__(self, datas: pd.DataFrame):
        self.datas = datas

        self.stock_return = {}
        self.px_volume = {}
        self.volatility = {}
        for key in datas.keys():
            if key in {'1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN'}:
                self.stock_return[key] = datas[key]
            elif key in {'PX_VOLUME'}:
                self.px_volume = datas[key]
            elif key in {'VOLATILITY_10D', 'VOLATILITY_30D'}:
                self.volatility[key] = datas[key]
        self.stock_return = pd.DataFrame(self.stock_return)
        self.px_volume = pd.DataFrame(self.px_volume)
        self.volatility = pd.DataFrame(self.volatility)

        self.draw_box(self.stock_return, 'stock_return')
        plt.show()

        self.draw_box(self.px_volume, 'px_volume')
        plt.show()

        self.draw_box(self.volatility, 'volatility')
        plt.show()

    def draw_box(self, data, title):
        data.plot.box(title=title)


def stock_sentiment_regression(x,y):
    lr = LinearRegression()
    lr.fit(x, y)
    z = np.linspace(min(x), max(x), 100)
    z_predict = lr.predict(z.reshape(-1, 1))
    plt.plot(z, z_predict, lw=5, c='g')
    plt.scatter(x, y, s=180, c='r')
    plt.show()

