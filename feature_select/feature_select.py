import pandas as pd 
import numpy as np 
from scipy.stats import pearsonr 
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold 
from sklearn.decomposition import PCA


def variance_demo():
    """
    删除低⽅差特征——特征选择
    :return: None
    """
    data = pd.read_csv("factor_returns.csv")
    print(data)
    # 1、实例化⼀个转换器类
    transfer = VarianceThreshold(threshold=1)
    # 2、调⽤fit_transform
    data = transfer.fit_transform(data.iloc[:, 1:10])
    print("删除低⽅差特征的结果：\n", data)
    print("形状：\n", data.shape)
    return None

def xgxs_demo():
    """
    ⽪尔逊相关系数、斯⽪尔曼相关系数
    :return: None
    """
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]
    print('⽪尔逊相关系数: ', pearsonr(x1, x2))
    print('斯⽪尔曼相关系数: ', spearmanr(x1, x2))

    return None 

def pca_demo():
    """
    对数据进⾏PCA降维
    :return: None
    """
    data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]
    # 1、实例化PCA, ⼩数——保留多少信息
    transfer = PCA(n_components=0.9)
    # 2、调⽤fit_transform
    data1 = transfer.fit_transform(data)
    print("保留90%的信息，降维结果为：\n", data1)
    # 1、实例化PCA, 整数——指定降维到的维数
    transfer2 = PCA(n_components=3)
    # 2、调⽤fit_transform
    data2 = transfer2.fit_transform(data)
    print("降维到3维的结果：\n", data2)

    return None 

if __name__ == '__main__':
    # variance_demo()
    xgxs_demo() 
    pca_demo()