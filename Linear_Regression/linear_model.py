import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression ## 正规方程
from sklearn.linear_model import SGDRegressor ## 梯度下降
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error


def linear_model():
    """
    线性回归: 正规方程
    return: None
    """

    # 获取数据
    data = load_boston()

    # 划分训练、测试集
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=2)

    # 特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 模型训练
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)

    # 均⽅误差
    error = mean_squared_error(y_test, y_predict)
    print("误差为:\n", error)

    return None


def linear_model2():
    """
    线性回归: SGD
    return: None
    """

    # 获取数据
    data = load_boston()

    # 划分训练、测试集
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=3)

    # 特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 模型训练
    estimator = SGDRegressor(max_iter=1000)
    estimator.fit(x_train, y_train)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)

    # 均⽅误差
    error = mean_squared_error(y_test, y_predict)
    print("误差为:\n", error)

    return None

def linear_model3():
    """
    线性回归: Ridge
    return: None
    """

    # 获取数据
    data = load_boston()

    # 划分训练、测试集
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=3)

    # 特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 模型训练
    estimator = Ridge(alpha=1)
    estimator.fit(x_train, y_train)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)

    # 均⽅误差
    error = mean_squared_error(y_test, y_predict)
    print("误差为:\n", error)

    return None




if __name__ == '__main__':
    linear_model()
    linear_model2()
    linear_model3()