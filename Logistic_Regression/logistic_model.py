import ssl

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ssl._create_default_https_context = ssl._create_unverified_context 

# 1.获取数据
names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
        'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
        'Normal Nucleoli', 'Mitoses', 'Class']

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                  names=names)

# 2.基本数据处理

# 2.1 缺失值处理
data = data.replace(to_replace="?", value=np.nan)
data = data.dropna()

# 2.2 确定特征值,目标值
x = data.iloc[:, 1:-1]
y = data["Class"]

# 2.3 分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.2)

# 3.特征工程(标准化)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 4.机器学习(逻辑回归)
'''
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l2', C = 1.0)
    solver可选参数:{'liblinear', 'sag', 'saga','newton-cg', 'lbfgs'}，
        默认: 'liblinear'；⽤于优化问题的算法。
        对于⼩数据集来说，“liblinear”是个不错的选择，⽽“sag”和'saga'对于⼤型数据集会更快。
        对于多类问题，只有'newton-cg'， 'sag'， 'saga'和'lbfgs'可以处理多项损失;“liblinear”仅限于“one-versus-rest”分类。
    penalty: 正则化的种类
    C: 正则化⼒度
'''
estimator = LogisticRegression()
estimator.fit(x_train, y_train)

# 5.模型评估
# 5.1 准确率
ret = estimator.score(x_test, y_test)
print("准确率为:\n", ret)

# 5.2 预测值
y_pre = estimator.predict(x_test)
print("预测值为:\n", y_pre)

# 5.3 精确率\召回率指标评价
'''
sklearn.metrics.classification_report(y_true, y_pred, labels=[], target_names=None )
    y_true: 真实⽬标值
    y_pred: 估计器预测⽬标值
    labels:指定类别对应的数字
    target_names: ⽬标类别名称
    return: 每个类别精确率与召回率
'''
ret = classification_report(y_test, y_pre, labels=(2, 4), target_names=("良性", "恶性"))
print(ret)

# 5.4 auc指标计算
'''
sklearn.metrics.roc_auc_score(y_true, y_score)
    计算ROC曲线⾯积，即AUC值
    y_true: 每个样本的真实类别，必须为0(反例),1(正例)标记
    y_score: 预测得分，可以是正类的估计概率、置信值或者分类器⽅法的返回值
'''
y_test = np.where(y_test>3, 1, 0)
auc = roc_auc_score(y_test, y_pre)
print('roc_auc_score', auc)
