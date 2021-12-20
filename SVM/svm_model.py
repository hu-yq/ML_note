import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

"""
sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3,coef0=0.0,random_state=None)

    C: 惩罚系数，⽤来控制损失函数的惩罚系数，类似于线性回归中的正则化系数。
        1、C越⼤，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增⼤，趋向于对训练集全分对的情况，
        这样会出现训练集测试时准确率很⾼，但泛化能⼒弱，容易导致过拟合。
        2、C值⼩，对误分类的惩罚减⼩，容错能⼒增强，泛化能⼒较强，但也可能⽋拟合。
    kernel: 算法中采⽤的核函数类型，核函数是⽤来将⾮线性问题转化为线性问题的⼀种⽅法。
        参数选择有RBF, Linear, Poly, Sigmoid或者⾃定义⼀个核函数。
            1、默认的是"RBF"，即径向基核，也就是⾼斯核函数；
            2、⽽Linear指的是线性核函数，
            3、Poly指的是多项式核，
            4、Sigmoid指的是双曲正切函数tanh核；
    degree:
        当指定kernel为'poly'时，表示选择的多项式的最⾼次数，默认为三次多项式；
"""
# 获取数据 
train = pd.read_csv(r"D:\jupyter-notebook\Machine_Learning_huyq\SVM\mnist\mnist_csv\mnist_train.csv") 
print("train shape: ", train.shape)

# 确定目标值和特征值 
train_image = train.iloc[:, 1:]
train_label = train.iloc[:, 0]

# 查看图像 

def to_plot(n):
    num = train_image.iloc[n,].values.reshape(28, 28)
    
    plt.imshow(num)
    plt.axis("off")
    plt.show()

to_plot(n=40)

# 数据归一化处理
# 对数据特征值归一化处理
train_image = train_image.values / 255 
train_label = train_label.values 

# 数据集分割 
x_train, x_val, y_train, y_val = train_test_split(train_image, train_label, train_size = 0.8, random_state=0)

# 特征降维和模型训练 
import time
from sklearn.decomposition import PCA

# 多次使用pca,确定最后的最优模型

def n_components_analysis(n, x_train, y_train, x_val, y_val):
    # 记录开始时间
    start = time.time()
    
    # pca降维实现
    pca = PCA(n_components=n)
    print("特征降维,传递的参数为:{}".format(n))
    pca.fit(x_train)
    
    # 在训练集和测试集进行降维
    x_train_pca = pca.transform(x_train)
    x_val_pca = pca.transform(x_val)
    
    # 利用svc进行训练
    print("开始使用svc进行训练")
    ss = SVC()
    ss.fit(x_train_pca, y_train)
    
    # 获取accuracy结果
    accuracy = ss.score(x_val_pca, y_val)
    
    # 记录结束时间
    end = time.time()
    print("准确率是:{}, 消耗时间是:{}s".format(accuracy, int(end-start)))
    
    return accuracy 

# 传递多个n_components,寻找合理的n_components:
n_s = np.linspace(0.70, 0.85, num=5)
accuracy = []

for n in n_s:
    tmp = n_components_analysis(n, x_train, y_train, x_val, y_val)
    accuracy.append(tmp)

# 准确率可视化展示
plt.plot(n_s, np.array(accuracy), "r")
plt.show()

# 确定最优模型
pca = PCA(n_components=0.80)

pca.fit(x_train)
print("pca.n_components_",pca.n_components_)

x_train_pca = pca.transform(x_train)
x_val_pca = pca.transform(x_val)
print(x_train_pca.shape, x_val_pca.shape)

# 训练比较优的模型,计算accuracy

ss1 = SVC()

ss1.fit(x_train_pca, y_train)

ss1.score(x_val_pca, y_val)
