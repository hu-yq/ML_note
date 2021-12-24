# 一、引入工具包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# 二、数据加载
data = pd.read_csv(r'./data/train.csv') ## 数据集地址： https://www.kaggle.com/c/rs6-attrition-predict
pd.set_option('display.max_columns', None)
# data.head() 



# 三、数据预处理
# 3.1 重复值处理
print("样本去重前样本数量：{}".format(data.shape[0]))
print("样本去重后样本数量：{}".format(data.drop_duplicates().shape[0]))

# 3.2 缺失值处理
missingDf = data.isnull().sum().sort_values(ascending = False).reset_index()
missingDf.columns = ['feature','missing_num']
missingDf['missing_percentage'] = missingDf['missing_num'] / data.shape[0]
# missingDf.head() 

# 3.3 异常值处理
# 筛选数值型特征 
numeric_columns = []
object_columns = []
for c in data.columns:
    if data[c].dtype == 'object':
        object_columns.append(c)
    else:
        numeric_columns.append(c) 
# 绘制箱型图查看异常值
fig = plt.figure(figsize=(20,30))
for i,col in enumerate(numeric_columns):
    ax = fig.add_subplot(9,3,i+1)
    sns.boxplot(data[col],orient='v',ax=ax)
    plt.xlabel(col)
plt.show()

# 四、特征选择
# 4.1 删除明显无关特征
print(data.describe()) 
data.drop(['user_id','EmployeeCount','EmployeeNumber','StandardHours','Over18'],axis=1,inplace=True)
# 4.2 查看数值型特征相关性
pearson_mat = data.corr(method='spearman')
plt.figure(figsize=(30,30))
ax = sns.heatmap(pearson_mat,square=True,annot=True,cmap='YlGnBu')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

# PerformanceRating:绩效评估
fig = plt.figure(figsize=(15,4)) # 建立图像
L1 = list(data['PerformanceRating'].unique())
for i,c in enumerate(L1):    
    ax = fig.add_subplot(1,3,i+1)
    p = data[data['PerformanceRating'] == c]['Attrition'].value_counts()
    ax.pie(p,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.2))
    ax.set_title(c)
plt.show() # 展示饼状图

data.drop(['JobLevel','TotalWorkingYears','YearsInCurrentRole','YearsWithCurrManager','PerformanceRating'],axis=1,inplace=True)

# 4.3 类别型特征探索性分析
#  
#  商务差旅频率与是否离职的关系 

# BusinessTravel:商务差旅频率
fig = plt.figure(figsize=(15,4)) # 建立图像
L1 = list(data['BusinessTravel'].unique())
for i,c in enumerate(L1):    
    ax = fig.add_subplot(1,3,i+1)
    p = data[data['BusinessTravel'] == c]['Attrition'].value_counts()
    ax.pie(p,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.2))
    ax.set_title(c)
plt.show() # 展示饼状图

# OverTime
fig = plt.figure(figsize=(15,4)) # 建立图像
L1 = list(data['OverTime'].unique())
for i,c in enumerate(L1):    
    ax = fig.add_subplot(1,2,i+1)
    p = data[data['OverTime'] == c]['Attrition'].value_counts()
    ax.pie(p,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.2))
    ax.set_title(c)
plt.show() # 展示饼状图

# JobSatisfaction:工作满意度
fig = plt.figure(figsize=(20,8)) # 建立图像
L1 = list(data['JobSatisfaction'].unique())
for i,c in enumerate(L1):    
    ax = fig.add_subplot(2,3,i+1)
    p = data[data['JobSatisfaction'] == c]['Attrition'].value_counts()
    ax.pie(p,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.2))
    ax.set_title(c)
plt.show() # 展示饼状图

# Gender
fig = plt.figure(figsize=(20,8)) # 建立图像
L1 = list(data['Gender'].unique())
for i,c in enumerate(L1):    
    ax = fig.add_subplot(2,3,i+1)
    p = data[data['Gender'] == c]['Attrition'].value_counts()
    ax.pie(p,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.2))
    ax.set_title(c)
plt.show() # 展示饼状图 

# Department 员工所在部门
fig = plt.figure(figsize=(15,4)) # 建立图像
L1 = list(data['Department'].unique())
for i,c in enumerate(L1):    
    ax = fig.add_subplot(1,3,i+1)
    p = data[data['Department'] == c]['Attrition'].value_counts()
    ax.pie(p,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.2))
    ax.set_title(c)
plt.show() # 展示饼状图

# EducationField 员工所学习的专业领域
fig = plt.figure(figsize=(20,8)) # 建立图像
L1 = list(data['EducationField'].unique())
for i,c in enumerate(L1):    
    ax = fig.add_subplot(2,3,i+1)
    p = data[data['EducationField'] == c]['Attrition'].value_counts()
    ax.pie(p,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.2))
    ax.set_title(c)
plt.show() # 展示饼状图

# JobRole
fig = plt.figure(figsize=(20,8)) # 建立图像
L1 = list(data['JobRole'].unique())
for i,c in enumerate(L1):    
    ax = fig.add_subplot(2,5,i+1)
    p = data[data['JobRole'] == c]['Attrition'].value_counts()
    ax.pie(p,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.2))
    ax.set_title(c)
plt.show() # 展示饼状图

# MaritalStatus
fig = plt.figure(figsize=(15,4)) # 建立图像
L1 = list(data['MaritalStatus'].unique())
for i,c in enumerate(L1):    
    ax = fig.add_subplot(1,3,i+1)
    p = data[data['MaritalStatus'] == c]['Attrition'].value_counts()
    ax.pie(p,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.2))
    ax.set_title(c)
plt.show() # 展示饼状图

# 五、特征工程
# 类别型特征转换 

# Attrition
data['Attrition'] = data['Attrition'].apply(lambda x:1 if x == "Yes" else 0)
# Gender
data['Gender'] = data['Gender'].apply(lambda x:1 if x == "Male" else 0)
# OverTime
data['OverTime'] = data['OverTime'].apply(lambda x:1 if x == "Yes" else 0)

for fea in  ['BusinessTravel', 'Department', 'EducationField','JobRole','MaritalStatus']:
    labels = data[fea].unique().tolist()
    data[fea] = data[fea].apply(lambda x:labels.index(x)) 

# 六、模型训练
# 6.1 切分特征和标签 

X = data.loc[:,data.columns != "Attrition"]
y = data['Attrition']

# 6.2 样本不均衡问题 
print(y.value_counts())
sm = SMOTE(random_state=20)
X, y = sm.fit_sample(X,y) 

# 6.3 切分训练集和测试集
X = pd.DataFrame(X)
y = pd.DataFrame(y)
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, y ,test_size = 0.3,random_state=0)

# 6.4 模型训练
model = DecisionTreeClassifier(random_state=0)
model.fit(Xtrain,Ytrain)
pred = model.predict(Xtest)

# 6.5 模型评估
y_pred_prob = model.predict_proba(Xtest)[:, 1]
auc_score = roc_auc_score(Ytest,y_pred_prob)#验证集上的auc值
auc_score 

# 6.6 使用网格搜索寻找最优参数对模型进行优化
gini_thresholds = np.linspace(0,0.5,20)
parameters = {
    'splitter':('best','random')
    ,'criterion':("gini","entropy")
    ,"max_depth":[*range(1,10)]
    ,'min_samples_leaf':[*range(1,50,5)]
    ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
}
clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, parameters, cv=10,scoring='roc_auc')
GS.fit(Xtrain,Ytrain)

# 最优评分：
print(GS.best_score_)
# 最优参数：
print(GS.best_params_) 

# 6.7 使用最优参数建立模型
model = DecisionTreeClassifier(random_state=0,criterion='gini',max_depth=9,min_impurity_decrease=0,min_samples_leaf=11,splitter='best')
model.fit(Xtrain,Ytrain)
model.score(Xtest,Ytest)
pred = model.predict(Xtest)
y_pred_prob = model.predict_proba(Xtest)[:, 1]
auc_score = roc_auc_score(Ytest,y_pred_prob)#验证集上的auc值
print(auc_score) 

# 七、绘制树形图
import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(clf,out_file=None,feature_names=feature_name,filled=True,rounded=True)
graph = graphviz.Source(dot_data)
