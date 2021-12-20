import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 


def test_PolynomialFeatures():
# PolynomialFeatures 这个类有 3 个参数：
    # degree：控制多项式的次数；
    # interaction_only：默认为 False，如果指定为 True，那么就不会有特征⾃⼰和⾃⼰结合的项，组合的特征中没有a 和 b ；
    # include_bias：默认为 True 。如果为 True 的话，那么结果中就会有 0 次幂项，即全为 1 这⼀列。

    X = np.arange(6).reshape(3, 2)
    # 设置多项式阶数为2,其他值默认
    # degree 多项式阶数
    poly = PolynomialFeatures(degree=2)
    res = poly.fit_transform(X)
    print(X,'\n', res)

    return None

def PolynomialFeatures_train():
    # 构造数据,数据可视化展示
    data = np.array([[ -2.95507616, 10.94533252],
                    [ -0.44226119, 2.96705822],
                    [ -2.13294087, 6.57336839],
                    [ 1.84990823, 5.44244467],
                    [ 0.35139795, 2.83533936],
                    [ -1.77443098, 5.6800407 ],
                    [ -1.8657203 , 6.34470814],
                    [ 1.61526823, 4.77833358],
                    [ -2.38043687, 8.51887713],
                    [ -1.40513866, 4.18262786]])

    X = data[:, 0].reshape(-1, 1) # 将array转换成矩阵
    y = data[:, 1].reshape(-1, 1)

    # 查看数据分布
    plt.plot(X, y, "b.")
    plt.xlabel('X')
    plt.ylabel('y')

    # 模型训练
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.intercept_, lin_reg.coef_) 

    X_plot = np.linspace(-3, 3, 1000).reshape(-1, 1)
    # 可以使⽤两种⽅法⽤于模型预测
    # y_plot = np.dot(X_plot, lin_reg.coef_.T) + lin_reg.intercept_
    y_plot = lin_reg.predict(X_plot)
    plt.plot(X_plot, y_plot,"red")
    plt.plot(X, y, 'b.')
    plt.xlabel('X')
    plt.ylabel('y')

    # 使⽤mse衡量其误差值:
    y_pre = lin_reg.predict(X)
    LinearRegression_mse = mean_squared_error(y, y_pre)
    print('mean_squared_error:',LinearRegression_mse) # 3.3363076332788495 

    # 使用多项式回归
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    print(X_poly)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    # [ 2.60996757] [[-0.12759678 0.9144504 ]]
    X_plot = np.linspace(-3, 3, 1000).reshape(-1, 1)
    X_plot_poly = poly_features.fit_transform(X_plot)
    y_plot = lin_reg.predict(X_plot_poly)
    plt.plot(X_plot, y_plot, 'red')
    plt.plot(X, y, 'b.')
    plt.show()

    # 使⽤mse衡量其误差值:
    y_pre = lin_reg.predict(X_poly)
    Poly_mse = mean_squared_error(y, y_pre)
    print('Poly_mean_squared_error:', Poly_mse)
    # 0.07128562789085331

    return None

if __name__ == '__main__':
    test_PolynomialFeatures()
    PolynomialFeatures_train()

