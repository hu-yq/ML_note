import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score


# 创建数据
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=9) 

plt.scatter(X[:, 0], X[:, 1], marker="o")
plt.show()


# kmeans训练,且可视化 聚类=2
y_pre = KMeans(n_clusters=2, random_state=9).fit_predict(X)

# 可视化展示
plt.scatter(X[:, 0], X[:, 1], c=y_pre)
plt.show()

# 用ch_scole查看最后效果
print(calinski_harabaz_score(X, y_pre)) 


# kmeans训练,且可视化 聚类=3
y_pre = KMeans(n_clusters=3, random_state=9).fit_predict(X)

# 可视化展示
plt.scatter(X[:, 0], X[:, 1], c=y_pre)
plt.show()

# 用ch_scole查看最后效果
print(calinski_harabaz_score(X, y_pre)) 

# kmeans训练,且可视化 聚类=4
y_pre = KMeans(n_clusters=4, random_state=9).fit_predict(X)

# 可视化展示
plt.scatter(X[:, 0], X[:, 1], c=y_pre)
plt.show()

# 用ch_scole查看最后效果
print(calinski_harabaz_score(X, y_pre))

