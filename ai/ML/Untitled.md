### Definition of Cluster

---

**定义 1**  设 $T$ 为给定的正数，若集合 $G$ 中任意两个样本 $x_i, x_j$，有  
$$
d_{ij} \leq T
$$
​	       则称 $G$ 为一个类或簇。

**定义 2**  设 $T$ 为给定的正数，若对集合 $G$ 中任意一个样本 $x_i$，$G$ 中的另一个样本 $x_j$ 满足  
$$
\frac{1}{n_G - 1} \sum_{x_j \in G} d_{ij} \leq T
$$
​	       其中 $n_G$ 为 $G$ 中样本的个数，则称 $G$ 为一个类或簇。

**定义 3**  设 $T$ 和 $V$ 为给定的两个正数，如果集合 $G$ 中任意两个样本 $x_i, x_j$ 的距离 $d_{ij}$ 满足  
$$
\frac{1}{n_G (n_G - 1)} \sum_{x_i \in G} \sum_{x_j \in G} d_{ij} \leq T\\
d_{ij} \leq V
$$
​	        则称 $G$ 为一个类或簇。

---





