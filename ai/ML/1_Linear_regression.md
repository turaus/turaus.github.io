## 1. 线性回归
线性回归是机器学习最基础的内容，它的很多算法和思想，在后续很多的高级算法中都有体现。
- 阶段概述：
    - 本阶段讲解，**多元线性回归**，**梯度下降法**，**归一化**，**正则化**，**Lasso回归**，**Ridge回归**，**多项式回归**。
- 达成目标：
    - 通过本阶段学习，从推导出多元线性回归算法的损失函数，到实现开发和应用算法，再到对算法从数据预处理上，以及损失函数上的优化都将整体彻底掌握。对后面学习更多算法，甚至深度学习都将起到举一反三的效果。

### 1.1 线性回归基础

#### 1.1.1 线性回归概述
**多元线性回归**公式：
$$
\hat{y} = \beta_0 + \beta_1 X
$$
- parameter:
  - $\hat{y}$ ：预测值(Predicted value)
  - $\beta_0$ ：偏置项(Intercept)
  - $\beta_1$ ：权重(Slope)
  - $X$ ：特征变量(Predicter)

前导：

1. 回归一词的由来：

    **回归**简单来说就是“回归平均值”(regression to the mean)
    但是这里的mean并不是把历史数据直接当成未来的预测值，而是会把**期望值**当作预测值

2. **中心极限定理（central limit theorem）**

    是概率论中讨论随机变量序列部分和分布渐近于正态分布的一类定理。这组定理是数理统计学和误差分析的理论基础，指出了**大量随机变量**累积分布函数逐点**收敛到正态分布**的积累分布函数的条件。

    在自然界与生产中，一些现象受到许多相互独立的随机因素的影响，如果每个因素所产生的影响都很微小时，总的影响可以看作是服从正态分布的。中心极限定理就是从数学上证明了这一现象。

#### 1.1.2 误差与损失函数

**误差**可以表示为：
$$
\mathcal{E}_i = \mid \mathtt{y}_i - \hat{\mathtt{y}}_i \mid
$$
由中心极限定理可知，对于足够大样本空间，可以认为误差服从正态分布，即为 **$\mathcal{E} \sim N(\mu, \sigma^2)$**
所以有：
$$
f(\epsilon \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(\epsilon - \mu)^2}{2\sigma^2}}
$$
通常认为这个假设的期望$\mu$为0，而不能限制方差$\sigma^2$的大小，所以有：
$$
f(\epsilon \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{\epsilon ^2}{2\sigma^2}}
$$

下面根据**最大似然估计**的方法找最优解$\theta$
似然函数为：
$$
L_{\theta}(\epsilon_1, \epsilon_2,\dots,\epsilon_{m})
= f(\epsilon_1, \dots,\epsilon_{m} \mid \mu, \sigma^2)
= \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^{2}}} \exp\left(-\frac{\epsilon_i^{2}}{2\sigma^{2}}\right)
$$
代入误差的表达式，有：
$$
L_{\theta}(\epsilon_1, \epsilon_2,\dots,\epsilon_{m}) 
= \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^{2}}} \exp\left(-\frac{(y_i - \theta^{T} x_i)^{2}}{2\sigma^{2}}\right) 
$$

两边同时取对数 :
$$
\begin{split}
\mathcal{l}(\theta)
&= \ln L(\theta)\\
&= \ln \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^{2}}} \exp\left(-\frac{(y_i - \theta^{T} x_i)^{2}}{2\sigma^{2}}\right)\\
&= \sum_{i=1}^{m} \ln \frac{1}{\sqrt{2\pi\sigma^{2}}} \exp\left(-\frac{(y_i - \theta^{T} x_i)^{2}}{2\sigma^{2}}\right)\\
&= m \ln \frac{1}{\sqrt{2\pi\sigma^{2}}} - 
    \frac{1}{\sigma^2} \cdot \frac{1}{2} \sum_{i=1}^{m}(y_i - \theta^{T} x_i)^{2}
\end{split}
$$

最优化(寻找使似然函数最大的参数$\theta$)：
$$
\begin{split}
\theta^{*}
&= \underset{\theta}{\arg \max}\  L_{\theta}(\epsilon_1,\dots,\epsilon_{m})\\[5pt]
&= \underset{\theta}{\arg \max}\  \ln L_{\theta}(\epsilon_1,\dots,\epsilon_{m})\\
&= \underset{\theta}{\arg \max}\  \ln \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^{2}}} \exp\left(-\frac{(y_i - \theta^{T} x_i)^{2}}{2\sigma^{2}}\right) \\
&= \underset{\theta}{\arg \max}\  \left(m \ln \frac{1}{\sqrt{2\pi\sigma^{2}}} - 
    \frac{1}{\sigma^2} \cdot \frac{1}{2} \sum_{i=1}^{m}(y_i - \theta^{T} x_i)^{2} \right)
\end{split}
$$

所以有**损失函数**：
$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{m}(h_\theta(x_i) - y_i)^{2}
$$
为消除样本容量的影响，将系数更正为$\frac{1}{m}$，那么有：
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x_i) - y_i)^{2}
$$
这可以看作是由均方误差（Mean Squared Error）来决定的损失函数，所以该损失函数名为**MSE损失函数**，是衡量一个模型性能好坏的一个重要指标。

---

#### 1.1.3 解析解的推导
我们现在有了损失函数形式，也明确了目标就是要最小化损失函数，那么接下来问题就是 $\theta$ 什么时候可以使得损失函数最小了。
$$
\begin{split}
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i)^{2} &= \frac{1}{m} (X \theta - y)^{T} (X \theta - y)\\
&= \frac{1}{m} (\theta^{T} X^{T} X \theta - \theta^{T} X^{T} y - y^{T} X \theta + y^{T}y)
\end{split}
$$

对损失函数求导
$$
\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} X^{T} (X \theta - y)
$$
令它等于0得：
$$
\theta = (X^{T}X)^{-1} X^{T}y
$$
> **注意：**
>
> - 上述解析解涉及矩阵逆的计算，解析解只在**满秩**或**半正定**的时候才成立。
> - 但是大多数实际问题都不满足这个要求，这会得到多个解，这就需要我们对其引入**正则化**。
> - 局部最优解与全局最优解：当损失函数是凸函数的时候，局部最优解就是全局最优解。
>     - 凸函数的判定，最为典型的方法是看黑塞矩阵（Hessian Matrix）是否半正定
>     - 黑塞矩阵是由目标函数在点 X 处的二阶偏导数组成的对称矩阵
>     - 由于线性回归的损失函数实质上就是$A^{T}A$,所以线性回归的损失函数一定是半正定的

---

### 1.2 进阶
上面已经推到完线性回归基础的解析解，但是在实际应用中很难满足使用条件，下面讲的梯度下降法、归一化、正则化、Lasso回归等，是帮助应用的一些手段。
#### 1.2.1 梯度下降法
- 解决问题：梯度下降法（Gradient Descent）是无约束最优化问题的求解算法。
- 使用的情景：
  上面所讲的多元线性回归推导过程中，最让我们的计算得以简化的一个条件是——线性回归基本模型的损失函数是个**凸函数**，可以快速地通过求一个极值的方法确定最优解。但对于更常见的**非凸函数**而言，我们会得到多个极值点，是计算变得复杂。而梯度下降法就是为解决这个问题而出现的，它利用迭代的手段，去逼近我们想要的最优解。
- 思想：类似于**数字炸弹**的游戏，采用"**猜**"的方式去逼近最优的答案。
  - 判断迭代的方向是否正确（及好像数字炸弹游戏中，主持人的作用）：
    1. 损失函数Loss是否在变小
    2. 梯度的绝对值是否在减小
- 梯度下降法**基本公式**： $ W_{j}^{t+1} = W_{j}^{t} - \eta \cdot g_{j} $
  - $g_{j}$：梯度（gradient）$g_{j} = \dfrac{\partial loss}{\partial W_{j}}$
  - $\eta$：学习率（learning rate），需要设置合适的参数，确保收敛到全局最优解。
  - $W_{j}$：为向量 $\theta$中的某一个，$j = 1,2,\dots, m$
- 梯度下降法流程
  1. 瞎蒙，Random随机θ，随机一组数值$W_0,\cdots,W_n$（Random Initial Value）<br>
  2. 求梯度,$g_{j} = \dfrac{\partial loss}{\partial W_{j}}$<br>
  3. if $g<0$, theta 变大，if $g>0$, theta 变小<br>
  4. 判断是否收敛（convergence），如果收敛跳出迭代，如果没有达到收敛，回第2步继续
---
**梯度下降法在线性回归的应用**
<div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">

**损失函数的导函数**

各个维度的$w_{j}$求导得：

$$
\begin{aligned}
g_{j} 
&= \frac{\partial loss}{\partial W_{j}} \\[5pt]
&= \frac{\partial}{\partial W_{j}} \frac{1}{2}(h_{w}(x) - y)^{2}\\[8pt]
&= (h_{w}(x) -  y) x_{j}
\end{aligned}
$$

将各个维度的$w_{j}$统一为：

$$
\theta_{j}^{t+1} = \theta_{j}^{t} - \eta \cdot (h_{\theta}(x)-y) \cdot x_{j}
$$

</div>

<div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">

**三种梯度下降法**

- 全量梯度下降（Batch Gradient Descent）：
    $$
    \theta_{j}^{t+1} = \theta_{j}^{t} - \eta \cdot \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)}) \cdot x_{j}^{(i)}
    $$

- 随机梯度下降（Stochastic Gradient Descent）：
    $$
    \theta_{j}^{t+1} = \theta_{j}^{t} - \eta \cdot (h_{\theta}(x^{(i)})-y^{(i)}) \cdot x_{j}^{(i)}
    $$

- 小批量梯度下降（Mini-Batch Gradient Descent）：
    $$
    \theta_{j}^{t+1} = \theta_{j}^{t} - \eta \cdot \sum_{i=1}^{batch \_ size}(h_{\theta}(x^{(i)})-y^{(i)}) \cdot x_{j}^{(i)}
    $$

</div>

<div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">

**轮次与批次**

- 轮次（epoch）：轮次顾名思义是把我们已有的训练集数据学习多少轮

- 批次（batch）：批次这里指的的我们已有的训练集数据比较多的时候，一轮要学习太多数据，那就把一轮次要学习的数据分成多个批次，一批一批数据的学习

</div>

---

#### 1.2.2 归一化
- 目的：归一化（Normalization）的一个目的是使得最终梯度下降的时候可以不同维度θ参数可以在接近的调整幅度上，使不同维度上的影响不会因数量产生偏差。
- 本质：无量纲化。
- 最大值最小值归一化（min-max scaling）：
  $$
  x_{i.j}^{*} = \frac{x_{i,j} - x_{j}^{min}}{x_{j}^{max}-x_{j}^{min}}
  $$
  **优点**是一定可以把数值归一到0到1之间
  **缺点**是如果有一个*离群值*,会使得一个数值为1，其它数值都几乎为0，所以受离群值的影响比较大。
  `from sklearn.preprocessing import MinMaxScaler`
- 标准归一化
  $$
  X_{new} = \frac{X_{i}-Mean}{Deviation} 
  $$
  相对于最大值最小值归一化来说，因为标准归一化是除以的是标准差，而标准差的计算会考虑到所有样本数据，所以受到离群值的影响会小一些，这就是除以方差的好处！但是如果是使用标准归一化不一定会把数据缩放到0到1之间了。
  `from sklearn.preprocessing import StandardScaler`

- 强调
  我们在做特征工程的时候，很多时候如果对训练集的数据进行了预处理，比如这里讲的归一化，那么未来对测试集的时候，和模型上线来新的数据的时候，都要进行相同的数据预处理流程，而且所使用的均值和方差是来自当时训练集的均值和方差！
  因为我们人工智能要干的事情就是从训练集数据中找规律，然后利用找到的规律去预测未来。这也就是说假设训练集和测试集以及未来新来的数据是属于同分布的！从代码上面来说如何去使用训练集的均值和方差呢？如果是上面代码的话，就需要把scaler对象持久化，回头模型上线的时候再加载进来去对新来的数据使用。
  
---

#### 1.2.3 正则化
- 前导知识
  - **过拟合**（over fit）：拟合过度，训练集的准确率升高的同时，测试集的准确率反而降低。学的过度了，做过的卷子都能再次答对，考试碰到新的没见过的题就考不好。
  
  - **欠拟合**（under fit）：还没有拟合到位，训练集和测试集的准确率都还没有到达最高。学的还不到位。
  
  - **鲁棒性**（Robust）：模型的泛化能力。
    
    > **说明**：
    >
    > 以下面两个式子描述同一条直线：
    > $0.5x_{1} + 0.4x_{2} + 0.3 = 0$
    > $5x_{1} + 4x_{2} + 3 = 0$
    > 第一个更好，因为下面的公式的系数是上面的十倍，**当$w$越小公式的容错的能力就越好**。
    >
    > 因为把测试集带入公式中如果测试集原来是100在带入的时候发生了一些偏差，比如说变成了101，第二个模型结果就会比第一个模型结果的偏差大多。
    
    所以，根据公式$\hat{y} = w^{T} x$,当变量出现一点错误时，会被$w$放大而影响到$\hat{y}$。但是$w$也不能太小，太小时没办法做分类。
  
- 目的：正则化（Regularization）就是防止过拟合，增加模型的鲁棒性，让模型的泛化能力和推广能力更加的强大

- 实质：
  正则化（鲁棒性调优）的本质就是牺牲模型在训练集上的正确率来**提高推广能力**，W在数值上越小越好，这样能抵抗数值的扰动。同时为了保证模型的正确率W又不能极小。
  故而人们将原来的损失函数加上一个惩罚项，这里面损失函数就是原来固有的损失函数，比如回归的话通常是MSE，分类的话通常是CrossEntropy交叉熵，然后在加上一部分惩罚项来使得计算出来的模型W相对小一些来带来泛化能力。

  惩罚项有**L1正则项**或者**L2正则项**
  $$
  L_{1} = \sum_{i=0}^{m} \mid w_{i} \mid \\
  L_{2} = \sum_{i=0}^{m} w_{i}^{2}
  $$
  其实L1和L2正则的公式数学里面的意义就是范数，代表空间中向量到原点的距离，
  *L-P范数*
  $$
  L_{P} = \|x\|_{P} = \left( \sum_{i=1}^{n} |x_i|^{P} \right)^{1/P}
  $$
  详情见：
  [范数动画详解](https://www.bilibili.com/video/BV1GM4y1c78K/?spm_id_from=333.1387.homepage.video_card.click&vd_source=62b6bb4c48ac16b4c1e4b27a2fce3817)
  [范数与正则化](http://www.cnblogs.com/MengYan-LongYou/p/4050862.html)

当我们把多元线性回归损失函数加上L2正则的时候，就诞生了Ridge岭回归。
当我们把多元线性回归损失函数加上L1正则的时候，就孕育出来了Lasso回归。
<img src="../source/imgs/1.4.3_Ridge_Lasso.png" alt="Ridge vs. Lasso" style="display: block; margin: 0 auto; max-width: 100%;" width=700>
- 其实L1和L2正则项惩罚项可以加到任何算法的损失函数上面去提高计算出来模型的泛化能力的。

- **L1 稀疏L2平滑**（从梯度的角度出发）：

    通常我们会说L1正则会使得计算出来的模型有的$W$趋近于0，有的$W$相对较大，相当于将$W$参数向两个极端发展；

    而L2会使得$W$参数整体变小。

- total loss，根据我们对模型泛化能力的需求，常在惩罚项乘上一个权重系数$\lambda$。$\lambda$越大，表示越看重模型的泛化能力，通常可以设置为0.4。
  $$
  J_{total\_ loss} = \|Xw-y\|_{2}^{2} + \lambda \|w\|_{P}
  $$
  所以$w$为：
  $$
  w^{*} = \underset{\theta}{\arg \min}\  \|Xw-y\|_{2}^{2} + \lambda \|w\|_{P}
  $$
---

#### 1.2.4 多元线性回归的衍生算法
- Ridge回归（MSE+L2）
  `from sklearn.linear_model import Ridge`
  $$
  \min_{w} \|Xw-y\|_{2}^{2} + \alpha \|w\|_{2}^{2}
  $$

- Lasso回归（MSE+L1）
  `from sklearn.linear_model import Lasso`
  $$
  \min_{w} \frac{1}{2n_{samples}}\|Xw-y\|_{2}^{2} + \alpha \|w\|_{1}
  $$

- 弹性网络回归（ElasticNet Regression）=> 同时使用了L1正则项和L2正则项
  `from sklearn.linear_model import ElasticNet`
  $$
  \min_{w} \frac{1}{2n_{samples}}\|Xw-y\|_{2}^{2} + \alpha \rho \|w\|_{1} +  \frac{\alpha(1-\rho)}{2}\|w\|_{2}^{2}
  $$
  <br>
  总结：

    | 模型       | 正则项 | 特征选择 | 处理共线性 | 使用场景                          |
    | :--------- | :----: | :------: | :--------: | :-------------------------------- |
    | Ridge      |   L2   |    否    |     好     | 共线性数据，防止过拟合            |
    | Lasso      |   L1   |    是    |    不好    | 高维数据，特征选择                |
    | ElasticNet | L1+L2  |    是    |     好     | 高维+共线性数据，平衡选择与稳定性 |

- 多项式升维（polynomial regression）
  目的：解决欠拟合的问题。
  对于多项式回归来说主要是为了扩展线性回归算法来适应更广泛的数据集，应对数据非线性的问题。
  这是一种数据预处理的手段，在`sklearn.preprocessing`模块下
  以两个维度为例：
  - 多项式回归公式是：$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 $
  - 二阶多项式升维得到的公式为：$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1^2 +  w_4 x_2^2 + w_5  x_1 x_2$

---

**代码实现**

**使用Scikit-Learn进行LinearRegression、Ridge回归、Lasso回归和ElasticNet回归**
原文链接:https://blog.csdn.net/qq_30868737/article/details/109495544

**Linear Regression**
- 参数
    `LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)`
    <div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">

    • `fit_intercept` ：是否有截据，如果没有则直线过原点;

    • `normalize` ：是否将数据归一化;

    • `copy_X` ：默认为True，当为True时，X会被copied,否则X将会被覆写;

    • `n_jobs` ：默认值为1。计算时使用的核数

    </div>

    <br>
- 属性：
    <div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">
   • `coef`_ ：array,shape(n_features, ) or (n_targets, n_features)。回归系数(斜率)。
    <br>
   • `intercept_` ：截距
    </div>
   
   
    <br>
- LinearRegression方法：
    <div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">
    • `fit(x,y,sample_weight=None)` ：x和y以矩阵的形式传入，sample_weight则是每条测试数据的权重，同样以矩阵方式传入（在版本0.17后添加了sample_weight）。
    <br>• `predict(x)` ： 预测方法，用来返回预测值
    <br>• `get_params(deep=True)` ： 返回对regressor 的设置值
    <br>
    • `score(X,y,sample_weight=None)` ： 评分函数，将返回一个小于1的得分，可能会小于0
    </div>

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


X1 = 2*np.random.rand(100, 1)
X2 = 2*np.random.rand(100, 1)
X = np.c_[X1, X2]

y = 4 + 3*X1 + 5*X2 + np.random.randn(100, 1)

reg = LinearRegression(fit_intercept=True)
reg.fit(X, y)
print(reg.intercept_, reg.coef_)

X_new = np.array([[0, 0],
                  [2, 1],
                  [2, 4]])
y_predict = reg.predict(X_new)

# 绘图进行展示真实的数据点和我们预测用的模型
plt.plot(X_new[:, 0], y_predict, 'r-')
plt.plot(X1, y, 'b.')
plt.axis([0, 2, 0, 25])
plt.show()
```
**Ridge**
- 参数
  `Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto')`
    <div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">

    • `alpha` ：指定权重值，默认为1。

    • `fit_intercept` ：bool类型，是否需要拟合截距项，默认为True。

    • `normalize` ：bool类型，建模时是否对数据集做标准化处理，默认为False。

    • `copy_X` ：bool类型，是否复制自变量X的数值，默认为True。

    • `max_iter` ：指定模型的最大迭代次数。

    • `tol` ：指定模型收敛的阈值，默认为0.0001。

    • `solver` ：求解器，有auto, svd, cholesky, sparse_cg, lsqr几种，一般我们选择auto，一些svd，cholesky也都是稀疏表示中常用的omp求解算法中的知识，大家有时间可以去了解。

    </div>

**Lasso**
- 参数
   `Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection=‘cyclic’)`

    <div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">   

    • `alpha` ：指定权重值，默认为1。

    • `fit_intercept` ：bool类型，是否需要拟合截距项，默认为True。

    • `normalize` ：bool类型，建模时是否对数据集做标准化处理，默认为False。

    • `precompute` ：bool类型，是否在建模前计算Gram矩阵提升运算速度，默认为False。

    • `copy_X` ：bool类型，是否复制自变量X的数值，默认为True。

    • `max_iter` ：指定模型的最大迭代次数。

    • `tol` ：指定模型收敛的阈值，默认为0.0001。

    • `warm_start` ：bool类型，是否将前一次训练结果用作后一次的训练，默认为False。

    • `positive` ：bool类型，是否将回归系数强制为正数，默认为False。

    • `random_state` ：指定随机生成器的种子。

    • `selection` ：指定每次迭代选择的回归系数，如果为’random’，表示每次迭代中将随机更新回归系数；如果为’cyclic’，则每次迭代时回归系数的更新都基于上一次运算。

    </div>

**ElasticNet**
- 参数
    `ElasticNet(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=1e-4, warm_start=False, positive=False, random_state=None, selection=’cyclic’)`

    <div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">

    • `alpha` ：float, optional 混合惩罚项的常数，默认是1，看笔记的得到有关这个参数的精确数学定义。alpha = 0等价于传统最小二乘回归，通过LinearRegression求解。因为数学原因，使用alpha = 0的lasso回归时不推荐的，如果是这样，你应该使用 LinearRegression 。

    • `l1_ratio` ：float 弹性网混合参数，0 <= l1_ratio <= 1，对于 l1_ratio = 0，惩罚项是L2正则惩罚。对于 l1_ratio = 1是L1正则惩罚。

    • `fit_intercept` ：bool类型，是否需要拟合截距项，默认为True。

    • `normalize` ：bool类型，建模时是否对数据集做标准化处理，默认为False。

    • `precompute` ：bool类型，是否在建模前计算Gram矩阵提升运算速度，默认为False。

    • `copy_X` ：bool类型，是否复制自变量X的数值，默认为True。

    • `max_iter` ：指定模型的最大迭代次数。

    • `tol` ：指定模型收敛的阈值，默认为0.0001。

    • `warm_start` ：bool类型，是否将前一次训练结果用作后一次的训练，默认为False。

    • `positive` ：bool类型，是否将回归系数强制为正数，默认为False。

    • `random_state` ：指定随机生成器的种子。

    • `selection` ：指定每次迭代选择的回归系数，如果为’random’，表示每次迭代中将随机更新回归系数；如果为’cyclic’，则每次迭代时回归系数的更新都基于上一次运算。

    </div>

**使用SGDRegressor**        
`from sklearn.linear_model import SGDRegressor` 
随机梯度下降求解器（SGDRegressor）在很多模型中都有使用，适用于大规模数据和高维特征，因为它是增量式学习的，可以逐步更新模型参数而不需要一次性加载所有数据。
- 参数
    `SGDRegressor(loss='squared_error', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)`
    <div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">

    • `loss` ：损失函数，可选值：squared_error（默认）：普通最小二乘回归（均方误差）。huber：Huber 损失（对异常值鲁棒）。epsilon_insensitive：支持向量回归（SVR）的线性损失。squared_epsilon_insensitive：SVR 的平方损失。

    • `penalty` ：正则化类型，l2（默认）：L2 正则化（Ridge 回归）。l1：L1 正则化（Lasso 回归）。elasticnet：L1 + L2 组合正则化。None：无正则化。

    • `learning_rate` ：学习率，constant：固定学习率（需指定 eta0）。optimal：动态调整（基于 alpha 和 t）。invscaling：随时间递减（公式：eta0 / pow(t, power_t)）。adaptive：当损失稳定时自动减小学习率。

    • `max_iter`；最大迭代次数，默认1000.

    • `tol` ：损失下降容忍度（默认1e-3）。当损失变化小于该值时提前停止。

    • `random_state` ：固定随机种子，确保结果可复现。

    </div>


---

[**实战**](../.\source\py\实战保险花销预测.ipynb)
步骤：
1. EDA(Explore Data Analysis) 数据探索分析 => 将数据可视化，判断其分布，将其转化为类正态分布。
2. 数据清洗 => 删除、填充、异常值处理。
3. 特征工程
   1. 特征构造
   2. 特征转换
        <div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px ; padding: 10px;">

        | 技术                   | 适用场景                   | 示例                               | 导入方式                                                     |
        | :--------------------- | :------------------------- | :--------------------------------- | :----------------------------------------------------------- |
        | 标准化(StandardScaler) | 基于距离的模型（SVM、KNN） | 将特征缩放到均值为0、方差为1       | `from sklearn.preprocessing import StandardScaler`           |
        | 归一化 (MinMaxScaler)  | 神经网络、图像数据         | 缩放到[0,1]区间                    | `from sklearn.preprocessing import MaxAbsScaler`             |
        | 非线性变换             | 解决偏态分布               | $\log(x+1), \sqrt{x}$              | 使用 `numpy` 库进行科学计算                                  |
        | 类别编码               | 转换非数值特征             | One-Hot、目标编码(Target Encoding) | `from sklearn.preprocessing import LabelEncoder` <br> `from sklearn.preprocessing import OneHotEncoder` <br> `pandas.get_dummies` |
        | 分箱 (Binning)         | 将连续值离散化             | 年龄分段为[0-18,19-30,...]         |                                                              |

        </div>
   3. 特征选择
4. 模型评估

---
