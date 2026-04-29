## 2. 线性分类

- 阶段概述：本阶段讲解，**逻辑回归算法**，**Softmax回归算法**，**SVM支持向量机算法**，**SMO优化算法**。

- 达成目标：通过本阶段学习，推导逻辑回归算法、SVM算法的判别式和损失函数，算法的优化、实现算法和应用开发实战。将会对分类算法有深入认知，对于理解后续神经网络算法和深度学习学习至关重要。

### 2.1 广义线性模型解决二分类问题

#### 2.1.1 广义线性模型
我们观察对数线性回归模型：
$$
\ln \hat{y} = w^{T} x + b
$$
它的实质是让 $w^{T} x + b$ 去逼近 $y$ 。在形式上仍是线性回归，但本质上是在求取输入空间到输出空间的**非线性函数映射关系**。
所以推广到一般，考虑可微函数 $g(\cdot)$，令：
$$
\hat{y} = g^{-1}(w^{T}x + b)
$$
这样就得到了**广义线性模型**，函数 $g(\cdot)$ 称为“联系函数”。

- 注意，广义线性回归适用场景：
  - 目标变量 $y$ 服从[**指数族分布**](https://geekdaxue.co/read/kaiba-20hbu@aev2fm/unh5cn)（指数族分布有高斯分布、二项分布、伯努利分布、多项分布、泊松分布、指数分布、beta 分布、拉普拉斯分布、gamma分布等）；
  - 指数族分布的表达式：
  $$
  p(y \mid \eta) = b(y)  \cdot \exp \left( \eta^{T}T(y) - a(\eta) \right)
  $$
  >  $\eta$ ： 为自然参数；
  >  $T(y)$ ： 为充分统计量（sufficient statistic），一般情况下就是 $y$ 本身，即 $T(y)=y$  。
  >  $a(\eta)$ ： 为对数部分函数（log partition function），这部分确保 $p(y \mid \eta)$ 的积分为 1 ，起到归一化的作用。
  >  $b(y)$ ：不是很重要，通常取为 1 。

#### 2.1.2 逻辑回归
根据广义线性模型，做二分类任务的基本手段是找一个**单调可微的函数**将分类任务的真实标记$y$与线性回归模型的预测值$\hat{y}$联系起来。
首先，对于二分类任务，其输出标记 $y \in \{0, 1\}$ ，而线性模型产生的预测值 $\hat{y}$ 为实值，于是，需要将实值 $\hat{y}$ 转换0/1值，有理想的“**单位越阶函数**”（unit-step function）：
$$
y = 
\begin{cases} 
0, & z < 0; \\[3pt]
0.5, & z = 0; \\[3pt]
1, & z > 0. 
\end{cases}
$$

考虑联系函数需要可微，可以引入一种“S形曲线”（Sigmiod函数），最典型的是对数几率函数（logistic function）:
$$
h_{\theta}(x)  =  g(\theta^{T}x) =  \frac{1}{1+e^{-\theta^{T}x}}
$$

<img src="../source/imgs/2.1.2_unit-step function vs. logistic function.png" alt="unit-step function vs. logistic function" style="display: block; margin: 0 auto; max-width: 100%;" width=700>

逻辑回归（Logistic Regression）不是一个回归的算法，逻辑回归是一个分类的算法。
逻辑回归算法是基于多元线性回归的算法。所以，逻辑回归这个分类算法是线性的分类器。

---

**对数几率函数的推导**
由于研究的是二分类问题，可以假设真实目标标记$y$可近似服从伯努利分布（0-1分布），即$y \sim BBernoulli( \phi)$,有：

$$
\begin{aligned}
p(y \mid \phi) 
&= \phi^{y} (1- \phi)^{1-y} \\[8pt]
&= \exp \left\{ y \ln \phi + (1-y) \ln (1-\phi) \right\} \\[5pt]
&= \exp \left\{ \ln \left( \frac{\phi}{1-\phi} \right) \cdot y + \ln (1-\phi) \right\}
\end{aligned}
$$
对比指数族分布的通式，可知：$\eta = \theta^{T}x = \ln \left(\frac{\phi}{1-\phi} \right)$
变形可得对数几率函数 $\phi = \dfrac{1}{1 + e^{-\theta^{T}x} }$
<br>
回过头来看多元线性回归，我们假设目标变量y服从正态分布，$y \sim N(\mu , \sigma^2)$经过类似的推导，也可以得到多元线性回归的通式 $\hat{y} = \theta^{T}x$

---

#### 2.1.3 损失函数
对于真实标记 $y$ 与 预测值 $\hat{y}$

|       | 真实标记 |        预测值        |
| :---: | :------: | :------------------: |
| 正例  |    1     |  $ g(\theta^{T}x) $  |
| 反例  |    0     | $ 1-g(\theta^{T}x) $ |

即：
$p(y = 1 \mid x;\theta) =  g(\theta^{T}x) $ 
$p(y = 0 \mid x;\theta) =  1-g(\theta^{T}x)$
统一为：
$$
p(y \mid x;\theta) = \left( g(\theta^{T}x) \right)^{y} \cdot \left( 1-g(\theta^{T}x) \right)^{1-y}
$$
**似然函数**为：
$$
\begin{aligned}
L(\theta) 
&= \prod_{i=1}^{m} p(y^{(i)} \mid x^{(i)}; \theta)\\[5pt]
&= \prod_{i=1}^{m} \left( g \left(\theta^{T}x^{(i)} \right) \right)^{y^{(i)}} \cdot \left( 1-g \left(\theta^{T}x^{(i)} \right) \right)^{1-y^{(i)}} 
\end{aligned}
$$

两边同时取对数：
$$
\begin{aligned}
\ell(\theta)
&= \ln L(\theta) \\
&= \sum_{i=1}^{m} \left(y^{(i)} \ln \left( g \left(\theta^{T}x^{(i)} \right) \right) + (1-y^{(i)}) \ln \left( 1-g \left(\theta^{T}x^{(i)} \right) \right) \right)
\end{aligned}
$$

最优化：
$$
\theta^{*} = \underset{\theta}{\arg \max} \  \ell(\theta) = \underset{\theta}{\arg \min} \  - \ell(\theta)
$$

损失函数：
$$
J(\theta) = - \sum_{i=1}^{m} \left(y^{(i)} \ln \left( g \left(\theta^{T}x^{(i)} \right) \right) + (1-y^{(i)}) \ln \left( 1-g \left(\theta^{T}x^{(i)} \right) \right) \right)
$$
可用**梯度下降法**求解。
先看联系函数--对数几率函数 $g(z) = \dfrac{1}{ 1+e^{-z} }$ 的导数：
$$
\begin{aligned}
g'(z)
&= \frac{\mathrm{d}}{\mathrm{d}z} \left( \frac{1}{1 + e^{-z}} \right) \\[5pt]  % 增加行间距
&= -\frac{1}{(1 + e^{-z})^2} \cdot \left( -e^{-z} \right) \\[5pt]
&= \frac{1}{1 + e^{-z}} \cdot \left( 1 - \frac{1}{1 + e^{-z}} \right) \\[5pt]
&= g(z) \cdot \bigl( 1 - g(z) \bigr)  % 使用 \bigl \bigr 强调括号
\end{aligned}
$$

对损失函数求偏导：
$$
\begin{aligned}
\frac{\partial}{\partial \theta_j} J(\theta) 
&= -\frac{1}{m} \sum_{i=1}^m \left( y^{(i)} \frac{1}{h_\theta(x^{(i)})} \frac{\partial}{\partial \theta_j} h_\theta(x^{(i)}) - (1-y^{(i)}) \frac{1}{1-h_\theta(x^{(i)})} \frac{\partial}{\partial \theta_j} h_\theta(x^{(i)}) \right) \\[10pt]
&= -\frac{1}{m} \sum_{i=1}^m \left( y^{(i)} \frac{1}{g(\theta^T x^{(i)})} - (1-y^{(i)}) \frac{1}{1-g(\theta^T x^{(i)})} \right) \frac{\partial}{\partial \theta_j} g(\theta^T x^{(i)}) \\[10pt]
&= -\frac{1}{m} \sum_{i=1}^m \left( y^{(i)} \frac{1}{g(\theta^T x^{(i)})} - (1-y^{(i)}) \frac{1}{1-g(\theta^T x^{(i)})} \right) g(\theta^T x^{(i)})(1-g(\theta^T x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)} \\[10pt]
&= -\frac{1}{m} \sum_{i=1}^m \left( y^{(i)} (1-g(\theta^T x^{(i)})) - (1-y^{(i)}) g(\theta^T x^{(i)}) \right) x_j^{(i)} \\[10pt]
&= -\frac{1}{m} \sum_{i=1}^m (y^{(i)} - g(\theta^T x^{(i)})) x_j^{(i)} \\[10pt]
&= \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
\end{aligned}
$$

得到：
$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$
**不难发现，这个导函数与多元线性回归推导出来的形式上一致。**

> 含正则项的损失函数（以L2为例）：
> $$
> J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \frac{1}{h_\theta(x^{(i)})} \frac{\partial}{\partial \theta_j} h_\theta(x^{(i)}) + (1-y^{(i)}) \frac{1}{1-h_\theta(x^{(i)})} \frac{\partial}{\partial \theta_j} h_\theta(x^{(i)}) \right] + \frac{\lambda}{m} \sum_{i=1}^{m} \theta_{j}^{2}
> $$
>
> $$
> \frac{\partial}{\partial \theta_{j}} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_{j}
> $$

---

**鸢尾花数据集实战**

```python
'''
The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============
'''
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
print(list(iris.keys()))
print(iris['DESCR'])
print(iris['feature_names'])

X = iris['data'][:, 3:]
print(X)

print(iris['target'])
# y = (iris['target'] == 2).astype(np.int)
y = iris['target']
print(y)

# 分类学习器通常称为“分类器”（classifier）
# binary_classifier = LogisticRegression(solver='sag', max_iter=1000)
# binary_classifier.fit(X, y)
# 多分类的分类器，"ovr"对应的时OvR的LR模型，"multinomial"对应的Softmax回归模型
# multi_classifier = LogisticRegression(solver='sag', max_iter=1000, multi_class='multinomial')
multi_classifier = LogisticRegression(solver='sag', max_iter=1000, multi_class='ovr')
multi_classifier.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(X_new)
y_proba = multi_classifier.predict_proba(X_new)
print(y_proba)
y_hat = multi_classifier.predict(X_new)
print(y_hat)
```

---

#### 2.1.4 利用二分类解决多分类问题

多分类学习的基本思路是“**拆解法**”,即将多分类任务拆为若干个二分类任务。

最经典的拆分策略有三种：
- “**一对一**”（One vs. One，简称OvO）；
- “**一对余**”（One vs. Rest，简称OvR）；
- “**多对多**”（Many vs. Many，简称MvM）。

1. OvO：将多个类别两两配对，从而产生 $\frac{N(N-1)}{2}$ 个二分类任务，得到 $\frac{N(N-1)}{2}$ 个结果，最终结果可由“投票”产生。

2. OvR：每次将一个类作为正例，其他类为反例，训练 $N$ 个分类器。在测试时，若仅有一个分类器预测为正类，则对应的类别表及作为最终结果；若有多个分类器预测为正类，则通常考虑个分类器预测置信度，选择置信度最大的类别表及作为分类结果。
<img src="../source/imgs/2.1.4_OvO_OvR.png" alt="OvO_OvR" style="display: block; margin: 0 auto; max-width: 100%;" width=700>

> OvO 与 OvR 两种策略的预测性能，在多数情况下两者差不多。
> 对于类别较少的情况，OvR只需训练 $N$ 个训练器，内存开销和测试时间更小；
> 但是当类别很多事，由于OvR的每个分类器多使用全部训练样例，训练时间开销反而更大。

3. MvM：每次将若干个类作为正类，若干个其他类作为反类。可以看出MvM是OvR和OvO更一般的形式。<br>
    对于MvM正、反类的构造一种常用的技术是“纠错输出码”（Error Correcting Output Codes，简称ECOC）.主要分为两步：<br>
    编码：对N个类别做M次划分，每次划分将一部分类别划分为正类，一部分类别划分为反类，从而形成一个二分类训练集；这样一共产生M个训练集，可训练出M个分类器。
    解码：测试时，M个分类器分别对测试样本x进行预测，这样预测的结果就形成了一个编码。将这个编码与每个类别各自的编码进行比较，找到距离最短的类别作为最终分类的结果。

---

### 2.2 Softmax回归解决多分类问题
对于服从**伯努利分布**的目标变量，可以通过 **Logistic回归** 建模。
对于服从**多项式分布**的目标变量，处理多分类问题，可以通过 **Sotfmax** 回归建模。

#### 2.2.1 前导知识--多项式分布
多项式分布（Multinomial Distribution）是二项分布（Binomial Distribution）的推广。
- 二项分布
    - 定义：n 次独立伯努利试验中成功次数的分布。
    - 随机变量：$X \sim Binomial(n,p)$，$X$表示成功次数。
    - PMF：
    $$
    P(X = k) = \dbinom{n}{k} \, p^{k} \, (1 - p)^{n - k}, \quad k \in \bigl\{0, 1, \ldots, n\bigr\}
    $$
    - 性质：$E(X) = np \ Var(X) = np(1-p)$

- 多项式分布
    - 定义：二项分布的推广，描述多类别（k≥2）独立试验中各类别出现次数的联合分布。
    - 随机变量：$ X=(X_1,\cdots, X_k) \sim Multinomial(n,p)$，其中$X_i$表示第$i$类结果的次数，$p = (p_1, \cdots, p_k)$为各类概率。
    - PMF：
    $$
    P(X = x) = \frac{n!}{x_1! \cdots x_k!} p_1^{x_1} \cdots p_k^{x_k}, \quad \sum_{i=1}^k x_i = n
    $$

    - 性质： $ \sum\limits_{i=1}^{k} p_i = 1$

<div style="width: 100%; overflow-x: auto; white-space: nowrap; border: 1px ; padding: 10px;">

| 分布       | 试验类型       | 结果类别数      | 随机变量维度 | PMF公式 | 主要用途                 |
|------------|----------------|-----------------|--------------|-----------------------|--------|
| 0-1分布    | 单次试验       | 2               | 标量         | $$P(X=x) = p^x(1-p)^{1-x}, \quad x \in \{0,1\}$$                       | 单次二值结果             |
| 二项分布   | $n$ 次试验     | 2               | 标量         | $$P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}, \quad k \in \{0,1,\dots,n\}$$  | 某类结果的累计次数       |
| 多项式分布 | $n$ 次试验     | $k \geq 2$      | **向量**         | $$P(\mathbf{X}=\mathbf{x}) = \frac{n!}{\prod_{i=1}^k x_i!}\prod_{i=1}^k p_i^{x_i}$$ | 多类别的联合计数分布     |

</div>

#### 2.2.2 多项分布转变为指数分布族的推导

从联合概率密度函数出发：
$$
\begin{aligned}
P(y; \varphi) &= \varphi_1^{\mathbb{I}(y=1)} \varphi_2^{\mathbb{I}(y=2)} \cdots \varphi_{k-1}^{\mathbb{I}(y=k-1)} \varphi_k^{\mathbb{I}(y=k)} \\[5pt]
&= \varphi_1^{\mathbb{I}(y=1)} \varphi_2^{\mathbb{I}(y=2)} \cdots \varphi_{k-1}^{\mathbb{I}(y=k-1)} \varphi_k^{1-\sum_{i=1}^{k-1} \mathbb{I}(y=i)} \\[5pt]
&= \exp\left(\ln \left( \varphi_1^{\mathbb{I}(y=1)} \varphi_2^{\mathbb{I}(y=2)} \cdots \varphi_{k-1}^{\mathbb{I}(y=k-1)} \varphi_k^{1-\sum_{i=1}^{k-1} \mathbb{I}(y=i)} \right)\right) \\[5pt]
&= \exp\left(\sum_{i=1}^{k-1} \mathbb{I}(y=i)\ln\varphi_i + \left(1-\sum_{i=1}^{k-1} \mathbb{I}(y=i)\right)\ln\varphi_k\right) \\[3pt]
&= \exp\left(\sum_{i=1}^{k-1} \mathbb{I}(y=i)\ln\left(\frac{\varphi_i}{\varphi_k}\right) + \ln\varphi_k\right) \\[3pt]
&= \exp\left(\sum_{i=1}^{k-1} T(y)\ln\left(\frac{\varphi_i}{\varphi_k}\right) + \ln\varphi_k\right) \\[8pt]
&= \exp\left(\eta^T T(y) - a(\eta)\right)
\end{aligned}
$$

> $\mathbb{I}(\cdot)$ 是指示函数，$\mathbb{I}[\text{true}]=1$ ，$\mathbb{I}[\text{false}]=0$ 。

自然参数：
$$
\eta = 
\begin{bmatrix}
\ln(\varphi_1/\varphi_k) \\[5pt]
\ln(\varphi_2/\varphi_k) \\[5pt]
\vdots \\
\ln(\varphi_{k-1}/\varphi_k)
\end{bmatrix}
$$
整理得：
$$
\eta_i = \ln\frac{\varphi_i}{\varphi_k} \quad \Rightarrow \quad \varphi_i = \varphi_k e^{\eta_i}
$$
由归一化条件：
$$
\begin{aligned}
\sum_{j=1}^k \varphi_j &= \sum_{j=1}^k \varphi_k e^{\eta_j} = 1 \\[5pt]
\Rightarrow \varphi_k &= \frac{1}{\sum_{j=1}^k e^{\eta_j}} \\[5pt]
\varphi_i &= \frac{e^{\eta_i}}{\sum_{j=1}^k e^{\eta_j}}
\end{aligned}
$$

至此，我们就得到了 softmax 回归的公式：
$$
h_\theta(x^{(i)}) = 
\begin{bmatrix}
p(y^{(i)} = 1|x^{(i)}; \theta) \\
p(y^{(i)} = 2|x^{(i)}; \theta) \\
\vdots \\
p(y^{(i)} = k|x^{(i)}; \theta)
\end{bmatrix} 
= \frac{1}{\sum_{j=1}^k e^{\theta_j^T x^{(i)}}} 
\begin{bmatrix}
e^{\theta_1^T x^{(i)}} \\
e^{\theta_2^T x^{(i)}} \\
\vdots \\
e^{\theta_k^T x^{(i)}}
\end{bmatrix}
$$

---

#### 2.2.3 损失函数
似然函数：
$$
L(\theta) = \prod_{i=1}^{m}p(y^{(i)} \mid x^{(i)} ; \theta) = \prod_{i=1}^{m} \prod_{j=1}^{k} \phi_{j}^{\mathbb{I}(y^{(i)} = j )}
$$

对数似然函数：
$$
\begin{aligned}
\ell(\theta) 
&= \ln L(\theta)
= \sum_{i=1}^{m} \sum_{j=1}^{k} \mathbb{I}(y^{(i)} = j ) \cdot \ln \left( \frac{e^{\theta_j^{T} x^{(i)}}}{\sum_{l=1}^{k} e^{\theta_l^{T} x^{(i)}}} \right)
\end{aligned}
$$

所以损失函数为：
$$
J(\theta) = -\frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=1}^{k} \mathbb{I}(y^{(i)} = j ) \cdot \ln \left( \frac{e^{\theta_j^{T} x^{(i)}}}{\sum_{l=1}^{k} e^{\theta_l^{T} x^{(i)}}} \right) \right]
$$

**Logistic Regression 是 Softmax Regression 在 k=2 时的一个特例**

当使用 LR 和 SR 去解决多分类问题的区别和算法选择：

- LR 使用 OvR 策略时，是将多分类问题转化为 n 个**独立**的二分类问题，它给出的是重新划分标签后属于这个类的概率，可能会碰到两种类的概率差不多的情况，这时就需要根据置信度去实现。
- SR 综合考量各个类别，倾向将所有的结果单独划分到一个类别中去。
- 选择算法时，需要根据实际需求选择：
  - 若需要将一个东西准确的划分到某一个类，可以选择 Softmax 回归，例如生物学的物种划分、邮件类别的划分等。
  - 若是使用算法给出某一个类的可能性，来辅助判断，可以选择 Logistic 回归，例如AI医疗，通过图像或病症的描述（这种因不同类别但是存在交叉的特征时）给出可能的疾病。
---

### 2.3. 支持向量机
支持向量机（Support Vector Machine，SVM）：**本身是一个二元分类算法，是对感知器算法模型的一种扩展**，现在的SVM 算法支持线性分类和非线性分类的分类应用，并且也能够直接将SVM应用于回归应用中，同时通过 OvR 或者 OvO 的方式我们也可以将SVM 应用在多元分类领域中。在不考虑集成学习算法，不考虑特定的数据集的时候，在分类算法中SVM可以说是特别优秀的。

思想：从向量角度出发，SVM模型将样本点映射到高维空间的向量，通过寻找一个超平面将不同类别的样本分开，通过这个超平面来做分类。

> 做分类的算法大致上有：LR、KNN、DT（决策树）、RT（随机森林）、XGBoost，还有很多概率模型，如贝叶斯。

#### 2.3.1 前导知识
- **距离**
    几何距离：二维平面中，点 $(x_i,y_i)$ 到直线 $ax + by + c = o$的距离为：
    $$
    d = \frac{| a x_i + b y_i + c |}{\sqrt{a^2+b^2}}
    $$

    推广到高维空间中，任意一个点$x^{(i)}$到超平面$w^{T}x + b = 0$的距离为：
    $$
    \gamma = \frac{|wx^{(i)}+b|}{\| w\|}
    $$

- [**拉格朗日乘子法**](https://blog.csdn.net/LittleEmperor/article/details/105057670)
  
  <br>

    <div style="width: 100%; border: 1px solid #ddd; padding: 10px;">

    <p>在解决有约束条件的最优化问题时，有时能使用消元法，利用约束条件将目标函数转化为无约束的极值求解问题，但这局限性比较大，大部分情况下很难适用，比如等式约束为高次耦合非线性。<p>

    所以，在大多数情况下，我们使用**拉格朗日乘子法**

  ---

    **带约束条件的最优化问题泛化表示**：

  $$
    \begin{cases} 
    \min f(X) & X \in \mathbb{E}^n \\[5pt]
    \text{S.t. } c_i(X) \leq 0 & i = 1, 2, \cdots, m \quad \& \quad h_j(X) = 0 & j = 1, 2, \cdots, l
    \end{cases}
  $$

    构造拉格朗日函数：
  $$
    L(X,\alpha,\beta) = f(X) + \sum_{i=1}^{m}\alpha_i c_i(X) + \sum_{\beta_j=1}^{l}\beta_j h_j(X)
  $$

    极值条件为：

  $$
    \begin{cases}
    \nabla_x L(x,\alpha,\beta) = 0\\[5pt]
    \nabla_{\alpha} L(x,\alpha,\beta) = 0\\[5pt]
    \nabla_{\beta} L(x,\alpha,\beta) = 0
    \end{cases}
  $$

  ---

   L1、L2正则实质上就是使用拉格朗日乘数法得到的一种结果。

    </div>

- **拉格朗日对偶**

    <div style="width: 100%; border: 1px solid #ddd; padding: 10px;">

    [PDF](../source/article/lagrangian_multiplier.pdf)
    
    [MD](../source/article/lagrangian_multiplier.md)

    </div>
    
    <br>
    
- **坐标上升算法**
  
    <div style="width: 100%; border: 1px solid #ddd; padding: 10px;">
    
    坐标上升法（Coordinate Ascent, CA）是一种用于优化问题的迭代方法，常用于机器学习、统计推断和信号处理等领域。它是坐标下降法（Coordinate Descent, CD）的对偶方法，而坐标下降法更为常见。两者的核心思想类似，都是在每次迭代时，仅优化一个变量的方向，使目标函数单调上升（或下降），而保持其他变量固定。
    
    **基本原理**
    
    坐标上升法的核心思想是，在高维优化问题中，每次固定其他变量，仅对一个变量进行优化更新，以逐步提高目标函数的值。假设目标函数为$ f(x_1, x_2, \dots, x_n)$，那么在每次迭代中，我们选择一个变量 x_i 并在其方向上优化，使得：
    $$
    x_i^{(t+1)} = \underset{x_i}{\arg \max} \  f(x_1^{(t)}, x_2^{(t)}, \dots, x_i, \dots, x_n^{(t)})
    $$
    然后依次更新其他变量，直到满足收敛条件。
    
    与梯度上升（Gradient Ascent）的区别
    - 梯度上升 计算目标函数在所有变量方向上的梯度，并同时更新所有变量。
    - 坐标上升 仅优化一个变量的方向，使得每次更新更简单，适用于维度较高但梯度计算困难的问题。
    
    **应用**
    - 期望最大化（EM）算法：在EM算法的M步中，经常使用坐标上升方法优化参数，使对数似然函数单调上升。
    - 变分推断（Variational Inference）：用于最大化证据下界（ELBO），优化概率模型的近似后验分布。
    - 推荐系统：如矩阵分解模型的优化，可用坐标上升法交替更新用户和物品的嵌入向量。
    
    **优缺点**
    
    优点：
    - 在梯度计算困难或不可行时仍然适用。
    - 每次迭代仅优化一个变量，计算较为简单。
    
    缺点：
    - 可能收敛较慢，尤其在变量间耦合较强时。
    - 不一定保证收敛到全局最优解。
    
    </div>
    
    <br>
    
- **SMO算法**
  
    <div style="width: 100%; border: 1px solid #ddd; padding: 10px;">

    [PDF](../source/article/SMO.pdf)
    
    [MD](../source/article/SMO.md)
    
    序列最小优化（Sequential Minimal Optimization，SMO）是基于 CA 改进的一种算法的基本思路是先固定$\alpha_i$之外的所有参数，然后求$\alpha_i$上的极值。
    
    由于存在约束 $\sum\limits_{i=1}^{m} \alpha_i y^{(i)} = 0$，所以若固定$\alpha_i$之外的所有参数，$\alpha_i$可由其他变量导出。
    
    所以对于原始目标函数优化的对偶问题：
    
    $$
    \alpha^{*} = \min_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)} ,x^{(j)}) - \sum_{i=1}^{m} \alpha_i\\[8pt]
    s.t. \quad
    \begin{cases}
    \sum\limits_{i=1}^{m} \alpha_i y^{(i)} = 0\\[10pt]
    0 \leq \alpha_i \leq C
    \end{cases}
    $$
    
    SMO 每次选择两个变量$\alpha_i$和$\alpha_j$（**启发式的**），并固定其他参数，SMO不断执行下面两个步骤直至收敛：
    
    - 选择接下来要更新的一对$\alpha_i$和$\alpha_j$：采用启发式的方法进行选择，以使目标函数最大程度地接近其全局最优值。
    - 将目标函数对$\alpha_i$和$\alpha_j$进行优化，保持其它所有的参数不变
    
    >“**启发式**”的解释：<br>
    >注意到只需选取的 $\alpha_i$ 和 $\alpha_j$ 中有一个不满足 KKT 条件，目标函数就会在迭代后减小。直观来看，KKT条件违背的程度越大，变量更新后可能导致的目标函数减幅越大，于是 SMO 先选取违背KKT条件程度最大的变量，第二个变量应选择一个使目标函数值见效最快的变量，但是由于比较各变量所对应的目标函数减幅的复杂度过高，因此 SMO 采用了这样的一个启发式：**使选取的两变量所对应样本之间的间隔最大**。<br>
    >
    >一种直观的解释是：这样的两个变量有很大的差别，与对两个相似的变量进行更新比较，对它们更新会给目标函数值带来更大的变化。
    
    当固定其他参数后，仅优化两个参数的过程能做到十分高效。具体来说，仅考虑 $\alpha_i$ 和 $\alpha_j$ 时，对偶问题的约束可重写为：
    
    $$
    \alpha_i y^{(i)} + \alpha_j y^{(j)} = c = - \sum_{k \neq i,j} \alpha_k y^{(k)} \ , \quad \alpha_i \geq 0 \ , \quad \alpha_j \geq 0
    $$
    
    这一用该式消去对偶问题的$\alpha_j$，得到一个关于$\alpha_i$的单变量二次规划问题，仅有的约束是$\alpha_i \geq 0$：
    
    $$
    \alpha^{*} = \min_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i (c - \alpha_i y^{(i)}) y^{(i)} K(x^{(i)} ,x^{(j)}) - \sum_{i=1}^{m} \alpha_i
    $$
    
    </div>

---

#### 2.3.2 感知器算法思想
感知器（Perceptron）的思想很简单：在任意空间中，感知器模型寻找的就是一个超平面，能够把所有的二元类别分割开。感知器模型的前提是：数据是线性可分的。
<img src="../source/imgs/2.3.1_Linear_Nonlinear.png" alt="Linear vs. Nonlinear" style="display: block; margin: 0 auto; max-width: 100%;">

对于 $m$ 个样本，每个样本 $n$ 维特征，二分类标记 $y \in \{-1,1\}$ 的训练集$D$ ：
$$
D = \{ (x^{(1)},y^{(1)}), (x^{(2)},y^{(2)}), \cdots , (x^{(m)},y^{(m)})\}\\[5pt]
\quad y^{(i)} \in \{1,-1\}
\quad x^{(i)} = (x_{1}^{(i)}, x_{2}^{(i)}, \cdots, x_{n}^{(i)})
$$
目标是找到一个超平面：
$$
\theta_{0} + \theta_{1} x_1 + \cdots + \theta_{n} x_n = 0 \quad \to \quad \theta x = 0
$$
让一个类别的样本满足 $\theta x < 0$，另一个类别的样本满足 $\theta x > 0$

所以感知器模型判别式可以表达为：
$$
y = sign(\theta x) = 
\begin{cases}
+1, \quad \theta x > 0\\[5pt]
-1, \quad \theta x < 0
\end{cases}
$$

> 回过头来看**逻辑回归的几何意义**，又何尝不是向量空间找到一个超平面，超平面一侧的点计算分数结果为负，另一侧结果分数为正，只不过最后不直接看 sign 符号，而是根据 sigmoid 函数将分数映射到 0-1 之间通过最大似然来赋予概率意义。

**损失函数**
由于感知器的标签y仅仅是表示两个不同的类，并没有相关的概率意义，所以我们抛弃最大似然估计的方法来求损失函数。

对于正确的分类，我们发现 $y \cdot \theta x > 0$，对于错误的分类，有 $y \cdot \theta x < 0$

所以我们可以定义我们的损失函数为：期望使分类错误的所有样本到超平面的距离之和最小。

引入标记$y$可以将求距离的绝对值去掉，所以有：
$$
\gamma_{right} = \frac{y^{(i)}(wx^{(i)}+b)}{\| w\|}\\[8pt]
\gamma_{wrong} = - \frac{y^{(i)}(wx^{(i)}+b)}{\| w\|}
$$

所以损失函数：
$$
L(\theta) = \sum_{i=1}^{m} -\frac{y^{(i)}(\theta x^{(i)})}{\| \theta\|}
$$

因为此时分子和分母中都包含了 $θ$ 值，当分子扩大 N 倍的时候，分母也会随之扩大，也就是说分子和分母之间存在倍数关系，所以可以固定分子或者分母为 1，然后求另一个即分子或者分母的倒数的最小化作为损失函数。为简单起见，将分母固定为 1，简化后的损失函数为：
$$
L(\theta) = \sum_{i=1}^{m} -y^{(i)} \theta x^{(i)}
$$

使用梯度下降法来求最优解。
$$
\frac{\partial}{\partial \theta} L(\theta) = - \sum_{i=1}^{m} y^{(i)} \cdot x^{(i)}
$$

---

#### 2.3.3 SVM算法
SVM 也是通过寻找超平面，用于解决二分类问题的分类算法，模型判别式与感知器相同，但 SVM 的损失函数与感知器和逻辑回归都不同。

- LR：通过最大似然估计寻找超平面；
- Perceptron：通过判错的点来寻找超平面；
- SVM：通过支持向量寻找超平面。

正是这三者造成了损失函数的不同。
感知器和逻辑回归都是通过最小化损失函数来得到 $\theta$ ，而 SVM 有两种手段，一种是先找支撑向量,另一种是直接最小化一个损失函数（为合页损失，hinge loss）。

SVM 较感知器的优势是 SVM 的泛化能力更强。
SVM 的思想：让离超平面比较近的点尽可能的远离这个超平面，增强模型的鲁棒性。 $\quad \Rightarrow \quad$  有约束条件的二次优化问题--“离超平面最近”、“尽可能远” $\quad \Rightarrow \quad$ 求解方法--**拉格朗日乘子法**
$$
\max_{w,b} \gamma_{min} = \frac{y_{min}(w^T x_{min} + b)}{\| w \|}\\[8pt]
s.t. \quad \gamma^{'(i)} = y^{(i)} \left(w^Tx^{(i)} + b \right) \ge \gamma^{'}_{min} \quad (i = 1,2,\cdots, m)
$$

<div style=" border: 1px solid #ddd; padding: 10px;">
**支持向量**（Support Vector）：距离超平面最近的几个满足判别式的点称为“支持向量”。

<img src="../source/imgs/2.3.3_SV_margin.png" alt="SV & maigin" style="display: block; margin: 0 auto; max-width: 100%;">

两个异类支持向量到超平面的距离之和称为“间隔（margin）”：
$$
\gamma = \frac{2}{\|w\|}
$$
</div>

---

##### 硬分隔 SVM
一组 $(w,b)$ 只能确定一个超平面，而一个超平面可由无数组 $(w,b)$ 表达，不同的 $(w,b)$ 使最近的点的 $\gamma^{'}$ 不同，所以只用找到一组 $(w,b)$ 使得 $\gamma^{'}_{min}=1$，就可以将最优化问题转化为：

$$
\begin{aligned}
&\left\{
\begin{aligned}
&\max \frac{2}{\|w\|_2} \\
&\text{s.t.} \quad y^{(i)}\left(w^T x^{(i)} + b\right) \geq 1
\end{aligned}
\right. \\[10pt]
\Leftrightarrow \quad &
\left\{
\begin{aligned}
&\min \frac{1}{2} \|w\|_2^2 \\
&\text{s.t.} \quad y^{(i)}\left(w^T x^{(i)} + b\right) \geq 1
\end{aligned}
\right.
\end{aligned}
$$

构造拉格朗日函数：

$$
L(w,b,\alpha) = \frac{1}{2} \| w \|_{2}^{2} - \sum_{i=1}^{m} \alpha_i \left[ y^{(i)} \left( w^{T} x^{(i)} + b \right) \right] \quad \& \quad \alpha_i \ge 0
$$

可将原始有约束的最优化问题转化为对拉格朗日函数进行无约束的最优化问题（即二次规划问题）：

$$
\min_{w,b} \max_{a_i \geq 0} L(w, b, a)
$$

由于我们的原始问题满足 f(x) 为凸函数，那么可以将原始问题的极小极大优化转换为对偶函数的极大极小优化进行求解：

$$
\begin{aligned}
&\min_{w,b} \max_{a_i \geq 0} L(w, b, a) \text{原始问题}\\
\Rightarrow
&\max_{a_i \geq 0} \min_{w,b} L(w, b, a) \text{对偶问题}
\end{aligned}
$$

**第一步--求极小 $\quad \Rightarrow \quad$ $\min\limits_{w,b} L(w, b, \alpha)$**

$$
\begin{aligned}
\frac{\partial L}{\partial w} = 0 \quad &\Rightarrow \quad w = \sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)}\\
\frac{\partial L}{\partial b} = 0 \quad &\Rightarrow \quad \sum_{i=1}^{m} \alpha y^{(i)} = 0
\end{aligned}
$$

反代回 $L(w, b, \alpha)$：

$$
\begin{aligned}
L(w, b, \alpha) 
&= \frac{1}{2} \| w \|_{2}^{2} - \sum_{i=1}^{m} \alpha_i \left[ y^{(i)} \left( w^{T} x^{(i)} + b \right) \right]\\[8pt]
&= \frac{1}{2} w^T w - \sum_{i=1}^{m} \alpha_i y^{(i)} w^T x^{(i)} - \sum_{i=1}^{m} \alpha_i y^{(i)} b + \sum_{i=1}^{m} \alpha_i\\[8pt]
&= \frac{1}{2} w^T \sum_{i=1}^{m} \alpha_i y{(i)} x^{(i)} - \sum_{i=1}^{m} \alpha_i y^{(i)} w^T x^{(i)} + \sum_{i=1}^{m} \alpha_i\\[8pt]
&= -\frac{1}{2} w^T \sum_{i=1}^{m} \alpha_i y{(i)} x^{(i)} + \sum_{i=1}^{m} \alpha_i\\[8pt]
&= -\frac{1}{2} \left(\sum_{i=1}^{m} \alpha_i y{(i)} x^{(i)} \right)^{T} \left(\sum_{i=1}^{m} \alpha_i y{(i)} x^{(i)} \right) + \sum_{i=1}^{m} \alpha_i\\[8pt]
&= -\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} \langle x^{(i)} ,x^{(j)}\rangle + \sum_{i=1}^{m} \alpha_i
\end{aligned}
$$

约束条件为：

$$
\begin{aligned}
s.t. \quad \sum_{i=1}^{m} \alpha_i y^{(i)} &= 0\\
\alpha_i &\ge 0, \quad i = 1,2,\cdots,m
\end{aligned}
$$

**第二步--对对偶函数的优化问题**：

$$
\alpha^{*} = \max_{\alpha} -\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} \langle x^{(i)} ,x^{(j)}\rangle + \sum_{i=1}^{m} \alpha_i\\
\begin{aligned}
s.t. \quad \sum_{i=1}^{m} \alpha_i y^{(i)} &= 0\\
\alpha_i &\ge 0, \quad i = 1,2,\cdots,m
\end{aligned}
$$

取个负号，转化为求极小值得问题：

$$
\alpha^{*} = \min_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} \langle x^{(i)} ,x^{(j)}\rangle - \sum_{i=1}^{m} \alpha_i\\
\begin{aligned}
s.t. \quad \sum_{i=1}^{m} \alpha_i y^{(i)} &= 0\\
\alpha_i &\ge 0, \quad i = 1,2,\cdots,m
\end{aligned}
$$

通常使用 **SMO**（Sequential Minimal Optimization）算法进行求解，可以求得一组 $\alpha^{*}$ 使得函数最优化。

**确定超平面**：

$$
w^{*} = \sum_{i=1}^{m} \alpha^{*} y^{(i)} x^{(i)}
$$

对于偏置项 $b$，注意到对任意支持向量 $(x_s,y_s)$ 都有 $y_sf(s_s) = 1$,即：

$$
y_s \left( \sum_{i \in S} \alpha_i y^{(i)} (x^{(i)})^T x_s + b \right) = 1 \quad \text{其中} S = \{i \: | \: \alpha_i > 0, i = 1,2,\cdots,m\}\text{为所有支持向量得下标集}
$$

选任意支持向量求解 $b_s^{*}$：

$$
b_s^{*} = y_s - \sum_{i \in S} \alpha_i y^{(i)} (x^{(i)})^T x_s + b
$$
上式中：由于 $y_s \pm 1$，所以$\frac{1}{y_s} = y_s$

为增强其鲁棒性，可以使用所有支持向量求解的平均值作为最终的结果 $b^{*}$：

$$
b^{*} = \frac{1}{|S|} \sum_{i \in S} b_s^{*} = \frac{1}{|S|} \sum_{i \in S} \left(y_s - \sum_{i \in S} \alpha_i y^{(i)} (x^{(i)})^T x_s + b \right)
$$

上述是硬分隔SVM的求解过程。

<div style=" border: 1px solid #ddd; padding: 10px;">

**求解流程小结：**

1. 原始目标：求得一组 $w$ 和 $b$ 使得分隔 margin 最大
2. 转换目标：通过拉格朗日函数构造目标函数，问题由求得 $(w,b)$ 转换为求 $\alpha$
   
    $$
    \alpha^{*} = \max_{\alpha} -\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} \langle x^{(i)} ,x^{(j)}\rangle + \sum_{i=1}^{m} \alpha_i\\
    \begin{aligned}
    s.t. \quad \sum_{i=1}^{m} \alpha_i y^{(i)} &= 0\\
    \alpha_i &\ge 0, \quad i = 1,2,\cdots,m
    \end{aligned}
    $$

3. 利用SMO算法求得 $\alpha^{*}$
4. 利用求得的 $\alpha^{*}$ 求得 $w^{*}$ 和 $b^{*}$
    $$
    w^{*} = \sum_{i=1}^{m} \alpha^{*} y^{(i)} x^{(i)}
    $$

    $$
    b^{*} = \frac{1}{|S|} \sum_{i \in S} b_s^{*} = \frac{1}{|S|} \sum_{i \in S} \left(y_s - \sum_{i \in S} \alpha_i y^{(i)} (x^{(i)})^T x_s + b \right)
    $$
    </div>

---

##### 软间隔 SVM

有些时候，线性不可分是由噪声点决定的。

<img src="../source/imgs/2.3.3_soft_SVM.png" alt="soft_SVM" style="display: block; margin: 0 auto; max-width: 100%;">

我们允许某些样本不满足约束

$$
y^{(i)}\left(w^T x^{(i)} + b\right) \geq 1
$$

而我们也希望在最大化间隔的同时，不满足约束的样本应尽可能少，于是优化目标可写为：

$$
\min_{w,b} \frac{1}{2} \|w\|^2 + C\sum_{i=1}^{m} \ell_{0/1} \left( y^{(i)}(w^T x^{(i)} + b) -1  \right)
$$

其中 $C>0$，是个常数，$\ell_{0/1}$ 称为“0/1损失函数”：

$$
\ell_{0/1} = 
\begin{cases}
1,&\text{if z < 0;}\\
0,&\text{otherwise}  
\end{cases}
$$

显然，当 $C$ 为无穷大时，所有样本点均满足约束条件，失去分类的意义；当 $C$ 为有限值，表明只有某一下点才可以不满足约束条件。

但是， **$\ell_{0/1}$ 非凸、非连续，数学性质不好**，通常采用其他的函数来代替 $\ell_{0/1}$ ，称为“代替损失（surrogate loss）”。代替损失函数一般具有较好的数学性质，它们通常是凸的连续函数且是 $\ell_{0/1}$ 的上界。

三种常用的替代损失函数如下：

- 合页损失（hinge loss）：$\ell_{hinge}(z) = \max(1, 1-z)$
- 指数损失（exponential loss）：$\ell_{exp}(z) = exp(-z)$
- 对数损失（logistic loss）：$\ell_{log}(z) = log(1+exp(-z))$

<img src="../source/imgs/2.3.3_surrogate_loss.png" alt="soft_SVM" style="display: block; margin: 0 auto; max-width: 100%;"><br>

若**采用hinge损失**，优化目标可写为：

$$
\min_{w,b} \frac{1}{2} \|w\|^2 + C\sum_{i=1}^{m} \max \left(0, 1- y^{(i)}(w^T x^{(i)} + b)  \right)
$$

引入松弛变量 $\xi_i \geq 0$（以松弛变量 $\xi$ 表示异常点嵌入间隔面的深度），目标函数重写为：

$$
\min \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^{m} \xi_i \\[10pt]
$$

约束条件放松为：

$$
\begin{aligned}
s.t. &\quad y^{(i)}\left(w^T x^{(i)} + b\right) \geq 1 - \xi_i\\[5pt]
&\quad \xi_i \geq 0, \quad i = 1,2,\cdots,m
\end{aligned}
$$

**构造拉格朗日函数**：
$$
L(w,b,\xi, \alpha, \mu) = \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^{m} \xi_i - \sum_{i=1}^{m} \alpha_i \left[y^{(i)}\left(w^T x^{(i)} + b\right) - 1 + \xi_i \right] - \sum_{i=1}^{m} \mu_i \xi_i \quad (\alpha_i \geq 0,\quad \mu_i \geq 0)
$$

**优化原始问题**：
$$
\min_{w, b, \xi} \: \max_{\substack{\alpha_i \geq 0 \\ \mu_i \geq 0}} L(w,b,\xi, \alpha, \mu)
$$

**对偶问题**：
$$
\max_{\substack{\alpha_i \geq 0 \\ \mu_i \geq 0}} \: \min_{w, b, \xi} L(w,b,\xi, \alpha, \mu)
$$

**对偶问题求解**：
$$
\begin{aligned}
\nabla_{w} L = 0 \quad &\Rightarrow \quad w = \sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)}\\[5pt]
\nabla_{b} L = 0 \quad &\Rightarrow \quad \sum_{i=1}^{m} \alpha_i y^{(i)} = 0\\[5pt]
\nabla_{\xi} L = 0 \quad &\Rightarrow \quad C - \alpha_i - \mu_i = 0
\end{aligned}
$$

**反代回拉格朗日函数**得：
$$
L = -\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} \langle x^{(i)} ,x^{(j)}\rangle + \sum_{i=1}^{m} \alpha_i
$$

其形式与硬分隔得一样，但约束条件发生变化：

$$
s.t. \quad
\begin{cases}
\sum\limits_{i=1}^{m} \alpha_i y^{(i)} = 0\\[8pt]
C - \alpha_i - \mu_i = 0\\[5pt]
\alpha_i \geq 0\\[5pt]
\mu_i \geq 0
\end{cases}
$$

由于新得目标函数中没有出现常数 $C$，将约束条件中的 $C$ 消掉得待优化函数：

$$
\alpha^{*} = \min_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} \langle x^{(i)} ,x^{(j)}\rangle - \sum_{i=1}^{m} \alpha_i\\[8pt]
s.t. \quad
\begin{cases}
\sum\limits_{i=1}^{m} \alpha_i y^{(i)} = 0\\[10pt]
0 \leq \alpha_i \leq C
\end{cases}
$$

这个仍可以用SMO来求解。

超平面的解与硬分隔SVM的一致。

<div style=" border: 1px solid #ddd; padding: 10px;">

**线性支持向量机小结**

1. 硬间隔SVM与软间隔SVM对比

| 特性                | 硬间隔SVM (Hard Margin SVM)                 | 软间隔SVM (Soft Margin SVM)                |
|---------------------|--------------------------------------------|--------------------------------------------|
| **适用场景**        | 数据严格线性可分                            | 数据近似线性可分/含噪声                    |
| **原始目标函数**    | $\min \dfrac{1}{2} \|\| w\|\|_{2}^{2} $         | $\min \dfrac{1}{2} \|\|w\|\|_{2}^{2} + C\sum\xi_i$ |
| **约束条件**        | $y_i(w^Tx_i + b) \geq 1$ | $y_i(w^Tx_i + b) \geq 1-\xi_i$ , $\xi_i \geq 0$ |
| **拉格朗日函数**    | $L = \dfrac{1}{2}\|\|w\|\|_{2}^{2} - \sum\alpha_i[y_i(w^Tx_i + b)-1]$ | $L = \dfrac{1}{2}\|\|w\|\|_{2}^{2} + C\sum\xi_i - \sum\alpha_i[y_i(w^Tx_i + b)-1+\xi_i] - \sum\mu_i\xi_i$ |
| **对偶问题目标**    | $\max \sum\alpha_i - \dfrac{1}{2}\sum\sum\alpha_i\alpha_jy_iy_jx_i^Tx_j$ | $\max \sum\alpha_i - \dfrac{1}{2}\sum\sum\alpha_i\alpha_jy_iy_jx_i^Tx_j$ |
| **对偶约束条件**    | $\alpha_i \geq 0$, $\sum\alpha_iy_i = 0$   | $0 \leq \alpha_i \leq C$, $\sum\alpha_iy_i = 0$ |
| **KKT条件**         | $\alpha_i[y_i(w^Tx_i + b)-1] = 0$ | $\alpha_i[y_i(w^Tx_i + b)-1+\xi_i] = 0$ , $\mu_i\xi_i = 0$ |
| **超平面解**        | $w^{*} = \sum\alpha_iy_ix_i$ <br> $b^{*} = y_i - w^Tx_i$ (对支持向量) | $w^{*} = \sum\alpha_iy_ix_i$ <br> $b^{*}$ 由$0 < \alpha_i < C$的支持向量确定 |
| **支持向量特性**    | 位于间隔边界上 ($y_i(w^Tx_i + b) = 1$) | 三种类型：<br> 1. 边界支持向量 ($0 < \alpha_i < C$)<br> 2. 非边界支持向量 ($\alpha_i = C$)<br> 3. 误分类支持向量 ($\xi_i > 0$) |
| **参数影响**        | 无调节参数                                 | 惩罚参数 $C$ 控制分类错误容忍度：<br> - $C \to \infty$：逼近硬间隔<br> - $C \to 0$：允许更多分类错误 |
| **几何解释**        | 寻找最大间隔超平面                         | 间隔最大化与分类错误最小化的权衡           |
| **主要优势**        | 理论最优解（当数据线性可分）               | 对噪声和异常值鲁棒，泛化性能更好           |
| **主要局限**        | 对非线性可分数据完全失效                   | $C$值需通过交叉验证调优                    |

2. 关键差异说明

- 松弛变量 $\xi_i$
    软间隔SVM引入松弛变量 $\xi_i \geq 0$ 量化分类错误程度：
    - $\xi_i = 0$：样本分类正确且在间隔外
    - $0 < \xi_i < 1$：样本分类正确但在间隔内
    - $\xi_i \geq 1$：样本被误分类

- 惩罚参数 $C$
  - **$C$ 的物理意义**：单位分类错误的惩罚权重
  - **调优方法**：通常通过网格搜索在 $[10^{-3}, 10^{3}]$ 对数空间寻找最优值
  - **平衡原理**：$ \frac{1}{C} $ 等价于正则化强度

- 支持向量类型（软间隔）
  
    | 类型                | $\alpha_i$ 范围       | $\xi_i$ 值     | 几何位置               |
    |---------------------|----------------------|---------------|-----------------------|
    | 边界支持向量        | $0 < \alpha_i < C$   | $\xi_i = 0$   | 恰在间隔边界上        |
    | 非边界支持向量      | $\alpha_i = C$       | $\xi_i = 0$   | 在间隔边界正确侧      |
    | 误分类支持向量      | $\alpha_i = C$       | $\xi_i > 0$   | 在间隔边界错误侧/误分类区 |

> **实践建议**：现实数据中严格线性可分的情况罕见，软间隔SVM（L1正则化）是实际应用中的标准选择。通过交叉验证选择适当的$C$值，可在模型复杂度和泛化能力之间取得平衡。

</div>

---

##### 非线性 SVM

对于线性不可分的问题，我们自然可以想到**升维**的思路，将线性不可分的问题转变为线性可分的问题。

- **初始角度**
    <br>
    对于线性的SVM来说，最优化问题为：
    $$
    \alpha^{*} = \min_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} \langle x^{(i)} ,x^{(j)}\rangle - \sum_{i=1}^{m} \alpha_i\\[8pt]
    s.t. \quad
    \begin{cases}
    \sum\limits_{i=1}^{m} \alpha_i y^{(i)} = 0\\[10pt]
    0 \leq \alpha_i \leq C
    \end{cases}
    $$

    我们可以想到利用 $\phi(x)$ 对训练集升维，最优化问题就转变为：
    $$
    \alpha^{*} = \min_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} \langle \phi(x^{(i)}) ,\ \phi(x^{(j)}) \rangle - \sum_{i=1}^{m} \alpha_i\\[8pt]
    s.t. \quad
    \begin{cases}
    \sum\limits_{i=1}^{m} \alpha_i y^{(i)} = 0\\[10pt]
    0 \leq \alpha_i \leq C
    \end{cases}
    $$

    但是粗暴的升维会给问题带来巨大的“**维度爆炸**”问题，时间空间消耗很可怕。

<br>

- **引入核函数**
    <br>    
    我们发现在SVM学习过程中，只需要求得  $\langle \phi(x^{(i)}) ,\ \phi(x^{(j)}) \rangle$  的结果，并不需要知道具体的 $\phi(x)$ 是什么。于是先驱们决定，跳过 $\phi(x)$ 直接定义 $\langle \phi(x^{(i)}) ,\ \phi(x^{(j)}) \rangle$ 的结果，这样既可以达到升维的效果，又可以避免维度爆炸的问题

    定义：

    $$
    K(x,z) = \phi(x) \phi(z)
    $$

    此时，对偶问题的目标函数变成了：

    $$
    \alpha^{*} = \min_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)} ,x^{(j)}) - \sum_{i=1}^{m} \alpha_i\\[8pt]
    s.t. \quad
    \begin{cases}
    \sum\limits_{i=1}^{m} \alpha_i y^{(i)} = 0\\[10pt]
    0 \leq \alpha_i \leq C
    \end{cases}
    $$

    判别函数（超平面）变成：

    $$
    y = \left(\sum\limits_{i=1}^{m} \alpha^{*}_{i} y^{(i)} K(x^{(i)},\  x) \right) + b^{*}
    $$

> **常用的核函数有**：
    线性核函数：
     $$
        K(x,z) = x \cdot z
     $$
    多项式核函数：
     $$
        K(x,z) = (\gamma x \cdot z + r)^d
     $$
    高斯核函数：
     $$
        K(x,z) = \exp{ \{-\gamma \|x - z \|^2 \}}
     $$
    sigmiod核函数：
     $$
     K(x,z) = \tanh{(\gamma x \cdot z + r)}
     $$

---

##### SVM算法流程总结

<div style=" border: 1px solid #ddd; padding: 10px;">

1. **选择某个核函数及其对应的超参数**
2. **选择惩罚系数$C$**
3. **构造最优化问题**：
    $$
    \alpha^{*} = \min_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)} ,x^{(j)}) - \sum_{i=1}^{m} \alpha_i\\[8pt]
    s.t. \quad
    \begin{cases}
    \sum\limits_{i=1}^{m} \alpha_i y^{(i)} = 0\\[10pt]
    0 \leq \alpha_i \leq C
    \end{cases}
    $$
4. **利用SMO算法求解出一组$\alpha^{*}$**
5. **利用求得的$\alpha^{*}$求得 $w^{*}$ 和 $b^{*}$**
    $$
    w^{*} = \sum_{i=1}^{m} \alpha^{*} y^{(i)} x^{(i)}
    $$

    $$
    b^{*} = \frac{1}{|S|} \sum_{i \in S} b_s^{*} = \frac{1}{|S|} \sum_{i \in S} \left(y_s - \sum_{i \in S} \alpha_i y^{(i)} (x^{(i)})^T x_s + b \right)
    $$
6. **学得的超平面和最终判别式**为：
    $$
    \sum\limits_{i=1}^{m} \alpha^{*}_{i} y^{(i)} K(x^{(i)},\  x) + b^{*} = 0
    $$

    $$
    f(x) = sign(\sum\limits_{i=1}^{m} \alpha^{*}_{i} y^{(i)} K(x^{(i)},\  x) + b^{*})
    $$

</div>

---

##### SVM概率化输出
标准SVM的无阈值输出为：

$$
f(x) = h(x) + b
$$

其中：

$$
h(x) = \sum_{i} y^{(i)} \alpha_i K(x^{(i)},x)
$$

Platt 利用 **Sigmoid-fitting** 的方法，将标准的SVM输出结果进行后处理，转换成后验概率：

$$
P(y=1 \mid f) = \frac{1}{1+ \exp(Af + B)}
$$

sigmoid-fitting 方法的优点在于保持SVM稀疏性的同时，可以良好的估计后验概率。

用极大似然估计来估计公式中的参数$A,B$。

定义训练集为$(f_i, t_i)$，其中$t_i$为目标概率输出值，将$y_i$缩放到$(0,1)$之间，有：

$$
t_i = \frac{y_i + 1}{2} \quad y_i \in {-1, 1}
$$

极小化训练集上的负对数似然函数：

$$
\min - \sum_i t_i \ln(p_i) + (1-t_i)\ln(1-p_i)
$$

其中：

$$
p_i = \frac{1}{1+\exp(A f_i + B)}
$$