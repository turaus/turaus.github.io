##  4. 神经网络

神经网络算法大部分归类为**有监督学习**，所以它处理的对象为回归或分类任务。

### 4.1 概述

人工神经网络(Artificial Neural Network， 简写为ANN)，也简称为神经网络(NN)，直接设计灵感来源于生物神经系统的基本原理，尤其是大脑中神经元之间的信息传递方式。

<img src="../source/imgs/4.1_neural_connection.jpg" alt="neural_connection" style="display: block; margin: 0 auto; max-width: 100%;" width=800>

在生物神经网络中，每个神经元与其他神经元相连，信号（电信号）由树突接收，并在细胞体中进行处理，若电位超过某一阈值，那么它就会被激活，传递“兴奋”给下一个神经元。

树突 $\quad \Rightarrow \quad$ 输入层（接收信号）

细胞体 $\quad \Rightarrow \quad$ 隐藏层（整合处理信号）

突触 $\quad \Rightarrow \quad$ 输出层（输出信号）

**单个神经元**（Neuron，或uint）结构示意图：

<img src="../source/imgs/4.1_neural_unit.jpg" alt="neural_unit" style="display: block; margin: 0 auto; max-width: 100%;" width=700>

将上述生物神经元处理信号的过程抽象出来，就得到沿用至今的“**M-P神经元模型**”，神经元接受来自 $n$ 个其他神经元的输入信号，这些信号通过**带权重的连接**（这个权重就是神经网络需要寻找的参数矩阵）进行传递，神经元接收到的总输入将与神经元的阈值进行比较，然后通过**激活函数**（Activation Function）处理以产生神经元的输出。

将多个神经元通过一定的层次连接起来，就构成了神经网络，下面是**神经网络**结构示意图：

<img src="../source/imgs/4.1_neural_networks_structure.jpg" alt="neural_networks_structure" style="display: block; margin: 0 auto; max-width: 100%;" width=800>

> 上图中所示结构是一种全连接（Full connected），它由以下两点特征：
>
> - 同一层的神经元之间没有连接
> - 第 $N$ 层的每个神经元和第 $N-1$ 层的所有神经元相连

---

### 4.2 激活函数

#### 4.2.1 激活函数的作用

<img src="../source/imgs/4.2.1_activation_function.jpg" alt="activation_function" style="display: block; margin: 0 auto; max-width: 100%;" width=700>

假如没有经过激活函数的处理，如上述简单的神经网络，不管中间经过了多少层，最终的结果一定可以用最开始的输入线性表示，这使得多层的神经网络失去意义，这样为何不采用线性回归的算法呢？其次，这样将无法处理非线性问题，引入激活函数就是引入**非线性变换**。

常见的激活函数类型有：

- 应对二分类问题的Sigmoid函数
- 应对多分类问题的Softmax函数
- 应对回归问题的恒等函数，即$f(x)=x$

下面将对各函数进行详细说明。

#### 4.2.2 Sigmoid函数

数学表达式为：
$$
f(x) = \text{sigmoid} (x) = \frac{1}{1+e^{-x}} = \frac{e^x}{1+e^x}\\
f^{'}(x) = \frac{1}{1+e^{-x}}(1-\frac{1}{1+e^{-x}}) = f(x)(1-f(x))
$$
<img src="../source/imgs/4.2.2_sigmoid_image.jpg" alt="sigmoid_image" style="display: block; margin: 0 auto; max-width: 100%;" width=700>

依据图像分析Sigmoid函数，Sigmoid函数具有以下明显特征：

- **函数是平滑过渡的 $S$ 型曲线，适于梯度下降法等优化算法**。
- 函数可以将任意的输入映射到 $(0,1)$ 之间，该特性使其天然适合表示概率（如二分类任务中输出概率），输出可直接解释为事件发生的可能性。
- 函数不是0中心的，均值不为0，会导致后续层的梯度更新出现一些问题。
- 导函数呈钟形曲线，最大值在 $x=0$​ 处，导数的最大值为0.25，向两侧逐渐衰减至0。反向传播时，中间区域的梯度较大，参数更新速度较快；但输入值远离0时（$\mid x \mid>5$），梯度接近0，称为梯度饱和现象，导致梯度消失，深层网络难以训练。
- 导数的取值不大，最大也只有1/4，深层次的神经网络在链式求导时，因多次乘以较小的数，进一步**加剧梯度消失**问题。

#### 4.2.3 tanh函数

双曲正切函数 $\tanh(x)$ 与sigmoid函数类似，但性质更好。

数学表达式为：
$$
f(x) = \tanh(x) = \dfrac{e^x - e^{-x}}{e^x + e^{-x}}\\
f^{'}(x) = 1- f^2(x)
$$
<img src="../source/imgs/4.2.3_tanh_image.jpg" alt="tanh_image" style="display: block; margin: 0 auto; max-width: 100%;" width=700>

tanh函数具有以下明显特征：

- 平滑过渡的 $S$ 型曲线,函数在整个实数域内连续且无限可导。
- 函数可以**将任意的输入映射到 $(-1,1)$ 之间，零中心化**。
- 与sigmoid函数相比，tanh导函数更陡，梯度饱和现象出现更早（大约在 $\mid x \mid > 3$处就出现）。
- tanh导函数能取到1，比sigmoid更适合训练。

#### 4.2.4 ReLU函数

数学表达式为：
$$
f(x) = \max (0,x)\\[5pt]
f^{'}(x) = 
\begin{cases}
1 \quad \quad if \ \  x>0\\[3pt]
0 \quad \quad if \ \  x \leq 0
\end{cases}
$$
ReLU函数在神经网络中的优势：

- 缓解梯度消失问题，ReLU在正值区域的梯度恒为1，避免了梯度消失，尤其适合深层网络。

- 计算高效，仅需判断输入是否大于0，计算速度远超sigmoid和tanh（无需指数运算）。

- 减轻过拟合，当输入为负时输出0，使得部分神经元“死亡”。但是如果神经元输出始终为0（例如初始化权重过小或学习率过高），梯度在反向传播时也为0，导致神经元“永久死亡”（Dead ReLU 问题）。

    > 为应对这个问题，可以使用 **Leaky ReLU 公式**：$f(x) = \max (\alpha x, x)$， 其中 $\alpha$ 是小的正数。可以在一种程度上规避”永久死亡的问题“

#### 4.2.5 Softmax函数

数学表达式为：
$$
\text{softmax}(z_j) = \dfrac{e^{z_j}}{\sum_{k=1}^{K}e^{z_k}}
$$
每个输出值在0到1之间，且所有输出之和为1，代表每个类别的概率，输出直观的概率解释。但类别过多时计算成本高，且指数计算可能不够高效。

详细内容见：[2_Linear_classification.md  2.2 Softmax回归解决多分类问题](./2_Linear_classification.md)

#### 4.2.6 小结

| 激活函数       | 公式                                                                 | 优点                                       | 缺点                                      | 适用场景                     |
|:--------------:|----------------------------------------------------------------------|--------------------------------------------|-------------------------------------------|------------------------------|
| **Sigmoid**    | $ \sigma(x) = \dfrac{1}{ 1+e^{-x} } $                                | 输出在$(0,1)$，适合概率输出                | 梯度消失、计算量大、非零中心化            | 二分类输出层、早期神经网络   |
| **Tanh**       | $ \tanh(x) = \dfrac{ e^x-e^{-x} }{ e^x+e^{-x} } $                      | 输出零中心化,映射到$(-1,1)$，梯度优化上比Sigmoid强 | 梯度消失问题仍存在                        | 隐藏层（较少用）             |
| **ReLU**       | $ \text{ReLU}(x) = \max(0,x) $                                     | 计算快、缓解梯度消失                        | Dead ReLU 问题   | 隐藏层（主流选择）           |
| **Leaky ReLU** | $ \text{LReLU}(x) = \max(\alpha x,x) $ | 缓解神经元死亡（$\alpha$常取0.01）         | 需调参 $\alpha$，效果不稳定              | 隐藏层（ReLU的改进版）       |
| **Parametric ReLU** | 类似Leaky ReLU，但α可学习                                        | 自适应负区间斜率                            | 增加计算复杂度                            | 深层网络（如ResNet）         |
| **Swish**      | $\text{Swish}(x)= x\cdot\sigma(\beta x)$，β可调             | 平滑、实验性能优于ReLU                      | 计算量稍大                                | 隐藏层（Google推荐）         |
| **GELU**       | $ \text{GELU}(x) = x\cdot\Phi(x) $，$\Phi$为标准正态CDF       | 结合随机正则化思想，适合预测模型            | 计算复杂                                  | Transformer<br>如BERT、GPT |
| **Softmax**    | $ \text{softmax}(x_i) = \dfrac{ e^{x_i} }{ \sum_{k=1}^{K}e^{x_j} }$ | 输出多分类概率分布                          | 仅适用于输出层                            | 多分类输出层                 |

激活函数的使用方法
1. 隐藏层激活函数

      - 默认选择：优先使用 ReLU（计算快、效果好）
      - 改进选择：若遇到神经元死亡（如输出全为0），改用 Leaky ReLU 或 ELU
      - 深层网络：可尝试 Swish 或 GELU（需更多计算资源）
2. 输出层激活函数

      - 二分类：Sigmoid（输出概率）

      - 多分类：Softmax（输出互斥概率）

      - 回归问题：线性激活（无激活函数）

3. 应用场景
    1. 计算机视觉（CNN）：ReLU（速度快）、Swish（高性能）
    2. 自然语言处理（RNN/Transformer）：GELU（BERT、GPT）、Tanh（早期LSTM）
    3. 强化学习：ReLU（稳定训练）
    4. 生成对抗网络（GAN）：Leaky ReLU（防止梯度消失）


---



### 4.3 初始化参数

所谓初始化参数，它的对象是神经网络的每个神经元连接的权重。

参数初始化是神经网络训练的起点，合适的初始化可以加速训练收敛，帮助模型找到更好的局部最优解，直接影响模型的效率。

**不正确初始化的权重会导致梯度消失或爆炸问题**，从而对训练过程产生负面影响。（对于梯度消失问题，权重更新很小，导致收敛速度变慢——这使得损失函数的优化变慢，在最坏的情况下，可能会阻止网络完全收敛。相反，使用过大的权重进行初始化可能会导致在前向传播或反向传播过程中梯度值爆炸。）

**初始化要求**

- 保持每层激活值的均值接近于0
- 保持每层激活值的方差保持稳定

**常用的初始化方法**

- 全零或等值初始化 	`nn.init.ones_()` `nn.init.zeros_()` `n.init.constant_()`

    由于初始化的值全都相同，每个神经元学到的东西也相同，将导致“对称性(Symmetry)”问题。$\Rightarrow$ 不推荐

- 从均匀分布（Uniform Distribution）中采样    `nn.init.uniform_() `

- 从正态分布（Normal Distribution）中采样      `nn.init.normal_()`

    均值为零，标准差设置一个小值。这样的做好的好处就是有相同的偏差，权重有正有负。比较合理。

- **Xavier/Glorot初始化**
    1. 采用均匀分布采样  $W \sim U(-\dfrac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}},\ \dfrac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}})$	`nn.init.xavier_uniform_()  `
        1. 采用正态分布采样  $W \sim N(0,\ \sqrt{\dfrac{2}{n_{in} + n_{out}}})$	`nn.init.xavier_normal_()`
- **kaiming/He初始化**
    1. 采用均匀分布采样  $W \sim U(-\dfrac{\sqrt{6}}{\sqrt{n_{in} }},\ \dfrac{\sqrt{6}}{\sqrt{n_{in} }})$	`nn.init.kaiming_uniform_()`
    2. 采用正态分布采样  $W \sim N(0,\ \sqrt{\dfrac{2}{n_{in}}})$	`nn.init.kaiming_normal_()`

[Example: 4.2_parameter_initialize.ipynb](../source/py/4.2_parameter_initialize.ipynb)

构建方法:

- 所有自定义的神经网络模型必须继承自 `torch.nn.Module` ，这是 PyTorch 的基类.
- 网络层的定义在`__init__.py` 中(如全连接层 `nn.Linear `、卷积层` nn.Conv2d` 等)
- 前向传播的逻辑在` forward `中实现，当调用模型实例时（如 `model(input_data) `），PyTorch 会自动调用 `forward `方
    法

### 4.4 损失函数

由于神经网络的任务有两类——分类和回归，对应的损失函数也有两类，分类任务用交叉熵损失（Cross-Entropy Loss），回归任务用均方误差（MSE）或平均绝对误差（MAE）来衡量。



#### 4.4.1 交叉熵损失函数

**二分类交叉熵损失函数** (Binary Cross-Entropy Loss)，也称为对数损失 (Log Loss)，用于衡量模型预测概率与真实标签之间的差异。

损失函数为：
$$
\ell = \sum_{i=1}^{m} \left(y^{(i)} \ln \hat{y}^{(i)} \right) + (1-y^{(i)}) \ln \left( 1-\hat{y}^{(i)} \right)
$$
多分类任务通常使用softmax将scores转换为概率的形式，所以多分类的交叉熵损失也叫做 **softmax 损失**，它的计算方法是：
$$
\ell = -\sum_{i=1}^{m} y^{(i)} \ln \left( \text{Softmax}(f(x^{(i)})) \right)
$$

#### 4.4.2 平均绝对误差损失函数

$$
\text{MAE} = \dfrac{1}{m} \sum_{i=1}^{m} \mid y^{(i)} - f(x^{(i)}) \mid
$$

这里的绝对值运算对应于L1范数，因此MAE也被称为 L1  LOSS。

> 使用场景：
>
> 适用于**需要鲁棒性较强的回归任务**，尤其是数据存在异常值时（如金融风控）

#### 4.4.3 均方误差损失函数

$$
\text{MSE} = \dfrac{1}{m} \sum_{i=1}^{m} \left(y^{(i)} - f(x^{(i)}) \right)
$$

> 使用场景：
> 适用于**对异常值不敏感且需要平滑梯度优化的场景**，如预测连续值（房价、温度等）

详细内容见：[1_Linear_regression.md  1.1.2 误差与损失函数](./1_Linear_regression.md)

#### 4.4.4 Smooth L1

Smooth L1，是L1损失的平滑版本，公式为：
$$
\ell = 
\begin{cases}
0.5(y^{(i)} - f(x^{(i)}))^2 \ , &\quad if \mid x \mid < 1\\[5pt]
\mid y^{(i)} - f(x^{(i)}) \mid - 0.5 \ , &\quad otherwise
\end{cases}
$$

>使用场景：
>常用于目标检测（如Faster R-CNN），解决MAE在零点不平滑的问题

<img src="../source/imgs/4.4.4_MSE_MAE_SMOOTH.jpg" alt="MSE_MAE_SMOOTH" style="display: block; margin: 0 auto; max-width: 100%;">

Smooth L1在误差较小时使用平方项，保证梯度平滑；误差较大时使用线性项，避免梯度爆炸。计算效率高，适合大规模数据。但是对超参数（如分段阈值）敏感，需根据任务调整。

---

### 4.5 优化算法

#### 4.5.1 梯度下降法

基本公式：
$$
 W_{j}^{t+1} = W_{j}^{t} - \eta \cdot g_{j} 
$$
详细内容见：[1_Linear_regression.md  1.2.1 梯度下降法](./1_Linear_regression.md)



在神经网络算法中，梯度的计算采用“**反向传播**”的思想。

<img src="../source/imgs/4.5.1_back_propagation.jpg" alt="back_propagation" style="display: block; margin: 0 auto; max-width: 100%;" width=700>

- 前向传播与反向传播的关系

    反向传播依赖于前向传播（Forward Propagation）。在训练过程中：

    **前向传播**是输入数据通过网络的每一层，计算每一层的输出，最终得到网络的预测结果和损失。

    **反向传播**是根据损失，从输出层开始，逐层向输入层传播误差，计算每个参数的梯度(对损失的贡献), 然后根据这些梯度调整参数，使得损失逐步减小。

    简单来说，**前向传播是“从输入到输出”，反向传播是“从输出到输入”**。

    
    
    **反向传播的核心思想基于链式法则，它允许我们在复杂的多层网络中高效地计算梯度，而不需要显式地对每个参数求偏导数**。

以下面这个例子来说明反向传播的思想。

<img src="../source/imgs/4.5.1_back_propagation_application.jpg" alt="back_propagation_application" style="display: block; margin: 0 auto; max-width: 100%;" width=700>

该模型输入 $x = 2$，权重 $w = 5$，偏置 $b = 0.1$，真实值 $y = 1$

---

- 前向传播：

    线性加权和：
    $$
    \hat{y} = wx + b = 1.1
    $$
    采用 Sigmoid 激活函数，输出为：
    $$
    a = \sigma (\hat{y}) = \dfrac{1}{1+e^{-\hat{y}}} \approx 0.75
    $$
    

​	以均方误差来衡量模型好坏（作为损失函数）：
$$
\ell = (a - y)^2 = 0.0625
$$
​	前向传播从输入到输出，得到模型的损失值，该值越小越好，但是很难通过这种方法来优化，因为每一层网络都对结果有贡献，	最终的表达式（其实很难得到这个表达式）很难求梯度，所以很难优化。

---

- 反向传播

    反向传播的核心是通过链式法则逐层计算损失函数 $\ell$ 对模型参数（权重 $w$ 和偏置 $b$）的梯度，从而**更新参数**以最小化损失。

    损失函数是均方误差，其对 $a$ 的导数为：
    $$
    \left. \frac{\partial \ell}{\partial a} \right|_{a=0.75} = \left. 2(a - y) \right|_{a=0.75} = -0.5
    $$
    $a$  对 $\hat{y}$ 求导：
    $$
    \left. \frac{\partial a}{\partial \hat{y}} \right|_{\hat{y} = 1.1} 
    = \left. a \cdot (1-a) \right|_{a=0.75} = 0.1875
    $$

    >这里特殊点在于Sigmoid函数的导数  **$\sigma ' = \sigma \cdot(1 - \sigma)$** 。
    
     $\hat{y}$ 对权重 $w$ 求导：
    $$
    \frac{\partial \hat{y}}{\partial w} = x = 2
    $$
    由链式求导法则，得损失函数 $\ell$ 对权 $w$ 得导数为：
    $$
    \dfrac{\partial \ell}{\partial w} = \dfrac{\partial \ell}{\partial a} \cdot \dfrac{\partial a}{\partial \hat{y}} \cdot \dfrac{\partial \hat{y}}{\partial w} = -0.1875
    $$
    链式法则将复杂的导数分解为简单的部分导数相乘，逐层传递梯度。
    
    **负梯度表示需要增加权重 $w$ 以减少损失。**
    
    用梯度下降更新权重，取学习率 $\eta = 0.1$，：
    $$
    w^{new} = w - \eta \cdot \dfrac{\partial \ell}{\partial w} = 0.51875
    $$
    同理得到新得偏置：
    $$
    b^{new} = 0.109375
    $$
    再次从前向传播得方向看，有：
    $$
    \hat{y}^{new} \approx 1.1469\\[3pt]
    a^{new} \approx 0.757\\[3pt]
    \ell^{new} \approx 0.0576
    $$
    $\ell$ 降低，说明更新的方向正确。
    
    ---

>在使用梯度下降算法中，可能会碰到以下情况：
>
>- 平缓区域，梯度值较小，参数优化缓慢
>- “鞍点” ，梯度为 0，参数无法优化
>- 局部最小值
>
>根据这些问题，出了一些对梯度下降算法的优化方法，比如：动量法（Momentum）、Adagrad、RMSprop、Adam等

---

#### 4.5.2 动量优化算法

动量法（Momentum）的灵感来源于物理学中的动量概念。当一个物体运动时，其动量会使其保持原有方向的趋势，从而平滑运动轨迹。在优化中，动量通过累积历史梯度的指数加权平均，使参数更新方向更加一致，从而：

- 加速收敛：在梯度方向一致时累积动量，增大更新步长。

- 减少震荡：在梯度方向变化时，动量抵消部分震荡，使更新更平滑。

在介绍动量法之前，先介绍“**指数加权平均**”这个概念。

<div style="width: 100%; border: 1px solid #ddd; padding: 10px;">

指数加权平均（Exponentially Weighted Moving Average，简称 EWMA）是一种对时间序列数据进行平滑处理的方法，其核心特点是**赋予近期数据更高的权重**，**而历史数据的权重随时间呈指数级衰减**。


数学表达为：
$$
V_t
= \begin{cases}
\theta_0 \  &\quad t = 0\\[3pt]
\beta \cdot V_{t-1} + (1- \beta) \cdot \theta_t \  &\quad t>0
\end{cases}
$$

>- $V_t$：当前时刻的指数加权平均值
>- $V_{t-1}$：前一刻的指数加权平均值
>- $\theta_t$：当前时刻的实际数据
>- $\beta$：衰减因子（0到1之间），决定历史数据的权重下降速度
>
>    有人把 $\beta$ 形象的称为“摩擦系数”，能直观体现 $\beta$ 取不同值时的效果
>
>    - 为 **0** 时，退化为未优化前的梯度更新
>
>    - 为 **1** 时， 表示完全没有摩擦，这样会存在大的问题
>
>    - 取 **0.9** 是一个较好的选择。可能是 **0.9** 的 **60** 次方约等于 **0.001**，相当仅考虑最近的60轮迭代所产生的的梯度，这个数值看起来相对适中合理

下面这个例子显示 EWMA 的效果          	[指数加权平均 作者：LiuHDme  from CSDN]([指数加权平均-CSDN博客](https://blog.csdn.net/LiuHDme/article/details/104744836))

<img src="../source/imgs/4.5.2_EWMA.png" alt="EWMA" style="display: block; margin: 0 auto; max-width: 100%;" width=700>

蓝点：采样

红线：$\beta = 0.9$ 时EWMA结果

绿线：$\beta = 0.98$ 时EWMA结果

</div>



回到动量法，它的数学表达为：
$$
v_t = \beta \cdot v_{t-1} + \eta \cdot \nabla J(W_t)\\[5pt]
W_{t+1} = W_t - v_t
$$

>- **动量项 $v_t$**：由历史动量 $\beta \cdot v_{t-1}$ 和当前梯度 $\eta \cdot \nabla W_t$ 加权求和得到，反映了**梯度方向的持续性和衰减性**
>- **参数更新 $W_{t+1}$**：参数基于动量项调整，而非直接使用当前梯度。这种设计能平滑震荡、加速收敛

**动量法的核心优势**：

1. 平滑震荡：动量项通过指数衰减平均**历史**梯度，减少参数更新方向的突变。
2. 加速收敛：在梯度方向一致的区域（如平坦区域），动量项逐步**累积**，推动参数快速接近极值点。
3. 逃离局部极值/鞍点：即使当前梯度趋近于0，**历史**动量仍能维持更新方向。

详细内容见：[深度学习中的Momentum算法原理-CSDN博客](https://blog.csdn.net/gaoxueyi551/article/details/105238182)

---

#### 4.5.3 自适应学习率优化算法

自适应学习率优化算法（Adaptive Gradient Algorithm，简称 AdaGrad）是一种用于优化神经网络和其他机器学习模型的梯度下降算法，其核心特点是**能够自适应地调整学习率**。

AdaGrad 的核心思想是**根据参数的历史梯度信息自适应地调整学习率**。具体来说，对于那些经常更新的参数（梯度较大的参数），学习率会逐渐减小，以避免震荡或过大的步长；而对于更新较少的参数（梯度较小的参数），学习率会相对较大，以加速收敛。

传统梯度下降算法使用固定的学习率 $\eta$ 来更新所有参数。AdaGrad 通过引入一个基于历史梯度的缩放因子，动态调整每个参数的学习率，从而使更新步长更符合数据的特性。

数学表达为：
$$
\theta_{t+1} = \theta_t - \dfrac{\eta}{\sqrt{s_t + \epsilon}}\cdot g_t
$$

>1.  $g_t$ 为损失函数的梯度：
>    $$
>    g_t = \nabla J(\theta_t)
>    $$
>
>2. $s_t$ 为 Adagrad 维护的一个累积变量，记录每个参数的历史梯度平方和：
>    $$
>    s_t = s_{t-1} + g_t^2
>    $$
>    这里，$g_t^2$ 表示梯度的**逐元素**平方。对于多维参数，$s_t$ 是一个向量或矩阵，与 $\theta$ 的维度相同。
>
>3. $\sqrt{s_t + \epsilon}$ 称为缩放因子，其中 $\epsilon$ 为一个极小的的常数（一般取 $10^{-8}$），用来防止除零。$\dfrac{\eta}{\sqrt{s_t + \epsilon}}$ 可以看作每个参数的有效学习率，它会随着 $s_t$ 的增大而减小。

**AdaGrad 优缺点：**

- AdaGrad 具有以下显著优点：
    1. **自适应学习率**：无需手动调整学习率，算法自动根据梯度历史调整每个参数的学习率。
    2. **适合稀疏数据**：对于稀疏特征或梯度较小的参数，Adagrad能加速收敛，非常适合NLP、推荐系统等场景。
    3. **简单易实现**：算法逻辑简单，计算开销较低，适合快速原型开发。
    4. **对初始学习率不敏感**：由于学习率会自适应调整，初始学习率 的选择对结果影响较小。
- AdaGrad 的缺点
    1. **学习率单调递减**：有效学习率持续减小，在训练后期，学习率可能变得极小，导致模型停止学习（**早停问题**）。
    2. **对非凸问题表现不佳**：在非凸优化问题（如深度神经网络）中，AdaGrad可能过早收敛到次优解，因为学习率衰减过快。
    3. **内存需求**：Adagrad 需要存储每个参数的历史梯度平方和 $s_t$ ，对于高维模型，这会增加内存开销。
    4. **不适合所有任务**：对于密集数据或需要长时间训练的任务，AdaGrad的表现可能不如其他算法（如RMSProp或Adam）

---

#### 4.5.4 均方根传播优化算法

均方根传播（Root Mean Square Propagation，简称 RMSProp）优化算法是一种广泛应用于深度学习的优化算法，**旨在通过自适应地调整学习率来加速梯度下降的收敛**。

RMSProp的核心思想是**通过计算梯度的指数平均来动态调整每个参数的学习率**（EWMA + AdaGrad），从而平滑梯度更新，减少振荡，并加速收敛。与 AdaGrad 不同，RMSProp 不累积所有历史梯度的平方，而是使用一个衰减的移动平均，关注近期梯度信息，避免学习率过早或过度缩小。

数学表达为：
$$
\theta_{t+1} = \theta_t - \dfrac{\eta}{\sqrt{v_t + \epsilon}}\cdot g_t
$$

>$v_t$ 为使用指数移动平均更新梯度平方的估计：
>$$
>v_t = \beta \  v_{t-1} + (1-\beta)g_t^2
>$$

**RMSProp 的优缺点：**

- RMSProp 的优点
    1. **自适应学习率**
    2. **快速收敛**：通过关注近期梯度，RMSProp 避免了 AdaGrad 学习率过快衰减的问题，适合非凸优化问题（如深度神经网络）。RMSProp 在许多任务中比 SGD 和 AdaGrad 收敛更快。
- RMSProp 的缺点
    1. **缺乏理论支持**：RMSProp 是一种启发式算法，未在学术论文中正式发表，缺乏严格的数学证明支持其收敛性。其性能依赖于经验调参和具体问题。
    2. **超参数敏感性**：虽然超参数较少，但学习率 $\eta$ 和衰减率 $\beta$ 的选择仍可能显著影响性能，需要针对具体任务调整。
    3. **非通用的最优解**：RMSProp 并非所有优化问题的理想选择。例如，在某些凸优化问题中，AdaGrad 可能更适合；在通用深度学习任务中，Adam 通常优于 RMSProp。
    4. **局部最小值风险**：尽管 RMSProp 擅长处理非凸问题，但仍可能陷入次优的局部最小值或鞍点，尤其在高维空间中。

[Example: 4.5.4_RMSProp.ipynb](../source/py/4.5.4_RMSProp.ipynb)

---

#### 4.5.5 自适应矩估计优化算法

Adam（Adaptive Moment Estimation，自适应矩估计）是一种在深度学习中广泛使用的优化算法。Momentum 善于处理梯度的方向和大小，而 RMSProp 善于调整学习率以应对数据的稀疏性。而 Adam 结合了 Momentum 和 RMSProp 两种优化算法的优点，同时减少它们的缺点，提供一种更加鲁棒的优化解决方案。

- Momentum：通过累积历史梯度的指数移动平均（一阶矩），加速梯度下降，类似于“惯性”
- RMSprop：通过梯度平方的指数移动平均（二阶矩），自适应地调整学习率，适应不同参数的梯度尺度



Adam的核心思想是**维护两个指数移动平均量**：

1. **一阶矩**：记录梯度的方向和大小，类似于动量法的均值
2. **二阶矩**：记录梯度的尺度，类似于RMSprop，用于自适应学习率

数学表达为：
$$
\theta_t = \theta_{t-1} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

> 1. 一阶矩（动量）：通过指数加权平均累积历史梯度，加速平坦方向的收敛
>     $$
>     m_t = \beta_1 \ m_{t-1}\  + (1-\beta_1)g_t
>     $$
>
> 2. 二阶矩（方差）： 捕捉梯度幅度的变化，调整学习率以应对不同参数的特性
>     $$
>     v_t = \beta_2 \ v_{t-1} + (1-\beta_2)g_t^2
>     $$
>
> 3. 偏差修正机制：
>
>     由于 $m_0 = 0$ 和 $v_0 = 0$ ，所以在训练初始阶段，$m_t$ 与 $v_t$ 会严重偏向于 0，尤其是在 $\beta_1$ 和$\beta_2$ 接近 1 时，这会导致 $m_t$ 无法准确反映梯度的真实均值，$v_t$ 无法准确反映梯度平方的真实均值，进而导致更新步长（学习率调整后的梯度）不准确，优化过程可能不稳定或收敛缓慢。所以需要一个偏差修正的机制，如下：
>     $$
>     \hat{m}_t = \dfrac{m_t}{1-\beta_1^t}\\[3pt]
>     \hat{v}_t = \dfrac{v_t}{1-\beta_2^t}\\
>     $$

**Adam 的优缺点：**

- Adam 的优点
    1. **自适应学习率**：尤其适合高维参数空间。
    2. **高效收敛**：在图像分类、自然语言处理等任务中表现出快速收敛性。
    3. **鲁棒性强**：对初始学习率不敏感，且能处理噪声数据和稀疏梯度。
- Adam 的缺点
    1. **局部最优风险**：某些任务中可能过早收敛至次优解，尤其在极大规模数据集上不如带动量的SGD
    2. **超参数调整**：尽管默认参数（$\beta_1 = 0.9, \ \beta_2 = 0.999$）适用多数场景，但特定问题仍需微调

[Example: 4.5.5_Adam.ipynb](../source/py/4.5.5_Adam.ipynb)

---

#### 4.5.6 小结

1. **动量法 (Momentum Method)**

    核心思想
    在每次更新中，将当前梯度与历史梯度的加权平均（动量项）结合，减少震荡并加速向目标方向移动。类似物理中的"球滚下山坡"，动量帮助模型"记住"之前的运动方向。

    数学表达
    $$
    v_t = \beta \cdot v_{t-1} + \eta \cdot \nabla W_t \\[5pt]
    W_{t+1} = W_t - v_t
    $$

2. **Adagrad (Adaptive Gradient Algorithm)**

    核心思想
    根据参数的梯度历史自适应地调整学习率，特别适合处理稀疏数据或凸优化问题。

    数学表达
    $$
    \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot g_t
    $$

3. **RMSProp (Root Mean Square Propagation)**

    核心思想
    Adagrad的改进版本，通过引入指数移动平均代替梯度平方和的累积，限制历史梯度的影响范围。

    数学表达
    $$
    v_t = \beta v_{t-1} + (1 - \beta) g_t^2 \\[5pt]
    \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t
    $$

4. **Adam (Adaptive Moment Estimation)**

    核心思想
    结合动量法和RMSProp的优点，用梯度的指数移动平均（一阶动量）加速梯度更新，用梯度平方的指数移动平均（二阶动量）自适应调整学习率。

    数学表达
    $$
    m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\[3pt]
    v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\[5pt]
    \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \\
    \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
    \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t + \epsilon}} \hat{m}_t
    $$

---

优化算法对比总结

| 算法 | 核心思想 | 优点 | 缺点 | 适用场景 |
|:----:|:--------:|------|------|----------|
| **Momentum** | 累积历史梯度方向，加速收敛 | 加速收敛、减少震荡、实现简单 | 超参数敏感、可能超调、非凸问题效果有限 | 平坦或狭长谷的损失函数 |
| **Adagrad** | 自适应学习率，基于梯度平方和调整 | 自适应、无需调学习率、适合稀疏数据 | 学习率衰减过快、非凸问题效果差、内存需求大 | 稀疏数据、凸优化问题 |
| **RMSProp** | 用指数移动平均替代累积，改进Adagrad | 学习率衰减合理、适合非凸问题、计算高效 | 仍需调参、对噪声敏感、可能陷入局部极值 | 深度学习、非凸优化、时间序列 |
| **Adam** | 结合动量法和RMSProp，跟踪一阶和二阶动量 | 收敛快、鲁棒性强、默认参数通用、适合复杂模型 | 理论收敛性争议、内存需求高、特定任务可能不如SGD | 大多数深度学习任务、复杂非凸优化 |

选择建议

1. Momentum：适合简单模型或初步实验，需手动调参

2. Adagrad：适合稀疏数据和凸优化，但在深度学习中较少单独使用

3. RMSProp：适合大多数深度学习任务，参数调整相对简单

4. Adam：**默认推荐算法，适合大多数场景，尤其是复杂模型和大规模数据**。如果 Adam 表现不佳，可尝试 RMSProp 或SGD + Momentum 

---

### 4.6 学习率衰减

**学习率衰减**（Learning Rate Decay）是深度学习优化中的一种重要技术，用于在训练过程中逐渐降低学习率，以帮助模型更好地收敛到损失函数的全局或局部最优解。

学习率衰减的核心思想是：

- 初期高学习率：训练初期，模型参数远离最优解，较**大**的学习率可以加速梯度下降，快速逼近较优区域。

- 后期低学习率：随着训练进行，模型逐渐接近最优解，较**小**的学习率可以避免大幅震荡，精细调整参数以稳定收敛。

几种常见的衰减策略有以下三种：固定步长衰减（Step Decay）、指定间隔学习率衰减、指数衰减（Exponential Decay）和余弦衰减（Cosine Annealing）。

1. **固定步长衰减**
    $$
    \eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}
    $$

    - $\gamma$ ：衰减因子（通常取 0.1 或 0.5）
    - $s$ ：衰减步长
    - $\lfloor \cdot \rfloor$ ：向下取整

2. **指定间隔学习率衰减**

    指定间隔学习率衰减可以理解为固定步长变体。

    预定义的训练轮数（epoch）或步数（称为“里程碑”，milestones）处，将学习率乘以一个衰减因子 $\gamma$ 。

    与普通的固定步长不同，允许用户指定多个不均匀的里程碑点，在这些点上进行学习率衰减。

3. **指数衰减**
    $$
    \eta_t = \eta_0 \cdot \gamma^t
    $$

4. **余弦衰减**

    学习率按照余弦函数周期性变化，从初始值逐渐下降到一个最小值（可以是 0 或自定义值）。
    $$
    \eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min}) \cdot \left(1+\cos (\frac{t}{T}\pi) \right)
    $$

---

### 4.7 正则化

**正则化**（Regularization）是深度学习中用于防止模型过拟合的重要技术。

正则化的核心目标是通过在模型训练过程中引入约束或惩罚，降低模型复杂度，从而提升模型的泛化能力。

常见方法为：

- **显式正则化**：直接在损失函数中添加惩罚项（如 L1 / L2 正则化）
- **隐式正则化**：通过修改训练过程间接约束模型（如数据增强、早停法、随机失活、批标准化等）

下面主要介绍隐式正则化

1. **数据增强**：通过对训练数据进行随机变换，增加数据的多样性，从而提高模型对不同输入的鲁棒性和泛化能力

    - 图像：随机翻转、旋转、缩放、裁剪、颜色抖动

    - 文本：同义词替换、随机插入/删除、回译

    - 语音：添加噪声、改变音调或速度

2. **提前停止**：在训练过程中监控验证集上的性能（如损失或准确率），当验证集性能在若干轮内不再提升时，停止训练以避免过拟合

3. **随机失活**（inverted Dropout）：在训练过程中，随机以一定概率 $p$（通常为 0.2~0.5）丢弃神经网络中的部分神经元及其连接，使得网络在每次前向传播时都使用不同的子网络结构从，而减少神经元之间的共适应性，增强模型的泛化能力
    $$
    h_i^{'} = \begin{cases}
    0 \ , &\quad \text{以概率}p\\
    \dfrac{h_i}{1-p}\ , &\quad \text{以概率}1-p
    \end{cases}
    $$
    训练时，每个神经元的输出 $h_i$ 除以概率 $p$ 保留，以概率 $1-p$ 置零；推理时，输出直接使用 $h_i$ ，无需随机丢弃。

    Dropout 广泛用于全连接层，常用于较深的网络，如 CNN 和 Transformer 。

4. **批标准化**（Batch Normalization）：在每一层的输入进行标准化（均值为 0，方差为 1），然后进行线性变换，以减少内部协变量偏移，从而稳定训练过程。BatchNorm 还具有一定的正则化效果，因为批次内的噪声可以看作一种随机性。

### 4.8 模型构建

- 所有自定义的神经网络模型必须继承自 `torch.nn.Module`，这是 PyTorch 的基类
- 网络层的定义在`__init__.py`中(如全连接层 `nn.Linear`、卷积层 `nn.Conv2d` 等)
- 前向传播的逻辑在 `forward` 中实现，当调用模型实例时（如 `model(input_data)`），PyTorch 会自动调用 `forward` 方法



参数（参数是模型在训练过程中自动学习的变量）：主要包括 **权重**（weights）和**偏置**（biases）

超参数

1. 网络结构相关：隐藏层数量，每层神经元数量，激活函数的选择

1. 训练过程相关：学习率（learning rate），批量大小（batch size），训练轮数（epochs），优化器选择（optimizer），损失函数选择（loss function)



```python
import torch

class CustomModel(torch.nn.Module):

  	def __init__(self):
        super().__init__() # 调用父类的初始化方法
        self.linear1 = torch.nn.Linear(3, 4)
        self.linear2 = torch.nn.Linear(4, 3)
        self.output = torch.nn.Linear(3, 2)
    
  	def forward(self, x):
        x = self.linear1(x) 
        x = torch.relu(x)  # 使用 ReLU 激活函数
        x = self.linear2(x)
        x = torch.relu(x)  # 使用 ReLU 激活函数
        x = self.output(x) # 输出层
        torch.softmax(x,dim=1) # 使用 Softmax 激活函数
        return x
# 构建数据
x = torch.randn(10,3)
# 构建模型
model = CustomModel()
# 前向传播
output = model(x)
# 打印输出
print(output.shape)

```

