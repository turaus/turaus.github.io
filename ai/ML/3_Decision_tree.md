## 3. 决策树

- 阶段概述：本阶段讲解，**决策树算法**、**随机森林算法**、**Adaboost算法**、**GBDT算法**、**XGBoost算法**。

- 达成目标：通过本阶段学习，掌握非线性决策树系列算法，重点掌握Kaggle神奇XGBoost算法，理解GBDT和XGBoost涉及的公式推导，本阶段的掌握将大大提升学员数据挖掘的能力，对于后续理解Kaggle实战阶段内容会起到很大的帮助。

### 3.1 概述

决策树是属于有监督机器学习的一种，起源非常早，符合直觉并且非常直观，模仿人类做决策的过程，早期人工智能模型中有很多应用，现在更多的是使用基于决策树的一些集成学习的算法。这章我们把决策树算法理解透彻非常有利于后面去学习集成学习。

- 阶段概述：本阶段讲解，**决策树算法**、**随机森林算法**、**Adaboost算法**、**GBDT算法**、**XGBoost算法**。

- 达成目标：通过本阶段学习，掌握非线性决策树系列算法，重点掌握Kaggle神奇XGBoost算法，理解GBDT和XGBoost涉及的公式推导，本阶段的掌握将大大提升学员数据挖掘的能力，对于后续理解Kaggle实战阶段内容会起到很大的帮助。

- 模型特点：
    - 可以处理非线性的问题
    - 可解释性强，没有比较抽象的系数 $\theta$
    - 模型简单，模型预测效率高  $\Rightarrow$  `if... else...`
    - 不容易显示的使用函数表达，不可微

### 3.2 决策树模型基础

#### 3.2.1 决策树基础结构
决策树是一种基于树结构的监督学习模型，由以下数学元素构成：

- **树结构**：$T = (V, E)$
  - $V$：节点集合（包括根节点、内部节点、叶节点）
  - $E$：有向边集合（表示特征分割路径）

- **节点类型**：
  - 根节点：$v_{\text{root}}$（包含全部样本）
  - 内部节点：$v_j$（应用分裂规则）
  - 叶节点：$v_{\text{leaf}}$（输出预测结果）

#### 3.2.2 数学表达

- **整体方式表达**：
  
    $$
    G(\mathbf{x}) = \sum_{t=1}^T q_t(\mathbf{x}) \cdot g_t(\mathbf{x})
    $$

    | 符号 | 含义 | 数学描述 |
    |------|------|----------|
    | $T$ | 叶节点总数 | 等于路径数量 |
    | $q_t(\mathbf{x})$ | 路径指示函数 | $\mathbb{I}[\mathbf{x} \text{ 属于第 } t \text{ 条路径}]$ |
    | $g_t(\mathbf{x})$ | 叶节点预测值 | 常数函数 $g_t(\mathbf{x}) = c_t$ |
    | $c_t$ | 叶节点输出值 | 分类：多数类概率<br>回归：样本均值 |

    >**路径指示函数 $q_t(\mathbf{x})$**
    >定义样本$\mathbf{x}$是否通过路径$t$到达叶节点：$q_t(\mathbf{x}) = \prod\limits_{k=1}^{d_t} \mathbb{I}[\text{condition}_k(\mathbf{x})]$
    >其中：
    >$\quad d_t$：路径$t$的深度
    >$\quad \text{condition}_k$：路径上第$k$个节点的分裂条件
    >
    >
    >
    >**叶节点值 $g_t(\mathbf{x})$**
    >对于到达叶节点$t$的所有样本：
    >$$
    >g_t(\mathbf{x}) = \begin{cases} 
  >\underset{k}{\arg \max}\  p_k^{(t)} & \text{(分类)} \\[10pt]
  >\dfrac{1}{|S_t|}\sum\limits_{i\in S_t} y_i & \text{(回归)}
  >\end{cases}
  >$$
  
- **迭代方式表达**：
    $$
    G(\mathbf{x}) = \sum_{c=1}^C \mathbb{I}[b(\mathbf{x}) = c] \cdot G_c(\mathbf{x})
    $$

    | 符号 | 含义 | 数学描述 |
    |:----:|------|----------|
    | $G(\mathbf{x})$ | 整棵树的预测函数 | $G: \mathbb{R}^d \to \mathcal{Y}$ |
    | $b(\mathbf{x})$ | 分支判断函数 | $b: \mathbb{R}^d \to \{1,2,\dots,C\}$ |
    | $G_c(\mathbf{x})$ | 第$c$分支的子树预测 | $G_c: \mathbb{R}^d \to \mathcal{Y}$ |
    | $\mathbb{I}[\cdot]$ | 指示函数 | $\mathbb{I}[\text{true}]=1$ ，$\mathbb{I}[\text{false}]=0$ |
    | $C$ | 分支数量 | 由特征空间划分决定 |

    从迭代的角度，可以构建决策树的基本模型，以伪代码的形式给出：

    ```python
    def DecisionTree(data D = {(x_n, y_n)}_{n=1}^N):
    if termination_criteria_met(D):          # 终止条件检查
        return base_hypothesis(D)            # 返回叶节点预测
    
    else:
        # 1. 学习分支判断函数
        b = learn_branching_criteria(D)      
        
        # 2. 根据分支条件划分数据集
        D_c = { (x_n, y_n) ∈ D | b(x_n) = c } for c=1..C
        
        # 3. 递归构建子树
        G_c = DecisionTree(D_c) for each c
        
        # 4. 返回完整决策树
        return G(x) = sum_{c=1}^C I[b(x)=c] * G_c(x)
    ```

    <div style="width: 100%; border: 1px solid #ddd; padding: 10px;">

    算法关键组件:

    - 分支数量选择 (Number of Branches)
        | 类型 | 分支数$C$ | 特点 | 适用场景 |
        |------|-----------|------|----------|
        | **二叉树** | 2 | 每次分裂产生两个子节点 $\Rightarrow$ 结构简单，计算高效 | 最常用，适用于所有特征类型 $\Rightarrow$（数值型/类别型） |
        | **多叉树** | >2 | 一次分裂产生多个子节点 $\Rightarrow$ 减少树深度 | 类别型特征（如one-hot编码）$\Rightarrow$ 离散特征取值较多时 |
        | **混合树** | 可变 | 不同节点分支数不同 $\Rightarrow$ 自适应数据分布 | 复杂数据分布 $\Rightarrow$ 特征类型混合的场景 |

    - 分支条件设计 (Branching Criteria)
        **数学表达**：
        $$
        b(\mathbf{x}) = \begin{cases} 
        1 & \text{if } f_j(\mathbf{x}) \leq \tau_j \\[5pt]
        2 & \text{if } f_j(\mathbf{x}) > \tau_j \\
        \vdots & \\[5pt]
        C & \text{其他条件}
        \end{cases}
        $$

        **常见分裂类型**：
        | 类型 | 数学形式 | 示例 | 适用特征 |
        |------|----------|------|----------|
        | **数值特征分裂** | $x_j \leq \tau$ | 年龄 ≤ 30 | 连续型特征 |
        | **类别特征分裂** | $x_j \in S$ | 颜色 ∈ {红,蓝} | 离散型特征 |
        | **组合特征分裂** | $x_i + x_j \leq \tau$ | 收入+资产 ≤ 50万 | 特征交互 |
        | **非线性分裂** | $x_j^2 \leq \tau$ | 面积² ≤ 100 | 多项式特征 |

    - 终止条件 (Termination Criteria)
        | 条件类型 | 数学表达 | 说明 | 目的 |
        |----------|----------|------|------|
        | **最大深度** | $d \geq d_{\max}$ | 树达到预设最大深度 | 控制模型复杂度 |
        | **最小样本** | $\|S\| < n_{\min}$ | 节点样本数小于阈值 | 防止过拟合 |
        | **纯度阈值** | $H(S) < \epsilon$ | 节点熵小于阈值 | 提前停止分裂 |
        | **增益阈值** | $\Delta I < \delta$ | 信息增益小于阈值 | 避免无效分裂 |
        | **无改进** | $\Delta \text{MSE} < \delta$ | 均方误差改进不足 | 回归树专用 |

    - 叶节点预测 (Base Hypothesis)
        **数学形式**：
        $$
        g_t(\mathbf{x}) = \begin{cases} 
        \underset{k}{\arg \max}\ \dfrac{1}{|S_t|} \sum \mathbb{I}(y_i=k) & \text{(分类)} \\[5pt]
        \dfrac{1}{|S_t|} \sum y_i & \text{(回归)} \\[10pt]
        \log \dfrac{p}{1-p} & \text{(概率预测)}
        \end{cases}
        $$

        **叶节点类型**：
        
        | 类型 | 输出 | 特点 | 适用场景 |
        |------|------|------|----------|
        | **常数叶** | 标量值 | 简单高效，标准实现 | 大多数分类/回归问题 |
        | **模型叶** | 简单模型 | 叶节点拟合局部模型 | 复杂局部模式 |
        | **概率叶** | 概率分布 | 输出类别概率 | 不确定性估计、集成学习基模型 |  
    
    </div>

---

### 3.3 决策树模型关键

#### 3.3.3 划分选择

如何选择最优化分属性呢，一般而言，我们希望决策树的分支节点所包含的样本尽可能属于同一类别，即节点的“纯度（purity）”越来越高。

1. **信息增益（ID3）**

    <div style="width: 100%; border: 1px solid #ddd; padding: 10px;">
    在信息论里熵叫作信息量，即**熵是对不确定性的度量**。从控制论的角度来看，应叫不确定性。
    
    在信息世界，熵越高，则能传输越多的信息，熵越低，则意味着传输的信息越少。$\Rightarrow$ 信息量 = 熵 = 不确定性 $\Rightarrow$ 熵表示不纯度（Impurity）<br>
    
    **信息熵**（Information Entropy）是度量样本集合纯度最常用的一种指标。
    假定当前样本集合 $D$ 中第$k$类样本所占的比例为$p_k \ (k = 1,2,\cdots, \mid \mathcal{Y} \mid)$，则集合$D$的信息熵定义为：
    $$
    \text{Ent}(D) = -\sum_{k=1}^{\mid \mathcal{Y} \mid} p_k \log_2 p_k
    $$
    **$\text{Ent}(D)$ 的值越小，则 $D$ 的纯度越高。<br>**
    假定离散属性 $a$ 有 $V$ 个可能的取值 $\{a^1,a^2,\cdots,a^v \}$，若使用 $a$ 来对样本集合 $D$ 进行划分，则会产生 $V$ 个分支节点，其中第 $v$ 个分支节点包含了 $D$ 中所有在属性 $a$ 上取值为 $a^v$ 的样本，记为 $D^v$。我们可以计算出 $D^v$ 的信息熵，并且考虑到不同分支节点所包含的样本数不一样，给分支节点赋予权重 $\dfrac{\mid D^v \mid}{\mid D \mid}$，区分样本数带来的影响，得到用属性 $a$ 对样本集 $D$ 进行划分所获得的**信息增益**（information gain）：
    $$
    \text{Gain}(D,a) = \text{Ent}(D) - \sum_{v=1}^{V} \dfrac{\mid D^v \mid}{\mid D \mid} \text{Ent}(D^v)
    $$
    
    一般而言，信息增益越大，意味着使用属性 $a$ 来划分所划分所获得的纯度提升越大，因此，我们可以用信息增益来进行决策树的划分属性选择，即：
    
    $$
    a^{*} = \underset{a \in A}{\arg \max} \ \text{Gain}(D,a)
    $$
    
    著名的**ID3决策树学习算法**就是以信息增益为准则来选择划分属性的。
    
    </div>
    <br>
    
2. **增益率（C4.5）**
    <div style="width: 100%; border: 1px solid #ddd; padding: 10px;">
    
    > **信息增益准则对可取值数目较多的属性有所偏好**（例如，当样本拥有编号、地址、手机号等具有唯一性的属性时，将这种属性当作划分的依据分出来的结果肯定是最“干净”的，但是这样训练出来的模型泛化性能非常糟糕）。
    
    为减少这种偏好可能带来的不利影响，著名的C4.5决策树算法不直接使用信息增益，而是使用增益率（Gain Ratio）来选择划分属性。<br>
    增益率定义为：
    $$
    \text{Gain\_ratio}(D,a) = \frac{\text{Gain}(D,a)}{\text{IV}(a)}
    $$
    
    其中 $\text{IV}(a)$ 称为属性 $a$ 的固有值（Intrinsic Valve）：
    
    $$
    \text{IV}(a) = - \sum\limits_{v=1}^{V} \dfrac{\mid D^v \mid}{\mid D \mid} \log_2 \dfrac{\mid D^v \mid}{\mid D \mid}
    $$
    
    属性 $a$ 的可能取值数目越多（即 $V$ 越大），则 $\text{IV}(a)$ 的值通常会越大。
    <br>
    
    >需要注意的是，增益率准则对取值数目较少的属性可能有偏好，因此C4.5算法并不是直接选择增益率最大的候选划分属性，而是使用了一个启发式：**先将候选划分属性中找到信息增益高于平均水平的属性，再从中选择增益率高的。**
    
    </div>
    <br>
    
3. **基尼指数（CART）**
    <div style="width: 100%; border: 1px solid #ddd; padding: 10px;">
    
    基尼系数是指国际上通用的、用以衡量一个国家或地区居民**收入差距**的常用指标。<br>
    > 基尼系数最大为 “1”，最小等于 “0”。基尼系数越接近 0 表明收入分配越是趋向平等。国际惯例把 0.2 以下视为收入绝对平均，0.2-0.3 视为收入比较平均；0.3-0.4 视为收入相对合理；0.4-0.5 视为收入差距较大，当基尼系数达到 0.5 以上时，则表示收入悬殊。
    
    我们将经济学的基尼系数的概念引入到机器学习中来，当**基尼系数越小，代表 $D$ 集合中的数据越纯**。<br>
    数据集 $D$ 的纯度可由**基尼值**来度量：
    $$
    \begin{aligned}
    \text{Gini}(D) &= \sum_{k=1}^{\mid \mathcal{Y} \mid} \sum_{k^{'} \neq k} p_k p_{k^{'}}\\
    &= \sum\limits_{k=1}^{\mid \mathcal{Y} \mid} p_k (1-p_k)\\
    &= 1- \sum_{k=1}^{\mid \mathcal{Y} \mid} p_k^2
    \end{aligned}
    $$
    
    Gini 是一种概率化的表达，它在进行计算的时候效率高。
    
    属性 $a$ 的**基尼系数**定义为：
    $$
    \text{Gini\_index} (D,a) = \sum_{v=1}^{V} \dfrac{\mid D^v \mid}{\mid D \mid} \text{Gini}(D^v)
    $$
    
    于是，在候选属性集合 $A$ 中，选择一个使得划分后基尼指数最小的属性作为最优化分属性：
    
    $$
    a^{*} = \underset{a \in A}{\arg \min} \ \text{Gini\_index}(D,a)
    $$
    
    以著名的鸢尾花数据集为例：
    <br>
    <img src="../source/imgs/3.2.3_Gini_index_CART.png" alt="Gini_index_CART" style="display: block; margin: 0 auto; max-width: 100%;" width="600">
    
    </div>
    <br>
    
4. 分裂手段的选择

    决策树划分准则的关键是看经过一个操作之后，如何来衡量划分后的结果比划分前更纯净了。

    现在最常用的决策树树形结构是二叉树结构，所以分裂的指标一般是信息增益（Entropy）和基尼指数（Gini）。

    信息熵和基尼指数都是用来衡量数据集纯度的指标。在数据维度较低、数据比较清晰的情况下，信息熵和基尼系数没有太大的区别。但是在处理数据维度很大、噪音很大的数据时，基尼系数通常表现更好。

    但两者应用时产生的差异很小。

    信息熵与基尼系数的对比:
    - 信息熵：计算复杂度较高，但在清晰数据上表现好。
    - 基尼系数：计算简单，且在噪音数据上更鲁棒。

---

#### 3.3.4 剪枝处理

**剪枝（Pruning）是决策树学习算法对付“过拟合”的主要手段，也是根本方法。**

在决策树学习过程中，为了尽可能正确的分类训练样本，节点划分过程将不断重复，有时会造成决策树分支过多，导致因训练样本学的太过了，在应用到测试集的时候反而效果不好。因此，可**通过主动去掉一些分支来降低过拟合的风险**（从另一个角度来看，也是决定决策树学习的深度，到什么程度停止）。

决策树剪枝的基本策略有**预剪枝**（prepruning）和**后剪枝**（post-pruning）：

- 预剪枝是指在决策树生成过程中，对每个结点**划分前先进行估计**，若当前节点的划分不能带来决策树泛化性能的提升，则停止划分并将当前节点标记为叶节点。

- 后剪枝则是先**从训练集生成一颗完整的决策树，然后自底层向上对非叶节点进行考察**，若将该节点对应的子树替换为叶节点能带来决策树泛化性能提升，则将该子树替换为叶节点。

- 预剪枝和后剪枝最基础的手段是靠**精确度（也可以说错误率）的评估**。

    对样例集 $D$ ，分类错误率定义为：

    $$
    E(f;D) = \dfrac{1}{m}\sum_{i=1}^{m} \mathbb{I}(f(\mathbf{x}_i \neq y_i))
    $$

    精度则定义为：

    $$
    acc(f,D) = \dfrac{1}{m} \sum_{i=1}^{m} \mathbb{I}(\mathbf{x}_i = y_i)
    $$

    更一般的，对于数据分布 $D$ 和概率密度函数 $p(\cdot)$ ，错误率与精度可分别描述为：
    $$
    E(f;D) = \int_{x \sim D} \mathbb{I}(f(x) \neq y) p(x) dx\\[5pt]
    \begin{aligned}
    acc(f;D) &= \int_{x \sim D} \mathbb{I}(f(x) = y) p(x) dx\\[5pt]
    &= 1- E(f;D)
    \end{aligned}
    $$

- 性能比较

    - 时间开销

        预剪枝：测试时间开销降低，训练时间开销降低

        后剪枝：测试时间开销降低，训练时间开销增加

    - 过/欠拟合风险

        预剪枝：过拟合风险降低，欠拟合风险增加（**贪心**——一次收益不是很好的划分后面可能具有很高收益的划分）

        后剪枝：过拟合风险降低，欠拟合风险基本不变

<div style="width: 100%; border: 1px solid #ddd; padding: 10px;">

- 决策树分类器模型的调用：<br>  

    `class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0, *monotonic_cst=None)`<br>    
    [DecisionTreeClassifier — scikit-learn 1.7.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#decisiontreeclassifier)<br>    
    主要参数如下： 

    | 参数 | 选项 | 默认 | 说明 |
    |------|------|------|------|
    | **`criterion`** | "gini", "entropy" | "gini" | 分裂质量衡量标准 |
    | **`splitter`** | "best", "random" | "best" | 分裂点选择策略 |
    | **`max_features`** | int/float/str | None | 考虑的最大特征数 |
    | **`class_weight`** | dict/balanced | None | 类别权重处理不平衡数据 |
    | **`ccp_alpha`** | float | 0.0 | **后剪枝**复杂度参数 |

    <br>
    主要预剪枝参数如下：  

    | 超参数 | 类型 | 默认值 | 作用 | 推荐范围 |
    |--------|------|--------|------|----------|
    | **`max_depth`** | int | None | 树的最大深度 | 3-10 (深树易过拟合) |
    | **`min_samples_split`** | int/float | 2 | 节点分裂所需最小样本数 | 2-20 (或0.01-0.1) |
    | **`min_samples_leaf`** | int/float | 1 | 叶节点所需最小样本数 | 1-10 (或0.005-0.05) |
    | **`min_weight_fraction_leaf`** | float | 0.0 | 叶节点最小权重和 | 0.0-0.5 |
    | **`max_leaf_nodes`** | int | None | 最大叶节点数量 | 10-100 |
    | **`min_impurity_decrease`** | float | 0.0 | 分裂所需最小不纯度减少量（收益Grain） | 0.0-0.1 |   

    <br>
    示例：
    
    ```python
    # 适度预剪枝（推荐）
    DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_leaf_nodes=20
    )
    
    # 严格预剪枝（欠拟合风险）
    DecisionTreeClassifier(
        max_depth=2,
        min_samples_split=50,
        min_samples_leaf=20
    )
    # 可能成为浅层决策树桩
    ```

</div>



后剪枝：

常用的三种后剪枝方法：

1. REP—**错误率**降低剪枝
2. PEP—悲观剪枝（C4.5）
3. CCP（ Cost-Complexity Pruning）—代价复杂度剪枝（CART）



代价复杂度剪枝（CART）：

剪枝策略：在**模型复杂度**和**预测精度**之间寻找最优平衡。

该算法为子树 $T_t$ 定义了代价（cost）和复杂度（complexity）以及一个可由用户设置的衡量代价与复杂度之间关系的参数 $\alpha$ ，其中，**代价指在剪枝过程中因子树 $T_t$ 被叶节点替代而增加的错分样本**，**复杂度表示剪枝后子树 $T_t$ 减少的叶结点数**，$\alpha$ 则表示剪枝后树的复杂度降低程度与代价间的关系，定义为：
$$
R_{\alpha}(T) = R(T) + \alpha \times |\widetilde{T}|
$$

> - $R(T)$：子树T的预测误差（回归常用MSE）
>
> - $|\widetilde{T}|$：子树T的叶节点数量（复杂度度量）
>
> - $\alpha$：复杂度惩罚系数（≥0）
>
>     $\alpha=0$：只考虑误差，保留完整树（过拟合风险）
>
>     $\alpha\rightarrow ∞$：无限惩罚复杂度$\rightarrow$单节点树（欠拟合）

对于每个非叶节点 $t$ ：
$$
\alpha = \dfrac{R(t)-R(T_t)}{\mid \widetilde{T_t} \mid - 1}
$$

> 其中，
>
> - $\mid \widetilde{T_t} \mid$：子树 $T_t$ 中叶节点数；
>
> - $R(t)$：节点 $t$ 剪枝后的误差（作为叶节点），计算公式为$R(t) = t(t) \cdot p(t)$
>
>     ​               $r(t)$为节点 $t$ 错分样本率，$p(t)$为落入节点 $t$ 的样本所占样本的比例；
>
> - $R(T_t)$：以 $t$ 为根的子树 $T_t$ 的误差，计算公式为$R(T_t) = \sum R(i)$，$i$ 为子树 $T_t$ 下的叶节点。

CCP剪枝算法分为两个步骤：

- 对于完全决策树T的每个非叶结点计算 $\alpha$ 值，循环剪掉具有最小 $\alpha$ 值的子树，直到剩下根节点。在该步可得到一系列的剪枝树$\{T_0,T_1,T_2,\cdots,T_m \}$,其中 $T_0$ 为原有的完全决策树，$T_m$ 为根结点，$T_i+1$ 为对  $T_i$  进行剪枝的结果；
- 从子树序列中，根据真实的误差估计选择最佳决策树。

这个剪枝的方式对应到`sklearn.tree.DecisionTreeClassifier`里面的`ccp_alpha`参数。



[Example: Post pruning decision trees with cost complexity pruning — scikit-learn 1.7.1 documentation](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py)

[Source code Downloaded from scikit-learn.org](../source/py/plot_cost_complexity_pruning.py)

---

#### 3.3.5 回归树

回归决策树主要用 CART（classification and regression tree）算法，内部结点特征的取值为“是”和“否”， 为二叉树结构。

所谓回归，就是根据特征向量来决定对应的输出值。回归树就是将特征空间划分成若干单元，每一个划分单元有一个特定的输出。因为每个结点都是“是”和“否”的判断，所以划分的边界是平行于坐标轴的。对于测试数据，我们只要按照特征将其归到某个单元，便得到对应的输出值。

回归树用到的划分准则为**MSE**。



`class sklearn.tree.DecisionTreeRegressor(*, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0, monotonic_cst=None)`



具体参考——[决策树-回归（作者：禺垣笔记）](https://zhuanlan.zhihu.com/p/42505644)

---

#### 3.3.6 经典决策树算法

- ID3和C4.5

    ID3（Iterative Dichotomiser 3，迭代二叉树3代）由Ross Quinlan于1986年提出。1993年，他对ID3进行改进设计出了C4.5算法。

    我们已经知道ID3与C4.5的不同之处在于，ID3根据信息增益选取特征构造决策树，而C4.5则是以信息增益率为核心构造决策树。既然C4.5是在ID3的基础上改进得到的，那么这两者的优缺点分别是什么？

    **使用信息增益会让ID3算法更偏向于选择值多的属性**。信息增益反映给定一个条件后不确定性减少的程度，必然是分得越细的数据集确定性更高，也就是信息熵越小，信息增益越大。因此，在一定条件下，值多的属性具有更大的信息增益。而C4.5则使用信息增益率选择属性。信息增益率通过引入一个被称作分裂信息(Split information)的项来惩罚取值较多的属性，分裂信息用来衡量属性分裂数据的广度和均匀性。这样就改进了ID3偏向选择值多属性的缺点。**相对于ID3只能处理离散数据，C4.5还能对连续属性进行处理**，具体步骤为：

    1. 把需要处理的样本(对应根节点)或样本子集(对应子树)按照连续变量的大小从小到大进行排序。
    2. 假设该属性对应的不同的属性值一共有N个，那么总共有N−1个可能的候选分割阈值点，每个候选的分割阈值点的值为上述排序后的属性值中两两前后连续元素的中点，根据这个分割点把原来连续的属性分成bool属性。实际上可以不用检查所有N−1个分割点。(连续属性值比较多的时候，由于需要排序和扫描，会使C4.5的性能有所下降。)
    3. 用信息增益比率选择最佳划分。

    

    C4.5其他优点

    - 在树的构造过程中可以进行剪枝，缓解过拟合；

    - 能够对连续属性进行离散化处理（二分法）；

    - 能够对缺失值进行处理；

    

- CART

    CART（Classification And Regression Tree，分类回归树）由L.Breiman，J.Friedman，R.Olshen和C.Stone于1984年提出，是一种应用相当广泛的决策树学习方法。值得一提的是，CART和C4.5一同被评为数据挖掘领域十大算法。

    **CART算法采用一种二分递归分割的技术，将当前的样本集分为两个子样本集，使得生成的的每个非叶子节点都有两个分支。因此，CART算法生成的决策树是结构简洁的二叉树。**

    作为一种决策树学习算法，CART与ID3以及C4.5不同，在分类任务中，它使用基尼系数对属性进行选择，GINI系数越小则划分越合理；在回归任务中，它使用平方误差对属性进行选择，越小的平方误差说明划分越合理。

