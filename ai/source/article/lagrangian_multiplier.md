# 拉格朗日乘数法与对偶问题

## 拉格朗日乘数法

### 等式约束优化问题

对于等式约束优化问题：

$$
\begin{aligned}
\min \ & f(x) \\
\text{s.t.} \ & h_i(x) = 0, \quad i = 1,...,m
\end{aligned}
$$

构造拉格朗日函数：

$$
\mathcal{L}(x,\lambda) = f(x) + \sum_{i=1}^m \lambda_i h_i(x)
$$

极值点满足的一阶必要条件：

$$
\begin{cases}
\nabla_x \mathcal{L} = \nabla f(x) + \sum_{i=1}^m \lambda_i \nabla h_i(x) = 0 \\[5pt]
h_i(x) = 0, \quad i=1,...,m
\end{cases}
$$

### 不等式约束优化问题（KKT条件）

对于不等式约束优化问题：

$$
\begin{aligned}
\min \ & f(x) \\[5pt]
\text{s.t.} \ & g_j(x) \leq 0, \quad j = 1,...,k \\
& h_i(x) = 0, \quad i = 1,...,m
\end{aligned}
$$

构造拉格朗日函数：

$$
\mathcal{L}(x,\lambda,\mu) = f(x) + \sum_{j=1}^k \mu_j g_j(x) + \sum_{i=1}^m \lambda_i h_i(x)
$$

KKT必要条件：

$$
\begin{cases}
\nabla_x \mathcal{L} = \nabla f(x) + \sum_{j=1}^k \mu_j \nabla g_j(x) + \sum_{i=1}^m \lambda_i \nabla h_i(x) = 0 \\[5pt]
h_i(x) = 0, \quad i=1,...,m \\[5pt]
g_j(x) \leq 0, \quad j=1,...,k \\[5pt]
\mu_j \geq 0, \quad j=1,...,k \\[5pt]
\mu_j g_j(x) = 0, \quad j=1,...,k
\end{cases}
$$

## 对偶问题

### 拉格朗日对偶函数

定义拉格朗日对偶函数：

$$
g(\lambda,\mu) = \inf_{x} \mathcal{L}(x,\lambda,\mu) = \inf_{x} \left[ f(x) + \sum_{j=1}^k \mu_j g_j(x) + \sum_{i=1}^m \lambda_i h_i(x) \right]
$$

对偶函数总是是原问题最优值的下界：

$$
g(\lambda,\mu) \leq p^*, \quad \forall \mu \geq 0
$$

### 对偶问题

拉格朗日对偶问题：

$$
\begin{aligned}
\max \ & g(\lambda,\mu) \\[5pt]
\text{s.t.} \ & \mu \geq 0
\end{aligned}
$$

记对偶问题的最优值为 $d^*$，则始终有弱对偶性：

$$
d^* \leq p^*
$$

当满足某些约束规范条件时（如Slater条件），有强对偶性：

$$
d^* = p^*
$$

<div style="width: 100%; white-space: nowrap; border: 1px solid #ddd; padding: 10px;">

**Slater条件（约束规范）**

Slater条件是保证强对偶性成立的重要条件：

**定义**：如果存在一个可行点 $x$ 满足：
1. 所有不等式约束严格成立：$g_j(x) < 0$ （对于所有 $j=1,...,k$）
2. 所有等式约束成立：$h_i(x) = 0$ （对于所有 $i=1,...,m$）

且：
- 目标函数 $f$ 和不等式约束 $g_j$ 是凸函数
- 等式约束 $h_i$ 是仿射函数

则称Slater条件成立。

**重要性**：
- 当Slater条件满足时，强对偶性 $d^* = p^*$ 必定成立
- 对于凸优化问题，Slater条件是最常用的强对偶性保证条件
</div>


### 对偶间隙

定义对偶间隙为：

$$
p^* - d^* \geq 0
$$

当对偶间隙为零时，强对偶成立。

### 互补松弛条件

在强对偶成立时，最优解满足：

$$
\mu_j^* g_j(x^*) = 0, \quad j=1,...,k
$$

这意味着：
- 要么约束 $g_j(x^*)=0$（紧约束）
- 要么对应的乘子 $\mu_j^*=0$（松约束）


## 参考视频：
[拉格朗日乘数法可视化解析](https://www.bilibili.com/video/BV15T411f7DY/?spm_id_from=333.337.search-card.all.click&vd_source=62b6bb4c48ac16b4c1e4b27a2fce3817)
[拉格朗日乘数法与对偶问题](https://www.bilibili.com/video/BV1HP4y1Y79e/?spm_id_from=333.337.search-card.all.click&vd_source=62b6bb4c48ac16b4c1e4b27a2fce3817)