# ROUND 1

<img src="https://files.deepseeksvc.com/api/file?file_id=f56fdc00-03d8-4434-8b9f-eccf6bc94fc1&state=gtZC7JCumAsd2rad8LEOixtsl5JH0%2Fq6271T9UY42eTmul9E1pNdlbY0YevALocTNC0Lpspoaw%2FrBgW6FfjNiuTXltsoHoLP9QWxuhjmaPX2H2coQ9UivFWSXFTFgyqm4yYcH2TVd9HqG3CgxBdfq9X0aDJDmg2OKKqLwGzRHY%2FrC%2BKtgKmQQopcq9n2QqZKvGDvIBnWX5rKnEj9Ryeh3prPbitkaIAIgirP5MPz74cqWlkuYHFm4NmfOZK4j6xRI1p%2BF9idRKBG8WwtKpdfNa0OF3sriG0LpK5yBdFx0FTM31AGNdHjm36A6se4E71sni2Pw%2Be4j2u9f1ebYKmMhnAVBFMFC9vGFC%2FuJHujeWgzWWNFXJ8BYutx&ty=p" alt="img" style="zoom:50%;" />

- > 这个特征方程是怎么给出来的，给我讲解一下这方面的知识



你的问题非常核心：如何从下重式罗经的动力学方程组推导出特征方程，并解释“忽略常数 \(\omega_2\)”的含义。下面我分步讲解。

---

## 一、下重式罗经的完整动力学方程（PDF 第36页）

文档给出的方程组为：

\[
\begin{cases}
\dot{\theta} = \omega_1 \alpha \\[4pt]
\dot{\alpha} = -\dfrac{M}{H}\,\theta - \omega_2
\end{cases}
\]

其中：
- \(\theta\)：高度角（偏下为正）
- \(\alpha\)：方位角（偏西为正）
- \(\omega_1 = \omega_{ie} \cos\phi\)（地球自转角速度的水平分量）
- \(\omega_2 = \omega_{ie} \sin\phi\)（地球自转角速度的垂直分量）
- \(M = m g a\)（重心下移产生的比例系数）
- \(H\)：陀螺动量矩

这是一个**非齐次线性常微分方程组**，因为 \(- \omega_2\) 是常数项。

---

## 二、为什么可以“忽略常数 \(\omega_2\)”来研究摆动特性？

常数 \(\omega_2\) 只影响系统的**平衡位置**（稳态解），不影响系统的**动态响应特性**（如振荡频率、阻尼比）。  
具体来说：

- 设系统的平衡点为 \((\theta_r, \alpha_r)\)，令 \(\dot{\theta}=0,\ \dot{\alpha}=0\)：
  \[
  \begin{cases}
  0 = \omega_1 \alpha_r \\
  0 = -\dfrac{M}{H}\theta_r - \omega_2
  \end{cases}
  \Rightarrow
  \begin{cases}
  \alpha_r = 0 \\
  \theta_r = -\dfrac{H\omega_2}{M}
  \end{cases}
  \]
  这就是文档第34页给出的稳态解。

- 定义**偏差变量**：
  \[
  \Delta\theta = \theta - \theta_r,\quad \Delta\alpha = \alpha - \alpha_r
  \]
  代入原方程，常数项 \(- \omega_2\) 与 \(-\dfrac{M}{H}\theta_r\) 抵消，得到**齐次方程**：
  \[
  \begin{cases}
  \Delta\dot{\theta} = \omega_1 \Delta\alpha \\[4pt]
  \Delta\dot{\alpha} = -\dfrac{M}{H} \Delta\theta
  \end{cases}
  \]

  该齐次方程决定了系统在平衡点附近的**自由摆动特性**。特征方程正是从这个齐次系统导出的。

因此，“忽略 \(\omega_2\)”实质上就是**研究偏差量的动态行为**，常数项只影响平衡点位置，不影响特征根。

---

## 三、从齐次方程组推导特征方程

齐次方程组写为矩阵形式：

\[
\frac{d}{dt}\begin{bmatrix} \Delta\theta \\ \Delta\alpha \end{bmatrix}
= \begin{bmatrix}
0 & \omega_1 \\
-\dfrac{M}{H} & 0
\end{bmatrix}
\begin{bmatrix} \Delta\theta \\ \Delta\alpha \end{bmatrix}
\]

设解的形式为 \(\begin{bmatrix} \Delta\theta \\ \Delta\alpha \end{bmatrix} = \begin{bmatrix} \Theta \\ A \end{bmatrix} e^{s t}\)，代入得：

\[
\begin{cases}
s\Theta = \omega_1 A \\
s A = -\dfrac{M}{H} \Theta
\end{cases}
\]

将第一式 \(A = \dfrac{s}{\omega_1} \Theta\) 代入第二式：

\[
s \cdot \frac{s}{\omega_1} \Theta = -\frac{M}{H} \Theta
\Rightarrow \frac{s^2}{\omega_1} \Theta = -\frac{M}{H} \Theta
\]

假设 \(\Theta \neq 0\)，两边乘以 \(\omega_1\)：

\[
s^2 = -\frac{M \omega_1}{H}
\]
\[
s^2 + \frac{M \omega_1}{H} = 0
\]

这就是**特征方程**。

或者直接计算矩阵的特征多项式：

\[
\det\begin{pmatrix}
s & -\omega_1 \\
\frac{M}{H} & s
\end{pmatrix} = s^2 + \frac{M\omega_1}{H} = 0
\]

---

## 四、特征根与物理意义

特征方程的解为：

\[
s_{1,2} = \pm j \sqrt{\frac{M \omega_1}{H}}
\]

- 两个特征根是**纯虚数**，说明系统是**无阻尼等幅振荡**（中心型平衡点）。
- 振荡角频率：
  \[
  \omega_n = \sqrt{\frac{M \omega_1}{H}}
  \]
- 振荡周期（第37页）：
  \[
  T_0 = \frac{2\pi}{\omega_n} = 2\pi \sqrt{\frac{H}{M \omega_1}}
  \]
  代入 \(\omega_1 = \omega_{ie}\cos\phi\) 即得文档公式。

物理意义：如果不加额外阻尼，陀螺主轴指北端会在平衡点 \((\alpha_r=0,\ \theta_r=-H\omega_2/M)\) 附近做**无衰减的椭圆摆动**，永远不会静止在子午面上。这就是为什么实际罗经必须引入阻尼（如安许茨的长轴阻尼或斯佩里的短轴阻尼）。

---

## 五、总结

| 步骤 | 操作                                      | 结果                                                         |
| ---- | ----------------------------------------- | ------------------------------------------------------------ |
| 1    | 写出完整方程                              | \(\dot{\theta}=\omega_1\alpha,\ \dot{\alpha}=-(M/H)\theta-\omega_2\) |
| 2    | 忽略常数 \(\omega_2\)（实际是考虑偏差量） | 得到齐次方程                                                 |
| 3    | 设为指数解 \(e^{st}\)                     | 导出特征方程 \(s^2 + (M\omega_1/H)=0\)                       |
| 4    | 求解特征根                                | \(s = \pm j\sqrt{M\omega_1/H}\) → 等幅振荡                   |

希望这个逐步推导能帮你彻底理解特征方程的来源。如果需要，我可以进一步解释为什么无阻尼振荡周期与舒拉调谐（84.4分钟）相关。

# ROUND 2

- > 这用到了什么数学知识？微分方程的特征根求解方法嘛



是的，你总结得非常到位。从下重式罗经的动力学方程组推导特征方程，主要用到了以下数学知识：

---

## 1. **常系数线性微分方程组**
方程组：
\[
\begin{cases}
\dot{\theta} = \omega_1 \alpha \\
\dot{\alpha} = -\dfrac{M}{H}\theta
\end{cases}
\]
是**一阶常系数线性齐次微分方程组**。可写成矩阵形式：
\[
\frac{d}{dt}\begin{bmatrix} \theta \\ \alpha \end{bmatrix} = 
\begin{bmatrix} 0 & \omega_1 \\ -\frac{M}{H} & 0 \end{bmatrix}
\begin{bmatrix} \theta \\ \alpha \end{bmatrix}
\]

---

## 2. **特征方程与特征根方法**
求解这类方程组的标准方法是：
- 假设解的形式为 \(\begin{bmatrix} \theta \\ \alpha \end{bmatrix} = \begin{bmatrix} \Theta \\ A \end{bmatrix} e^{s t}\)，其中 \(s\) 是待定常数（特征根）。
- 代入原方程，得到关于 \(\Theta, A\) 的线性代数方程组：
\[
\begin{cases}
s\Theta = \omega_1 A \\
sA = -\frac{M}{H} \Theta
\end{cases}
\]
- 写成矩阵形式：
\[
\begin{bmatrix} s & -\omega_1 \\ \frac{M}{H} & s \end{bmatrix}
\begin{bmatrix} \Theta \\ A \end{bmatrix} = 0
\]
- 要有非零解，系数矩阵的行列式必须为零：
\[
\det\begin{bmatrix} s & -\omega_1 \\ \frac{M}{H} & s \end{bmatrix} = s^2 + \frac{M\omega_1}{H} = 0
\]
这就是**特征方程**。

---

## 3. **线性代数（行列式、特征值）**
- 系数矩阵的特征值就是满足 \(\det(A - sI)=0\) 的 \(s\)。
- 这里 \(A = \begin{bmatrix} 0 & \omega_1 \\ -\frac{M}{H} & 0 \end{bmatrix}\)，特征多项式为：
\[
\det\begin{pmatrix} -s & \omega_1 \\ -\frac{M}{H} & -s \end{pmatrix} = s^2 + \frac{M\omega_1}{H} = 0
\]

---

## 4. **复数和简谐振动**
特征根 \(s_{1,2} = \pm j \sqrt{\frac{M\omega_1}{H}}\) 是纯虚数，对应解为：
\[
\theta(t) = C_1 \cos(\omega_n t) + C_2 \sin(\omega_n t)
\]
其中 \(\omega_n = \sqrt{\frac{M\omega_1}{H}}\)。这表示**无阻尼等幅振荡**（简谐运动）。

---

## 总结：用到的核心数学知识

| 知识点                   | 作用                                   |
| ------------------------ | -------------------------------------- |
| 常系数线性微分方程组     | 描述系统动态                           |
| 矩阵形式与特征值         | 将方程组转化为代数问题                 |
| 行列式                   | 求特征方程                             |
| 复数与欧拉公式           | 理解纯虚根对应振荡                     |
| 二阶常微分方程特征多项式 | 等价于将两个一阶方程合并为一个二阶方程 |

所以，你所说的“**微分方程的特征根求解方法**”正是核心，同时还结合了线性代数中矩阵特征值的基本概念。这些是控制理论和振动分析中最基础的工具。