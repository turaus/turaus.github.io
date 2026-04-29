import pandas as pd
import numpy as np
import re
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz    # 决策树可视化
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split    # 分割训练集和测试集
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap,to_rgb

iris = load_iris()
data = pd.DataFrame(iris.data)
data.columns = iris.feature_names
data['Species'] = load_iris().target
# print(data)

x = data.iloc[:, 0:4]
y = data.iloc[:, -1]
# y = pd.Categorical(data[4]).codes
# print(x.head())     # sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# print(y.head())


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

tree_clf = DecisionTreeClassifier(max_depth=8, criterion='gini')
tree_clf.fit(x_train, y_train)

print(x_train.shape)

custom_colors = ['#FFD700', '#32CD32', '#4169E1']  # 金色, 酸橙色, 皇家蓝

def create_custom_cmap(base_colors):
    cmaps = []
    for color in base_colors:
        # 创建从白色到目标颜色的渐变
        colors = [(1, 1, 1), to_rgb(color)]  # 白色 -> 目标颜色
        cmaps.append(ListedColormap(colors))
    return cmaps

# 创建自定义颜色映射
custom_cmaps = create_custom_cmap(custom_colors)

plt.figure(figsize=(15,20))
ax = plt.gca()

tree_plot = plot_tree(
    tree_clf,
    feature_names=iris.feature_names[:],
    class_names=iris.target_names,
    rounded=True,        # 圆角节点
    filled=True,         # 颜色填充
    proportion=True,     # 显示比例而非绝对数量
    precision=2,         # 数值精度（小数位数）
    impurity=True,       # 显示基尼系数
    label="root",        # 或 "all" 控制标签显示
    node_ids=False,      # 隐藏节点ID
    fontsize=10,          # 字体大小
    ax=ax
)

# 自定义节点颜色
for node in tree_plot:
    text = node.get_text()
    
    # 使用正则表达式提取值（处理整数和浮点数）
    value_match = re.search(r'value = \[([\d\.,\s]+)\]', text)
    if value_match:
        values_str = value_match.group(1)
        
        # 提取所有数值（可能是整数或浮点数）
        values = []
        for x in values_str.split(','):
            try:
                # 尝试转换为浮点数
                values.append(float(x.strip()))
            except ValueError:
                # 如果转换失败，尝试作为整数处理
                values.append(float(x.strip().split('.')[0]))
        
        # 计算总样本数/比例和
        total = sum(values)
        
        # 确定主要类别
        if total > 0:
            proportions = [v / total for v in values]
            main_class = np.argmax(proportions)
            intensity = proportions[main_class]
            
            # 设置自定义颜色
            facecolor = custom_cmaps[main_class](intensity)
            node.set_bbox(dict(facecolor=facecolor, alpha=0.8, edgecolor='gray', linewidth=1))
            
            # 更改文本颜色以提高对比度
            text_color = 'black' if intensity < 0.7 else 'white'
            node.set_color(text_color)

# 添加图例
legend_elements = [
    plt.Rectangle((0,0), 1, 1, color=custom_colors[0], label='setosa'),
    plt.Rectangle((0,0), 1, 1, color=custom_colors[1], label='versicolor'),
    plt.Rectangle((0,0), 1, 1, color=custom_colors[2], label='virginica')
]
plt.legend(handles=legend_elements, title='Classes', fontsize=12, loc='lower right')

# 添加标题
plt.title('Custom Colored Decision Tree for Iris Dataset', fontsize=16, pad=20)

# 调整布局并保存
plt.tight_layout()

plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()
# .\dot -Tpng E:\学习\py文档\机器学习\4.决策树\决策树_代码\iris_tree.dot -o E:\学习\py文档\机器学习\4.决策树\决策树_代码\iris_tree1.png

y_test_hat = tree_clf.predict(x_test)
print("acc score:", accuracy_score(y_test, y_test_hat))

print(tree_clf.feature_importances_)

# print(tree_clf.predict_proba([[5, 1.5]]))
# print(tree_clf.predict([[5, 1.5]]))

# 寻找较合适的深度
# depth = np.arange(1, 15)
# err_list = []
# for d in depth:
#     print(d)
#     clf = DecisionTreeClassifier(criterion='gini', max_depth=d)
#     clf.fit(x_train, y_train)
#     y_test_hat = clf.predict(x_test)
#     result = (y_test_hat == y_test)
#     err = 1 - np.mean(result)
#     # print(100 * err)
#     err_list.append(err)
#     print(d, ' 错误率：%.2f%%' % (100 * err))

# mpl.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(facecolor='w')
# plt.plot(depth, err_list, 'ro-', lw=2)
# plt.xlabel('决策树深度', fontsize=15)
# plt.ylabel('错误率', fontsize=15)
# plt.title('决策树深度和过拟合', fontsize=18)
# plt.grid(True)
# plt.show()


