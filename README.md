# 学习笔记静态网站

这是一个基于GitHub Pages的个人学习笔记网站，采用纯静态HTML/CSS/JavaScript构建，无需任何构建工具或依赖。

## 项目结构

```
笔记转网页/
├── index.html          # 主页面
├── style.css           # 主样式文件
├── script.js           # 交互脚本
├── underwater-nav/     # 水下机器人导航基础模块
│   ├── introduction.html
│   ├── module.css
│   ├── sensors.html
│   ├── algorithms.html
│   └── applications.html
├── ai/                 # 人工智能模块
│   ├── ml-basics.html
│   ├── deep-learning.html
│   ├── computer-vision.html
│   └── nlp.html
└── marine-control/     # 海洋机器人操纵与控制模块
    ├── kinematics.html
    ├── dynamics.html
    ├── control.html
    └── path-planning.html
```

## 功能特点

- ✨ **响应式设计** - 完美适配桌面端和移动端
- 🎨 **模块化配色** - 每个学习模块有独特的配色方案
- 📱 **左侧导航栏** - 快速检索和定位内容
- 🔄 **平滑滚动** - 流畅的页面导航体验
- 🎯 **卡片式布局** - 清晰展示各模块内容

## 快速开始

### 1. 推送到GitHub

```bash
# 初始化Git仓库
git init

# 添加所有文件
git add .

# 提交更改
git commit -m "初始化学习笔记网站"

# 添加远程仓库
git remote add origin https://github.com/turaus/turaus.github.io.git

# 推送到GitHub
git push -u origin main
```

### 2. 启用GitHub Pages

1. 访问你的GitHub仓库：`https://github.com/turaus/turaus.github.io`
2. 点击 "Settings" 标签
3. 在左侧菜单找到 "Pages"
4. 在 "Source" 部分，选择 "main" 分支
5. 点击 "Save"
6. 等待几分钟后，你的网站将在 `https://turaus.github.io` 上线

## 如何添加新内容

### 创建新的学习笔记页面

1. 在对应模块目录下创建新的HTML文件
2. 复制 `underwater-nav/introduction.html` 作为模板
3. 修改页面内容和标题
4. 在主页 `index.html` 中添加指向新页面的链接

### 自定义样式

- 修改 `style.css` 来调整全局样式
- 每个模块可以有自己的 `module.css` 用于特定样式

### 配色方案

项目使用CSS变量定义配色，可以在 `style.css` 中修改：

```css
:root {
    /* 水下机器人模块配色 - 深蓝海洋色系 */
    --underwater-primary: #006994;
    --underwater-secondary: #0077B6;
    --underwater-accent: #00B4D8;
    --underwater-bg: #E3F2FD;

    /* 人工智能模块配色 - 科技紫色系 */
    --ai-primary: #6B5B95;
    --ai-secondary: #8B7BB5;
    --ai-accent: #A78BFA;
    --ai-bg: #F3E5F5;

    /* 海洋机器人操纵与控制模块配色 - 海洋绿色系 */
    --marine-primary: #2E7D32;
    --marine-secondary: #43A047;
    --marine-accent: #66BB6A;
    --marine-bg: #E8F5E9;
}
```

## 移动端适配

网站已针对移动设备进行优化：
- 响应式导航栏
- 自动调整的卡片布局
- 触摸友好的交互设计

## 技术栈

- **HTML5** - 语义化标签
- **CSS3** - Flexbox、Grid、CSS变量、渐变、阴影
- **JavaScript (ES6+)** - 平滑滚动、交互效果
- **无框架依赖** - 轻量快速，易于维护

## 许可证

MIT License

---

📝 持续更新中 | 记录学习的每一步