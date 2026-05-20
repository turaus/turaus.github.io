#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown to HTML 转换器 (海洋机器人主题风格)
功能：
- 支持标准Markdown语法 (标题、列表、粗体、表格、代码块等)
- 数学公式 (行内 $...$ 或块级 $$...$$) 通过 MathJax 渲染为 SVG
- 自动生成侧边栏目录 (基于标题层级)
- 代码块高亮 (使用 highlight.js)
- 响应式布局，移动端适配
- 采用与“传感器特征分析”HTML相同的视觉风格 (卡片、渐变侧边栏等)

依赖库:
    pip install markdown

使用方法:
    python md_to_html.py input.md output.html
"""

import argparse
import os
import sys
import markdown
from markdown.extensions.toc import TocExtension

# ---------- HTML 模板 ----------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <!-- MathJax 3 (SVG 渲染) -->
    <script>
        MathJax = {{
            tex: {{
                packages: {{'[+]': ['base', 'ams', 'newcommand']}},
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true
            }},
            svg: {{
                fontCache: 'global',
                scale: 1.0,
                minScale: 0.8
            }},
            options: {{
                ignoreHtmlClass: 'no-mathjax',
                processHtmlClass: 'math'
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <!-- highlight.js 代码高亮 (自动识别语言) -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
    <style>
        :root {{
            --primary: #0b3d5f;
            --primary-light: #145a7a;
            --accent: #e67e22;
            --bg: #f4f7fa;
            --card-bg: #ffffff;
            --text: #2c3e50;
            --border: #dce4ec;
            --sidebar-width: 260px;
            --shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            --radius: 10px;
            --transition: 0.2s ease;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
            display: flex;
            min-height: 100vh;
        }}

        /* 侧边导航栏 */
        .sidebar {{
            width: var(--sidebar-width);
            background: linear-gradient(180deg, #0a3144 0%, #0b3d5f 100%);
            color: #ecf0f1;
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            overflow-y: auto;
            padding: 2rem 1rem 2rem 1.5rem;
            z-index: 10;
            box-shadow: 2px 0 12px rgba(0, 0, 0, 0.15);
            transition: transform var(--transition);
        }}

        .sidebar h2 {{
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 1.5rem;
            color: #ffffff;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.6rem;
        }}

        /* Toc 目录样式 (由 markdown 扩展生成) */
        .sidebar .toc {{
            list-style: none;
        }}
        .sidebar .toc ul {{
            list-style: none;
            padding-left: 1rem;
        }}
        .sidebar .toc li {{
            margin-bottom: 0.25rem;
        }}
        .sidebar .toc a {{
            display: block;
            color: #bdc3c7;
            text-decoration: none;
            padding: 0.5rem 0.8rem;
            border-radius: 6px;
            font-size: 0.92rem;
            transition: all var(--transition);
            border-left: 3px solid transparent;
        }}
        .sidebar .toc a:hover,
        .sidebar .toc a.active {{
            background: rgba(255, 255, 255, 0.08);
            color: #ffffff;
            border-left-color: var(--accent);
        }}
        /* 多级缩进微调 */
        .sidebar .toc .toc-level-2 {{
            padding-left: 0.8rem;
        }}

        /* 主内容区域 */
        .main-content {{
            margin-left: var(--sidebar-width);
            flex: 1;
            padding: 2rem 2.5rem;
            max-width: 1100px;
        }}

        .header {{
            margin-bottom: 2rem;
        }}

        .header h1 {{
            font-size: 2.2rem;
            color: var(--primary);
            font-weight: 700;
            letter-spacing: 0.3px;
        }}

        .header .subtitle {{
            color: #5d6d7e;
            font-size: 1rem;
            margin-top: 0.3rem;
        }}

        /* 卡片容器包裹渲染的 Markdown 内容 */
        .card {{
            background: var(--card-bg);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 1.8rem 2rem;
            margin-bottom: 1.8rem;
            border: 1px solid var(--border);
            transition: box-shadow var(--transition);
        }}

        .card:hover {{
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        }}

        /* Markdown 内标题样式 */
        .card h1 {{
            font-size: 1.8rem;
            color: var(--primary);
            border-left: 5px solid var(--accent);
            padding-left: 0.8rem;
            margin: 1.5rem 0 1rem 0;
        }}
        .card h2 {{
            font-size: 1.5rem;
            color: var(--primary);
            border-left: 5px solid var(--accent);
            padding-left: 0.8rem;
            margin: 1.4rem 0 1rem 0;
        }}
        .card h3 {{
            font-size: 1.25rem;
            color: #2c3e50;
            margin: 1.2rem 0 0.6rem;
            font-weight: 600;
        }}
        .card h4, .card h5, .card h6 {{
            font-weight: 600;
            margin: 1rem 0 0.4rem;
            color: #34495e;
        }}
        .card p {{
            margin-bottom: 0.8rem;
        }}
        .card ul, .card ol {{
            margin: 0.5rem 0 1rem 1.8rem;
        }}
        .card li {{
            margin: 0.2rem 0;
        }}
        /* 表格样式 */
        .card table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background-color: #fff;
            font-size: 0.9rem;
            overflow-x: auto;
            display: block;
        }}
        .card th, .card td {{
            border: 1px solid var(--border);
            padding: 0.6rem 0.8rem;
            text-align: left;
        }}
        .card th {{
            background-color: #eef2f5;
            font-weight: 600;
        }}
        .card tr:nth-child(even) {{
            background-color: #f9fbfd;
        }}
        /* 代码块样式 */
        .card pre {{
            background: #f6f8fa;
            border-radius: 6px;
            padding: 1rem;
            overflow-x: auto;
            font-size: 0.85rem;
            line-height: 1.45;
            margin: 1rem 0;
        }}
        .card code {{
            font-family: 'SF Mono', 'Courier New', monospace;
            background: #f0f2f5;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .card pre code {{
            background: transparent;
            padding: 0;
        }}
        /* 引用块 */
        .card blockquote {{
            border-left: 4px solid var(--accent);
            background: #fef9e7;
            margin: 1rem 0;
            padding: 0.5rem 1rem;
            color: #5d4e2e;
        }}
        /* 响应式 */
        @media screen and (max-width: 768px) {{
            .sidebar {{
                transform: translateX(-100%);
                position: fixed;
                z-index: 20;
            }}
            .sidebar.open {{
                transform: translateX(0);
            }}
            .main-content {{
                margin-left: 0;
                padding: 1.2rem;
            }}
            .mobile-nav-toggle {{
                display: block;
                position: fixed;
                top: 1rem;
                left: 1rem;
                background: var(--primary);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0.5rem 0.8rem;
                z-index: 30;
                cursor: pointer;
                font-size: 1rem;
            }}
            .card {{
                padding: 1.2rem;
            }}
        }}
        @media screen and (min-width: 769px) {{
            .mobile-nav-toggle {{
                display: none;
            }}
        }}
        /* 数学公式块内部允许横向滚动 */
        .math-block {{
            overflow-x: auto;
        }}
        footer {{
            text-align: center;
            margin-top: 2rem;
            color: #7f8c8d;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <button class="mobile-nav-toggle" onclick="document.querySelector('.sidebar').classList.toggle('open')">☰ 导航</button>
    <nav class="sidebar">
        <h2>📑 目录</h2>
        {toc_html}
    </nav>
    <main class="main-content">
        <div class="header">
            <h1>{page_heading}</h1>
            <div class="subtitle">由 Markdown 自动生成 | 海洋机器人环境感知风格</div>
        </div>
        <div class="card">
            {body_html}
        </div>
        <footer>
            <p>📄 文档由 Markdown 转换 • 公式采用 SVG 渲染 • 样式基于传感器特征分析主题</p>
        </footer>
    </main>
    <script>
        // 侧边栏导航激活高亮 (基于滚动监听)
        const tocLinks = document.querySelectorAll('.sidebar .toc a');
        const sections = Array.from(document.querySelectorAll('.card h1, .card h2, .card h3')).map(el => {{
            if (el.id) return el;
            return null;
        }}).filter(Boolean);
        window.addEventListener('scroll', () => {{
            let current = '';
            for (let section of sections) {{
                const sectionTop = section.offsetTop - 100;
                if (pageYOffset >= sectionTop) {{
                    current = '#' + section.id;
                }}
            }}
            tocLinks.forEach(link => {{
                link.classList.remove('active');
                if (link.getAttribute('href') === current) {{
                    link.classList.add('active');
                }}
            }});
        }});
        // 移动端点击链接后关闭侧边栏
        if (window.innerWidth <= 768) {{
            tocLinks.forEach(link => {{
                link.addEventListener('click', () => {{
                    document.querySelector('.sidebar').classList.remove('open');
                }});
            }});
        }}
    </script>
</body>
</html>
"""


def convert_md_to_html(md_text: str, page_title: str = "Markdown 文档") -> str:
    """
    使用 markdown 库将 md 文本转换为 html 内容，同时生成目录。
    返回完整的 html 字符串。
    """
    # 配置 markdown 扩展：
    # - toc: 生成目录 (锚点基于标题)
    # - tables: 支持表格
    # - fenced_code: 支持 ``` 代码块
    # - nl2br: 保留换行 (可选)
    md = markdown.Markdown(
        extensions=[
            TocExtension(
                permalink=False,
                title="",
                baselevel=1,
                separator="-",
                toc_depth="2-4"      # 目录显示 h1~h4 层级
            ),
            'tables',
            'fenced_code',
            'nl2br'
        ],
        extension_configs={
            'toc': {
                'title': '',
                'anchorlink': False,
                'permalink': False
            }
        }
    )
    body_html = md.convert(md_text)
    toc_html = md.toc
    # 如果 TOC 为空，则给出提示
    if not toc_html:
        toc_html = '<div style="color:#bdc3c7; padding:0.5rem;">无标题 (请使用 # 定义标题)</div>'
    else:
        # 为 toc 容器添加 class 方便样式控制 (默认 ul 带 toc 类)
        toc_html = f'<div class="toc">{toc_html}</div>'
    
    # 提取文档中第一个 h1 作为页面主标题，否则使用传入的 page_title
    heading = page_title
    if body_html:
        # 简单查找 <h1> 标签内容 (不引入额外库)
        import re
        match = re.search(r'<h1[^>]*>(.*?)</h1>', body_html, re.IGNORECASE)
        if match:
            heading = re.sub(r'<[^>]+>', '', match.group(1))  # 去除可能的内嵌标签
    
    # 填充模板
    full_html = HTML_TEMPLATE.format(
        page_title=page_title,
        page_heading=heading,
        toc_html=toc_html,
        body_html=body_html
    )
    return full_html


def main():
    parser = argparse.ArgumentParser(
        description="将 Markdown 文件转换为具有海洋机器人主题风格的 HTML，支持数学公式(SVG)和代码高亮。"
    )
    parser.add_argument("input", help="输入的 Markdown 文件路径")
    parser.add_argument("output", nargs="?", default=None, help="输出的 HTML 文件路径 (默认: 输入文件名同目录 .html)")
    parser.add_argument("--title", default=None, help="HTML 页面标题 (默认使用输入文件名)")
    args = parser.parse_args()
    
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"错误: 文件 '{input_path}' 不存在")
        sys.exit(1)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    # 确定输出文件名
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".html"
    
    # 页面标题
    if args.title:
        page_title = args.title
    else:
        page_title = os.path.basename(input_path)
    
    # 转换
    html_content = convert_md_to_html(md_text, page_title=page_title)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 转换成功！HTML 已保存至: {output_path}")


if __name__ == "__main__":
    # 如果没有命令行参数，展示帮助信息
    if len(sys.argv) == 1:
        print("海洋机器人风格 Markdown 转换器\n")
        print("用法示例:")
        print("  python md_to_html.py README.md")
        print("  python md_to_html.py doc.md output.html --title '我的文档'")
        print("\n依赖安装: pip install markdown")
        sys.exit(0)
    main()