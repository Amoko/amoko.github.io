---
title: Sublime Text, Spyder Configuration
layout: default
tags: [dev config,]
---

## 1 Sublime Text 3

### 1.1 添加新的Build System

选择 Tools - Build System- New Build System

粘贴下列内容，请根据你的目录修改路径。

Linux
```python
{
    "shell_cmd": "/home/you/anaconda3/bin/python -u \"$file\"",
    "selector": "source.python",
    "file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)"
}
```
Windows
```python
{
    "cmd": ["C:/Users/you/Anaconda3/python", "-u", "$file"],
    "selector": "source.python",
    "file_regex": "^\\s*File \"(...*?)\", line ([0-9]*)"
}
```

### 1.2 添加sublimerge插件

Ctrl + Shift + P 打开命令面板

输入 install 调出 Install Package

搜索 sublimerge 安装。

打开要比较的文件，右键 - compare



## 2 Spyder 

### 2.1 添加模块自动补全

找到此文件  ~\Anaconda3\Lib\site-packages\spyder\utils\introspection\module_completion.py

将需要自动补全的模块名添加到mods变量中。