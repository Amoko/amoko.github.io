---
title: Sublime Text, Spyder Configuration
layout: default
tags: [dev config,]
---

## 1 Sublime Text 3

### 1.1 添加新的Build System

Tools - Build System- New Build System

粘贴以下配置（注意python安装路径）

**Linux**

```python
{
    "shell_cmd": "/home/you/anaconda3/bin/python -u \"$file\"",
    "selector": "source.python",
    "file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)"
}
```
**Windows**

```python
{
    "cmd": ["C:/Users/you/Anaconda3/python", "-u", "$file"],
    "selector": "source.python",
    "file_regex": "^\\s*File \"(...*?)\", line ([0-9]*)"
}
```

### 1.2 添加sublimerge插件

Preferences - Package Control - Install Package

搜索 sublimerge 安装。

### 1.3 查看文件编码

Preference - Settings

``` json
"show_encoding": true,
```



## 2 Spyder 

### 2.1 添加模块自动补全

~\Anaconda3\Lib\site-packages\spyder\utils\introspection\module_completion.py

将需要自动补全的模块名添加到mods变量中。