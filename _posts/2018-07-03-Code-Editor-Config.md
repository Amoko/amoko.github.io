---
title: Sublime Text, Spyder
layout: default
tags: [dev config,]
---

### Sublime Text 3 添加新的Build System

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



### Spyder 添加模块自动补全

找到此文件  ~\Anaconda3\Lib\site-packages\spyder\utils\introspection\module_completion.py

将需要自动补全的模块名添加到mods变量中。