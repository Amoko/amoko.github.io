# Set Python Build System in Sublime Text

for Linux
```python
{
    "shell_cmd": "/home/you/anaconda3/bin/python -u \"$file\"",
    "selector": "source.python",
    "file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)"
}
```
for Windows
```python
{
    "cmd": ["C:/Users/you/Anaconda3/python", "-u", "$file"],
    "selector": "source.python",
    "file_regex": "^\\s*File \"(...*?)\", line ([0-9]*)"
}
```



