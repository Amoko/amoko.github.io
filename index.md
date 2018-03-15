# Yonji

# WAR IS PEACE
# FREEDOM IS SLAVERY
# IGNORANCE IS STRENGTH

# March.13 2018

<div class="blog-index">  
  {% assign post = site.posts.first %}
  {% assign content = post.content %}
  {% include post_detail.html %}
</div>

---
layout: default
---

<div class="posts">
  {% for post in site.posts %}
    <article class="post">

      <h1><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h1>

      <div class="entry">
        {{ post.excerpt }}
      </div>

      <a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read More</a>
    </article>
  {% endfor %}
</div>


You can use the [editor on GitHub](https://github.com/Amoko/amoko.github.io/edit/master/index.md) to maintain and preview the content for your website in Markdown files.


### Markdown


```markdown
Syntax highlighted code block

あ　い　う　え　お

```


### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Amoko/amoko.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.



<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-115616798-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-115616798-1');
</script>
