# Yonji's Blog
---

<div class="posts">
  {% for post in site.posts %}
    <article class="post">
      
      <h2><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h2>
      <div class="entry">
        {{ post.excerpt }}
      </div>
      <hr />
      
    </article>
  {% endfor %}
</div>


## This project in [Git Hub](https://github.com/Amoko/amoko.github.io/)


<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-115616798-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-115616798-1');
</script>
