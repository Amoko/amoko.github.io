
<div class="posts">
  {% for post in site.posts %}
    <article class="post">
      <h2><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h2>
      <p>{{ post.excerpt }}</p>
      <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
      ***
    </article>
  {% endfor %}
</div>


#### This project in [GitHub](https://github.com/Amoko/amoko.github.io/)


<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-115616798-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-115616798-1');
</script>
