# Yonji's Blog

<div class="posts">
  {% for post in site.posts %}
    <article class="post">

      <h1><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h1>

      <div class="entry">
        {{ post.excerpt }}
      </div>
      
      <!--
      <a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read More</a>
      -->
    </article>
  {% endfor %}
</div>


## My [GitHub](https://github.com/Amoko/amoko.github.io/)


<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-115616798-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-115616798-1');
</script>
