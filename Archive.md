---
title: Archive
layout: default
---

<div class="posts">
  {% for post in site.posts %}
    <article class="post">
      <h2><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h2>
      <!--
      <p>{{ post.excerpt }}</p>
      -->
      <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
    </article>
    <hr />
  {% endfor %}
</div>
