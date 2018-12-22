---
layout: default
---


<div>
	<h2>
		tags:
		{% for tag in site.tags %}
			<a href="#{{ tag[0] }}" title="{{ tag[0] }}" rel="{{ tag[1].size }}">
				{{ tag[0] }}
			</a>
			|
		{% endfor %}
	</h2>
</div>


<div>
	{% for tag in site.tags %}
		<h2 id="{{ tag[0] }}">{{ tag[0] }}</h2>
		{% for post in tag[1] %}
			<li>
				<time datetime="{{ post.date | date:"%Y-%m-%d" }}">
				{{ post.date | date:"%Y-%m-%d" }}</time>
				<a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
			</li>
		{% endfor %}
	{% endfor %}
</div>

<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-115616798-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-115616798-1');
</script>