---
layout: page
title: About
description: 整理笔记的地方
keywords: Clementine
comments: true
menu: 关于
permalink: /about/
---

研究自然语言处理， 主要方向是语言结构预测， 熟悉依存句法分析。
Deep Learning 学习者， 熟悉 dynet, pytorch 和 tensorflow, 喜欢观察轮子和造轮子。
openSUSE 和 KDE 用户， 喜欢 simple by default, powerful when needed 的东西.
喜欢简洁方便的文档写作， latex, markdown, pandoc 之类。
## 联系

{% for website in site.data.social %}
* {{ website.sitename }}：[@{{ website.name }}]({{ website.url }})
{% endfor %}

## Skill Keywords

{% for category in site.data.skills %}
### {{ category.name }}
<div class="btn-inline">
{% for keyword in category.keywords %}
<button class="btn btn-outline" type="button">{{ keyword }}</button>
{% endfor %}
</div>
{% endfor %}
