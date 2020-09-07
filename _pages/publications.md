---
title: List of publications
permalink: /publications/
---

## Theses

{% bibliography --query @thesis[status!=editorial && status!=other && author^=*Stoller*] %}

## Journal articles 

{% bibliography --query @article[status!=editorial && status!=other && author^=*Stoller*] %}

## Conference papers

{% bibliography --query @inproceedings[status!=editorial && status!=other && author^=*Stoller*] %}
