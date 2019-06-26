---
title: List of publications
permalink: /publications/
---
 
## Journal articles 

{% bibliography --query @article[status!=editorial && status!=other && author^=Stoller] %}

## Conference papers

{% bibliography --query @inproceedings[status!=editorial && status!=other && author^=Stoller] %}
