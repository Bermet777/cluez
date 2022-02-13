from collections import Counter
from django import template

register = template.Library()


@register.filter(name='keyword_count')
def keyword_count(sub):
    for key in sub:
        sub[key] = int(sub[key])
    return dict(Counter({})+Counter(sub))
