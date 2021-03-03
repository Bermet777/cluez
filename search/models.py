from datetime import datetime, timedelta

from django.contrib.auth import get_user_model
from django.db import models
from django.contrib.postgres.fields import HStoreField

# Create your models here.
User = get_user_model()


class SearchResult(models.Model):
    article = models.TextField()
    link = models.TextField()
    domain = models.TextField()
    description = models.TextField()
    topic = models.TextField()
    subject = models.TextField()
    text = models.TextField()
    check_url = models.TextField()
    data = HStoreField(null=True, blank=True)


class Search(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    results = models.ManyToManyField(SearchResult)
    input = models.TextField()
    lang = models.CharField(max_length=100)
    region = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    finished = models.BooleanField(default=False)
    result = models.TextField(null=True, blank=True, default=None)
