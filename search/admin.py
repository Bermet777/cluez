from django.contrib import admin
from django import forms

from . import models


class SearchAdminForm(forms.ModelForm):

    class Meta:
        model = models.Search
        fields = "__all__"


class SearchAdmin(admin.ModelAdmin):
    form = SearchAdminForm


class SearchResultAdminForm(forms.ModelForm):

    class Meta:
        model = models.SearchResult
        fields = "__all__"


class SearchResultAdmin(admin.ModelAdmin):
    form = SearchResultAdminForm


admin.site.register(models.Search, SearchAdmin)
admin.site.register(models.SearchResult, SearchResultAdmin)
