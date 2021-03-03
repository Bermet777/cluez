from django.urls import path

from .views import searches, search_ind, search_org, search_results

app_name = "searches"
urlpatterns = [
    path("list/", view=searches, name="searches"),
    path("<int:search_id>/", view=search_results, name="search_detail"),
    path("individual/", view=search_ind, name="search_individual"),
    path("organization/", view=search_org, name="search_organization"),
]
