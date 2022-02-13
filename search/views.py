from datetime import datetime, timedelta
from collections import Counter
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.http.response import HttpResponseRedirect, Http404
from django.shortcuts import render
from django.template import loader
from django.template.defaultfilters import register
from django.utils import timezone
from django.utils.safestring import mark_safe
from pandas.tseries.offsets import BDay

from .models import Search, SearchResult
from .tasks import search_task, search_get_serp
from .forms import SearchIndividualForm, SearchOrgForm


# @register.filter
# def highlight_search(text, search):
#     highlighted = text.replace(search, '<span class="highlight">{}</span>'.format(search))
#     return mark_safe(highlighted)



@login_required
def searches(request):
    time_threshold = timezone.now() - BDay(3)
    searches = Search.objects.filter(user=request.user, created_at__gt=time_threshold).order_by('-created_at')

    template = loader.get_template('searches.html')
    context = {
        'searches': searches,
    }
    return HttpResponse(template.render(context, request))


@login_required
def search_results(request, search_id):
    time_threshold = timezone.now() - BDay(3)
    search = Search.objects.filter(user=request.user, pk=search_id, created_at__gt=time_threshold)
    if search:
        search = search[0]
    else:
        raise Http404
    search_results = search.results.all()

    data = {}
    for result in search_results:
        sub = result.data
        for key in sub:
            sub[key] = int(sub[key])
        data = dict(Counter(data)+Counter(sub))

    template = loader.get_template('search_detail.html')
    context = {
        'search': search,
        'search_result': search_results,
        'data': data
    }
    return HttpResponse(template.render(context, request))


@login_required
def search_ind(request):
    from django_q.tasks import async_task
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = SearchIndividualForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            search_input = '%s %s' % (form.cleaned_data['first_name'], form.cleaned_data['last_name'])
            if form.cleaned_data['language'] in ['hu']:
                search_input = '%s %s' % (form.cleaned_data['last_name'], form.cleaned_data['first_name'])
            search = Search.objects.create(
                user=request.user,
                input=search_input,
                lang=form.cleaned_data['language'],
                region=form.cleaned_data['region'],
            )
            search.save()
            # process the data in form.cleaned_data as required
            #search_get_serp.apply_async(args=(search.pk,), link=search_task.s(search.pk))
            async_task(search_get_serp, search.pk)
            return HttpResponseRedirect('/search/list/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = SearchIndividualForm()

    return render(request, 'search_ind.html', {'form': form})


@login_required
def search_org(request):
    from django_q.tasks import async_task
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = SearchOrgForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            search = Search.objects.create(
                user=request.user,
                input=form.cleaned_data['business_name'],
                lang=form.cleaned_data['language'],
                region=form.cleaned_data['region'],
            )
            search.save()
            # process the data in form.cleaned_data as required
            #search_get_serp.apply_async(args=(search.pk,), link=search_task.s(search.pk))
            async_task(search_get_serp, search.pk)

            return HttpResponseRedirect('/search/list/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = SearchOrgForm()

    return render(request, 'search_org.html', {'form': form})
