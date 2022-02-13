import asyncio
import json
import os
import uuid

import requests
import gc
import re
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
from datetime import datetime

from django import db
from django.db import connections
import celery

from psycopg2._psycopg import OperationalError
from aiohttp import ClientSession, BasicAuth
from asgiref.sync import sync_to_async

from .client import RestClient
from .helpers import sent_to_words, text_from_html

from config import celery_app
from .lang_models import LanguageModels
from .models import Search, SearchResult
from celery.signals import worker_process_init

old_send = requests.Session.send


def new_send(*args, **kwargs):
    if kwargs.get("timeout", None) is None:
        kwargs["timeout"] = 100
    return old_send(*args, **kwargs)


requests.Session.send = new_send

lm = LanguageModels()


class FaultTolerantTask(celery.Task):
    """ Implements after return hook to close the invalid connection.
    This way, django is forced to serve a new connection for the next
    task.
    """
    abstract = True

    def after_return(self, *args, **kwargs):
        # connection.close()
        db.close_old_connections()


@worker_process_init.connect()
def configure_worker(signal=None, sender=None, **kwargs):
    print("Loader called")
    for conn in db.connections.all():
        conn.close_if_unusable_or_obsolete()

    # Calling db.close() on some DB connections will cause the inherited DB
    # conn to also get broken in the parent process so we need to remove it
    # without triggering any network IO that close() might cause.
    # for c in db.connections.all():
    #     if c and c.connection:
    #         try:
    #             os.close(c.connection.fileno())
    #         except (AttributeError, OSError, TypeError, db.InterfaceError):
    #             pass
    #     try:
    #         c.close()
    #     except db.InterfaceError:
    #         pass
    #     except db.DatabaseError as exc:
    #         str_exc = str(exc)
    #         if 'closed' not in str_exc and 'not connected' not in str_exc:
    #             raise


@celery_app.task(
    base=FaultTolerantTask,
    bind=True,
    soft_time_limit=42,  # raises celery.exceptions.SoftTimeLimitExceeded inside the coroutine
    time_limit=300,  # breaks coroutine execution
)
async def my_task(self, *args, **kwargs):
    await asyncio.sleep(5)


@celery_app.task
async def my_simple_task(*args, **kwargs):
    await asyncio.sleep(5)


async def run_script(search, item, index, session):
    url = item['url']
    if url.endswith('pdf') or url.endswith('xlsx') or url.endswith('xls'):
        return None
    try:
        async with session.get(url, verify_ssl=False, timeout=10) as resp:
            print("Get website content, status:{}".format(resp.status))
            response = await resp.text()
            result_text = re.sub('\W+', ' ', text_from_html(response)[0:10000]
                            .replace('\n', ' ').replace('\r', ' ').replace('\t',' ').replace("  ", " "))
    except UnicodeDecodeError as e:
        return None
    except Exception as e:
        print(e)
        return None
    if len(result_text) < 200:
        return None

    # Clean text
    result_text = ''.join(c for c in result_text if not c.isdigit())

    # auth=BasicAuth(login='admin', password='pass')
    detected_lang = lm.detect_language(result_text)["language"]
    #print(result_text)
    async with session.post("http://spacy-{}.cluez.io/tokenizer".format(detected_lang),
                            auth=BasicAuth(login='admin', password='pass'),
                            json={"text": result_text}) as resp:
        print("Get tokenizer response, status:{}".format(resp.status))
        token_response = await resp.text()
        #print(token_response)
    try:
        token_response = json.loads(token_response)
    except json.decoder.JSONDecodeError as e:
        print(e)
        return None

    return item, result_text, token_response['tokens'], detected_lang


async def async_search(response, search):
    global lm

    input_key = search.input
    language = search.lang

    # if error then please create log command
    if (response["tasks"][0]["result"] is None) or (response["tasks"][0]["result"][0]["items"] is None):
        search.result = "No results"
        search.finished = True
        await sync_to_async(search.save)()

    else:
        async with ClientSession() as session:
            results = await asyncio.gather(*[run_script(search, item, i, session) for i, item in
                                             enumerate(response["tasks"][0]['result'][0]["items"])])

        print("Async finished - run cleaning etc..")
        # Filter out Nones
        results = [x for x in results if x is not None]

        if len(results) == 0:
            ##sometimes occure because of filtering user agent by sites. See desription of the problem: https://stackoverflow.com/questions/27231113/999-error-code-on-head-request-to-linkedin
            search.result = "No results"
            search.finished = True
            await sync_to_async(search.save)()
            return

        else:
            lem = []
            num = []
            ID = []
            lang = []
            b = []
            texx = []
            items = []
            n = 0
            for item, result_text, token_response, detected_lang in results:
                for token in token_response:
                    lem.append(token.split())
                    b.append(token.split())
                num.append(len(b))
                ID.extend(np.repeat(n, len(b)))
                lang.extend(np.repeat(detected_lang, len(b)))
                n += 1
                texx.append(result_text)
                items.append(item)

            ID = pd.DataFrame(ID)
            ID = ID.loc[:, [0]]
            lem = pd.DataFrame(lem)
            lem = lem.loc[:, [0]]
            lang = pd.DataFrame(lang)
            lang = lang.loc[:, [0]]
            dd = pd.concat([ID, lem, lang], axis=1)
            dd.columns = ["ID", "unigrams", "language"]

            print("lematize ok", datetime.now())
            dd = dd[(dd.language == "en") & (~dd['unigrams'].isin(lm.stopword_en))].append(
                [dd[(dd.language == "fr") & (~dd['unigrams'].isin(lm.stopword_fr))]
                    , dd[(dd.language == "hu") & (~dd['unigrams'].isin(lm.stopword_hu))]
                    , dd[(dd.language == "pl") & (~dd['unigrams'].isin(lm.stopword_pl))]
                    , dd[(dd.language == "es") & (~dd['unigrams'].isin(lm.stopword_es))]
                    , dd[(dd.language == "de") & (~dd['unigrams'].isin(lm.stopword_de))]], ignore_index=True)

            dd['length'] = dd.unigrams.str.len()
            dd = dd.drop(dd[dd["length"] <= 3].index)
            dd = dd.dropna()

            print("stop_words ok", datetime.now())

            output_series = dd.groupby(['ID'])['unigrams'].apply(list)
            data_words = list(sent_to_words(output_series))
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=70)  # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words], threshold=70)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            data_words = trigram_mod[bigram_mod[data_words]]
            print("data_words ok", datetime.now())

            article_list = []
            for q in range(len(data_words)):
                article_list.extend(list(data_words[q]))
            uniq_list = list(set(article_list))
            uniq_num = [0] * len(uniq_list)

            dwapp = []
            for i in range(len(data_words)):
                dwapp.extend(data_words[i])

            for j in range(len(uniq_list)):
                uniq_num[j] = uniq_num[j] + sum(x in uniq_list[j] for x in dwapp)
            print("uniq", datetime.now())

            uniq_list = pd.DataFrame(uniq_list)
            uniq_num = pd.DataFrame(uniq_num)
            dd2 = pd.concat([uniq_list, uniq_num], axis=1)
            print(dd2)
            if dd2.empty:
                search.finished = True
                await sync_to_async(search.save)()
                return
            dd2.columns = ["unique_words", "frequency"]
            bad_words = dd2.loc[(dd2['frequency'] >= len(texx) * 0.7)]
            bad_words_list = bad_words['unique_words'].tolist()

            bad_words2 = dd2.loc[(dd2['frequency'] <= 2)]
            bad_words2_list = bad_words2['unique_words'].tolist()
            keys = lm.keywords[lm.keywords["language"] == language]["keywords"].values[0]
            keys_set = set(keys)
            bad_words_set = set([*bad_words_list, *bad_words2_list])
            bad_words = list(bad_words_set.difference(keys_set))
            print(keys_set)

            aaa = []
            for t in range(len(data_words)):
                aaa.append(data_words[t])

            for t in range(len(aaa)):
                aaa[t] = [item for item in aaa[t] if item not in bad_words]

            print("badwords_ok", datetime.now())
            output_series = pd.Series(aaa)
            data_words = list(sent_to_words(output_series))

            # remove frequent and rare words
            # Create Dictionary
            id2word = corpora.Dictionary(data_words)
            # Create Corpus
            texts = data_words

            # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in texts]

            print("corpus ok", datetime.now())
            if len(texx) > 10:
                hdp = gensim.models.hdpmodel.HdpModel(corpus=corpus, id2word=id2word, alpha=.5)
                ii = [None] * len(corpus)
                for i in range(len(corpus)):
                    ii[i] = hdp[corpus[i]]
                get_document_topics = ii
            else:
                pass
            print("hdp ok", datetime.now())

            #texx2 = [texx[i] for i in list(set(dd.ID))]


            target = re.sub('"', '', lm.keywords[lm.keywords["language"] == language]["keywords"].values[0]).split(
                " OR ")
            target[0] = target[0][1:]
            target.append(input_key)

            for o in range(len(aaa)):
                topic = 99
                if len(texx) > 10 and len(get_document_topics[o]) > 0:
                    topic = get_document_topics[o][0][0]

                data = {}
                for i, t in enumerate(target):
                    if t in texx[o]:
                        if i == len(target) - 1:
                            t = "Full name match"
                        data[t] = 1
                    else:
                        data[t] = 0
                try:
                    search_result = {
                        "article": items[o]["title"],
                        "link": items[o]["url"],
                        "domain": items[o]["domain"],
                        "description": items[o]["description"],
                        "topic": topic,
                        "subject": input_key,
                        "text": texx[o],
                        "check_url": response["tasks"][0]["result"][0]['check_url'],
                        "data": data
                    }
                    result = False
                    counter = 0
                    while result is False or counter < 5:
                        result = await append_search_result(search.pk, search_result)
                        counter += 1
                except OperationalError as e:
                    print(e)
        search.finished = True
        await sync_to_async(search.save)()


@sync_to_async
def append_search_result(search_pk, search_result_json):
    try:
        search = Search.objects.get(pk=search_pk)
        # search_result_json = json.loads(search_result_json)
        search_result = SearchResult(
            article=search_result_json['article'],
            link=search_result_json['link'],
            domain=search_result_json['domain'],
            description=search_result_json['description'],
            topic=search_result_json['topic'],
            subject=search_result_json['subject'],
            text=search_result_json['text'],
            check_url=search_result_json['check_url'],
            data=search_result_json['data']
        )
        for item in search.results.all():
            if search_result.link == item.link:
                return True
        search_result.save()
        search.results.add(search_result)
        search.save()
        return True
    except (db.OperationalError, db.InterfaceError, db.InternalError) as e:
        print('Caught DB connection error, resetting connection')
        print(e)
        db.close_old_connections()
        return False


@celery_app.task(base=FaultTolerantTask, default_retry_delay=30, max_retries=3, soft_time_limit=3600, time_limit=3600)
def search_get_serp(search_pk):
    """TBD Doc"""
    search = Search.objects.get(pk=search_pk)
    global lm

    input_key = search.input
    language = search.lang

    gc.collect()

    print("Start Time =", datetime.now())

    search_input = '"%s" AND ' % input_key

    keys = lm.keywords[lm.keywords["language"] == language]["keywords"].values[0]
    client = RestClient("hello@duegle.com", "02f57675d732fb96")  ###változtasd meg fizetősre
    post_data = dict()
    # You can set only one task at a time
    post_data[0] = dict(
        language_code=language,  # en
        location_code=2840,  # us
        keyword=search_input + " " + keys
    )
    response = client.post("/v3/serp/google/organic/live/regular", post_data)
    print("response")
    print(response)
    search_task(response, search_pk)
    #return response


@celery_app.task(base=FaultTolerantTask, default_retry_delay=30, max_retries=3, soft_time_limit=3600, time_limit=3600)
def search_task(response, search_pk=None, first_run=True):
    """TBD Doc"""
    try:
        start_time = datetime.utcnow()
        search = Search.objects.get(pk=search_pk)
        asyncio.run(async_search(response, search))
        search.duration = datetime.utcnow() - start_time
        search.save()
    except (db.OperationalError, db.InterfaceError, db.InternalError) as e:
        print('Caught DB connection error, resetting connection')
        print(e)
        db.close_old_connections()
        if first_run:
            search_task(response, search_pk, False)

