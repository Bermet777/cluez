import requests
import gc
import re
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
from datetime import datetime
from psycopg2._psycopg import OperationalError

from .client import RestClient
from .helpers import sent_to_words, text_from_html

from config import celery_app
from celery.signals import worker_process_init
from .lang_models import LanguageModels
from .models import Search, SearchResult

old_send = requests.Session.send


def new_send(*args, **kwargs):
    if kwargs.get("timeout", None) is None:
        kwargs["timeout"] = 100
    return old_send(*args, **kwargs)


requests.Session.send = new_send

lm = None


@worker_process_init.connect()
def setup(**kwargs):
    print('initializing NLP parser')
    global lm
    lm = LanguageModels()
    print('done initializing NLP parser')


@celery_app.task(default_retry_delay=30, max_retries=3, soft_time_limit=3600, time_limit=3600)
def search_task(search_pk):
    """TBD Doc"""
    global lm

    search = Search.objects.get(pk=search_pk)

    input_key = search.input
    language = search.lang

    gc.collect()
    log = pd.DataFrame({
        'step': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'time': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    log["subject"] = input_key
    log.iloc[0, 1] = datetime.now()
    log.iloc[0, 0] = "Start Time"

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

    # if error then please create log command
    if (response["tasks"][0]["result"][0]["items"] is None) or (response["tasks"][0]["result"] is None):
        search.result = "No results"
        search.save()

    else:
        url = pd.DataFrame(index=range(len(response["tasks"][0]['result'][0]["items"])), columns=range(0))
        for x in range(0, len(response["tasks"][0]['result'][0]["items"])):
            url.loc[x, 1] = response["tasks"][0]['result'][0]["items"][x]["url"]
        texx = [None] * len(response["tasks"][0]['result'][0]["items"])
        print("Response ok", datetime.now())
        n = 0
        for x in range(0, len(response["tasks"][0]['result'][0]["items"])):
            if url.iloc[x].values[0][-3:] not in ["pdf", "xlsx", "xls"]:
                try:
                    texx[x - n] = re.sub('\W+', ' ',
                                         text_from_html(requests.get(url.loc[x, 1], headers=lm.headers).text)[
                                         0:10000].replace('\n', ' ').replace('\r', ' ').replace('\t',
                                                                                                ' ').replace(
                                             "  ", " "))
                except:
                    texx[x - n] = (url.loc[x, 1])
            else:
                n = n + 1
        texx = texx[0:len(texx) - n]
        print("Read ok", datetime.now())

        n = 0
        for z in range(len(texx)):
            if len(texx[z - n]) < 200:
                texx.pop(z - n)
                n = n + 1
            else:
                pass
        print("cut short article ok", datetime.now())

        if len(texx) == 0:
            ##sometimes occure because of filtering user agent by sites. See desription of the problem: https://stackoverflow.com/questions/27231113/999-error-code-on-head-request-to-linkedin

            search.result = "No results"
            search.save()

        else:
            texx_clean = texx
            for g in range(len(texx_clean)):
                texx_clean[g] = ''.join(c for c in texx[g] if not c.isdigit())

            print("clean ok", datetime.now())
            lem = []
            num = []
            ID = []
            lang = []
            n = 0
            for h in range(len(texx_clean)):
                detected_lang = lm.detect_language(texx[h])["language"]
                if detected_lang == "en":
                    # pip install en_core_web_sm

                    doc = lm.nlp_en(texx_clean[h])
                    b = []
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h - n]))
                    lang.extend(np.repeat(detected_lang, num[h - n]))
                elif detected_lang == "hu":
                    # pip install https://github.com/oroszgy/spacy-hungarian-models/releases/download/hu_core_ud_lg-0.3.1/hu_core_ud_lg-0.3.1-py3-none-any.whl
                    # note: the incompatibility warning is a bug, see: https://github.com/oroszgy/spacy-hungarian-models/issues/15
                    try:
                        doc = lm.nlp_hu(texx_clean[h])
                    except Exception as e:
                        print(e)
                    b = []
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h - n]))
                    lang.extend(np.repeat(detected_lang, num[h - n]))
                elif detected_lang == "pl":
                    # C:\windows\system32>pip install C:\Users\csetri1dani428\downloads\pl_spacy_model-0.1.0.tar.gz
                    # python3 -m spacy download pl_core_news_sm

                    doc = lm.nlp_pl(texx_clean[h])
                    b = []
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h - n]))
                    lang.extend(np.repeat(detected_lang, num[h - n]))
                elif detected_lang == "de":
                    # python -m spacy download de_core_news_sm
                    doc = lm.nlp_de(texx_clean[h])
                    b = []
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h - n]))
                    lang.extend(np.repeat(detected_lang, num[h - n]))
                elif detected_lang == "fr":
                    # python -m spacy download fr_core_news_sm
                    doc = lm.nlp_fr(texx_clean[h])
                    b = []
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h - n]))
                    lang.extend(np.repeat(detected_lang, num[h - n]))
                elif detected_lang == "es":
                    # python -m spacy download es_core_news_sm
                    doc = lm.nlp_es(texx_clean[h])
                    b = []
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h - n]))
                    lang.extend(np.repeat(detected_lang, num[h - n]))
                else:
                    n = n + 1

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
            log.iloc[6, 1] = datetime.now()
            log.iloc[6, 0] = "uniq ok"

            uniq_list = pd.DataFrame(uniq_list)
            uniq_num = pd.DataFrame(uniq_num)
            dd2 = pd.concat([uniq_list, uniq_num], axis=1)
            dd2.columns = ["unique_words", "frequency"]
            bad_words = dd2.loc[(dd2['frequency'] >= len(texx) * 0.7)]
            bad_words_list = bad_words['unique_words'].tolist()

            bad_words2 = dd2.loc[(dd2['frequency'] <= 2)]
            bad_words2_list = bad_words2['unique_words'].tolist()
            aaa = []
            for t in range(len(data_words)):
                aaa.append(data_words[t])

            for t in range(len(aaa)):
                aaa[t] = [item for item in aaa[t] if item not in bad_words_list]
                aaa[t] = [item for item in aaa[t] if item not in bad_words2_list]

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

            texx = [texx[i] for i in list(set(dd.ID))]
            ji = lm.keywords[lm.keywords["language"] == "en"]["keywords"].values[0]
            ji = re.sub('"', '', ji)
            ji = ji.split(" OR ")
            ji[0] = ji[0][1:]
            ji.append(input_key)

            target = re.sub('"', '', lm.keywords[lm.keywords["language"] == language]["keywords"].values[0]).split(" OR ")
            target[0] = target[0][1:]
            target.append(input_key)

            for o in range(len(aaa)):
                topic = 99
                if len(texx) > 10 and len(get_document_topics[o]) > 0:
                    topic = get_document_topics[o][0][0]

                data = {}
                for i, t in enumerate(target):
                    if i == len(target)-1:
                        t = "Full name match"
                    if t in texx[o]:
                        data[t] = 1
                    else:
                        data[t] = 0
                try:
                    search_result = SearchResult.objects.create(
                        article=response["tasks"][0]['result'][0]["items"][o]["title"],
                        link=response["tasks"][0]['result'][0]["items"][o]["url"],
                        domain=response["tasks"][0]['result'][0]["items"][o]["domain"],
                        description=response["tasks"][0]['result'][0]["items"][o]["description"],
                        topic=topic,
                        subject=input_key,
                        text=texx[o],
                        check_url=response["tasks"][0]["result"][0]['check_url'],
                        data=data
                    )
                    search_result.save()
                    search.results.add(search_result)
                    search.save()
                except OperationalError as e:
                    print(e)
        search.finished = True
        search.save()



