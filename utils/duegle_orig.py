#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from search.client import RestClient
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment
import gc
import re
import numpy as np
import spacy
from spacy_langdetect import LanguageDetector
from datetime import datetime
# Gensim
import gensim
import gensim.corpora as corpora
gc.collect()
import requests
DEFAULT_TIMEOUT = 100
old_send = requests.Session.send
def new_send(*args, **kwargs):
    if kwargs.get("timeout", None) is None:
        kwargs["timeout"] = DEFAULT_TIMEOUT
    return old_send(*args, **kwargs)
requests.Session.send = new_send
#pip install https://github.com/oroszgy/spacy-hungarian-models/releases/download/hu_core_ud_lg-0.3.1/hu_core_ud_lg-0.3.1-py3-none-any.whl


# In[ ]:


keywords=pd.read_csv("keywords2.csv", sep=',', encoding='utf8')
a=pd.read_csv("TEST SUBJECTS2.csv", sep='\t')
a.reset_index(drop=True, inplace=True)
print(a)
print(a['Name'])

# In[ ]:


#base
#pip install selectolax
from selectolax.parser import HTMLParser

def get_text_selectolax(html):
    tree = HTMLParser(html)

    if tree.body is None:
        return None

    for tag in tree.css('script'):
        tag.decompose()
    for tag in tree.css('style'):
        tag.decompose()

    text = tree.body.text(separator='\n')
    return text

headers = {
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en-US,en;q=0.8',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
     }

def langdet(string):
    doc=nlp(string)
    return doc._.language

from spacy.lang.fr.stop_words import STOP_WORDS
stopfr=list(STOP_WORDS)
from spacy.lang.en.stop_words import STOP_WORDS
stopen=list(STOP_WORDS)
from spacy.lang.de.stop_words import STOP_WORDS
stopde=list(STOP_WORDS)
from spacy.lang.hu.stop_words import STOP_WORDS
stophu=list(STOP_WORDS)
from spacy.lang.pl.stop_words import STOP_WORDS
stoppl=list(STOP_WORDS)


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

#bs4: text_from_html(requests.get(url.loc[x,1], headers=headers).text)
#korábban:get_text_selectolax(requests.get(url.loc[x,1], headers=headers).text)

print("base ok",datetime.now())


# In[ ]:


key_out=pd.DataFrame(a[:1], columns=["article","link","domain","description","topic","subject name","Text","litigation","fraud","tax evasion","money laundering","embezzlement","bribery","corruption","Full name match"])
time_log=pd.DataFrame(a, columns=["subject","time","results"])
for s in range(len(a)): #range(len(a)):
    X=a['Name'][s]
    print(X)
    log=pd.DataFrame({
    'step': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'time': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    log["subejct"]=X
    print("Start Time =", datetime.now())
    log.iloc[0,1]=datetime.now()
    log.iloc[0,0]="Start Time"
    X='"'+ X+'"'+ " "+ "AND"+ " "
    Y=a["Language"][s]
    key=keywords[keywords["language"]==Y]["keywords"].values[0]
    client = RestClient("hello@duegle.com", "02f57675d732fb96") ###változtasd meg fizetősre
    post_data = dict()
# You can set only one task at a time
    post_data[len(post_data)] = dict(
        language_code=Y, #en
        location_code=2840,#us
        keyword=X+" "+key
    )
    response = client.post("/v3/serp/google/organic/live/regular", post_data)

    #due to https://www.google.com/search?num=100&hl=hu&gl=US&ei=4gsYYKjYA5CMsQXKpZ24CA&q=%22Trans-Sped+Logisztikai+Szolg%C3%A1ltat%C3%B3+K%C3%B6zpo%22+AND+++peres+OR+csal%C3%A1s+OR+ad%C3%B3elker%C3%BCl%C3%A9s+OR+p%C3%A9nzmos%C3%A1s+OR+sikkaszt%C3%A1s+OR+veszteget%C3%A9s+OR+korrupci%C3%B3&oq=%22Trans-Sped+Logisztikai+Szolg%C3%A1ltat%C3%B3+K%C3%B6zpo%22+AND+++peres+OR+csal%C3%A1s+OR+ad%C3%B3elker%C3%BCl%C3%A9s+OR+p%C3%A9nzmos%C3%A1s+OR+sikkaszt%C3%A1s+OR+veszteget%C3%A9s+OR+korrupci%C3%B3&gs_lcp=CgZwc3ktYWIQAzIECAAQRzIECAAQRzIECAAQR1CDH1iQIWCyJGgAcAN4AIABAIgBAJIBAJgBAaABAaoBB2d3cy13aXrIAQPAAQE&sclient=psy-ab&ved=0ahUKEwio0cja7sjuAhUQRqwKHcpSB4cQ4dUDCA0&uact=5
    #if error then please create log command
    if (response["tasks"][0]["result"][0]["items"]==None) or (response["tasks"][0]["result"]==None):
        ji=keywords[keywords["language"]=="en"]["keywords"].values[0]
        ji=re.sub('"', '', ji)
        ji=ji.split(" OR ")
        ji[0]=ji[0][1:]
        ji.append(a.iloc[s][0])
        key_table=pd.DataFrame(index = range(1),columns= ji)
        key_table.insert(0, "article", "Any")
        key_table.insert(1, "link", "Any")
        key_table.insert(2, "domain", "Any")
        key_table.insert(3, "description", "Any")
        key_table.insert(4, "topic", "Any")
        key_table.insert(5, "subject name", "Any")
        key_table.insert(6, "Text", "Any")
        key_table.insert(7, "Check url", "Any")
        for o in range(1):
            key_table.iloc[o,0]="NA"
            key_table.iloc[o,1]="NA"
            key_table.iloc[o,2]="NA"
            key_table.iloc[o,3]="NA"
            key_table.iloc[o,4]="NA"
            key_table.iloc[o,5]=a['Name'][s]
            key_table.iloc[o,6]="No results error. Please check the subject manually by check url"
            key_table.iloc[o,7]=response["tasks"][0]["result"][0]['check_url']
            key_table.iloc[o,15]=0
        #key_table=key_table[key_table.iloc[:,key_table.shape[1]-1]==1].reset_index()
        key_table=key_table.rename(columns={key_table.columns[15]: 'Full name match'})
        key_out=key_out.append(key_table,ignore_index=True)
        print(a['Name'][s]+ "Response error, plese see the key_out",datetime.now())

    else:
        url = pd.DataFrame(index=range(len(response["tasks"][0]['result'][0]["items"])),columns=range(0))
        for x in range(0, len(response["tasks"][0]['result'][0]["items"])):
            url.loc[x,1]=response["tasks"][0]['result'][0]["items"][x]["url"]
        texx=[None]*len(response["tasks"][0]['result'][0]["items"])
        print("Response ok",datetime.now())
        log.iloc[1,1]=datetime.now()
        log.iloc[1,0]="Response ok"
        n=0
        for x in range(0, len(response["tasks"][0]['result'][0]["items"])):
            if url.iloc[x].values[0][-3:] not in ["pdf","xlsx","xls"]:
                try:
                    texx[x-n]=re.sub('\W+',' ',text_from_html(requests.get(url.loc[x,1], headers=headers).text)[0:10000].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace("  "," "))
                except:
                    texx[x-n]=(url.loc[x,1])
            else:
                n=n+1
        texx=texx[0:len(texx)-n]
        print("Read ok",datetime.now())

        n=0
        for z in range(len(texx)):
            if len(texx[z-n])<200:
                texx.pop(z-n)
                n=n+1
            else:
                pass
        print("cut short article ok",datetime.now())
        log.iloc[2,1]=datetime.now()
        log.iloc[2,0]="cut short article ok"
        #n=0
        #for p in range(len(texx_clean)):
        #    if a.iloc[0][0] in texx_clean[p-n]:
        #        pass)
        #    else:
        #        texx_clean.pop(p-n)
        #        n=n+1

        #print("Name in the first 10000 characters ok")

        if len(texx)==0:
            ##sometimes occure because of filtering user agent by sites. See desription of the problem: https://stackoverflow.com/questions/27231113/999-error-code-on-head-request-to-linkedin
            ji=keywords[keywords["language"]=="en"]["keywords"].values[0]
            ji=re.sub('"', '', ji)
            ji=ji.split(" OR ")
            ji[0]=ji[0][1:]
            ji.append(a.iloc[s][0])
            key_table=pd.DataFrame(index = range(1),columns= ji)
            key_table.insert(0, "article", "Any")
            key_table.insert(1, "link", "Any")
            key_table.insert(2, "domain", "Any")
            key_table.insert(3, "description", "Any")
            key_table.insert(4, "topic", "Any")
            key_table.insert(5, "subject name", "Any")
            key_table.insert(6, "Text", "Any")
            key_table.insert(7, "Check url", "Any")

            for o in range(1):
                key_table.iloc[o,0]="NA"
                key_table.iloc[o,1]="NA"
                key_table.iloc[o,2]="NA"
                key_table.iloc[o,3]="NA"
                key_table.iloc[o,4]="NA"
                key_table.iloc[o,5]=a['Name'][s]
                key_table.iloc[o,6]="No results error. For more details see check url. It could happen if only response error recieved from sties or there are only linkedin results. They usually filter user agents and make every webscrapping stop"
                key_table.iloc[o,7]=response["tasks"][0]["result"][0]['check_url']
                key_table.iloc[o,15]=0
            #key_table=key_table[key_table.iloc[:,key_table.shape[1]-1]==1].reset_index()
            key_table=key_table.rename(columns={key_table.columns[15]: 'Full name match'})
            key_out=key_out.append(key_table,ignore_index=True)
            print(a['Name'][s]+ "No results error. ",datetime.now())

        else:
            texx_clean = texx
            for g in range(len(texx_clean)):
                texx_clean[g] = ''.join(c for c in texx[g] if not c.isdigit())

            print("clean ok",datetime.now())

            log.iloc[3,1] = datetime.now()
            log.iloc[3,0] = "clean ok"

            lem = []
            num = []
            ID = []
            lang = []
            n = 0
            for h in range (len(texx_clean)):
                if langdet(texx[h])["language"]== "en":
                    #pip install en_core_web_sm
                    import en_core_web_sm
                    doc=nlp(texx_clean[h])
                    b=[]
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h-n]))
                    lang.extend(np.repeat(langdet(texx[h])['language'], num[h-n]))
                elif langdet(texx[h])["language"]== "hu":
                    #pip install https://github.com/oroszgy/spacy-hungarian-models/releases/download/hu_core_ud_lg-0.3.1/hu_core_ud_lg-0.3.1-py3-none-any.whl
                    #note: the incompatibility arning is a bug, see: https://github.com/oroszgy/spacy-hungarian-models/issues/15
                    import hu_core_ud_lg
                    doc=nlp(texx_clean[h])
                    b=[]
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h-n]))
                    lang.extend(np.repeat(langdet(texx[h])['language'], num[h-n]))
                elif langdet(texx[h])["language"]== "pl":
                    #C:\windows\system32>pip install C:\Users\csetri1dani428\downloads\pl_spacy_model-0.1.0.tar.gz
                    #python3 -m spacy download pl_core_news_sm
                    import pl_core_news_sm
                    doc=nlp(texx_clean[h])
                    b=[]
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h-n]))
                    lang.extend(np.repeat(langdet(texx[h])['language'], num[h-n]))
                elif langdet(texx[h])["language"]== "de":
                    #python -m spacy download de_core_news_sm
                    import de_core_news_sm
                    doc=nlp(texx_clean[h])
                    b=[]
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h-n]))
                    lang.extend(np.repeat(langdet(texx[h])['language'], num[h-n]))
                elif langdet(texx[h])["language"]== "fr":
                    #python -m spacy download fr_core_news_sm
                    import fr_core_news_sm
                    doc=nlp(texx_clean[h])
                    b=[]
                    for token in doc:
                        lem.append(token.lemma_.split())
                        b.append(token.lemma_.split())
                    num.append(len(b))
                    ID.extend(np.repeat(h, num[h-n]))
                    lang.extend(np.repeat(langdet(texx[h])['language'], num[h-n]))
                else:
                    n=n+1

            ID=pd.DataFrame(ID)
            ID=ID.loc[:,[0]]
            lem=pd.DataFrame(lem)
            lem=lem.loc[:,[0]]
            lang=pd.DataFrame(lang)
            lang=lang.loc[:,[0]]
            dd=pd.concat([ID,lem,lang], axis=1)
            dd.columns= ["ID", "unigrams","language"]

            print("lematize ok",datetime.now())
            log.iloc[4,1]=datetime.now()
            log.iloc[4,0]="lematize ok"
            dd=dd[(dd.language=="en") & (~dd['unigrams'].isin(stopen))].append([dd[(dd.language=="fr") & (~dd['unigrams'].isin(stopfr))]
                                                                         ,dd[(dd.language=="hu") & (~dd['unigrams'].isin(stophu))]
                                                                         ,dd[(dd.language=="pl") & (~dd['unigrams'].isin(stoppl))]
                                                                         ,dd[(dd.language=="de") & (~dd['unigrams'].isin(stopde))]], ignore_index=True)

            dd['length'] = dd.unigrams.str.len()
            dd = dd.drop(dd[dd["length"] <= 3].index)
            dd=dd.dropna()

            print("stop_words ok",datetime.now())
            log.iloc[5,1]=datetime.now()
            log.iloc[5,0]="stop_words ok"

            output_series = dd.groupby(['ID'])['unigrams'].apply(list)
            data_words = list(sent_to_words(output_series))
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=70) # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words], threshold=70)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            data_words=trigram_mod[bigram_mod[data_words]]
            print("data_words ok",datetime.now())

            article_list=[]
            for q in range(len(data_words)):
                article_list.extend(list(data_words[q]))
            uniq_list=list(set(article_list))
            uniq_num=[0]*len(uniq_list)

            dwapp=[]
            for i in range(len(data_words)):
                dwapp.extend(data_words[i])

            for j in range(len(uniq_list)):
                uniq_num[j] = uniq_num[j]+sum(x in uniq_list[j] for x in dwapp)
            print("uniq",datetime.now())
            log.iloc[6,1]=datetime.now()
            log.iloc[6,0]="uniq ok"

            uniq_list=pd.DataFrame(uniq_list)
            uniq_num=pd.DataFrame(uniq_num)
            dd2=pd.concat([uniq_list,uniq_num], axis=1)
            dd2.columns= ["unique_words", "frequency"]
            bad_words=dd2.loc[(dd2['frequency'] >= len(texx)*0.7)]
            bad_words_list = bad_words['unique_words'].tolist()

            bad_words2=dd2.loc[(dd2['frequency'] <= 2)]
            bad_words2_list = bad_words2['unique_words'].tolist()
            aaa=[]
            for t in range(len(data_words)):
                aaa.append(data_words[t])

            for t in range(len(aaa)):
                aaa[t] = [item for item in aaa[t] if item not in bad_words_list]
                aaa[t] = [item for item in aaa[t] if item not in bad_words2_list]

            print("badwords_ok",datetime.now())
            log.iloc[7,1]=datetime.now()
            log.iloc[7,0]="badwords ok"
            output_series = pd.Series(aaa)
            data_words = list(sent_to_words(output_series))

            #remove frequent and rare words
            # Create Dictionary
            id2word = corpora.Dictionary(data_words)
            # Create Corpus
            texts = data_words

            # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in texts]

            print("corpus ok",datetime.now())
            if len(texx)>10:
                hdp = gensim.models.hdpmodel.HdpModel(corpus=corpus, id2word=id2word, alpha = .5)
                ii=[None]*len(corpus)
                for i in range(len(corpus)):
                    ii[i]=hdp[corpus[i]]
                get_document_topics=ii
            else:
                pass
            print("hdp ok",datetime.now())
            log.iloc[8,1]=datetime.now()
            log.iloc[8,0]="hdp ok"

            texx=[texx[i] for i in list(set(dd.ID))]
            ji=keywords[keywords["language"]=="en"]["keywords"].values[0]
            ji=re.sub('"', '', ji)
            ji=ji.split(" OR ")
            ji[0]=ji[0][1:]
            ji.append(a.iloc[s][0])
            target=re.sub('"', '',keywords[keywords["language"]==a["Language"][s]]["keywords"].values[0]).split(" OR ")
            target[0]=target[0][1:]
            target.append(a.iloc[s][0])
            key_table=pd.DataFrame(index = range(len(aaa)),columns= ji)
            key_table.insert(0, "article", "Any")
            key_table.insert(1, "link", "Any")
            key_table.insert(2, "domain", "Any")
            key_table.insert(3, "description", "Any")
            key_table.insert(4, "topic", "Any")
            key_table.insert(5, "subject name", "Any")
            key_table.insert(6, "Text", "Any")
            key_table.insert(7, "Check url", "Any")

            for o in range(len(aaa)):
                key_table.iloc[o,0]=response["tasks"][0]['result'][0]["items"][o]["title"]
                key_table.iloc[o,1]=response["tasks"][0]['result'][0]["items"][o]["url"]
                key_table.iloc[o,2]=response["tasks"][0]['result'][0]["items"][o]["domain"]
                key_table.iloc[o,3]=response["tasks"][0]['result'][0]["items"][o]["description"]
                key_table.iloc[o,5]=a['Name'][s]
                key_table.iloc[o,6]=texx[o]
                key_table.iloc[o,7]=response["tasks"][0]["result"][0]['check_url']
                if len(texx)>10:
                    if len(get_document_topics[o])>0:
                        key_table.iloc[o,4]=get_document_topics[o][0][0]
                    else:
                        key_table.iloc[o,4]=99
                else:
                    key_table.iloc[o,4]=99

                for i in range(len(target)):
                    if target[i] in texx[o]:
                        key_table.iloc[o,i+8] =1
                    else:
                        key_table.iloc[o,i+8] =0
            #key_table=key_table[key_table.iloc[:,key_table.shape[1]-1]==1].reset_index()
            key_table=key_table.rename(columns={key_table.columns[15]: 'Full name match'})
            key_out=key_out.append(key_table,ignore_index=True)
            print("keytable ok",datetime.now())

            log.iloc[9,1]=datetime.now()
            log.iloc[9,0]="keytable ok"
            time_log.iloc[s,1]=(log[log.step=="keytable ok"].time[9]-log[log.step=="Start Time"].time[0]).total_seconds()
            time_log.iloc[s,0]=a['Name'][s]
            time_log.iloc[s,2]=len(texx)
time_log.to_csv("time_log.csv")
key_out.to_csv("key_out.csv")
import os
f=pd.read_csv("key_out.csv").iloc[1:]#lső sor nan a létrehozásnál betett sortól
li=list(f["subject name"].unique())
fileSize=[]
for i in range(len(li)):
    sep_table=f[f["subject name"]==li[i]]
    filename = li[i]+".csv"
    filename=filename.replace('"',"" ) #a fájl nevében szerepel idézőjel ami miatt windws serveren ez hibára fog futni, célszerű a kettős kereséseket elkerülni
    sep_table.to_csv(filename)
    fileInfo = os.stat(filename)
    fileSize.append(fileInfo.st_size/1000)
d = {'Name':li,'size':fileSize}
d= pd.DataFrame(d)
d.to_csv("filesize.csv")

