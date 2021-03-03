import datetime

import pandas as pd
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as sw_fr
from spacy.lang.en.stop_words import STOP_WORDS as sw_en
from spacy.lang.de.stop_words import STOP_WORDS as sw_de
from spacy.lang.hu.stop_words import STOP_WORDS as sw_hu
from spacy.lang.pl.stop_words import STOP_WORDS as sw_pl
from spacy_langdetect import LanguageDetector


class LanguageModels:

    def __init__(self):
        self.stopword_fr = list(sw_fr)
        self.stopword_en = list(sw_en)
        self.stopword_de = list(sw_de)
        self.stopword_hu = list(sw_hu)
        self.stopword_pl = list(sw_pl)

        # Base nlp, to detect lang only
        self.base_nlp = spacy.load("en_core_web_sm")
        self.base_nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

        # Load keywords from csv
        self.keywords = pd.read_csv("search/keywords2.csv", sep=',', encoding='utf8')
        self.headers = {
            'Accept-Encoding': 'gzip, deflate, sdch',
            'Accept-Language': 'en-US,en;q=0.8',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
        }

    def detect_language(self, s):
        doc = self.base_nlp(s)
        return doc._.language

    def clean_vars(self):
        self.key_out = pd.DataFrame(
            columns=["article", "link", "domain", "description", "topic", "subject name", "Text",
                     "litigation", "fraud", "tax evasion", "money laundering", "embezzlement",
                     "bribery", "corruption", "Full name match"])
        self.time_log = pd.DataFrame(columns=["subject", "time", "results"])

    @property
    def nlp_en(self):
        return spacy.load('en_core_web_sm')

    @property
    def nlp_de(self):
        return spacy.load('de_core_news_sm')

    @property
    def nlp_fr(self):
        return spacy.load('fr_core_news_sm')

    @property
    def nlp_hu(self):
        return spacy.load('hu_core_ud_lg')

    @property
    def nlp_pl(self):
        return spacy.load('pl_core_news_sm')

    @property
    def nlp_es(self):
        return spacy.load('pl_core_news_sm')


