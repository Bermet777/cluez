import json

import requests


def check(string):
    session = requests.Session()
    session.auth = ('admin', 'pass')

    auth = session.post('http://' + 'spacy-en.cluez.io')
    response = session.post('http://' + 'spacy-en.cluez.io' + '/tokenizer', json={"text": "Skip to content AMC Central Europe AMC Networks"})

    print(response.content)
    print(json.loads(response.content))


async def get_tokens(string):
    pass


if __name__ == '__main__':
    check("asd")
