import tensorflow.keras as k
from sklearn.feature_extraction.text import CountVectorizer


def get_char_vectorizer(corpus):
    char_vectorizer = k.preprocessing.text.Tokenizer(
        char_level=True, oov_token='<OOV>', filters='\t\n')
    char_vectorizer.fit_on_texts(corpus)

    return char_vectorizer

def get_tokens(url):
    # tokens = []
    url = str(url)
    url = url.replace('.', '/')
    url = url.replace('=', '/')
    url = url.replace('&', '/')
    url = url.replace('?', '/')
    url = url.replace('-', '/')
    url = url.replace('@', '/')
    url = url.replace(':', '/')
    tokens = url.split('/')
    return tokens

def get_word_vectorizer_v2(corpus):


    word_vectorizer = CountVectorizer(
        stop_words=None,
        min_df=5,
        tokenizer=get_tokens,
        # analyzer='word',
        max_features=500
    )
    word_vectorizer.fit(corpus)

    return word_vectorizer


def get_word_vectorizer(corpus):
    word_vectorizer = CountVectorizer(
        stop_words=None,
        min_df=5,
        token_pattern=r'&\w+;|[:/&?=.\[\]\\]|%\w{2}|[-_\w\d]+',
        analyzer='word',
        max_features=500
    )
    word_vectorizer.fit(corpus)

    return word_vectorizer
