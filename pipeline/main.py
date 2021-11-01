from prefect import task, Flow
from datetime import timedelta
from prefect.schedules import IntervalSchedule
from twitter.twitter import Twitter
from database.connect import Database

twitter_extractor = Twitter()
database = Database()


@task()
def extract(q='@zara'):
    """

    :return:
    """

    _tweets = twitter_extractor.search_by_word(q)

    return _tweets


@task()
def transform(tweets):
    """

    :return:
    """

    return tweets


@task()
def load(collection_name, tweets):
    """

    :return:
    """

    for _tweet in tweets:
        database.save_document_with_collection(collection_name, _tweet._json)


def build_flow():
    schedule = IntervalSchedule(interval=timedelta(minutes=1))

    with Flow('ETL', schedule) as _flow:
        _zara_tweets = extract(q='@zara')
        _hm_tweets = extract(q='@hm')
        _mango_tweets = extract(q='@Mango')

        _zara_tweets = transform(_zara_tweets)
        _hm_tweets = transform(_hm_tweets)
        _mango_tweets = transform(_mango_tweets)

        load('comments', _zara_tweets)
        load('HM', _hm_tweets)
        load('Mango', _mango_tweets)

    return _flow
