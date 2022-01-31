from prefect import task, Flow
from datetime import timedelta
from prefect.schedules import IntervalSchedule
from apps.twitter.twitter import Twitter
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
    schedule = IntervalSchedule(interval=timedelta(minutes=10))

    with Flow('ETL', schedule) as _flow:
        _zara_tweets = extract(q='@zara')
        _hm_tweets = extract(q='@hm')
        _mango_tweets = extract(q='@Mango')
        _nike_tweets = extract(q='@Nike')
        _adidas_tweets = extract(q='@adidas')

        load('zara', _zara_tweets)
        load('hm', _hm_tweets)
        load('mango', _mango_tweets)
        load('nike', _nike_tweets)
        load('adidas', _adidas_tweets)

    return _flow


flow = build_flow()
flow.run()
