import tweepy
from config import variables
from database.connect import Database


class Twitter:

    def __init__(self):
        auth = tweepy.OAuthHandler(variables.api_key, variables.api_secret)
        auth.set_access_token(variables.access_token, variables.access_token_secret)

        self.api = tweepy.API(auth)

        self.database = Database(collection_name='HM')

    def search_by_word(self, q='', max_id=None):
        return self.api.search_tweets(q=q, lang='es', result_type='recent')

    def get_tweets(self, max_id=None):
        tweets = self.search_by_word(q='@hm')

        for tw in tweets:
            self.database.save_document(tw._json)

        self.get_tweets(max_id=tweets.max_id)
