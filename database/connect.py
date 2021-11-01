import pymongo
from config import variables


class Database:
    def __init__(self, collection_name='comments'):
        self.client = pymongo.MongoClient(
            f"mongodb+srv://"
            f"{variables.username_database}:"
            f"{variables.password_database}@cluster0.25joh.gcp.mongodb.net/"
            f"{variables.database_name}?retryWrites=true&w=majority")

        self.database = self.client.get_database(variables.database_name)
        self.collection = self.database.get_collection(collection_name)

    def get_database(self):
        return self.database

    def get_collection(self, collection='comments'):
        return self.database.get_collection(collection)

    def save_document_with_collection(self, collection_name, document):
        collection = self.database.get_collection(collection_name)
        return collection.insert_one(document)

    def save_document(self, document):
        return self.collection.insert_one(document)
