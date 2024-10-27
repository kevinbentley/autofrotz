import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http import models
import uuid

# collection is the name of the collection in qdrant
# embeddings is an array of floating point values
# data is a dictionary (json format), it gets stored in the database with the vectors
def write_vector(client, collection, embeddings, data):
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=str(data["id"]),
                vector=embeddings.tolist(),
                payload=data
            )
            ]
    )

def make_collection(client, collection_name, vecsize):

    collections = client.get_collections()
    for collection in collections.collections:
        if(collection.name==collection_name):
            return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vecsize, distance=models.Distance.COSINE),
    )


def search(client, collection, vec, numResults):
    
    search_result = client.search(
        collection_name=collection,
        query_vector=vec, 
        limit=numResults)
    return search_result

def get_client(host, port):
    return QdrantClient(host=host, port=port)

def get_nearest(client, embedding, k, collection):
    pass

def get_all_documents(client, doc_collection,filter_collection):
    all = []
    next_page_offset = 0
    while True:        
        scrollResponse = client.scroll(
            offset=next_page_offset,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="collection", 
                        match=models.MatchValue(value=filter_collection)
                    ),
                ]
            ),
            collection_name=doc_collection,    
            limit=100,
            with_payload=True,
            with_vector=False)
        print(str(scrollResponse))
        if(scrollResponse==None):
            break
        for result in scrollResponse[0]:            
            all.append(result)
        next_page_offset = scrollResponse[1]
        if(next_page_offset==None):
            break
    return all

def get_by_id(client, collection, id, with_payload, with_vector):
    points = client.retrieve(collection,[id],with_payload,with_vector)
    return points