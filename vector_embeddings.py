import time
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue


client = QdrantClient(url="http://localhost:6333")


def createQdrant(user_id):
    try:
        client.get_collection(user_id)
        print("collection already exists")
        return True
    except:
        try:
            client.create_collection(
                collection_name=user_id,
                vectors_config=VectorParams(
                    size=384, distance=Distance.COSINE),
            )
            return True
        except Exception as e:
            print("Error Creating collection:", e)
            return None


def addToQdrant(user_id, data):
    # payload contains the status and vector contains the embeddings
    try:
        requestss = client.upload_collection(
            collection_name=user_id,
            payload=data["payload"],
            vectors=data["vector"],
        )
        print("added vector to qdrant DB")
        return True
    except Exception as e:
        print("Error adding:", e)
        return None


def searchQdrant(user_id, vector):
    try:
        search_result = client.query_points(
            collection_name=user_id,
            query=vector,
            query_filter=Filter(
                must=[FieldCondition(
                    key="status", match=MatchValue(value="True"))]
            ),
            with_payload=True,
            limit=5,
        ).points
        return search_result
    except Exception as e:
        print("error searching:", e)
        return None


def deleteQdrant(user_id, vector):
    try:
        delete_qdrant = client.set_payload(
            collection_name=user_id,
            payload=[{
                "status": "deleted"
            }],
            points=vector
        )
        print("Deleted payload successfully")
        return True
    except Exception as e:
        print("vector could not be deleted:", e)
        return None


# # code to view collections in console
# scroll_result, next_page = client.scroll(
#     collection_name="test",
#     with_payload=True,
#     with_vectors=True,
#     limit=100  # Adjust this as needed
# )

# for point in scroll_result:
#     print(f"ID: {point.id}")
#     print(f"Vector: {point.vector}")
#     print(f"Payload: {point.payload}")
#     print("-" * 30)
