from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList


client = QdrantClient(url="http://localhost:6333")

score_threshold = 0.35


def createQdrant(user_id):
    """creates Qdrant collection if it does not exists with the collection name {user_id}

    input-
    user_id: str

    return-
    Bool - true if successful else None
    """
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
    """adds values to exsisting Qdrant DB
    take in 2 args:
    user_id: str
    data: dict containing "payload" and "vector" key and value pairs

    returns:
    True if added successfully else None
    """
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


def searchQdrant(user_id, vector, isVectorRequired=False):
    """
    this searched the Qdrant Db for the given vector

    input:
    user_id: str
    vector: list['floats']
    isVectorRequired: Bool (default to false)

    Output:
    filtered_results: list['object']
    if isVectorRequired is True then return the vector value in the object
    """
    try:
        search_result = client.query_points(
            collection_name=user_id,
            query=vector,
            query_filter=Filter(
                must=[FieldCondition(
                    key="status", match=MatchValue(value="True"))]
            ),
            with_vectors=isVectorRequired,
            with_payload=True,
            limit=5,
        ).points
        filtered_results = [
            res for res in search_result if res.score >= score_threshold]
        return filtered_results
    except Exception as e:
        print("error searching:", e)
        return e


def deleteQdrant(user_id, point_id):
    """
    sets the payload of the given point_id

    input-
    user_id: str
    point_id: str

    return-
    Bool - true if successful else None
    """
    try:
        delete_qdrant = client.set_payload(
            collection_name=user_id,
            payload={
                "status": "deleted"
            },
            points=PointIdsList(points=[point_id])
        )
        print("Deleted payload successfully")
        return True
    except Exception as e:
        print("vector could not be deleted:", e)
        return None
