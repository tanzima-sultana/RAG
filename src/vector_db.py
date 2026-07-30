from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pickle 

from config import DOCKER_PORT

class VectorDB:
    def __init__(self, db_name):
        self.db_name = db_name
        self.client = QdrantClient(host="localhost", port=DOCKER_PORT)

    def insert_update_embeddings(self, name, chunk_ids, embedding, batch_size=256):
        for start in range(0, len(chunk_ids), batch_size):
            #end = start + batch_size
            end = min(start + batch_size, len(chunk_ids))
            points = [
                PointStruct(
                    id=i,
                    vector=embedding[i],
                    payload={"chunk_id": chunk_ids[i]}
                )
                for i in range(start, end)
            ]
            self.client.upsert(collection_name=name, points=points)
    
    def get_embeddings(self, name):
        info = self.client.get_collection(collection_name=name)
        print(info.points_count)
    
    def search(self, name, query_embedding, k):
        results = self.client.query_points(
            collection_name=name,
            query=query_embedding,
            limit=k,
        )
        return results.points

    def create_vector_db(self, embedding_path):

        name_etxn = embedding_path.removeprefix("embeddings/").removesuffix(".pkl").replace("/", "_")
        name = f"{self.db_name}_{name_etxn}"

        try:
            # Load embedding_map
            embeddings_map = None
            with open(embedding_path, "rb") as f:
                embeddings_map = pickle.load(f)

            # Load embedding
            chunk_ids = list(embeddings_map.keys())
            embedding = list(embeddings_map.values())

            embedding_dim = len(embedding[0]) 
            
            # Create client
            if not self.client.collection_exists(collection_name=name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
                )

            # Update or insert into vector db
            self.insert_update_embeddings(name, chunk_ids, embedding)

            # Get
            #self.get_embeddings(name)
        except Exception as e:
            print(f"vectorDB failed: {e}")
            return None

        return name 
    
