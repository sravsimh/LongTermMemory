from sentence_transformers import SentenceTransformer
import numpy as np
import time
# from vector_embeddings import createQdrant, addToQdrant, searchQdrant
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def createEmbeddings(data):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        model.encode(['warminggu'], convert_to_numpy=True)

        batchSize = 50
        allEmbeds = []

        startTime = time.time()
        for i in range(0, len(data), batchSize):
            batch = data[i:i+batchSize]
            emb = model.encode(batch, convert_to_numpy=True).astype('float32')
            allEmbeds.append(emb)
            # print(
            #     f"Encoded batch {i//batchSize + 1} / {len(data)//batchSize + 1}")

        embeddings = np.vstack(allEmbeds)
        # print(f"Done encoding in {time.time() - startTime:.2f}s")
        return embeddings
    except Exception as e:
        print("error creating embedding", e)
        return None
