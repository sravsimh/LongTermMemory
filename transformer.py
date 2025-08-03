from sentence_transformers import SentenceTransformer
import numpy as np
import time
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def createEmbeddings(data):
    """
    this creates vector embedding of the input data

    input-
    data: list['str']

    return-
    embeddigns: list[list['floats']]
    """

    try:
        model = SentenceTransformer(
            'all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
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
