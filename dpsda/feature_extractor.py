import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

# https://github.com/UKPLab/sentence-transformer


def extract_features(
        data,
        batch_size=1000,
        model_name="all-mpnet-base-v2"):
    # If available, the model is automatically executed on the GPU. You can specify the device for the model like this:

    model = SentenceTransformer(model_name)  # device='cuda',
    model.eval()

    with torch.no_grad():
        sentence_embeddings = []
        for i in tqdm(range(len(data) // batch_size+1)):
            embeddings = model.encode(
                data[i * batch_size:(i + 1) * batch_size])
            if len(embeddings) > 0:
                sentence_embeddings.append(embeddings)
    sentence_embeddings = np.concatenate(sentence_embeddings)
    del model
    return sentence_embeddings
