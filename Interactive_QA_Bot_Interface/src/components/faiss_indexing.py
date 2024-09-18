import sys
from src.exception.exception import customexception
from src.logger.logger import logger
import faiss
import numpy as np

class FAISSIndexing:
    def __init__(self):
        logger.info("FAISS Indexing initiated")
        pass

    def normalize(self, embedding):
        norm = np.linalg.norm(embedding)
        return embedding / norm

    # Function to build FAISS index for embeddings
    def build_faiss_index(self, embeddings):
        try:
            dimension = 1024
            index = faiss.IndexFlatIP(dimension)
            normalized_embeddings = [self.normalize(embed) for embed in embeddings]
            index.add(np.array(normalized_embeddings).astype(np.float32))
            return index
        except Exception as e:
            logger.error(f"Error while building FAISS index: {str(e)}")
            raise customexception(e, sys)