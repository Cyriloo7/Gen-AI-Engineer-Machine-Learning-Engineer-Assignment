import sys
from src.exception.exception import customexception
from src.logger.logger import logger
import faiss
import numpy as np

class FAISSIndexing:
    def __init__(self):
        logger.info("FAISS Indexing initiated")
        pass

    # Function to build FAISS index for embeddings
    def build_faiss_index(self, embeddings):
        try:
            dimension = 1024
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype(np.float32))
            return index
        except Exception as e:
            logger.error(f"Error while building FAISS index: {str(e)}")
            raise customexception(e, sys)