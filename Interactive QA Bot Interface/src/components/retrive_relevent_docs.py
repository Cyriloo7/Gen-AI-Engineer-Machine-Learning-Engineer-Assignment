import sys
from src.exception.exception import customexception
from src.logger.logger import logger
import numpy as np
import cohere

class RetriveReleventDocs:
    def __init__(self):
        logger.info("Retrieving documentation from the vector database")
        self.co = cohere.Client('SGXUJ2vUDqaNNpJwh1ffmo1PFkGmN50W6ghcW4UA')
        pass

    def retrieve(self, query, index, chunks, k=3):
        try:
            query_embed = self.co.embed(texts=[query], model="embed-english-v3.0", input_type="search_document").embeddings
            D, I = index.search(np.array(query_embed).astype(np.float32), k)
            results = [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
            results.sort(key=lambda x: x[1])
            return results
        
        except Exception as e:
            logger.error(f"Error occurred while retrieving relevant documents: {str(e)}")
            raise customexception(e, sys)