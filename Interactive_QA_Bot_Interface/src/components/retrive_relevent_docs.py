import sys
from src.exception.exception import customexception
from src.logger.logger import logger
import numpy as np
import cohere

class RetriveReleventDocs:
    def __init__(self):
        logger.info("Retrieving documentation from the vector database")
        self.co = cohere.Client('6aqPnWpIVEDJ4VxllhTTMLj0fhsG8xtNmOYZ100I')
        pass

    def normalize(self, embedding):
        norm = np.linalg.norm(embedding)
        return embedding / norm

    def retrieve(self, query, index, chunks, k=10, score_threshold=0.4):
        try:
            # Generate embedding for the query
            query_embed = self.co.embed(texts=[query], model="embed-multilingual-v3.0", input_type="search_document", truncate='END').embeddings

            # Search the FAISS index for the k most relevant document chunks
            normalized_query_embed = self.normalize(np.array(query_embed).astype(np.float32))
            D, I = index.search(normalized_query_embed, k)

            # Extract the chunks and their scores
            results = [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]

            # Sort results based on the similarity score in descending order
            results.sort(key=lambda x: x[1], reverse=True)

            # Filter results based on the similarity score threshold to ensure relevance
            filtered_results = [(chunk, score) for chunk, score in results if score >= score_threshold]

            if not filtered_results:
                logger.info(f"No relevant results found above the score threshold {score_threshold}.")
                return []  # Return an empty list if no relevant context is found

            return filtered_results

        except Exception as e:
            logger.error(f"Error occurred while retrieving relevant documents: {str(e)}")
            raise customexception(e, sys)
