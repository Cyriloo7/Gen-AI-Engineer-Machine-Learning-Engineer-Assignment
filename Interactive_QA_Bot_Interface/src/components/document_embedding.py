import cohere
import numpy as np
import sys
from src.exception.exception import customexception
from src.logger.logger import logger


class DocumentEmbedding:
    def __init__(self):
        logger.info("Document Embedding Class In Progress")
        self.co = "cVm83R5JH4xU5K7FgGqcJRn2YSrPYmYAx8f9hQUh"
        pass

    def create_embeddings(self, texts, batch_size=40):
        try:
            embeddings = []
            for i in range(0, len(texts), batch_size):

                batch = texts[i:i+batch_size]
                response = self.co.embed(texts=batch, model="embed-multilingual-v3.0", input_type="search_document", truncate='END')
                embeddings.extend(response.embeddings)

            return np.vstack(embeddings)
        except Exception as e:
            logger.error("Error in creating embeddings: ", exc_info=True)
            raise customexception(e, sys)
