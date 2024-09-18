from src.exception.exception import customexception
import sys
from src.logger.logger import logger


class SplitIntoChunks:
    def __init__(self):
        logger.info("Initializing SplitIntoChunks class")
        pass

    # Function to split document text into smaller chunks
    def split_text(self, text, chunk_size=500):
        try:
            logger.info("Splitting document text into smaller chunks")
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        except Exception as e:
            raise customexception(e, sys)