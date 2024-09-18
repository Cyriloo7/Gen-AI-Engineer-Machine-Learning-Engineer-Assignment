from src.exception.exception import customexception
import sys
from src.logger.logger import logger
import PyPDF2

class PDFFileReader:
    def __init__(self):
        logger.info("PDFFileReader constructor called")
        pass

    
    def extract_text_from_pdf(self, pdf_path):
        try:
            text = ""
            #with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"An error occurred while reading PDF: {str(e)}")
            raise customexception(e, sys)

