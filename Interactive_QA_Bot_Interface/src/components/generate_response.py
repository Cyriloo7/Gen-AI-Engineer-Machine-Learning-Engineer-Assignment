import sys
from src.exception.exception import customexception
from src.logger.logger import logger
import cohere 


class GenerateResponse:
    def __init__(self):
        logger.info("Generating response from prompt")
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        self.co = cohere.Client(cohere_api_key)
        #self.co = cohere.Client(${{secrets.COHERE_API}})
        pass

    def generate_response_from_prompt(self, prompt):
        try:
            stream = self.co.chat_stream(
                model='command-r-plus-08-2024',
                message=prompt,
                temperature=0.8,
                chat_history=[],
                prompt_truncation='AUTO',
                #connectors=[{"id":"web-search"}],
                max_tokens=4096
            )

            generated_text = ""
            for event in stream:
                if event.event_type == "text-generation":
                    generated_text += event.text


            return generated_text

        except customexception as e:
            logger.error(f"Error generating response: {e}")
            raise customexception(e, sys)
