from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    ASSIST_ID = os.getenv('OPENAI_ASSISTANT_ID')
    SEND_ID = os.getenv("SEND_ID")
    MESSAGE_SAND_URL = os.getenv("MESSAGE_SAND_URL")
    TOKEN_GET_URL = os.getenv("TOKEN_GET_URL")
    AMO_HOST = os.getenv("AMO_HOST")
    AMO_PASSWORD = os.getenv("AMO_PASSWORD")
    AMO_EMAIL = os.getenv("AMO_EMAIL")
    BASE_URL_LEAD = os.getenv("BASE_URL_LEAD")

