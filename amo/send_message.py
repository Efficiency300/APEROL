import re
import aiohttp
import logging
from config.config import Config
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
PHOTOS_DIR = BASE_DIR / "photos"

async def send_message(amojo_id, chat_token, message, chat_id):
    try:

        message_text = re.sub(r'【.*?】.*', '', message)

        data = {
            "message": message_text,
            "chat_id": chat_id,
            "chat_token": chat_token,
            "amojo_id": amojo_id,
            "token": Config.SEND_ID,
        }

        # Asynchronous HTTP POST request
        async with aiohttp.ClientSession() as session:
            async with session.post(Config.MESSAGE_SAND_URL, data=data) as response:
                if response.status == 200:
                    logger.info("Сообщение успешно отправлено")
                else:
                    response.raise_for_status()

    except aiohttp.ClientResponseError as http_err:
        logger.error("HTTP error occurred: %s", http_err)

