import aiohttp
from icecream import ic
from config.config import Config

async def amo_api_data():
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "amo_host": Config.AMO_HOST,
        "amo_password": Config.AMO_PASSWORD,
        "amo_email": Config.AMO_EMAIL
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(Config.TOKEN_GET_URL, json=payload, headers=headers) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return {
                        "amojo_id": response_data.get('amojo_id'),
                        "chat_token": response_data.get('chat_token')
                    }
                else:
                    ic(f"HTTP ошибка: {response.status} - {await response.text()}")

    except Exception as e:
        ic(f"Произошла неизвестная ошибка: {e}")


