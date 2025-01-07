import aiohttp
from icecream import ic
from config.config import Config

async def user_status(lead_id: str) -> str:
    url = f'{Config.BASE_URL_LEAD}/{lead_id}'

    headers = {
        "accept": "application/json",
        "authorization": Config.SEND_ID
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    status_id = data.get("status_id")
                    return status_id
                else:
                    ic(f"Unexpected status code: {response.status}")
                    return f"Error: Unexpected status code {response.status}"

    except aiohttp.ClientResponseError as http_err:
        ic(f"HTTP error occurred: {http_err}")
