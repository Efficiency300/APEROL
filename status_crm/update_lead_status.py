import aiohttp
from icecream import ic
from config.config import Config

async def update_lead_status(lead_id: str, new_status_id: int) -> None:
    url = f'{Config.BASE_URL_LEAD}/{lead_id}'

    headers = {
        'Authorization': Config.SEND_ID,
        'Content-Type': 'application/json'
    }

    data = {
        'status_id': new_status_id
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.patch(url, json=data, headers=headers) as response:
                if response.status == 200:
                    ic('Лид успешно перемещен на новый этап.')
                else:
                    ic(f'Неожиданная ошибка: {response.status}, {await response.text()}')

    except aiohttp.ClientResponseError as http_err:
        ic(f'HTTP error occurred: {http_err}')

