import asyncio
import json
import random
from collections import defaultdict
from services.MarkdownProcessor import MarkdownProcessor
from services.photo_service import PhotoService
from services.stt_service import STTService
from amo.send_message import send_message
from services.llm_service import thread
from utils.logger import setup_logger
from amo.amo_data import amo_api_data

logger = setup_logger()

message_buffers = defaultdict(list)
message_timers = {}
BUFFER_DELAY = 5
def is_image(url: str) -> bool:
    return isinstance(url, str) and url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
def is_voice(url: str) -> bool:
    return isinstance(url, str) and url.lower().endswith(('.m4a', '.mp3', '.wav', '.ogg'))

async def process_attachments(user_input_list: list,photo_service: PhotoService,stt_service: STTService) -> tuple[str, str, str]:
    """Обрабатывает вложения и формирует текстовые данные."""
    image_description = ""
    transcribed_text = ""
    processed_text_parts = []

    for msg in user_input_list:
        if isinstance(msg, dict):
            if msg.get("type") == "image":
                image_url = msg.get("content")
                image_response = await photo_service.process_image_from_url(image_url)
                if image_response:
                    image_data = json.loads(image_response)
                    image_description += f"\n\nКонтекст изображения: {image_data.get('description', '')}"
            elif msg.get("type") == "voice":
                voice_url = msg.get("content")
                transcribed = stt_service.transcribe(voice_url)
                if transcribed:
                    transcribed_text += f"\n\nТранскрибированное голосовое сообщение: {transcribed}"
        else:
            processed_text_parts.append(msg)

    return "\n".join(processed_text_parts), image_description, transcribed_text


async def send_text_in_parts(clean_text: str, amo_data: dict, chat_id: str) -> None:
    """
    Делим текст по '~' и отправляем каждую часть с задержкой 0.5-2 секунды.
    """
    parts = clean_text.split('~')  # Разделяем по тильде
    for part in parts:
        part = part.strip()
        if part:
            # Отправляем каждую часть пользователю
            await send_message(
                amo_data["amojo_id"],
                amo_data["chat_token"],
                part,
                chat_id,

            )
            # Пауза 0.5-2 секунды
            await asyncio.sleep(random.uniform(0.5, 2.0))

async def send_responses(clean_text: str, amo_data: dict, chat_id: str) -> None:
    # Вместо единой отправки сообщения, вызываем новый метод:
    await send_text_in_parts(clean_text, amo_data, chat_id)



async def process_messages(chat_id: str, entity_id: str) -> None:
    """Обрабатывает сообщения из буфера."""
    try:
        user_input_list = message_buffers.pop(chat_id, [])
        photo_service = PhotoService()
        stt_service = STTService()

        # Обработка вложений
        processed_text, image_description, transcribed_text = await process_attachments(
            user_input_list, photo_service, stt_service
        )

        # Формируем итоговый ввод
        user_input = f"{processed_text}{image_description}{transcribed_text}"
        logger.info(f"Итоговый ввод для LLM: {user_input}")

        # Получаем данные из AmoCRM
        amo_data = await amo_api_data()

        user_input = f"ответ клиента: {user_input}"
        ai_answer, list_ai = await thread(user_input, chat_id)
        clean_text = await MarkdownProcessor.strip_markdown(ai_answer)
        logger.info(f"Ответ LLM: {clean_text}")

        # Отправка ответа
        await send_responses(clean_text, amo_data, chat_id)



    except Exception as e:
        logger.error(f"Произошла ошибка при обработке: {e}", exc_info=True)


async def handle_incoming_message(message_text: str, chat_id: str, result: str, entity_id: str) -> None:
    """Обрабатывает входящее сообщение."""
    if message_text:
        message_buffers[chat_id].append(message_text)

    if result:
        if is_image(result):
            message_buffers[chat_id].append({"type": "image", "content": result})
        elif is_voice(result):
            message_buffers[chat_id].append({"type": "voice", "content": result})

    if chat_id in message_timers:
        message_timers[chat_id].cancel()

    message_timers[chat_id] = asyncio.create_task(timer_task(chat_id, entity_id))


async def timer_task(chat_id: str, entity_id: str) -> None:
    await asyncio.sleep(BUFFER_DELAY)
    await process_messages(chat_id, entity_id)


async def main(message_text: str, chat_id: str, result: str, entity_id: str) -> None:
    await handle_incoming_message(message_text, chat_id, result, entity_id)
