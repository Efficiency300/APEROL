import aiohttp
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class STTService:
    def __init__(self):
        self.client = AsyncOpenAI()

    async def transcribe(self, url, model="whisper-1"):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    audio_content = await response.read()
                    try:
                        transcript = await self.client.audio.transcriptions.create(
                            model=model,
                            file=("audio_file.m4a", audio_content)
                        )
                        return transcript.text
                    except Exception as e:
                        print(f"Ошибка транскрибирования: {e}")
                        return None
                else:
                    print("Ошибка при скачивании аудиофайла")
                    return None
