import json
from asyncio import Lock
from openai import AsyncOpenAI
from config.config import Config
from pathlib import Path
from services.prompt import prompt
from utils.JsonDataBase import JSONDatabase

BASE_DIR = Path(__file__).resolve().parent.parent
talk_id_json = f"{BASE_DIR}/config/thread_id.json"
file_lock = Lock()

db = JSONDatabase(talk_id_json)
FULL_PROMPT = prompt
async def thread(message_text: str, chat_id: str) -> list[str | dict]:

    try:
        async with file_lock:
            client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            thread_id = await db.get(chat_id) if await db.exists(chat_id) else None
            if not thread_id:
                thread = await client.beta.threads.create()
                thread_id = thread.id
                await db.add(chat_id, thread_id)

            await client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message_text
            )

            run = await client.beta.threads.runs.create_and_poll(
                thread_id=thread_id,
                model="gpt-4o",
                assistant_id=Config.ASSIST_ID,
                temperature=0.5,
                instructions=FULL_PROMPT
            )

            if run.status == "completed":
                messages = await client.beta.threads.messages.list(thread_id=thread_id)
                messages_json = json.loads(messages.model_dump_json())
                response = messages_json["data"][0]["content"][0]["text"]["value"]
                return [response, messages_json]

            return ["Model run not completed"]

    except Exception as e:
        return [str(e)]
