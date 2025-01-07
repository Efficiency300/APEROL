from urllib.parse import parse_qs
from fastapi import FastAPI, Request
from icecream import ic
import uvicorn
from pathlib import Path
from status_crm.update_lead_status import update_lead_status
from status_crm.user_status import user_status
from core.main import main
from utils.JsonDataBase import JSONDatabase

BASE_DIR = Path(__file__).resolve().parent.parent
talk_id_json = BASE_DIR / "config/talk_id.json"


peregovori = 66373106
abaa = 72102778
nerazbornoe = 72932214

app = FastAPI()
db = JSONDatabase(talk_id_json)

async def check_and_return(talk_id, current_time, result, entity_id):
    if await db.exists(talk_id):
        past_time = await db.get(talk_id)
        if int(past_time) >= current_time:
            return {"status": "success", "data": {"result": result, "entity_id": entity_id}}
    return None


@app.post("/")
async def client_data(r: Request):
    try:
        raw_body = await r.body()
        decoded_body = raw_body.decode("utf-8")
        parsed_data = parse_qs(decoded_body)
        message_text = parsed_data.get("message[add][0][text]", [""])[0]
        chat_id = parsed_data.get("message[add][0][chat_id]", [""])[0]
        talk_id = parsed_data.get("message[add][0][talk_id]", [""])[0]
        current_time = int(parsed_data.get("message[add][0][created_at]", ["0"])[0])
        entity_id = parsed_data.get("message[add][0][entity_id]", [None])[0]
        attachment_type = parsed_data.get("message[add][0][attachment][type]", [None])[0]
        attachment_link = parsed_data.get("message[add][0][attachment][link]", [None])[0]

        result = attachment_link if attachment_type in ["voice", "picture"] else ""

        if not chat_id or (not message_text and not attachment_link):
            return {"error": "Invalid data"}

        # Проверка статуса пользователя
        results = await user_status(entity_id)
        print(results)
        if results in [peregovori] and "test ai" in message_text.lower():
            check_result = await check_and_return(talk_id, current_time, result, entity_id)
            if check_result:
                return check_result

            await main(message_text, chat_id, result, entity_id)
            await db.add(talk_id, current_time)
            await update_lead_status(entity_id, abaa)
        
        if results in [abaa , nerazbornoe]:
            check_result = await check_and_return(talk_id, current_time, result, entity_id)
            if check_result:
                return check_result


            await main(message_text, chat_id, result, entity_id)
            await db.add(talk_id, current_time)

            return {"status": "success", "data": {"result": result, "entity_id": entity_id}}

    except Exception as e:
        import traceback
        ic(traceback.format_exc())
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)