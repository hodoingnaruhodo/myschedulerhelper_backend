# main v1
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# 우리 사이트 주소에서의 호출만 허용 (저장소명이 달라도 도메인만 맞으면 됩니다)
ALLOWED_ORIGINS = [
    "https://hodoingnaruhodo.github.io",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

SYSTEM = (
    "당신은 사용자의 업무 일정과 과거 실수 기록을 참고해 오늘 주의할 점을 조언하는 비서입니다. "
    "일정 자체를 바꾸자고 제안하지 말고, 각 업무에서 과거에 반복된 실수를 토대로 오늘 무엇을 조심할지 "
    "간결한 한국어로 정리하세요. 과장 없이 실용적으로, 항목별로 짧게 적습니다."
)


class AdviceRequest(BaseModel):
    schedule: list = []      # [{date,time,title,taskType,...}, ...]
    mistakes: dict = {}      # {카테고리: [{text,count,dates}, ...]}


@app.get("/health")
def health():
    # 서버를 깨우기 위한 가벼운 핑 엔드포인트
    return {"status": "ok"}


@app.post("/advice")
def advice(req: AdviceRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    lines = ["[오늘 일정]"]
    if req.schedule:
        for e in req.schedule:
            t = e.get("time", "")
            cat = e.get("taskType", "")
            title = e.get("title", "")
            lines.append(f"- {t} {title}" + (f" (카테고리: {cat})" if cat else ""))
    else:
        lines.append("- 등록된 일정 없음")

    lines.append("\n[과거 실수 기록 (카테고리별)]")
    if req.mistakes:
        for cat, items in req.mistakes.items():
            for m in items:
                cnt = m.get("count", 1)
                lines.append(f"- {cat}: {m.get('text','')} (총 {cnt}회)")
    else:
        lines.append("- 기록 없음")

    user_prompt = "\n".join(lines) + "\n\n위 정보를 참고해 오늘 각 업무에서 조심할 점을 조언해 주세요."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                max_output_tokens=1024,
            ),
        )
        return {"advice": response.text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {e}")
