#main v2
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

SYSTEM_ADVICE = (
    "당신은 사용자의 업무 일정과 과거 실수 기록을 참고해 오늘 주의할 점을 조언하는 비서입니다. "
    "일정 자체를 바꾸자고 제안하지 말고, 각 업무에서 과거에 반복된 실수를 토대로 오늘 무엇을 조심할지 "
    "간결한 한국어로 정리하세요. 과장 없이 실용적으로, 항목별로 짧게 적습니다."
)

SYSTEM_CHAT = (
    "당신은 사용자의 하루 업무 회고를 돕는 대화형 비서입니다. "
    "사용자가 오늘 한 업무와 실수, 아쉬운 점을 함께 짚어보며 구체적이고 실용적인 한국어 조언을 합니다. "
    "공감하되 과장 없이, 다음에 개선할 점을 명확하게 제시하세요. 답변은 간결하게 유지합니다."
)


class AdviceRequest(BaseModel):
    schedule: list = []
    mistakes: dict = {}


class ChatRequest(BaseModel):
    messages: list = []      # [{role:"user"/"assistant", content:"..."}]
    context: dict = {}       # {schedule, notes, mistakes}


@app.get("/health")
def health():
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
                lines.append(f"- {cat}: {m.get('text','')} (총 {m.get('count',1)}회)")
    else:
        lines.append("- 기록 없음")

    user_prompt = "\n".join(lines) + "\n\n위 정보를 참고해 오늘 각 업무에서 조심할 점을 조언해 주세요."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_ADVICE,
                max_output_tokens=1024,
            ),
        )
        return {"advice": response.text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {e}")


def build_context(context):
    lines = []
    sched = context.get("schedule", [])
    if sched:
        lines.append("[오늘 일정]")
        for e in sched:
            t = e.get("time", "")
            title = e.get("title", "")
            cat = e.get("taskType", "")
            lines.append(f"- {t} {title}" + (f" (카테고리: {cat})" if cat else ""))
    notes = context.get("notes", {})
    if notes:
        lines.append("\n[오늘 회고 메모]")
        for title, note in notes.items():
            if note:
                lines.append(f"- {title}: {note}")
    mistakes = context.get("mistakes", {})
    if mistakes:
        lines.append("\n[과거 실수]")
        for cat, items in mistakes.items():
            for m in items:
                lines.append(f"- {cat}: {m.get('text','')} ({m.get('count',1)}회)")
    return "\n".join(lines)


@app.post("/chat")
def chat(req: ChatRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    ctx = build_context(req.context)
    system = SYSTEM_CHAT + (("\n\n참고 정보:\n" + ctx) if ctx else "")

    contents = []
    for m in req.messages:
        role = "user" if m.get("role") == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m.get("content", ""))]))
    if not contents:
        contents = [types.Content(role="user", parts=[types.Part(text="오늘 하루를 함께 돌아보고 싶어요.")])]

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=1024,
            ),
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {e}")
