#main v3
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

SYSTEM_RECOMMEND = (
    "당신은 사용자의 오늘 빈 시간에 업무를 배치해 주는 일정 비서입니다. "
    "고정 일정(이미 정해진 약속)은 절대 바꾸거나 옮기지 마세요. 주어진 빈 시간대 안에서만 업무를 추천합니다. "
    "마감이 임박하고 남은 작업이 많은 업무를 우선하고, 완료한 업무의 예상 대비 실제 소요 시간과 피드백을 참고해 "
    "현실적으로 시간을 배분하세요. '몇 시~몇 시에 어떤 업무(가능하면 어떤 단계)를 하라'고 구체적으로, "
    "왜 그 순서인지 한 줄 이유를 덧붙여 간결한 한국어로 추천합니다."
)


class AdviceRequest(BaseModel):
    schedule: list = []
    mistakes: dict = {}


class ChatRequest(BaseModel):
    messages: list = []      # [{role:"user"/"assistant", content:"..."}]
    context: dict = {}       # {schedule, notes, mistakes}


class RecommendRequest(BaseModel):
    work_start: str = "08:30"
    work_end: str = "17:30"
    today: str = ""
    fixed: list = []         # 오늘 고정 일정 [{time, endTime, title}]
    tasks: list = []         # 진행 중 업무 [{name, deadline, steps:[...]}]
    done: list = []          # 완료 업무 [{name, steps:[{name, estHours, actualHours, feedback}]}]


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
                max_output_tokens=2048,
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
                max_output_tokens=2048,
            ),
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {e}")


def _to_min(hhmm):
    try:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _to_hhmm(mins):
    return f"{mins // 60:02d}:{mins % 60:02d}"


def compute_free_slots(work_start, work_end, fixed):
    ws, we = _to_min(work_start), _to_min(work_end)
    if ws is None or we is None or ws >= we:
        return []
    busy = []
    for e in fixed:
        s = _to_min(e.get("time", ""))
        if s is None:
            continue
        en = _to_min(e.get("endTime", "")) if e.get("endTime") else s + 60
        if en is None:
            en = s + 60
        s2, en2 = max(s, ws), min(en, we)
        if s2 < en2:
            busy.append((s2, en2))
    busy.sort()
    slots, cur = [], ws
    for s, en in busy:
        if s > cur:
            slots.append((cur, s))
        cur = max(cur, en)
    if cur < we:
        slots.append((cur, we))
    return [{"start": _to_hhmm(a), "end": _to_hhmm(b), "minutes": b - a} for a, b in slots]


def _dday(today, deadline):
    try:
        from datetime import date
        y, m, d = map(int, today.split("-"))
        y2, m2, d2 = map(int, deadline.split("-"))
        return (date(y2, m2, d2) - date(y, m, d)).days
    except Exception:
        return None


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="API 키가 설정되지 않았습니다.")

    slots = compute_free_slots(req.work_start, req.work_end, req.fixed)

    lines = [f"근무시간: {req.work_start} ~ {req.work_end}", f"오늘 날짜: {req.today}"]

    lines.append("\n[고정 일정 — 변경 금지]")
    if req.fixed:
        for e in req.fixed:
            t = e.get("time", "")
            end = e.get("endTime", "")
            when = t + ("~" + end if end else "")
            lines.append(f"- {when} {e.get('title','')}")
    else:
        lines.append("- 없음")

    lines.append("\n[오늘 빈 시간대]")
    if slots:
        for s in slots:
            lines.append(f"- {s['start']}~{s['end']} ({s['minutes']}분)")
    else:
        lines.append("- 빈 시간이 없습니다.")

    lines.append("\n[진행 중 업무]")
    if req.tasks:
        for t in req.tasks:
            dd = _dday(req.today, t.get("deadline", ""))
            ddtxt = (f"D-{dd}" if dd and dd > 0 else ("오늘 마감" if dd == 0 else (f"{-dd}일 지남" if dd is not None else "마감 없음")))
            steps = t.get("steps", [])
            remain = [s for s in steps if not s.get("done")]
            remain_h = sum(float(s.get("estHours", 0) or 0) for s in remain)
            lines.append(f"- {t.get('name','')} ({ddtxt}) · 남은 단계 {len(remain)}개 · 남은 예상 {remain_h}h")
            for s in remain:
                lines.append(f"    · {s.get('name','')} (예상 {s.get('estHours',0)}h)")
    else:
        lines.append("- 없음")

    lines.append("\n[완료 업무 피드백 — 시간 배분 참고]")
    if req.done:
        for t in req.done:
            for s in t.get("steps", []):
                est = s.get("estHours", 0)
                act = s.get("actualHours", 0)
                fb = s.get("feedback", "")
                note = f" / {fb}" if fb else ""
                lines.append(f"- [{t.get('name','')}] {s.get('name','')}: 예상 {est}h → 실제 {act}h{note}")
    else:
        lines.append("- 없음")

    prompt = "\n".join(lines) + "\n\n위 정보를 바탕으로, 고정 일정은 그대로 두고 빈 시간대에 어떤 업무를 언제 하면 좋을지 추천해 주세요."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_RECOMMEND,
                max_output_tokens=2048,
            ),
        )
        return {"recommendation": response.text, "free_slots": slots}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {e}")

