"""Streamlit 채팅 UI — POST /query(표준 규약)만 호출하는 얇은 클라이언트

- 질문 → POST {API_URL}/query → answer 표시, contexts는 "근거 리뷰", trace는 "처리 과정" 접기 영역
- API 주소는 환경 변수 API_URL (기본 http://localhost:8000)
- 되묻기 대화 이어가기 (PRD "되묻기 대화 이어가기"): API는 무상태다. 응답이 되묻기이면
  다음 입력을 "이전 질문 / (에이전트) 되묻기 / 사용자 답"으로 이어 붙여 하나의 question으로 다시 보낸다.
  되묻기는 최대 2턴까지 이어 붙이고, 그 뒤에는 새 질문으로 시작한다.
- 대화 내용은 브라우저 세션 메모리에만 있고 서버·파일에 저장하지 않는다 (SERVICE.md 정책 3).

실행: streamlit run src/app.py   (run.sh가 API 다음에 띄운다)
"""

from __future__ import annotations

import html
import os
import re

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT", "120"))
MAX_CLARIFY_TURNS = 2
SEP = " / "

DEMO_QUESTIONS = [
    ("원룸 청소기 추천", "원룸에서 쓸 35만원 이하 무선청소기 3개 추천해줘. 소음이 제일 중요해"),
    ("어디서 사야 쌀까", "클린웨이브 V7 사일런트 무선청소기 어디서 사는 게 제일 싸? 나는 누리카드랑 바로플러스 멤버십 있어"),
    ("조건 없이 물어보기", "청소기 추천해줘"),
]
DEMO_ICONS = [":material/meeting_room:", ":material/payments:", ":material/chat:"]  # 방(원룸) · 결제(어디서 싸게) · 말풍선(되묻기)
HOME_ICON = ":material/home:"  # 웹 표준 '처음 화면' 기호. 예시 버튼과 겹치지 않게 홈 전용으로만 쓴다
TOOL_STEPS = {"search_products", "fetch_reviews", "retrieve_reviews", "query_price_db", "calc_real_price"}
STEP_LABELS = {
    "guardrail_input": "입력 점검", "search_products": "상품 검색", "fetch_reviews": "리뷰 수집",
    "retrieve_reviews": "근거 검색", "query_price_db": "가격 이력", "calc_real_price": "실결제가 계산",
    "answer": "답변 작성", "guardrail_output": "출력 점검",
}


# ─────────────────────────────── 로직 (테스트 가능하게 UI와 분리) ───────────────────────────────


def is_clarify(body: dict) -> bool:
    """에이전트가 도구 없이 되물은 응답인지 (trace의 answer 단계 output이 '되묻기')."""
    return any(s.get("step") == "answer" and s.get("output") == "되묻기" for s in body.get("trace", []))


def build_question(pending: dict | None, user_text: str) -> str:
    """되묻기 중이면 '이전 질문 / (에이전트) 되묻기 / 사용자 답'으로 이어 붙인다."""
    if not pending:
        return user_text
    clarify = " ".join(pending["clarify"].replace("**", "").split())  # 줄바꿈·굵게 표시를 한 줄로
    return SEP.join([pending["question"], f"(에이전트) {clarify}", user_text])


def call_api(question: str) -> tuple[int | None, dict | None, str | None]:
    """(HTTP 상태, 응답 본문, 오류 메시지). 규약 형태의 오류 응답(503·500)도 본문을 돌려준다."""
    try:
        r = httpx.post(f"{API_URL}/query", json={"question": question}, timeout=REQUEST_TIMEOUT)
    except httpx.HTTPError as exc:
        return None, None, f"API에 연결하지 못했습니다 ({type(exc).__name__}). API 서버가 떠 있는지 확인해 주세요: {API_URL}"
    try:
        body = r.json()
    except ValueError:
        return r.status_code, None, f"API 응답을 읽지 못했습니다 (HTTP {r.status_code})."
    if r.status_code == 422:
        return r.status_code, None, "질문을 입력해 주세요 (빈 질문이거나 2,000자를 넘었습니다)."
    if not {"answer", "contexts", "trace"} <= set(body):
        return r.status_code, None, f"예상하지 못한 응답입니다 (HTTP {r.status_code})."
    return r.status_code, body, None


def next_pending(pending: dict | None, sent_question: str, body: dict) -> tuple[dict | None, bool]:
    """응답 뒤의 되묻기 상태와 '되묻기 턴 초과' 여부."""
    if not is_clarify(body):
        return None, False
    turns = (pending or {}).get("turns", 0) + 1
    if turns > MAX_CLARIFY_TURNS:
        return None, True
    return {"question": sent_question, "clarify": body["answer"], "turns": turns}, False


# ─────────────────────────────── 화면: 디자인 토큰·고정 HTML ───────────────────────────────
# 고정 HTML(헤더·빈 화면·배지)만 unsafe_allow_html로 그린다. answer·리뷰·trace는 API·리뷰 원문이라
# 신뢰하지 않는 데이터이므로 일반 st.markdown(HTML 이스케이프)으로만 표시한다.

THEMES = {
    # AI chatbot 테마: 중립 회색 + AI 보라 포인트. ok(실구매)·stamp(협찬·샘플)는 의미 색이라 유지
    "light": {"paper": "#F7F7F9", "surface": "#FFFFFF", "ink": "#16151C", "ink2": "#5B5966", "rule": "#E4E3EA",
              "accent": "#6D3FE0", "on-accent": "#FFFFFF", "accent-soft": "#F0EBFD", "ok": "#187A45", "ok-soft": "#E5F3EA",
              "stamp": "#B42318", "stamp-soft": "#FBE8E6"},
    "dark": {"paper": "#0F0F13", "surface": "#18181E", "ink": "#ECEBF2", "ink2": "#A6A4B3", "rule": "#2A2933",
             "accent": "#A889FF", "on-accent": "#0F0F13", "accent-soft": "#241D3D", "ok": "#5CD08E", "ok-soft": "#12301F",
             "stamp": "#FF8E7E", "stamp-soft": "#3B1B17"},
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');
:root { --jh-body: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif;
        --jh-display: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
        --jh-mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
        --jh-shadow: 0 1px 2px rgba(22,21,28,.05), 0 8px 24px rgba(22,21,28,.06); }
.stApp { background: var(--jh-paper); color: var(--jh-ink); }
.stApp p, .stApp li, .stApp label, .stApp textarea, .stApp h1, .stApp h2, .stApp h3, .stApp summary, .stApp dd, .stApp dt {
  font-family: var(--jh-body); word-break: keep-all; overflow-wrap: anywhere; }
/* 상단 영역: 배경·구분선은 Streamlit header 띠(3.75rem)가 그리고, 우리 타이틀 띠(.jh-appbar)는 그 위에 투명하게 겹친다.
   타이틀 띠는 클릭을 통과시키고(pointer-events: none) 로고 링크만 클릭을 받는다.
   → 로고 클릭은 Streamlit 툴바에 가로채이지 않고, 오른쪽 ⋮ 메뉴도 그대로 보이고 눌린다. Deploy 버튼은 숨긴다 */
[data-testid="stHeader"] { background: var(--jh-paper); border-bottom: 1px solid var(--jh-rule); }
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none !important; }
.jh-appbar { position: fixed; top: 0; left: 0; right: 0; height: 3.75rem; z-index: 999991; background: transparent;
             pointer-events: none; }
.jh-appbar a.jh-home { pointer-events: auto; }
.jh-appbar .jh-top { max-width: 1000px; height: 100%; margin: 0 auto; padding: 0 1rem; flex-wrap: nowrap; }
@media (max-width: 1100px) { .jh-appbar .jh-top { padding-right: 60px; } }  /* 오른쪽 ⋮ 메뉴와 겹치지 않게 */
.jh-appbar .jh-name { white-space: nowrap; }
@media (max-width: 900px) { .jh-appbar .jh-status { display: none; } }  /* 앱 명칭이 길어져 좁은 화면에서는 상태 문구를 숨김 */
@media (max-width: 640px) { .jh-appbar .jh-tag { display: none; } .jh-appbar .jh-name { font-size: 16px; } }
[data-testid="stMainBlockContainer"] { padding-top: calc(3.75rem + 10px) !important; padding-bottom: 7rem; }
[data-testid="stElementContainer"]:has(.jh-appbar) { margin: 0; height: 0; }
.stApp h1.jh-title.jh-chat-title { font-size: clamp(18px, 2.6vw, 22px); margin: 0 0 4px; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--jh-ink2); }
.stApp :focus-visible { outline: 3px solid var(--jh-accent) !important; outline-offset: 2px; }
.jh-num { font-family: var(--jh-mono); font-variant-numeric: tabular-nums; letter-spacing: -.01em; }

/* ── header (compact) ── */
.jh-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.jh-brand { display: flex; align-items: center; gap: 8px; }
.jh-mark { width: 32px; height: 32px; border-radius: 10px; display: grid; place-items: center;
           background: var(--jh-accent); color: var(--jh-on-accent); }
.jh-mark svg, .jh-ava svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.jh-name { font: 700 19px/1 var(--jh-display); color: var(--jh-ink); letter-spacing: -.02em; }
/* 로고·이름 = 처음 화면(랜딩) 링크 */
.stApp a.jh-home { display: inline-flex; align-items: center; gap: 8px; min-height: 44px; padding: 0 8px 0 6px; margin-left: -6px;
                   border-radius: 10px; color: var(--jh-ink); text-decoration: none; cursor: pointer; transition: background-color .15s; }
.stApp a.jh-home:hover { background: var(--jh-accent-soft); }
.stApp a.jh-home:hover .jh-name { color: var(--jh-accent); }
.jh-status { display: inline-flex; align-items: center; gap: 6px; margin-left: 4px; font-size: 13px; color: var(--jh-ink2); }
.jh-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--jh-ok); }
.jh-tag { display: inline-flex; align-items: center; padding: 3px 9px; border-radius: 6px; font-size: 12.5px; font-weight: 600;
          background: var(--jh-paper); color: var(--jh-ink2); border: 1px solid var(--jh-rule); white-space: nowrap; }
.jh-tag.sample { background: var(--jh-stamp-soft); color: var(--jh-stamp); border-color: transparent; }
.jh-tag.ok { background: var(--jh-ok-soft); color: var(--jh-ok); border-color: transparent; }
.jh-tag.ask { background: var(--jh-accent-soft); color: var(--jh-accent); border-color: transparent; }
.stApp h1.jh-title { font: 700 clamp(22px, 3.6vw, 28px)/1.3 var(--jh-body); margin: 14px 0 0; padding: 0; color: var(--jh-ink);
                     letter-spacing: -.02em; text-wrap: balance; }
.jh-sub { margin: 6px 0 0; font-size: 15px; color: var(--jh-ink2); }
.jh-label { margin: 18px 0 6px; font-size: 13px; font-weight: 600; color: var(--jh-ink2); }

/* ── demo chips ── */
/* 예시 질문 3개: 옅은 보라(accent-soft) 바탕 + 보라 글자(라이트 5.2:1, 다크 5.8:1). 마우스 올림·누름은 진한 보라로 채운다.
   '새 대화'는 대화를 지우는 버튼이라 회색 점선으로 한 단계 낮춘다. 포커스 링은 전역 :focus-visible 그대로 */
.st-key-jh-demo button { min-height: 44px; border-radius: 999px; border: 1px solid color-mix(in srgb, var(--jh-accent) 25%, transparent);
                         background: var(--jh-accent-soft); color: var(--jh-accent); transition: border-color .15s, background-color .15s, color .15s; }
.st-key-jh-demo button p, .st-key-jh-demo button span { color: inherit; }
.st-key-jh-demo button:hover, .st-key-jh-demo button:active { border-color: var(--jh-accent); background: var(--jh-accent); color: var(--jh-on-accent); }
.st-key-jh-demo button p { font-size: 14.5px; font-weight: 600; white-space: nowrap; }
.st-key-jh-demo .st-key-jh-new button { background: transparent; border: 1px dashed var(--jh-rule); color: var(--jh-ink2); }
.st-key-jh-demo .st-key-jh-new button p { font-weight: 500; }
.st-key-jh-demo .st-key-jh-new button:hover, .st-key-jh-demo .st-key-jh-new button:active {
  background: var(--jh-surface); border-color: var(--jh-ink2); color: var(--jh-ink); }

/* ── shared card ── */
.jh-card { background: var(--jh-surface); border: 1px solid var(--jh-rule); border-radius: 14px; box-shadow: var(--jh-shadow);
           margin: 0 0 14px; overflow: hidden; }
.jh-card-body { padding: 18px 20px; }
.jh-card h2, .jh-card h3 { margin: 0; padding: 0; color: var(--jh-ink); }

/* assistant identity row (welcome + every answer) */
.jh-bot-head { display: flex; align-items: center; gap: 10px; margin: 0 0 10px; scroll-margin-top: calc(3.75rem + 16px); }
.jh-ava { width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; flex: none;
          background: var(--jh-accent-soft); color: var(--jh-accent); }
.jh-ava svg { width: 16px; height: 16px; }
.jh-bot-head b { font-size: 14.5px; font-weight: 600; color: var(--jh-ink); }
.jh-bot-head span { font-size: 13px; color: var(--jh-ink2); }

/* ── landing (empty state) ── */
.st-key-jh-landing { padding: 4px 0 4px; }
.jh-hero { padding-top: 4px; }
.jh-pill { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px 5px 6px; border-radius: 999px; background: var(--jh-surface);
           border: 1px solid var(--jh-rule); font-size: 13px; font-weight: 500; color: var(--jh-ink2); }
.jh-pill b { font-weight: 600; color: var(--jh-accent); background: var(--jh-accent-soft); padding: 2px 8px; border-radius: 999px; }
.stApp h1.jh-hero-title { font: 700 clamp(28px, 4.4vw, 40px)/1.18 var(--jh-display); letter-spacing: -.035em; margin: 16px 0 12px;
                          padding: 0; color: var(--jh-ink); text-wrap: balance; }
.jh-lede { margin: 0 0 18px; font-size: 16.5px; line-height: 1.7; color: var(--jh-ink2); }
.jh-fine { list-style: none; display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 12px 0 0; padding: 0; }
.jh-fine li { display: inline-flex; align-items: center; gap: 5px; margin: 0; font-size: 13.5px; color: var(--jh-ink2); }
.jh-fine svg, .jh-step svg { width: 15px; height: 15px; flex: none; stroke: var(--jh-ok); fill: none; stroke-width: 2.4;
                             stroke-linecap: round; stroke-linejoin: round; }

/* conversational preview + CSS streaming demo */
.jh-preview { margin: 0; background: var(--jh-surface); border: 1px solid var(--jh-rule); border-radius: 18px; box-shadow: var(--jh-shadow); overflow: hidden; }
.jh-preview-head { display: flex; align-items: center; gap: 10px; padding: 12px 16px; border-bottom: 1px solid var(--jh-rule); }
.jh-preview-head b { display: block; font-size: 14.5px; font-weight: 600; color: var(--jh-ink); }
.jh-preview-head span { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--jh-ink2); }
.jh-thread { display: flex; flex-direction: column; gap: 10px; padding: 16px; }
.jh-ubub { align-self: flex-end; max-width: 88%; margin: 0; padding: 9px 13px; border-radius: 14px 14px 4px 14px; background: var(--jh-accent);
           color: var(--jh-on-accent); font-size: 14px; line-height: 1.55; }
.jh-steps { display: flex; flex-wrap: wrap; gap: 6px; }
.jh-step { display: inline-flex; align-items: center; gap: 5px; padding: 4px 9px; border-radius: 999px; border: 1px solid var(--jh-rule);
           background: var(--jh-paper); font-size: 12.5px; color: var(--jh-ink2); animation: jh-in .3s ease both; animation-delay: var(--d); }
.jh-bbub { align-self: flex-start; max-width: 94%; margin: 0; padding: 11px 13px; border-radius: 14px 14px 14px 4px; background: var(--jh-paper);
           font-size: 14.5px; line-height: 1.65; color: var(--jh-ink); }
.jh-bbub .w { animation: jh-in .18s ease both; animation-delay: calc(var(--i) * 45ms + 1300ms); }
.jh-caret { display: inline-block; width: .45em; height: 1em; vertical-align: -.12em; background: var(--jh-accent);
            animation: jh-blink 1s steps(1) infinite, jh-out .01s linear forwards; animation-delay: 1300ms, var(--end); }
.jh-mini { display: grid; grid-template-columns: 30px minmax(0, 1fr) auto; gap: 10px; align-items: center; padding: 10px 12px;
           border: 1px solid var(--jh-rule); border-radius: 12px; animation: jh-up .35s ease both; animation-delay: calc(var(--end) + 150ms); }
.jh-mini .jh-rank { width: 30px; height: 30px; font-size: 14px; border-radius: 8px; }
.jh-mini b { display: block; font-size: 14px; font-weight: 600; color: var(--jh-ink); }
.jh-mini span { font-size: 12px; color: var(--jh-ink2); }
.jh-mini-price { text-align: right; }
.jh-mini-price b { font-size: 15.5px; }
@keyframes jh-in { from { opacity: 0; } to { opacity: 1; } }
@keyframes jh-up { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes jh-blink { 50% { opacity: 0; } }
@keyframes jh-out { to { opacity: 0; visibility: hidden; } }

/* feature cards + integrations */
.jh-section { margin: 40px 0 0; }
.jh-kicker { margin: 0 0 6px; font: 600 12.5px/1 var(--jh-mono); color: var(--jh-accent); letter-spacing: .02em; }
.stApp h2.jh-h2 { font: 700 clamp(21px, 3vw, 26px)/1.3 var(--jh-display); letter-spacing: -.025em; margin: 0 0 16px; padding: 0;
                  color: var(--jh-ink); text-wrap: balance; }
.jh-feats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.jh-feat { display: flex; flex-direction: column; gap: 6px; padding: 18px; border-radius: 14px; border: 1px solid var(--jh-rule);
           background: var(--jh-surface); transition: border-color .2s, transform .2s; }
.jh-feat:hover { border-color: color-mix(in srgb, var(--jh-accent) 45%, var(--jh-rule)); transform: translateY(-2px); }
.jh-feat-ico { width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center; flex: none;
               background: var(--jh-accent-soft); color: var(--jh-accent); }
.jh-feat-ico svg { width: 19px; height: 19px; fill: none; stroke: currentColor; stroke-width: 1.8;
                   stroke-linecap: round; stroke-linejoin: round; }
.stApp .jh-feat h3 { margin: 6px 0 0; padding: 0; font-size: 15.5px; font-weight: 700; color: var(--jh-ink); }
.jh-feat p { margin: 0; font-size: 14px; line-height: 1.6; color: var(--jh-ink2); }
.jh-spec { margin-top: auto; padding-top: 8px; border-top: 1px dashed var(--jh-rule); font: 500 12px/1.5 var(--jh-mono); color: var(--jh-ink2); }
/* 기능 더 보기 (접기/펼치기) — 버튼 높이 44px, 포커스 링은 전역 :focus-visible 그대로 */
.jh-more { margin: 12px 0 0; }
.jh-more summary { list-style: none; display: inline-flex; align-items: center; gap: 6px; min-height: 44px; padding: 0 16px;
                   border-radius: 999px; border: 1px solid var(--jh-rule); background: var(--jh-surface); color: var(--jh-ink);
                   font-size: 14px; font-weight: 600; cursor: pointer; transition: border-color .15s, background-color .15s; }
.jh-more summary::-webkit-details-marker { display: none; }
.jh-more summary:hover { border-color: var(--jh-accent); background: var(--jh-accent-soft); }
.jh-more summary svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round;
                       stroke-linejoin: round; transition: transform .2s; }
.jh-more[open] summary svg { transform: rotate(180deg); }
.jh-more .c, .jh-more[open] .o { display: none; }
.jh-more[open] .c { display: inline; }
.jh-more[open] .jh-feats { margin-top: 12px; animation: jh-up .25s ease both; }
.jh-try-label { margin: 40px 0 6px; font-size: 13px; font-weight: 600; color: var(--jh-ink2); }
.st-key-jh-inline-input { margin-top: 8px; }
@media (max-width: 900px) { .jh-feats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) { .jh-feats { grid-template-columns: minmax(0, 1fr); } }

/* recommendation card */
.jh-rec-head { display: grid; grid-template-columns: 36px minmax(0, 1fr) auto; gap: 4px 14px; align-items: start; }
.jh-rank { width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center;
           background: var(--jh-accent); color: var(--jh-on-accent); font: 700 17px/1 var(--jh-display); }
.jh-rank.plain { background: var(--jh-accent-soft); color: var(--jh-accent); }
.jh-rec-head h3 { font-size: 17px; font-weight: 700; line-height: 1.4; }
.jh-rec-head .jh-summary { margin: 4px 0 0; font-size: 14.5px; line-height: 1.6; color: var(--jh-ink2); }
.jh-price { text-align: right; min-width: 0; }
.jh-price .k { display: block; font-size: 12px; color: var(--jh-ink2); }
.jh-price .v { display: block; font-size: 21px; font-weight: 600; color: var(--jh-ink); line-height: 1.3; }
.jh-price .w { display: block; font-size: 12.5px; color: var(--jh-ink2); }
.jh-note { margin: 12px 0 0; padding: 8px 12px; border-radius: 8px; background: var(--jh-paper); font-size: 13.5px; color: var(--jh-ink2); }
.jh-sec { padding: 14px 20px; border-top: 1px solid var(--jh-rule); }
.jh-sec h4 { margin: 0 0 8px; padding: 0; font-size: 13px; font-weight: 600; color: var(--jh-ink2); letter-spacing: .02em; }
.jh-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.jh-list li { position: relative; margin: 0; padding-left: 22px; font-size: 14.5px; line-height: 1.6; color: var(--jh-ink); }
.jh-list li::before { position: absolute; left: 0; top: 0; font-weight: 700; }
.jh-list.pro li::before { content: "+"; color: var(--jh-ok); }
.jh-list.con li::before { content: "−"; color: var(--jh-stamp); }
.jh-ref { font-family: var(--jh-mono); font-size: 11.5px; color: var(--jh-accent); margin-left: 4px; white-space: nowrap; }
.jh-kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 6px 12px; margin: 10px 0 0; font-size: 14px; }
.jh-kv dt { color: var(--jh-ink2); font-weight: 600; }
.jh-kv dd { margin: 0; color: var(--jh-ink); }
.jh-card details { border-top: 1px solid var(--jh-rule); }
.jh-card summary { list-style: none; cursor: pointer; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between;
                   gap: 8px; font-size: 14px; font-weight: 600; color: var(--jh-accent); min-height: 44px; }
.jh-card summary::-webkit-details-marker { display: none; }
.jh-card summary::after { content: "펼치기"; font-size: 12.5px; font-weight: 500; color: var(--jh-ink2); }
.jh-card details[open] summary::after { content: "접기"; }
.jh-card summary:hover { background: var(--jh-paper); }
.jh-malls { list-style: none; margin: 0; padding: 0 20px 12px; }
.jh-malls li { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2px 12px; margin: 0; padding: 10px 0;
               border-top: 1px dashed var(--jh-rule); }
.jh-malls li:first-child { border-top: 0; }
.jh-malls .m { font-size: 14.5px; font-weight: 600; color: var(--jh-ink); }
.jh-malls .m a { color: inherit; text-decoration: underline; text-decoration-color: var(--jh-rule); text-underline-offset: 3px; }
.jh-malls .t { font-size: 15px; font-weight: 600; text-align: right; color: var(--jh-ink); }
.jh-malls .c { grid-column: 1 / -1; font-size: 13px; color: var(--jh-ink2); }
.jh-malls li.best .t { color: var(--jh-ok); }
.jh-card-foot { padding: 10px 20px; background: var(--jh-paper); border-top: 1px solid var(--jh-rule); font-size: 12.5px;
                line-height: 1.55; color: var(--jh-ink2); }
.jh-sources ol { margin: 0; padding-left: 0; list-style: none; display: grid; gap: 6px; }
.jh-sources li { margin: 0; font-size: 14px; color: var(--jh-ink); }
.jh-sources li .n { font-family: var(--jh-mono); color: var(--jh-accent); margin-right: 6px; }
.jh-sources a { color: var(--jh-accent); }
.jh-notice { margin: 0 0 14px; padding: 10px 14px; border-radius: 10px; background: var(--jh-accent-soft); color: var(--jh-ink);
             font-size: 14px; line-height: 1.55; }
.jh-intro { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 0 0 12px; }
.jh-intro p { margin: 0; font-size: 16px; font-weight: 600; color: var(--jh-ink); }
@media (max-width: 640px) {
  .jh-rec-head { grid-template-columns: 32px minmax(0, 1fr); }
  .jh-price { grid-column: 2; text-align: left; }
  .jh-card-body, .jh-sec { padding-left: 16px; padding-right: 16px; }
  .jh-malls { padding-left: 16px; padding-right: 16px; }
  .jh-card summary { padding-left: 16px; padding-right: 16px; }
}

/* ── chat: user bubble right, assistant = cards without an outer bubble ── */
[data-testid="stChatMessage"] { padding: 0; gap: 0; background: transparent; scroll-margin-top: calc(3.75rem + 16px); }
.st-key-jh-scroll { position: absolute !important; width: 1px; height: 1px; overflow: hidden; opacity: 0; pointer-events: none; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] { display: none; }
[data-testid="stChatMessage"]:has([class*="st-key-jh-user"]) { width: fit-content; max-width: 85%; margin: 12px 0 12px auto;
  padding: 10px 16px; border-radius: 16px 16px 4px 16px; background: var(--jh-accent); }
[data-testid="stChatMessage"]:has([class*="st-key-jh-user"]) p { color: var(--jh-on-accent); font-size: 15.5px; line-height: 1.6; }
[data-testid="stChatMessage"]:has([class*="st-key-jh-bot"]) { margin: 4px 0 20px; }
[data-testid="stChatMessage"]:has([class*="st-key-jh-bot"]) p,
[data-testid="stChatMessage"]:has([class*="st-key-jh-bot"]) li { font-size: 15.5px; line-height: 1.7; color: var(--jh-ink); }
[data-testid="stChatMessage"] code { font-family: var(--jh-mono); font-size: 12.5px; color: var(--jh-accent);
                                     background: var(--jh-accent-soft); padding: 1px 6px; border-radius: 5px; }
[data-testid="stExpander"] details { border-radius: 12px; border-color: var(--jh-rule); background: var(--jh-surface); }
[data-testid="stExpander"] summary p { font-weight: 600; font-size: 14px; color: var(--jh-ink2); }
[data-testid="stExpander"] [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--jh-rule); border-radius: 10px; }

/* ── input ── */
/* 심플 입력창: 그림자·두꺼운 링 없이 얇은 테두리 하나. 포커스는 테두리 색만 바뀌고, 전송 버튼은 입력이 있을 때만 보라색 */
[data-testid="stChatInput"] { border-radius: 14px; border: 1px solid var(--jh-rule); background: var(--jh-surface); box-shadow: none;
                              padding: 2px 4px 2px 8px; transition: border-color .15s; }
[data-testid="stChatInput"] > div { background: transparent; border: 0; }
[data-testid="stChatInput"]:hover { border-color: color-mix(in srgb, var(--jh-ink2) 45%, var(--jh-rule)); }
[data-testid="stChatInput"]:focus-within { border: 1px solid var(--jh-accent) !important; box-shadow: none !important; }
/* 입력 중 굵은 선 제거: 전역 :focus-visible 링(3px)과 Streamlit 기본 포커스 테두리를 입력창 안에서만 끈다.
   포커스 표시는 바깥 테두리 색(보라, 1px)과 보라색 커서로 대신한다 */
[data-testid="stChatInput"] textarea:focus, [data-testid="stChatInput"] textarea:focus-visible,
[data-testid="stChatInput"] div:focus-within {
  outline: none !important; box-shadow: none !important; border-color: transparent !important; }
[data-testid="stChatInput"] textarea { font-size: 16px; line-height: 1.5; color: var(--jh-ink); background: transparent; caret-color: var(--jh-accent); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--jh-ink2); opacity: 1; }
[data-testid="stChatInputSubmitButton"] { width: 34px; height: 34px; min-width: 34px; border-radius: 10px; border: 0;
                                          background: var(--jh-accent); color: var(--jh-on-accent); transition: background-color .15s, color .15s; }
[data-testid="stChatInputSubmitButton"] svg { fill: currentColor; color: inherit; }
[data-testid="stChatInputSubmitButton"]:disabled { background: transparent; color: var(--jh-ink2); }
[data-testid="stChatInputSubmitButton"]:not(:disabled):hover { filter: brightness(1.08); }
[data-testid="stChatInputSubmitButton"]:focus-visible { outline: 2px solid var(--jh-accent); outline-offset: 2px; }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--jh-paper); }
/* 조회 결과 화면 하단 줄: [홈 버튼][입력창]. 입력창 높이 64px → 52px(약 20% 축소): 안쪽 위아래 여백 12px → 6px.
   글자 크기는 16px 유지(모바일 확대 방지). 홈 버튼은 입력창과 같은 52px 높이 */
.st-key-jh-input-chat [data-testid="stChatInput"] > div { padding-top: 6px !important; padding-bottom: 6px !important; }
/* 홈 버튼: 옅은 보라 바탕(accent-soft) + 보라 아이콘·글자 (대비 라이트 5.2:1 · 다크 5.8:1).
   흰 입력창 옆에서 바로 눈에 띄고, 진한 보라 전송 버튼보다는 한 단계 낮다. 마우스 올림·누름은 진한 보라 + 흰 글자(6.1:1 · 7.0:1) */
.st-key-jh-home button { height: 52px; min-height: 52px; padding: 0 14px; border-radius: 14px;
                         border: 1px solid color-mix(in srgb, var(--jh-accent) 25%, transparent);
                         background: var(--jh-accent-soft); color: var(--jh-accent); transition: border-color .15s, background-color .15s, color .15s; }
.st-key-jh-home button p, .st-key-jh-home button span { color: inherit; font-size: 14.5px; font-weight: 600; white-space: nowrap; }
.st-key-jh-home button:hover, .st-key-jh-home button:active { border-color: var(--jh-accent); background: var(--jh-accent); color: var(--jh-on-accent); }
.st-key-jh-chatbar { align-items: center; }
[data-testid="stBottomBlockContainer"] { max-width: 780px; padding-top: 12px; padding-bottom: 20px; }
[data-testid="stMainBlockContainer"] { max-width: 1000px; }
[data-testid="stChatMessage"]:has([class*="st-key-jh-bot"]) { max-width: 780px; }
@media (prefers-reduced-motion: reduce) { .stApp * { transition: none !important; animation: none !important; } .jh-caret { display: none; } }
"""

_CHECK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>'
_SPARK = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8Z"/>'
          '<path d="M19 15l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7Z"/></svg>')

BOT_HEAD_HTML = (f'<div class="jh-bot-head"><span class="jh-ava">{_SPARK}</span>'
                 '<b>진짜후기 AI</b><span>쇼핑 비교 어시스턴트</span></div>')

APP_NAME = "진짜후기 쇼핑 비교 에이전트"  # 앱 명칭 (브라우저 탭 제목·상단 타이틀)
TITLE = "광고 후기 빼고, 내 카드로 계산한 가격으로 Top 3"


HEADER_HTML = f"""
<header class="jh-appbar">
<div class="jh-top">
<div class="jh-brand"><a class="jh-home" href="./" target="_self" aria-label="{APP_NAME} 처음 화면으로 이동" title="처음 화면으로">
<span class="jh-mark" aria-hidden="true">{_SPARK}</span><span class="jh-name">{APP_NAME}</span></a>
<span class="jh-status"><i class="jh-dot" aria-hidden="true"></i>AI 쇼핑 어시스턴트</span></div>
<span class="jh-tag sample">샘플 데이터 · 가격·판매처·할인은 가상</span>
</div>
</header>
"""
# 대화 중 페이지 제목(h1). 랜딩에서는 히어로가 h1을 갖는다
CHAT_TITLE_HTML = f'<h1 class="jh-title jh-chat-title">{TITLE}</h1>'

# 새 답변이 오면 그 답변의 맨 위(어시스턴트 이름 줄 → 머리말 → 1위 카드)가 타이틀 띠 바로 아래에 오도록 스크롤한다.
# Streamlit 채팅은 입력창이 있으면 맨 아래로 자동 스크롤하므로, 렌더가 끝난 뒤 몇 번 더 맞춘다.
# 사용자가 그사이 직접 스크롤하면 더 움직이지 않는다. 고정 문자열이며 사용자·API 데이터는 넣지 않는다
SCROLL_TO_NEW_JS = """
<script>
(function () {
  var w = window.parent, d = w.document, userMoved = false;
  ['wheel', 'touchmove', 'keydown'].forEach(function (ev) {
    w.addEventListener(ev, function () { userMoved = true; }, {once: true, passive: true});
  });
  function go() {
    if (userMoved) return;
    var msgs = d.querySelectorAll('[data-testid="stChatMessage"]');
    if (!msgs.length) return;
    var last = msgs[msgs.length - 1];
    var target = last.querySelector('.jh-bot-head') || last;
    target.scrollIntoView({behavior: 'auto', block: 'start'});
  }
  [150, 500, 1100, 2000].forEach(function (t) { setTimeout(go, t); });
})();
</script>
"""


# ── 랜딩(대화 전 빈 화면): 히어로 · 대화 미리보기(스트리밍 데모) · 기능 카드 ──
HERO_HTML = f"""
<div class="jh-hero">
<span class="jh-pill"><b>AI 에이전트</b>협찬 후기를 거르고 실결제가로 비교해요</span>
<h1 class="jh-hero-title">{TITLE}</h1>
<p class="jh-lede">예산과 용도, 가진 카드·멤버십을 한 문장으로 적으면 실구매 후기만 근거로 판매처별 실결제가를 계산해 서로 다른 모델 3개를 추천합니다.</p>
</div>
"""

HERO_FINE_HTML = f"""
<ul class="jh-fine">
<li>{_CHECK}샘플 데이터로 바로 체험</li>
<li>{_CHECK}답변까지 최대 30초</li>
<li>{_CHECK}조건이 빠지면 되묻기</li>
</ul>
"""

_PREVIEW_Q = "원룸에서 쓸 35만원 이하 무선청소기 3개 추천해줘. 소음이 제일 중요해"
_PREVIEW_A = ("조용한 원룸용으로는 클린웨이브 V7 사일런트를 1위로 추천해요. 실구매 후기에서 "
              "\"밤 10시 넘어서 돌려도 옆집 눈치가 안 보인다\"는 평이 반복돼요. 판매처 3곳 중 바로마켓이 가장 저렴합니다.")
_PREVIEW_STEPS = ["상품 검색", "협찬 후기 제외", "실결제가 계산"]
_STREAM_START_MS, _STREAM_STEP_MS = 1300, 45


def _stream_spans(text: str) -> tuple[str, int]:
    """단어마다 지연이 다른 span으로 감싸 CSS만으로 '스트리밍'처럼 보이게 한다 (고정 문구만 사용)."""
    words = text.split(" ")
    spans = "".join(f'<span class="w" style="--i:{i}">{html.escape(w)} </span>' for i, w in enumerate(words))
    return spans, _STREAM_START_MS + len(words) * _STREAM_STEP_MS


def preview_html() -> str:
    spans, end_ms = _stream_spans(_PREVIEW_A)
    steps = "".join(f'<span class="jh-step" style="--d:{250 + i * 300}ms">{_CHECK}{s}</span>'
                    for i, s in enumerate(_PREVIEW_STEPS))
    return f"""
<figure class="jh-preview" aria-label="예시 대화 화면">
<div class="jh-preview-head"><span class="jh-ava">{_SPARK}</span><div><b>진짜후기 AI</b><span><i class="jh-dot" aria-hidden="true"></i>예시 화면 · 샘플 데이터</span></div></div>
<div class="jh-thread">
<p class="jh-ubub">{html.escape(_PREVIEW_Q)}</p>
<div class="jh-steps">{steps}</div>
<p class="jh-bbub">{spans}<span class="jh-caret" style="--end:{end_ms}ms" aria-hidden="true"></span></p>
<div class="jh-mini" style="--end:{end_ms}ms"><span class="jh-rank">1</span><div><b>클린웨이브 V7 사일런트</b><span>바로마켓 · 판매처 3곳 비교</span></div>
<div class="jh-mini-price"><span>최저 실결제가</span><b class="jh-num">242,000원</b></div></div>
</div>
</figure>
"""


_FEATURES = [
    ('<path d="M12 3 4.5 6v6c0 4.4 3.2 8 7.5 9 4.3-1 7.5-4.6 7.5-9V6Z"/><path d="m9 12 2 2 4-4"/>',
     "협찬 후기 걸러내기", "원고료·제품 제공 표기를 찾아 협찬 후기는 추천 근거에서 뺍니다.", "협찬 판별 정확도 목표 85% 이상"),
    ('<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18M7 15h4"/>',
     "내 결제 수단 기준 실결제가", "카드 할인, 멤버십, 배송비까지 판매처마다 끝까지 계산합니다.", "가격·순위는 코드가 계산"),
    ('<path d="M4 5a2 2 0 0 1 2-2h14v16H6a2 2 0 0 0-2 2Z"/><path d="M4 19V5M9 8h7M9 12h5"/>',
     "근거 리뷰 공개", "추천 이유마다 실구매 후기 번호를 붙이고 원문을 펼쳐 볼 수 있습니다.", "출처 번호 [1][2]로 연결"),
    ('<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12Z"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.3 2.4c-.5.2-.8.6-.8 1.1v.5M12 16h.01"/>',
     "부족한 조건은 되묻기", "예산이나 용도가 빠지면 추천 전에 한두 번 물어봅니다.", "되묻기 최대 2턴"),
    ('<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/>',
     "판매처 한눈에 비교", "같은 모델을 파는 판매처를 한 카드에 모아 가장 싼 곳을 표시합니다.", "서로 다른 모델 3개 추천"),
    ('<rect x="4.5" y="10.5" width="15" height="10" rx="2"/><path d="M8 10.5V7.5a4 4 0 0 1 8 0v3"/>',
     "안전장치", "개인정보는 가리고, 범위 밖 질문과 구매 대행 요청은 정중히 거절합니다.", "대화는 서버에 저장 안 함"),
]

def _ico(path: str) -> str:
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{path}</svg>'


_FEATURES_SHOWN = 3  # 핵심 차별점(협찬 걸러내기·실결제가·근거 공개)은 항상 보이고, 나머지는 접어 둔다
_CHEVRON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>'


def _feat_cards(rows) -> str:
    return "".join(f'<article class="jh-feat"><span class="jh-feat-ico">{_ico(p)}</span><h3>{t}</h3><p>{d}</p>'
                   f'<span class="jh-spec">{s}</span></article>' for p, t, d, s in rows)


_MORE_N = len(_FEATURES) - _FEATURES_SHOWN
# 접기는 HTML <details>: 다시 그리기(rerun) 없이 열리고 키보드(Tab·Enter·Space)·화면 낭독기를 기본 지원한다
FEATURES_HTML = (
    '<section class="jh-section" aria-labelledby="jh-feat-title">'
    '<p class="jh-kicker">AI가 하는 일</p><h2 id="jh-feat-title" class="jh-h2">추천만 하지 않고, 근거와 가격까지 확인합니다</h2>'
    f'<div class="jh-feats">{_feat_cards(_FEATURES[:_FEATURES_SHOWN])}</div>'
    f'<details class="jh-more"><summary><span class="o">기능 {_MORE_N}개 더 보기</span><span class="c">접기</span>{_CHEVRON}</summary>'
    f'<div class="jh-feats">{_feat_cards(_FEATURES[_FEATURES_SHOWN:])}</div></details>'
    "</section>"
)


def _html(s: str) -> str:
    """마크다운이 들여쓰기·빈 줄을 코드 블록으로 바꾸지 않게 한 줄로 합친다."""
    return "".join(line.strip() for line in s.strip().splitlines())


def _theme_tokens() -> dict:
    try:
        mode = st.context.theme.type or "light"
    except Exception:  # 구버전·AppTest에서 theme 정보가 없을 때
        mode = "light"
    return THEMES.get(mode, THEMES["light"])


def inject_style() -> None:
    tokens = ";".join(f"--jh-{k}:{v}" for k, v in _theme_tokens().items())
    st.markdown(f"<style>:root{{{tokens}}}{_html(CSS)}</style>", unsafe_allow_html=True)


# ─────────────────────────────── answer → 카드 (chains.render_answer 형식 파싱) ───────────────────────────────
# answer는 코드가 만든 고정 형식이지만 요약·근거 문장에 리뷰 유래 텍스트가 섞이므로
# 카드 HTML의 모든 값은 html.escape를 거치고, 링크는 http(s) URL만 허용한다.

_REC_HEAD = re.compile(r"^(\d+)\) (.+?) · 최저 실결제가 ([\d,]+원)(?:\((.+?)\))?(?: · 판매처 (\d+)곳 비교)?$")
_PRICE_HEAD = re.compile(r"^■ (.+?) · 가장 싼 곳: (.+?) ([\d,]+원) · 판매처 (\d+)곳$")
_MALL_LINE = re.compile(r"^- (.+?): (.+) = ([\d,]+원)(?: · (\S+))?$")
_REFS = re.compile(r"\s*((?:\[\d+\])+)$")
_SOURCE = re.compile(r"^(?:\[(\d+)\]|-) (.+?)(?: · (https?://\S+))?$")
_URL = re.compile(r"^https?://[^\s\"'<>]+$")


def _split_refs(text: str) -> tuple[str, str]:
    m = _REFS.search(text)
    return (text[:m.start()], m.group(1)) if m else (text, "")


def parse_answer(text: str) -> dict | None:
    """추천·가격 비교 answer를 {intro, cards, sources, notices}로 나눈다. 카드 형식이 없으면 None."""
    intro, cards, sources, notices = [], [], [], []
    card, mode, tail_title = None, None, None
    for raw in (text or "").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        head = _REC_HEAD.match(line) or _PRICE_HEAD.match(line)
        if head:
            if _REC_HEAD.match(line):
                rank, name, price, where, n = head.groups()
            else:
                name, mall, price, n = head.groups()
                rank, where = None, mall
            card = {"rank": rank, "name": name, "price": price, "where": where or "", "malls_n": n,
                    "summary": "", "notes": [], "malls": [], "pros": [], "cons": [], "kv": [], "foot": ""}
            cards.append(card)
            mode, tail_title = None, None
            continue
        if card is not None and line.startswith("   "):
            s = line.strip()
            if s.startswith("요약:"):
                card["summary"] = s[3:].strip()
            elif s.startswith("판매처별 실결제가"):
                mode = "malls"
            elif s == "추천 근거:":
                mode = "pros"
            elif s.startswith("- ") and mode == "malls" and (m := _MALL_LINE.match(s)):
                card["malls"].append(dict(zip(("mall", "calc", "total", "link"), m.groups())))
            elif s.startswith("- ") and mode == "pros":
                card["pros"].append(_split_refs(s[2:]))
            elif s.startswith("단점(실구매 후기):"):
                card["cons"] = [_split_refs(p.strip()) for p in s.split(":", 1)[1].split("; ") if p.strip()]
            elif s.startswith("이런 분께는 비추천:"):
                card["kv"].append(("비추천", s.split(":", 1)[1].strip()))
            elif s.startswith("확인 불가:"):
                card["kv"].append(("확인 불가", s.split(":", 1)[1].strip()))
            elif s.startswith("리뷰 구분:"):
                card["foot"] = s.split(":", 1)[1].strip()
            else:
                card["notes"].append(s.lstrip("※ ").strip())
            continue
        if not cards:  # 첫 카드 전: 머리말·안내
            intro.append(line)
            continue
        card, mode = None, None  # 카드 뒤: 출처·안내
        if line.startswith("※"):
            notices.append(line.lstrip("※ ").strip())
        elif line.endswith(":") and not line.startswith(("[", "- ")):
            tail_title = line[:-1]
        elif m := _SOURCE.match(line):
            sources.append({"group": tail_title or "출처", "n": m.group(1), "title": m.group(2), "link": m.group(3)})
        else:
            notices.append(line)
    return {"intro": intro, "cards": cards, "sources": sources, "notices": notices} if cards else None


def _e(s) -> str:
    return html.escape(str(s or ""), quote=True)


def _link(label: str, url: str | None) -> str:
    if url and _URL.match(url):
        return f'<a href="{_e(url)}" target="_blank" rel="noopener noreferrer">{_e(label)}</a>'
    return _e(label)


def card_html(c: dict) -> str:
    rank = (f'<span class="jh-rank" aria-label="{_e(c["rank"])}위">{_e(c["rank"])}</span>' if c["rank"]
            else '<span class="jh-rank plain" aria-hidden="true">₩</span>')
    price_k = "최저 실결제가" if c["rank"] else "가장 싼 곳"
    parts = [f'<article class="jh-card"><div class="jh-card-body"><div class="jh-rec-head">{rank}<div>'
             f'<h3>{_e(c["name"])}</h3>'
             + (f'<p class="jh-summary">{_e(c["summary"])}</p>' if c["summary"] else "")
             + f'</div><div class="jh-price"><span class="k">{price_k}</span><span class="v jh-num">{_e(c["price"])}</span>'
             + (f'<span class="w">{_e(c["where"])}</span>' if c["where"] else "")
             + "</div></div>"]
    parts += [f'<p class="jh-note">{_e(n)}</p>' for n in c["notes"]]
    if c["kv"]:
        parts.append('<dl class="jh-kv">' + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in c["kv"]) + "</dl>")
    parts.append("</div>")

    def items(rows):
        return "".join(f'<li>{_e(t)}{f"<span class=\"jh-ref\">{_e(r)}</span>" if r else ""}</li>' for t, r in rows)

    if c["pros"]:
        parts.append(f'<div class="jh-sec"><h4>추천 근거 · 실구매 후기</h4><ul class="jh-list pro">{items(c["pros"])}</ul></div>')
    if c["cons"]:
        parts.append(f'<div class="jh-sec"><h4>아쉬운 점 · 실구매 후기</h4><ul class="jh-list con">{items(c["cons"])}</ul></div>')
    if c["malls"]:
        n = c["malls_n"] or len(c["malls"])
        best = min(int(m["total"].replace(",", "").rstrip("원")) for m in c["malls"])
        rows = "".join(
            f'<li class="{"best" if int(m["total"].replace(",", "").rstrip("원")) == best else ""}">'
            f'<span class="m">{_link(m["mall"], m["link"])}</span><span class="t jh-num">{_e(m["total"])}</span>'
            f'<span class="c">{_e(m["calc"])}</span></li>' for m in c["malls"])
        parts.append(f'<details><summary>판매처 {_e(n)}곳 실결제가 비교</summary><ul class="jh-malls">{rows}</ul></details>')
    foot = "샘플 데이터" + (f" · {c['foot']}" if c["foot"] else "")
    parts.append(f'<div class="jh-card-foot">{_e(foot)}</div></article>')
    return "".join(parts)


def sources_html(sources: list[dict]) -> str:
    groups: dict[str, list[dict]] = {}
    for s in sources:
        groups.setdefault(s["group"], []).append(s)
    secs = []
    for title, rows in groups.items():
        lis = "".join(f'<li>{f"<span class=\"n\">[{_e(s["n"])}]</span>" if s["n"] else ""}{_link(s["title"], s["link"])}</li>'
                      for s in rows)
        secs.append(f'<div class="jh-sec jh-sources"><h4>{_e(title)}</h4><ol>{lis}</ol></div>')
    return f'<section class="jh-card">{"".join(secs)}</section>'


def render_answer_cards(answer: str, body: dict) -> bool:
    """카드 형식이면 카드로 그리고 True. 아니면 False (호출부가 일반 텍스트로 표시)."""
    parsed = parse_answer(answer)
    if not parsed:
        return False
    ctx_n = len(body.get("contexts", []))
    tools = sum(1 for s in body.get("trace", []) if s.get("step") in TOOL_STEPS)
    intro = parsed["intro"]
    st.markdown(intro[0] if intro else "추천 결과입니다.")  # 머리말은 일반 마크다운 (이스케이프)
    st.markdown(f'<div class="jh-intro"><span class="jh-tag ok">근거 리뷰 {ctx_n}개</span>'
                f'<span class="jh-tag">도구 {tools}회 호출</span><span class="jh-tag sample">샘플 데이터</span></div>'
                + "".join(f'<p class="jh-notice">{_e(x)}</p>' for x in intro[1:]), unsafe_allow_html=True)
    for c in parsed["cards"]:
        st.markdown(card_html(c), unsafe_allow_html=True)
    if parsed["sources"]:
        st.markdown(sources_html(parsed["sources"]), unsafe_allow_html=True)
    if parsed["notices"]:
        st.markdown("".join(f'<p class="jh-notice">{_e(n)}</p>' for n in parsed["notices"]), unsafe_allow_html=True)
    return True


def _md(text: str) -> str:
    """answer의 줄바꿈과 들여쓰기를 마크다운에서 그대로 보이게 한다."""
    return "  \n".join(line.replace("   ", "   ", 1) for line in (text or "").split("\n"))


def render_response(msg: dict) -> None:
    status, body = msg.get("status"), msg.get("body")
    if msg.get("error"):
        st.error(msg["error"], icon=":material/wifi_off:")
        return
    clarify = is_clarify(body)
    if status and status >= 500:
        st.warning(body["answer"], icon=":material/error:")
    elif clarify or not render_answer_cards(body["answer"], body):
        tag = '<span class="jh-tag ask">되묻기</span>' if clarify else '<span class="jh-tag sample">샘플 데이터</span>'
        with st.container(key=f"jh-text-{id(msg)}"):
            st.markdown(f'<div class="jh-intro">{tag}</div>', unsafe_allow_html=True)
            st.markdown(_md(body["answer"]))
    if msg.get("clarify_note"):
        st.info(msg["clarify_note"], icon=":material/forum:")
    ctx = body.get("contexts", [])
    with st.expander(f"근거 리뷰 ({len(ctx)}개)", icon=":material/rate_review:"):
        if not ctx:
            st.markdown("이 답변에 쓴 리뷰가 없습니다. 가격만 묻는 질문, 되묻기, 범위 밖 질문에는 리뷰를 쓰지 않습니다.")
        for c in ctx:
            source = "실제 블로그 참고 후기" if str(c["doc_id"]).startswith("blog-") else "실구매 후기 · 샘플 데이터"
            with st.container(border=True):
                st.markdown(f"`{c['doc_id']}` · {source}  \n{c['text']}")
    with st.expander(f"처리 과정 ({len(body.get('trace', []))}단계)", icon=":material/account_tree:"):
        for i, s in enumerate(body.get("trace", []), 1):
            label = STEP_LABELS.get(s["step"], s["step"])
            st.markdown(f"{i}. **{label}** `{s['step']}` — {s['input']}  \n→ {s['output']}")
        if msg.get("sent_question") and msg["sent_question"] != msg.get("user_text"):
            st.caption(f"API로 보낸 질문(이어 붙임): {msg['sent_question']}")


def handle(user_text: str, wait_text: str = "후보를 찾고 리뷰를 분석하는 중...") -> None:
    ss = st.session_state
    sent = build_question(ss.pending, user_text)
    with st.spinner(wait_text):
        status, body, error = call_api(sent)
    msg = {"role": "assistant", "status": status, "body": body, "error": error,
           "sent_question": sent, "user_text": user_text}
    if body is not None and status == 200:
        ss.pending, exceeded = next_pending(ss.pending, sent, body)
        if ss.pending:
            msg["clarify_note"] = (f"답을 입력하면 앞 질문과 이어서 보냅니다 "
                                   f"(되묻기 {ss.pending['turns']}/{MAX_CLARIFY_TURNS}턴).")
        elif exceeded:
            msg["clarify_note"] = "되묻기는 2턴까지만 이어 갑니다. 예산·용도를 담아 새로 질문해 주세요."
    ss.messages += [{"role": "user", "content": user_text}, msg]


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon=":material/shopping_cart:", layout="centered")
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("pending", None)

    inject_style()
    landing = not ss.messages
    st.markdown(_html(HEADER_HTML), unsafe_allow_html=True)
    if not landing:
        st.markdown(CHAT_TITLE_HTML, unsafe_allow_html=True)

    clicked = None
    if landing:  # 대화 전: 랜딩 (히어로 + 대화 미리보기 + 기능 카드)
        with st.container(key="jh-landing"):
            left, right = st.columns([1.05, 1], gap="large", vertical_alignment="center")
            with left:
                st.markdown(_html(HERO_HTML), unsafe_allow_html=True)
                st.markdown(_html(HERO_FINE_HTML), unsafe_allow_html=True)
            with right:
                st.markdown(_html(preview_html()), unsafe_allow_html=True)
            st.markdown(FEATURES_HTML, unsafe_allow_html=True)
        st.markdown('<p class="jh-try-label">직접 물어보기 · 예시 질문을 누르거나 아래 입력창에 적어 주세요</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="jh-label">예시 질문</p>', unsafe_allow_html=True)
    with st.container(key="jh-demo"):
        cols = st.columns(len(DEMO_QUESTIONS) + 1)
        for i, (col, (label, q)) in enumerate(zip(cols, DEMO_QUESTIONS)):
            if col.button(label, key=f"jh-demo-{i}", icon=DEMO_ICONS[i], use_container_width=True, help=q):
                clicked = q
        if cols[-1].button("새 대화", key="jh-new", icon=":material/refresh:", use_container_width=True):
            ss.messages, ss.pending = [], None
            st.rerun()

    # 예시 버튼: 대화 중에는 맨 아래 스피너가 화면 밖이라 반응이 없어 보였다.
    # 누른 버튼 바로 아래에서 진행 상황을 보여 주고, 답이 오면 새 질문 위치로 스크롤한다 (랜딩과 같은 체감)
    if clicked:
        ss.pending = None  # 데모 버튼은 항상 새 질문으로 보낸다
        handle(clicked, wait_text="예시 질문을 보냈어요. 후보를 찾고 리뷰를 분석하는 중이에요 (최대 30초)")
        ss.scroll_to_new = True
        st.rerun()

    for i, m in enumerate(ss.messages):
        is_user = m["role"] == "user"
        avatar = ":material/person:" if is_user else ":material/auto_awesome:"
        with st.chat_message(m["role"], avatar=avatar):
            with st.container(key=f"jh-{'user' if is_user else 'bot'}-{i}"):
                if is_user:
                    st.markdown(m["content"])
                else:
                    st.markdown(BOT_HEAD_HTML, unsafe_allow_html=True)
                    render_response(m)

    if ss.pop("scroll_to_new", False) and len(ss.messages) >= 2:  # 새 답변: 그 답변의 첫 결과부터
        with st.container(key="jh-scroll"):  # 화면에서 자리를 차지하지 않는 1px iframe (CSS로 숨김)
            st.iframe(SCROLL_TO_NEW_JS, height=1)

    placeholder = "추가로 알려줄 내용을 입력하세요" if ss.pending else "예산·용도·결제 수단을 함께 적어 주세요"
    if landing:
        # 랜딩: 입력창을 본문 안(예시 질문 아래)에 둔다. 화면 하단에 고정된 입력창이 있으면 Streamlit이
        # 페이지를 맨 아래로 자동 스크롤하므로, 랜딩은 스크롤 없이 맨 위(히어로·미리보기)에서 열린다
        with st.container(key="jh-inline-input"):
            typed = st.chat_input(placeholder, key="jh-input-landing")
    else:
        # 조회 결과 화면: 화면 하단 고정 줄에 [홈][입력창]을 나란히 둔다 (가로 배치라 모바일에서도 쌓이지 않음)
        with st.bottom, st.container(key="jh-chatbar", horizontal=True, wrap=False, vertical_alignment="center", gap="small"):
            if st.button("홈", key="jh-home", icon=HOME_ICON, help="처음 화면으로 (대화 지우기)"):
                ss.messages, ss.pending = [], None
                st.rerun()
            typed = st.chat_input(placeholder, key="jh-input-chat")
    if typed and typed.strip():
        handle(typed.strip())
        ss.scroll_to_new = True
        st.rerun()


if __name__ == "__main__":  # streamlit run·AppTest는 스크립트를 __main__으로 실행한다
    main()
