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

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT", "120"))
MAX_CLARIFY_TURNS = 2
SEP = " / "

DEMO_QUESTIONS = [
    ("원룸 청소기 추천", "원룸에서 쓸 35만원 이하 무선청소기 3개 추천해줘. 소음이 제일 중요해"),
    ("어디서 사야 쌀까", "클린웨이브 V7 사일런트 무선청소기 어디서 사는 게 제일 싸? 나는 누리카드랑 바로플러스 멤버십 있어"),
    ("조건 없이 질문 (되묻기)", "청소기 추천해줘"),
]


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


# ─────────────────────────────── 화면 ───────────────────────────────


def _md(text: str) -> str:
    """answer의 줄바꿈과 들여쓰기를 마크다운에서 그대로 보이게 한다."""
    return "  \n".join(line.replace("   ", "   ", 1) for line in (text or "").split("\n"))


def render_response(msg: dict) -> None:
    status, body = msg.get("status"), msg.get("body")
    if msg.get("error"):
        st.error(msg["error"])
        return
    if status and status >= 500:
        st.warning(body["answer"])
    else:
        st.markdown(_md(body["answer"]))
    if msg.get("clarify_note"):
        st.info(msg["clarify_note"])
    ctx = body.get("contexts", [])
    with st.expander(f"근거 리뷰 ({len(ctx)}개)"):
        if not ctx:
            st.caption("이 답변에 쓴 리뷰 청크가 없습니다 (가격만 묻는 질문·되묻기·가드레일 차단 등).")
        for c in ctx:
            st.markdown(f"**{c['doc_id']}**  \n{c['text']}")
    with st.expander(f"처리 과정 ({len(body.get('trace', []))}단계)"):
        for i, s in enumerate(body.get("trace", []), 1):
            st.markdown(f"{i}. **{s['step']}** — {s['input']}  \n→ {s['output']}")
        if msg.get("sent_question") and msg["sent_question"] != msg.get("user_text"):
            st.caption(f"API로 보낸 질문(이어 붙임): {msg['sent_question']}")


def handle(user_text: str) -> None:
    ss = st.session_state
    sent = build_question(ss.pending, user_text)
    with st.spinner("후보를 찾고 리뷰를 분석하는 중..."):
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
    st.set_page_config(page_title="진짜후기 쇼핑 비교", page_icon="🛒", layout="centered")
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("pending", None)

    st.title("진짜후기 쇼핑 비교")
    st.caption("협찬 리뷰를 걸러낸 실구매 후기와 내 결제 조건 기준 실결제가로 Top 3를 추천합니다. "
               "가격·판매처·할인은 샘플 데이터(가짜)입니다.")

    cols = st.columns(len(DEMO_QUESTIONS) + 1)
    clicked = None
    for col, (label, q) in zip(cols, DEMO_QUESTIONS):
        if col.button(label, use_container_width=True, help=q):
            clicked = q
    if cols[-1].button("새 대화", use_container_width=True):
        ss.messages, ss.pending = [], None
        st.rerun()

    for m in ss.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_response(m)

    placeholder = "추가로 알려줄 내용을 입력하세요" if ss.pending else "예산·용도·결제 수단을 함께 적어 주세요"
    typed = st.chat_input(placeholder)
    question = clicked or typed
    if question and question.strip():
        if clicked:  # 데모 버튼은 항상 새 질문으로 보낸다
            ss.pending = None
        handle(question.strip())
        st.rerun()

    st.caption(f"API: {API_URL}")


if __name__ == "__main__":  # streamlit run·AppTest는 스크립트를 __main__으로 실행한다
    main()
