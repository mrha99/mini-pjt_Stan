"""app.py(Streamlit UI) 테스트: 순수 로직 + AppTest로 화면 흐름 (API는 가짜 httpx.post)."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from src import app

APP = str(Path(__file__).resolve().parent.parent / "src" / "app.py")
CLARIFY = {"answer": "예산과 사용 환경(원룸·반려동물 등)을 알려주세요.", "contexts": [],
           "trace": [{"step": "guardrail_input", "input": "청소기 추천해줘", "output": "pass (rule)"},
                     {"step": "answer", "input": "에이전트 응답", "output": "되묻기"},
                     {"step": "guardrail_output", "input": "응답 초안 (llm)", "output": "PII 0건 마스킹, pass"}]}
RECOMMEND = {"answer": "조건에 맞는 무선청소기 3개를 추천합니다. (샘플 데이터)\n\n1) 클린웨이브 V7 사일런트 · 최저 실결제가 242,000원",
             "contexts": [{"doc_id": "rv-0002#c0", "text": "소음이 정말 작아요"}],
             "trace": [{"step": "guardrail_input", "input": "q", "output": "pass (rule)"},
                       {"step": "search_products", "input": "무선청소기, ~400,000원", "output": "후보 5개"},
                       {"step": "answer", "input": "후보 모델 3개", "output": "추천 카드 3개"},
                       {"step": "guardrail_output", "input": "응답 초안 (code)", "output": "PII 0건 마스킹, pass"}]}


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


@pytest.fixture
def fake_api(monkeypatch):
    """질문 내용에 따라 응답을 고르는 가짜 POST /query. 보낸 질문을 기록한다."""
    sent = []

    def post(url, json, timeout):
        assert url.endswith("/query")
        q = json["question"]
        sent.append(q)
        if q == "청소기 추천해줘" or (q.count(" / ") == 2 and "또 되묻기" in q):
            return FakeResponse(200, CLARIFY)
        if "또 되묻기" in q:
            return FakeResponse(200, CLARIFY)
        if "한도" in q:
            return FakeResponse(503, {"answer": "오늘 사용할 수 있는 AI 호출량을 모두 썼습니다.", "contexts": [],
                                      "trace": [{"step": "answer", "input": "x", "output": "오류"}]})
        return FakeResponse(200, RECOMMEND)

    monkeypatch.setattr(httpx, "post", post)
    return sent


def _texts(at) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n".join(e.value for e in at.info)


# ─────────────────────────────── 로직 ───────────────────────────────


def test_is_clarify_uses_trace_answer_step():
    assert app.is_clarify(CLARIFY) and not app.is_clarify(RECOMMEND)


def test_build_question_matches_eval_set_format():
    pending = {"question": "청소기 추천해줘", "clarify": "예산과 사용 환경을 알려주세요", "turns": 1}
    assert (app.build_question(pending, "40만원 이하, 원룸에서 혼자 써")
            == "청소기 추천해줘 / (에이전트) 예산과 사용 환경을 알려주세요 / 40만원 이하, 원룸에서 혼자 써")
    assert app.build_question(None, "질문") == "질문"


def test_next_pending_limits_two_turns():
    p1, ex = app.next_pending(None, "q1", CLARIFY)
    assert p1["turns"] == 1 and not ex
    p2, ex = app.next_pending(p1, "q2", CLARIFY)
    assert p2["turns"] == 2 and p2["question"] == "q2" and not ex
    p3, ex = app.next_pending(p2, "q3", CLARIFY)
    assert p3 is None and ex                                    # 3번째 되묻기는 이어 붙이지 않음
    assert app.next_pending(p2, "q", RECOMMEND) == (None, False)


def test_api_url_from_env(monkeypatch):
    import importlib
    monkeypatch.setenv("API_URL", "http://api:9000/")
    assert importlib.reload(app).API_URL == "http://api:9000"
    monkeypatch.delenv("API_URL")
    importlib.reload(app)


# ─────────────────────────────── 화면 (AppTest) ───────────────────────────────


def test_demo_button_shows_answer_contexts_and_trace(fake_api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert [b.label for b in at.button][:3] == [label for label, _ in app.DEMO_QUESTIONS]
    at.button[0].click().run()
    assert fake_api == [app.DEMO_QUESTIONS[0][1]]
    assert "조건에 맞는 무선청소기 3개를 추천합니다" in _texts(at)
    labels = [e.label for e in at.expander]
    assert "근거 리뷰 (1개)" in labels and "처리 과정 (4단계)" in labels
    assert "rv-0002#c0" in _texts(at) and "search_products" in _texts(at)


def test_clarify_then_answer_continues_to_recommendation(fake_api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("청소기 추천해줘").run()
    assert "예산과 사용 환경" in _texts(at) and "되묻기 1/2턴" in _texts(at)
    at.chat_input[0].set_value("40만원 이하, 원룸에서 혼자 써").run()
    assert fake_api[-1] == ("청소기 추천해줘 / (에이전트) 예산과 사용 환경(원룸·반려동물 등)을 알려주세요. / "
                            "40만원 이하, 원룸에서 혼자 써")
    assert "조건에 맞는 무선청소기 3개를 추천합니다" in _texts(at)
    assert at.session_state.pending is None                       # 추천이 나오면 되묻기 상태 해제


def test_demo_button_resets_pending_clarify(fake_api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("청소기 추천해줘").run()
    at.button[0].click().run()
    assert fake_api[-1] == app.DEMO_QUESTIONS[0][1]               # 데모 질문은 이어 붙이지 않음


def test_error_responses_shown(fake_api, monkeypatch):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("한도 테스트").run()
    assert any("호출량" in w.value for w in at.warning)            # 503 규약 응답 → 경고로 answer 표시

    def down(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "post", down)
    at.chat_input[0].set_value("청소기").run()
    assert any("API에 연결하지 못했습니다" in e.value for e in at.error)


def test_new_conversation_button_clears(fake_api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("청소기 추천해줘").run()
    at.button[len(app.DEMO_QUESTIONS)].click().run()
    assert at.session_state.messages == [] and at.session_state.pending is None


def test_build_question_flattens_multiline_clarify():
    pending = {"question": "청소기 추천해줘", "turns": 1,
               "clarify": "알려주세요!\n\n1. **예산**은요?\n2. **사용 환경**은요?"}
    assert (app.build_question(pending, "40만원 이하")
            == "청소기 추천해줘 / (에이전트) 알려주세요! 1. 예산은요? 2. 사용 환경은요? / 40만원 이하")
