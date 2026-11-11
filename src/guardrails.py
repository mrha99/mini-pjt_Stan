"""인젝션 탐지, PII 마스킹 (패턴 6)

지금은 PII 마스킹(mask_pii)만 있다. 도구 캐시·블로그 리뷰 저장(SERVICE.md 정책 3)에서 쓰며,
7단계에서 인젝션 탐지·범위 밖 요청 거절을 이 모듈에 추가한다.
"""

from __future__ import annotations

import re

_SIDO = (r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)"
         r"(?:특별시|광역시|특별자치시|특별자치도|도)?")

# 순서 중요: 주소·이메일을 먼저 지워야 그 안의 숫자가 전화번호·계좌번호로 잘못 잡히지 않는다.
PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[이메일]"),
    # 도로명/지번 주소: (시도) 시군구 … 로/길/동 번지 (동·호)
    ("address", re.compile(
        rf"(?:{_SIDO}\s*)?[가-힣]{{1,10}}(?:시|군|구)\s+(?:[가-힣]{{1,10}}(?:시|군|구)\s+)?"
        r"[가-힣0-9]{1,20}(?:로|길|동|읍|면|리)\s*\d{1,5}(?:-\d{1,5})?"
        r"(?:\s*,?\s*\d{1,4}동)?(?:\s*\d{1,5}호)?"), "[주소]"),
    ("phone", re.compile(r"(?<!\d)01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"), "[전화번호]"),
    ("phone", re.compile(r"(?<!\d)0(?:2|[3-6][1-5]|70|50\d?)[-.\s]\d{3,4}[-.\s]\d{4}(?!\d)"), "[전화번호]"),
    ("phone", re.compile(r"(?<!\d)1[5-9]\d{2}-\d{4}(?!\d)"), "[전화번호]"),
    # 계좌번호: 날짜(2026-11-11)는 제외
    ("account", re.compile(r"(?<![\d-])(?!(?:19|20)\d{2}-\d{1,2}-\d{1,2}(?![\d-]))"
                           r"\d{2,6}-\d{2,6}-\d{2,8}(?:-\d{1,4})?(?![\d-])"), "[계좌번호]"),
]


def mask_pii(text: str) -> str:
    """전화번호·이메일·주소·계좌번호를 마스킹한다 (응답·로그·캐시·저장 리뷰 공통)."""
    if not text:
        return text
    for _, pattern, repl in PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def find_pii(text: str) -> list[str]:
    """탐지된 PII 종류 목록 (trace 기록용, 원문은 반환하지 않음)."""
    return [kind for kind, pattern, _ in PII_PATTERNS if pattern.search(text or "")]
