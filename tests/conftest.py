"""테스트 공통 설정: 테스트 실행이 LangSmith trace를 남기지 않도록 끈다.

src.tools가 .env를 override=False로 읽으므로, import 전에 여기서 설정한 값이 우선한다.
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
