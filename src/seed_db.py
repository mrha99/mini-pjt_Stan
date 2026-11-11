"""샘플 카탈로그(data/products.json)로 최근 30일 가격 이력 DB(data/prices.db)를 생성한다.

가짜 데이터(과제용): 실제 시세가 아니다. 같은 입력이면 항상 같은 결과가 나오도록
product_id 기반 시드로 생성하며, 마지막 날(기준일) 가격은 products.json의 현재 가격과 같다.

usage: python -m src.seed_db [--days 30] [--end-date 2026-11-11] [--db data/prices.db]
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import zlib
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_PATH = ROOT / "data" / "products.json"
DB_PATH = ROOT / "data" / "prices.db"

SCHEMA = """
DROP TABLE IF EXISTS price_history;
DROP TABLE IF EXISTS meta;
CREATE TABLE price_history (
    product_id   TEXT    NOT NULL,
    model_id     TEXT    NOT NULL,
    mall         TEXT    NOT NULL,
    date         TEXT    NOT NULL,   -- YYYY-MM-DD
    price        INTEGER NOT NULL,   -- 판매가(원)
    shipping_fee INTEGER NOT NULL,   -- 배송비(원)
    PRIMARY KEY (product_id, date)
);
CREATE INDEX idx_price_history_model ON price_history (model_id, date);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _round_price(value: float) -> int:
    """100원 단위로 반올림한다."""
    return int(round(value / 100.0)) * 100


def price_series(product_id: str, current_price: int, days: int) -> list[int]:
    """product_id로 시드를 고정한 가격 경로. 마지막 값은 current_price."""
    rng = random.Random(zlib.crc32(product_id.encode("utf-8")))
    # 30일 전에는 현재가보다 0~12% 비쌌다가 점차 내려오는 추세 + 소폭 변동
    start_premium = rng.uniform(0.0, 0.12)
    # 한 번의 단기 특가(2~3일)를 넣어 "최근 최저가" 비교가 의미 있게 만든다
    sale_start = rng.randint(3, max(3, days - 6))
    sale_len = rng.randint(2, 3)
    sale_cut = rng.uniform(0.05, 0.15)

    prices = []
    for i in range(days - 1):
        progress = i / max(1, days - 1)
        base = current_price * (1 + start_premium * (1 - progress))
        noise = rng.uniform(-0.015, 0.015)
        factor = 1 + noise
        if sale_start <= i < sale_start + sale_len:
            factor -= sale_cut
        prices.append(max(1000, _round_price(base * factor)))
    prices.append(current_price)
    return prices


def seed(days: int = 30, end_date: str | None = None, db_path: Path = DB_PATH,
         products_path: Path = PRODUCTS_PATH) -> int:
    doc = json.loads(products_path.read_text(encoding="utf-8"))
    products = doc["products"]
    end = date.fromisoformat(end_date or doc["_meta"]["base_date"])
    dates = [(end - timedelta(days=days - 1 - i)).isoformat() for i in range(days)]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in products:
        for d, price in zip(dates, price_series(p["product_id"], p["price"], days)):
            rows.append((p["product_id"], p["model_id"], p["mall"], d, price, p["shipping_fee"]))

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO price_history VALUES (?, ?, ?, ?, ?, ?)", rows)
        conn.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [
                ("notice", "가짜 데이터(과제용)"),
                ("source", "data/products.json 샘플 카탈로그에서 src/seed_db.py로 생성"),
                ("start_date", dates[0]),
                ("end_date", dates[-1]),
                ("days", str(days)),
            ],
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="샘플 가격 이력 DB 생성 (가짜 데이터)")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--end-date", default=None, help="기준일 YYYY-MM-DD (기본: products.json base_date)")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    n = seed(days=args.days, end_date=args.end_date, db_path=args.db)
    print(f"price_history {n}행 생성 → {args.db}")


if __name__ == "__main__":
    main()
