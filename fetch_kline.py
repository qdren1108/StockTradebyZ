"""
抓取A股市值大于指定阈值且排除创业板的股票日K线数据并保存为CSV。
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

# --------------------------- 字段映射 --------------------------- #

COLUMN_MAP_HIST = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",     # 手
    "成交额": "amount",      # 元
    "换手率": "turnover",    # %
}

# --------------------------- 日志配置 --------------------------- #
LOG_FILE = Path("fetch.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetch_mktcap")

# --------------------------- 数据源适配 --------------------------- #

import akshare as ak  # type: ignore

# ---- 市值快照 ---- #

def _get_mktcap_ak() -> pd.DataFrame:
    """实时快照，列含 code, mktcap (单位: 亿)"""
    df = ak.stock_zh_a_spot_em()
    df = df[["代码", "总市值"]].rename(columns={"代码": "code", "总市值": "mktcap"})  # type: ignore
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    return df


def get_constituents(min_cap: float, small_player:bool) -> List[str]:
    df = _get_mktcap_ak()
    if small_player:
        cond = (
            (df["mktcap"] >= min_cap)
            & ~df["code"].str.startswith(("300", "301", "688", "8", "4"))
        )
    else:
        cond = (
            (df["mktcap"] >= min_cap)            
        )
    codes = df.loc[cond, "code"].str.zfill(6).tolist()
    with open("appendix.json", 'r', encoding='utf-8') as f:
        appendix_codes = json.load(f)["data"]
    appendix_codes = [c for c in appendix_codes if c not in codes]
    codes = appendix_codes+codes
    logger.info("AKShare 筛选市值≥ %.0f 亿+自选股票共 %d 只", (min_cap / 1e8), len(codes))
    return codes

# ---- 历史 K 线 ---- #

def get_kline(
    code: str,
    start: str,
    end: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    返回字段：date, open, close, high, low, volume, amount, turnover
    """
    raw = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start,
        end_date=end,
        adjust=adjust,
    )
    if raw.empty:
        return raw

    df = (
        raw[list(COLUMN_MAP_HIST)]
        .rename(columns=COLUMN_MAP_HIST)  # type: ignore
        .assign(date=lambda x: pd.to_datetime(x["date"]))
    )
    # 数值列统一转 float/int，方便后续处理
    numeric_cols = [c for c in df.columns if c != "date"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


# ---- 将快照写入 CSV ---- #

def append_today_row(row: pd.Series, out_dir: Path, force_update: bool = False):
    """若缺失今日记录或 force_update=True，则以快照数据写入/覆盖"""
    code = row["code"]
    csv_file = out_dir / f"{code}.csv"
    if not csv_file.exists():
        return

    df = pd.read_csv(csv_file, parse_dates=["date"])
    today = pd.Timestamp.today().normalize()

    exists_today = (df["date"] == today).any()
    if exists_today and not force_update:
        return  # 已有且不需强制刷新

    if exists_today:
        df = df[df["date"] != today]  # 删除旧行

    new_row = {
        "date": today,
        "open": row["open"],
        "high": row["high"],
        "low": row["low"],
        "close": row["close"],
        "volume": row["volume"],
        "amount": row["amount"],
        "turnover": row["turnover"],
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_file, index=False)
    logger.debug("%s 已写入今日快照", code)


DATA_SOURCE = "akshare"

# --------------------------- 工具函数 --------------------------- #

def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():  # type: ignore
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():  # type: ignore
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df


def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
    incremental: bool,
):
    """抓取单只股票 K 线并写 CSV"""
    csv_path = out_dir / f"{code}.csv"

    # ---------- ① 计算增量起始日期 ----------
    if incremental and csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["date"])
            last_date = existing["date"].max() + pd.Timedelta(days=1)
            if last_date > pd.to_datetime(end):
                # 历史已最新，但仍需后续刷新今日
                start = end  # 让 hist 查询范围缩到今日
            else:
                start = last_date.strftime("%Y%m%d")
        except Exception:
            logger.exception("读取 %s 失败，将重新下载", csv_path)

    # ---------- ② 抓取历史 ----------
    for attempt in range(1, 4):
        try:
            new_df = get_kline(code, start, end)
            if new_df.empty:
                break

            new_df = validate(new_df)

            # ---------- ③ 合并历史 ----------
            if csv_path.exists() and not new_df.empty:
                old_df = pd.read_csv(csv_path, parse_dates=["date"]) if incremental else pd.DataFrame()
                new_df = (
                    pd.concat([old_df, new_df], ignore_index=True)
                    .drop_duplicates(subset="date")
                    .sort_values("date")
                )

            new_df.to_csv(csv_path, index=False)
            break
        except Exception:
            logger.exception("%s 第 %d 次抓取失败", code, attempt)
            time.sleep(2)
    else:
        logger.error("%s 三次抓取均失败，已跳过！", code)


# --------------------------- 主函数 --------------------------- #

def main():
    parser = argparse.ArgumentParser(description="按市值筛选 A 股并抓取历史 K 线")
    parser.add_argument("--small-player", type=bool, default=True, help="为True则不获取创业板股票")
    parser.add_argument("--min-mktcap", type=float, default=2.5e10, help="最小总市值，默认2.5e10")
    parser.add_argument("--start", default="20250101", help="起始日期 YYYYMMDD (默认 20050101)")
    parser.add_argument("--end", default="today", help="结束日期 YYYY-MM-DD 或 'today' (默认 today)")
    parser.add_argument("--out", default="./data", help="输出目录 (默认 ./data)")
    parser.add_argument("--workers", type=int, default=20, help="并发线程数 (默认 20)")    
    args = parser.parse_args()

    start = args.start
    end = dt.date.today().strftime("%Y%m%d") if args.end.lower() == "today" else args.end

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "数据源: %s | 市值阈值: ≥%.0f 亿 | 日期范围: %s → %s | 输出目录: %s",
        DATA_SOURCE,
        (args.min_mktcap / 1e8),
        start,
        end,
        out_dir.resolve(),
    )

    codes = get_constituents(args.min_mktcap, args.small_player)
    if not codes:
        logger.error("筛选结果为空，请降低 --min-mktcap 阈值或检查数据源！")
        sys.exit(1)

    # ---------- 抓取历史 ----------
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(fetch_one, code, start, end, out_dir, True)
            for code in codes
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            pass
    
    logger.info("全部任务完成，数据已保存至 %s", out_dir.resolve())


if __name__ == "__main__":
    main()
